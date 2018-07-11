import numpy as np
import scipy.sparse as sparse
from itertools import count
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from gridlod import interp, coef, util, fem, world, linalg, helmholtz, transport, pg_helmholtz
from gridlod.world import World

def test_periodic_1d():
    # Example from Peterseim, Variational Multiscale Stabilization and the Exponential Decay of correctors, p. 2
    # Two modifications: A with minus and u(here) = 1/4*u(paper).
    NFine = np.array([200, 200])
    NList = [10, 20]
    epsilon = 1. / 32
    k = 2

    pi = np.pi

    xt = util.tCoordinates(NFine)
    xt_xcoord = xt[:, 0]
    xp = util.pCoordinates(NFine)
    xp_xcoord = xp[:, 0]
    aFine = (2 - np.cos(2 * pi * xt_xcoord / epsilon)) ** (-1)

    uSol = 4 * (xp_xcoord - xp_xcoord ** 2) - 4 * epsilon * (1 / (4 * pi) * np.sin(2 * pi * xp_xcoord / epsilon) -
                                               1 / (2 * pi) * xp_xcoord * np.sin(2 * pi * xp_xcoord / epsilon) -
                                               epsilon / (4 * pi ** 2) * np.cos(2 * pi * xp_xcoord / epsilon) +
                                               epsilon / (4 * pi ** 2))

    uSol = uSol / 4

    #fig = plt.figure(figsize=plt.figaspect(0.5))
    #ax = fig.add_subplot(1, 2, 1, projection='3d')
    #ax.plot_trisurf(xp_xcoord, xp[:, 1], uSol)

    for N in NList:
        NWorldCoarse = np.array([N, N])
        NCoarseElement = NFine // NWorldCoarse
        boundaryConditions = np.array([[0, 0], [1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        NpCoarse = np.prod(NWorldCoarse + 1)

        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement,
                                                                      boundaryConditions)
        aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine)

        pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 0)
        pglod.updateCorrectors(aCoef, clearFineQuantities=False)

        KFull = pglod.assembleMsStiffnessMatrix()
        MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)

        boundaryMap = boundaryConditions == 0
        fixed = util.boundarypIndexMap(NWorldCoarse, boundaryMap)
        free = np.setdiff1d(np.arange(0, NpCoarse), fixed)

        f = np.ones(NpCoarse)
        bFull = MFull * f

        KFree = KFull[free][:, free]
        bFree = bFull[free]

        xFree = sparse.linalg.spsolve(KFree, bFree)

        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        basisCorrectors = pglod.assembleBasisCorrectors()
        modifiedBasis = basis - basisCorrectors
        xFull = np.zeros(NpCoarse)
        xFull[free] = xFree
        uLodCoarse = basis * xFull
        uLodFine = modifiedBasis * xFull

        AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aFine)
        MFine = fem.assemblePatchMatrix(NFine, world.MLocFine)

        ErrorCoarse = np.sqrt(np.dot(uSol - uLodCoarse, MFine * (uSol - uLodCoarse)))
        ErrorFine = np.sqrt(np.dot(uSol - uLodFine, AFine * (uSol - uLodFine)))

        print("Coarse error: " + str(ErrorCoarse))
        print("Fine error: " + str(ErrorFine))

        #fig = plt.figure(figsize=plt.figaspect(0.5))
        #ax = fig.add_subplot(1, 2, 1, projection='3d')
        #ax.plot_trisurf(xp_xcoord, xp[:, 1], uLodCoarse)
        #plt.show()

def test_helmholtz():
    NFine = np.array([128, 128])
    NpFine = np.prod(NFine + 1)
    NtFine = np.prod(NFine)
    NWorldCoarse = np.array([16, 16])
    NCoarseElement = NFine // NWorldCoarse

    boundaryConditions = np.array([[1, 1],
                                   [1, 1]])
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    aBase = np.ones(NtFine)
    aBase.flatten()
    aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase)

    wavenumber = 20
    wavenumberBase_neg_squared = -1 * wavenumber**2 * np.ones(NtFine)
    wavenumberBase_neg_squared.flatten()
    wavenumberBase_neg_complex = -1 * 1j * wavenumber * np.ones(NtFine)
    wavenumberBase_neg_complex.flatten()
    waveCoeff_neg_squared = coef.coefficientFine(NWorldCoarse, NCoarseElement, wavenumberBase_neg_squared)
    waveCoeff_neg_complex = coef.coefficientFine(NWorldCoarse, NCoarseElement, wavenumberBase_neg_complex)

    coords = util.pCoordinates(NFine)
    xC = coords[:, 0]
    yC = coords[:, 1]
    g = np.zeros(NpFine, dtype=complex)
    #bottom boundary (?!)
    g[0:NFine[0]+1] = -1 * 1j * wavenumber * np.exp(-1 * 1j* wavenumber * xC[0:NFine[0]+1])
    # top boundary (?!)
    g[NpFine-NFine[0]-1:] = -1 * 1j * wavenumber * np.exp(-1 * 1j * wavenumber * xC[NpFine-NFine[0]-1:NpFine])
    #right boundary
    g[NFine[0]:NpFine:NFine[0]+1] = -2 * 1j * wavenumber * np.exp(-1 * 1j* wavenumber * xC[NFine[0]:NpFine:NFine[0]+1])

    uFine, _, _, _ = helmholtz.solveFine(world, aCoef.aFine, waveCoeff_neg_squared.aFine,
                                         waveCoeff_neg_complex.aFine,
                                         None, None, g, boundaryConditions)
    grid = uFine.reshape(NFine+1, order='C')

    plt.imshow(grid.real, extent=(xC.min(), xC.max(), yC.min(), yC.max()), cmap=plt.cm.hot)
    plt.colorbar()
    plt.show()

def test_helmholtz_lod():
    NFine = np.array([64, 64, 64])
    NpFine = np.prod(NFine + 1)
    NtFine = np.prod(NFine)
    NWorldCoarse = np.array([8, 8, 8])
    NpCoarse = np.prod(NWorldCoarse+1)
    NtCoarse = np.prod(NWorldCoarse)
    NCoarseElement = NFine // NWorldCoarse

    boundaryConditions = np.array([[1, 1],
                                   [1, 1],
                                   [1, 1]])
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    aBase = np.ones(NtFine)
    aBase.flatten()
    aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase)

    wavenumber = 2
    wavenumberBase_neg_squared = -1 * wavenumber**2 * np.ones(NtFine)
    wavenumberBase_neg_squared.flatten()
    wavenumberBase_neg_complex = -1 * 1j * wavenumber * np.ones(NtFine)
    wavenumberBase_neg_complex.flatten()
    waveCoeff_neg_squared = coef.coefficientFine(NWorldCoarse, NCoarseElement, wavenumberBase_neg_squared)
    waveCoeff_neg_complex = coef.coefficientFine(NWorldCoarse, NCoarseElement, wavenumberBase_neg_complex)


    coordsCorase = util.pCoordinates(NWorldCoarse)
    xcC = coordsCorase[:, 0]
    ycC = coordsCorase[:, 1]

    #bdry for LOD
    g_wo_k = np.zeros(NpCoarse, dtype=complex)
    # bottom boundary (?!)
    g_wo_k[0:NWorldCoarse[0] + 1] = -1 * 1j * np.exp(-1 * 1j * wavenumber * xcC[0:NWorldCoarse[0] + 1])
    # top boundary (?!)
    g_wo_k[NpCoarse - NWorldCoarse[0] - 1:] = -1 * 1j * np.exp(
        -1 * 1j * wavenumber * xcC[NpCoarse - NWorldCoarse[0] - 1:NpCoarse])
    # right boundary
    g_wo_k[NWorldCoarse[0]:NpCoarse:NWorldCoarse[0] + 1] = -2 * 1j * np.exp(
        -1 * 1j * wavenumber * xcC[NWorldCoarse[0]:NpCoarse:NWorldCoarse[0] + 1])

    IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement,
                                                                  boundaryConditions)
    k=2

    pglod = pg_helmholtz.PetrovGalerkinLOD(world, k, IPatchGenerator)
    pglod.updateCorrectors(aCoef, waveCoeff_neg_squared, waveCoeff_neg_complex, clearFineQuantities=True)
    uLODx, _ = pglod.solve(None, g_wo_k, boundaryConditions)

    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    #basisCorrectors = pglod.assembleBasisCorrectors()
    #modifiedBasis = basis - basisCorrectors

    #uLODcoarse = basis * uLODx
    #uLODfine = modifiedBasis * uLODx

    uLODfine = basis * uLODx - pglod.computeCorrection(basis * uLODx, basis * uLODx, basis * uLODx)


    del basis, pglod

    # uLODcoarsegrid = uLODcoarse.reshape(NFine+1)
    #uLODfinegrid = uLODfine.reshape(NFine + 1)

    coordsFine = util.pCoordinates(NFine)
    xfC = coordsFine[:, 0]
    yfC = coordsFine[:, 1]
    #bdry for Helmholtz
    g_fine = np.zeros(NpFine, dtype=complex)
    # bottom boundary (?!)
    g_fine[0:NFine[0] + 1] = -1 * wavenumber * 1j * np.exp(-1 * 1j * wavenumber * xfC[0:NFine[0] + 1])
    # top boundary (?!)
    g_fine[NpFine - NFine[0] - 1:] = -1 * wavenumber * 1j * np.exp(
        -1 * 1j * wavenumber * xfC[NpFine - NFine[0] - 1:NpFine])
    # right boundary
    g_fine[NFine[0]:NpFine:NFine[0] + 1] = -2 * wavenumber * 1j * np.exp(
        -1 * 1j * wavenumber * xfC[NFine[0]:NpFine:NFine[0] + 1])

    #reference solution
    uFine, AFine, MFine, _ = helmholtz.solveFine(world, aCoef.aFine, waveCoeff_neg_squared.aFine,
                                                 waveCoeff_neg_complex.aFine,
                                                 None, None, g_fine, boundaryConditions)
    uFineGrid = uFine.reshape(NFine + 1, order='C')
    MFine /= wavenumber ** 2

    errorL2 = np.sqrt(np.dot(MFine * (uFine - uLODfine), (uFine - uLODfine).conj()))

    errorH1semi = np.sqrt(np.dot(AFine * (uFine - uLODfine), (uFine - uLODfine).conj()))
    print(errorL2)
    print(errorH1semi)

    del MFine, AFine

    #fig = plt.figure()
    #ax1 = fig.add_subplot(1, 2, 1)
    #ax2 = fig.add_subplot(1, 2, 2)

    #im1 = ax1.imshow(uLODfinegrid.real, extent=(xcC.min(), xcC.max(), ycC.min(), ycC.max()), cmap=plt.cm.hot)
    #fig.colorbar(im1, ax=ax1)

    #im2 = ax2.imshow(uFineGrid.real, extent=(xcC.min(), xcC.max(), ycC.min(), ycC.max()), cmap=plt.cm.hot)
    #fig.colorbar(im2, ax=ax2)
    #plt.show()

def test_helmholtz_het_lod():
    NFine = np.array([512, 512])
    NpFine = np.prod(NFine + 1)
    NtFine = np.prod(NFine)
    #NList = [16, 32, 64, 128]
    NList = [128]

    scatterer_left = np.array([0.25, 0.0])
    scatterer_right = np.array([0.75, 1.0])
    inclusion_left = np.array([0.0, 0.25])
    inclusion_right = np.array([1.0, 0.75])
    delta = 1./16.
    inclusions = np.array((scatterer_right-scatterer_left)/delta, dtype=int)
    except_row = 8

    aBaseSquare = np.ones(NFine, dtype=complex)
    aBaseSquare[int(scatterer_left[1] * NFine[1]):int(scatterer_right[1] * NFine[1]),
          int(scatterer_left[0] * NFine[0]):int(scatterer_right[0] * NFine[0])] = 10
    for ii in range(inclusions[0]):
        for jj in range(inclusions[1]):
            if jj != except_row:
                startindexcols = int((scatterer_left[0] + delta * (ii + inclusion_left[0])) * NFine[0])
                stopindexcols = int((scatterer_left[0] + delta * (ii + inclusion_right[0])) * NFine[0])
                startindexrows = int((scatterer_left[1] + delta * (jj + inclusion_left[1])) * NFine[1])
                stopindexrows = int((scatterer_left[1] + delta * (jj + inclusion_right[1])) * NFine[1])
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = (delta **2) * (10-0.01j)
            else:
                startindexcols = int((scatterer_left[0] + delta * (ii + inclusion_left[0])) * NFine[0])
                stopindexcols = int((scatterer_left[0] + delta * (ii + inclusion_right[0])) * NFine[0])
                startindexrows = int((scatterer_left[1] + delta * (jj + 0.375)) * NFine[1])
                stopindexrows = int((scatterer_left[1] + delta * (jj + 0.625)) * NFine[1])
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = (delta ** 2) * (10 - 0.01j)

    aBase = aBaseSquare.ravel()

    wavenumber = 29
    wavenumberBase_neg_squared = -1 * wavenumber**2 * np.ones(NtFine)
    wavenumberBase_neg_squared.flatten()
    wavenumberBase_neg_complex = -1 * 1j * wavenumber * np.ones(NtFine)
    wavenumberBase_neg_complex.flatten()

    coordsFine = util.pCoordinates(NFine)
    xfC = coordsFine[:, 0]
    yfC = coordsFine[:, 1]

    #aBaseGrid=aBase.reshape(NFine)
    plt.imshow(aBaseSquare.real, extent=(xfC.min(), xfC.max(), yfC.min(), yfC.max()), cmap=plt.cm.hot)
    plt.show()

    g_fine = np.zeros(NpFine, dtype=complex)
    # bottom boundary (?!)
    g_fine[0:NFine[0] + 1] = -1 * wavenumber * 1j * np.exp(-1 * 1j * wavenumber * xfC[0:NFine[0] + 1])
    # top boundary (?!)
    g_fine[NpFine - NFine[0] - 1:] = -1 * wavenumber * 1j * np.exp(-1 * 1j * wavenumber * xfC[NpFine - NFine[0] - 1:NpFine])
    # right boundary
    g_fine[NFine[0]:NpFine:NFine[0] + 1] = -2 * wavenumber * 1j * np.exp(-1 * 1j * wavenumber * xfC[NFine[0]:NpFine:NFine[0] + 1])

    for N in NList:
        NWorldCoarse = np.array([N,N])
        NpCoarse = np.prod(NWorldCoarse+1)
        NCoarseElement = NFine // NWorldCoarse

        boundaryConditions = np.array([[1, 1],
                                      [1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase)
        waveCoeff_neg_squared = coef.coefficientFine(NWorldCoarse, NCoarseElement, wavenumberBase_neg_squared)
        waveCoeff_neg_complex = coef.coefficientFine(NWorldCoarse, NCoarseElement, wavenumberBase_neg_complex)

        coordsCoarse = util.pCoordinates(NWorldCoarse)
        xcC = coordsCoarse[:, 0]
        ycC = coordsCoarse[:, 1]

        g_wo_k = np.zeros(NpCoarse, dtype=complex)
        #bottom boundary (?!)
        g_wo_k[0:NWorldCoarse[0]+1] = -1 * 1j * np.exp(-1 * 1j* wavenumber * xcC[0:NWorldCoarse[0]+1])
        # top boundary (?!)
        g_wo_k[NpCoarse-NWorldCoarse[0]-1:] = -1 * 1j * np.exp(-1 * 1j * wavenumber * xcC[NpCoarse-NWorldCoarse[0]-1:NpCoarse])
        #right boundary
        g_wo_k[NWorldCoarse[0]:NpCoarse:NWorldCoarse[0]+1] = -2 * 1j * np.exp(-1 * 1j* wavenumber * xcC[NWorldCoarse[0]:NpCoarse:NWorldCoarse[0]+1])

        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement,
                                                                  boundaryConditions)
        k=2

        #LOD
        pglod = pg_helmholtz.PetrovGalerkinLOD(world, k, IPatchGenerator)
        pglod.updateCorrectors(aCoef, waveCoeff_neg_squared, waveCoeff_neg_complex, clearFineQuantities=True)
        uLODx, _ = pglod.solve(None, g_wo_k, boundaryConditions)

        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        #modifiedBasis = basis - pglod.assembleBasisCorrectors()

        #uLODcoarse = basis * uLODx
        #uLODfine = modifiedBasis * uLODx
        uLODfine = basis * uLODx - pglod.computeCorrection(basis * uLODx, basis * uLODx, basis * uLODx)

        del basis, pglod

        #uLODcoarsegrid = uLODcoarse.reshape(NFine+1)
        uLODfinegrid = uLODfine.reshape(NFine + 1)


        #reference solution
        uFine, AFine, MFine, _ = helmholtz.solveFine(world, aCoef.aFine, waveCoeff_neg_squared.aFine,
                                             waveCoeff_neg_complex.aFine,
                                             None, None, g_fine, boundaryConditions)
        uFineGrid = uFine.reshape(NFine + 1, order='C')
        MFine /= wavenumber **2

        #MFineL2 = fem.assemblePatchMatrix(NFine, world.MLocFine)
        errorL2 = np.sqrt(np.dot(MFine*(uFine - uLODfine), (uFine - uLODfine).conj()))

        #AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aCoef.aFine)
        errorH1semi = np.sqrt(np.dot(AFine * (uFine - uLODfine), (uFine - uLODfine).conj()))
        print(errorL2)
        print(errorH1semi)

        if N == 128:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)

            im1 = ax1.imshow(uLODfinegrid.real, extent=(xcC.min(), xcC.max(), ycC.min(), ycC.max()), cmap=plt.cm.hot)
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(uFineGrid.real, extent=(xcC.min(), xcC.max(), ycC.min(), ycC.max()), cmap=plt.cm.hot)
            fig.colorbar(im2, ax=ax2)
            plt.show()



if __name__ == '__main__':
    test_helmholtz_het_lod()
