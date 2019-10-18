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
    NFine = np.array([256, 256])
    NpFine = np.prod(NFine + 1)
    NtFine = np.prod(NFine)
    NWorldCoarse = np.array([8, 8])
    NpCoarse = np.prod(NWorldCoarse+1)
    NtCoarse = np.prod(NWorldCoarse)
    NCoarseElement = NFine // NWorldCoarse

    boundaryConditions = np.array([[1, 1],
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
    #NList = [4, 8, 16, 32, 64]
    NList = [64]

    scatterer_left = np.array([0.0, 0.0])
    scatterer_right = np.array([1.0, 1.0])
    inclusion_left = np.array([0.25, 0.25])
    inclusion_right = np.array([0.75, 0.75])
    delta = 1./8.
    inclusions = np.array((scatterer_right-scatterer_left)/delta, dtype=int)
    except_row = 4

    eps_matrix = 1
    eps_incl = 1

    aBaseSquare = np.ones(NFine, dtype=complex)
    aBaseSquare[int(scatterer_left[1] * NFine[1]):int(scatterer_right[1] * NFine[1]),
          int(scatterer_left[0] * NFine[0]):int(scatterer_right[0] * NFine[0])] = eps_matrix
    for ii in range(inclusions[0]):
        for jj in range(inclusions[1]):
            if jj != except_row:
                startindexcols = int((scatterer_left[0] + delta * (ii + inclusion_left[0])) * NFine[0])
                stopindexcols = int((scatterer_left[0] + delta * (ii + inclusion_right[0])) * NFine[0])
                startindexrows = int((scatterer_left[1] + delta * (jj + inclusion_left[1])) * NFine[1])
                stopindexrows = int((scatterer_left[1] + delta * (jj + inclusion_right[1])) * NFine[1])
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = (delta **2) * (eps_incl)
            else:
                startindexcols = int((scatterer_left[0] + delta * (ii + inclusion_left[0])) * NFine[0])
                stopindexcols = int((scatterer_left[0] + delta * (ii + inclusion_right[0])) * NFine[0])
                startindexrows = int((scatterer_left[1] + delta * (jj + inclusion_left[1])) * NFine[1])
                stopindexrows = int((scatterer_left[1] + delta * (jj + inclusion_right[0])) * NFine[1])
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = eps_matrix#(delta ** 2) * (eps_incl)

    aBase = aBaseSquare.ravel()

    #wavenumber = 29
    wavenumber = 9
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

        '''g_wo_k = np.zeros(NpCoarse, dtype=complex)
        #bottom boundary (?!)
        g_wo_k[0:NWorldCoarse[0]+1] = -1 * 1j * np.exp(-1 * 1j* wavenumber * xcC[0:NWorldCoarse[0]+1])
        # top boundary (?!)
        g_wo_k[NpCoarse-NWorldCoarse[0]-1:] = -1 * 1j * np.exp(-1 * 1j * wavenumber * xcC[NpCoarse-NWorldCoarse[0]-1:NpCoarse])
        #right boundary
        g_wo_k[NWorldCoarse[0]:NpCoarse:NWorldCoarse[0]+1] = -2 * 1j * np.exp(-1 * 1j* wavenumber * xcC[NWorldCoarse[0]:NpCoarse:NWorldCoarse[0]+1])

        IPatchGenerator = lambda i, N: interp.weightedL2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, aCoef,
                                                                  boundaryConditions)
        k=2

        #LOD
        pglod = pg_helmholtz.PetrovGalerkinLOD(world, k, IPatchGenerator)
        pglod.updateCorrectors(aCoef, waveCoeff_neg_squared, waveCoeff_neg_complex, clearFineQuantities=True)
        uLODx, _ = pglod.solve(None, g_wo_k, boundaryConditions)

        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        #modifiedBasis = basis - pglod.assembleBasisCorrectors()

        uLODcoarse = basis * uLODx
        #uLODfine = modifiedBasis * uLODx
        uLODfine = basis * uLODx - pglod.computeCorrection(basis * uLODx, basis * uLODx, basis * uLODx)

        del basis, pglod

        #uLODcoarsegrid = uLODcoarse.reshape(NFine+1)
        uLODfinegrid = uLODfine.reshape(NFine + 1)
        '''

        #reference solution
        uFine, AFine, MFine, _ = helmholtz.solveFine(world, aCoef.aFine, waveCoeff_neg_squared.aFine,
                                             waveCoeff_neg_complex.aFine,
                                             None, None, g_fine, boundaryConditions)
        uFineGrid = uFine.reshape(NFine + 1, order='C')
        MFine /= wavenumber **2

        '''MFineL2Aweighted = fem.assemblePatchMatrix(NFine, world.MLocFine, aCoef.aFine)
        errorL2 = np.sqrt(np.dot(MFine*(uFine - uLODfine), (uFine - uLODfine).conj()))
        errorL2Aweighted = np.sqrt(np.dot(MFineL2Aweighted * (uFine - uLODfine), (uFine - uLODfine).conj()))

        #AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aCoef.aFine)
        errorH1semi = np.sqrt(np.dot(AFine * (uFine - uLODfine), (uFine - uLODfine).conj()))
        print('error to uLODfine in L2: ', errorL2)
        print('error to uLODfine in weighted L2: ', errorL2Aweighted)
        print('error to uLODfine in weighted H1 semi: ', errorH1semi)

        coarseerrorL2 = np.sqrt(np.dot(MFine * (uFine - uLODcoarse), (uFine - uLODcoarse).conj()))
        coarseerrorL2Aweighted = np.sqrt(np.dot(MFineL2Aweighted * (uFine - uLODcoarse), (uFine - uLODcoarse).conj()))

        coarseerrorH1semi = np.sqrt(np.dot(AFine * (uFine - uLODcoarse), (uFine - uLODcoarse).conj()))
        print('error to uLODcoarse in L2: ', coarseerrorL2)
        print('error to uLODcoarse in weighted L2: ', coarseerrorL2Aweighted)
        print('error to uLODcoarse in weighted H1 semi: ', coarseerrorH1semi)
        '''
        if N == 64:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)

            #im1 = ax1.imshow(uLODfinegrid.real, extent=(xcC.min(), xcC.max(), ycC.min(), ycC.max()), cmap=plt.cm.hot)
            #fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(uFineGrid.real, extent=(xcC.min(), xcC.max(), ycC.min(), ycC.max()), cmap=plt.cm.hot)
            fig.colorbar(im2, ax=ax2)
            plt.show()

def test_helmholtz_het_1d_lod():
    NFine = np.array([1024])
    NpFine = np.prod(NFine + 1)
    NtFine = np.prod(NFine)
    #NList = [4, 8, 16, 32, 64, 128]
    NList = [128]

    scatterer_left = 0.25
    scatterer_right = 0.75
    inclusion_left = 0.25
    inclusion_right = 0.75
    delta = 1./8.
    inclusions = int((scatterer_right-scatterer_left)/delta)
    except_row = 2

    eps_matrix = 1
    eps_incl = 1

    aBase = np.ones(NFine, dtype=complex)
    aBase[int(scatterer_left * NFine[0]):int(scatterer_right * NFine[0])] = eps_matrix
    for ii in range(inclusions):
        if ii != except_row:
            startindexcols = int((scatterer_left + delta * (ii + inclusion_left)) * NFine[0])
            stopindexcols = int((scatterer_left + delta * (ii + inclusion_right)) * NFine[0])
            aBase[startindexcols:stopindexcols] = (delta **2) * eps_incl
        else:
            startindexcols = int((scatterer_left + delta * (ii + inclusion_left)) * NFine[0])
            stopindexcols = int((scatterer_left + delta * (ii + inclusion_right)) * NFine[0])
            aBase[startindexcols:stopindexcols] = eps_matrix


    wavenumber = 16
    wavenumberBase_neg_squared = -1 * wavenumber**2 * np.ones(NtFine)
    wavenumberBase_neg_squared.flatten()
    wavenumberBase_neg_complex = -1 * 1j * wavenumber * np.ones(NtFine)
    wavenumberBase_neg_complex.flatten()

    coordsFine = util.pCoordinates(NFine)
    tFine = util.tCoordinates(NFine)
    plt.plot(tFine, aBase.real)
    plt.show()

    #plt.imshow(aBase.real, extent=(coordsFine.min(), coordsFine.max(), coordsFine.min(), coordsFine.max()), cmap=plt.cm.hot)
    #plt.show()

    g_fine = np.zeros(NpFine, dtype=complex)
    # right boundary
    g_fine[NpFine-1] = -2 * wavenumber * 1j * np.exp(-1 * 1j * wavenumber * coordsFine[NpFine-1])
    #plt.plot(coordsFine, g_fine)
    #plt.show()

    for N in NList:
        NWorldCoarse = np.array([N])
        NpCoarse = np.prod(NWorldCoarse+1)
        NCoarseElement = NFine // NWorldCoarse

        boundaryConditions = np.array([[1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase)
        waveCoeff_neg_squared = coef.coefficientFine(NWorldCoarse, NCoarseElement, wavenumberBase_neg_squared)
        waveCoeff_neg_complex = coef.coefficientFine(NWorldCoarse, NCoarseElement, wavenumberBase_neg_complex)

        coordsCoarse = util.pCoordinates(NWorldCoarse)

        '''g_wo_k = np.zeros(NpCoarse, dtype=complex)
        #right boundary
        g_wo_k[NpCoarse-1] = -2 * 1j * np.exp(-1 * 1j* wavenumber * coordsCoarse[NpCoarse-1])

        IPatchGenerator = lambda i, N: interp.weightedL2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, aCoef,
                                                                  boundaryConditions)
        k=3

        #LOD
        pglod = pg_helmholtz.PetrovGalerkinLOD(world, k, IPatchGenerator)
        pglod.updateCorrectors(aCoef, waveCoeff_neg_squared, waveCoeff_neg_complex, clearFineQuantities=True)
        uLODx, _ = pglod.solve(None, g_wo_k, boundaryConditions)

        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        #modifiedBasis = basis - pglod.assembleBasisCorrectors()

        uLODcoarse = basis * uLODx
        #uLODfine = modifiedBasis * uLODx
        uLODfine = basis * uLODx - pglod.computeCorrection(basis * uLODx, basis * uLODx, basis * uLODx)

        del basis, pglod'''

        #uLODcoarsegrid = uLODcoarse.reshape(NFine+1)
        #uLODfinegrid = uLODfine.reshape(NFine + 1)


        #reference solution
        uFine, AFine, MFine, _ = helmholtz.solveFine(world, aCoef.aFine, waveCoeff_neg_squared.aFine,
                                             waveCoeff_neg_complex.aFine,
                                             None, None, g_fine, boundaryConditions)
        #uFineGrid = uFine.reshape(NFine + 1, order='C')
        '''MFine /= wavenumber **2

        MFineL2Aweighted = fem.assemblePatchMatrix(NFine, world.MLocFine, aCoef.aFine)
        errorL2 = np.sqrt(np.dot(MFine*(uFine - uLODfine), (uFine - uLODfine).conj()))
        errorL2Aweighted = np.sqrt(np.dot(MFineL2Aweighted * (uFine - uLODfine), (uFine - uLODfine).conj()))

        #AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aCoef.aFine)
        errorH1semi = np.sqrt(np.dot(AFine * (uFine - uLODfine), (uFine - uLODfine).conj()))
        print('error to uLODfine in L2: ', errorL2)
        print('error to uLODfine in weighted L2: ', errorL2Aweighted)
        print('error to uLODfine in weighted H1 semi: ', errorH1semi)

        coarseerrorL2 = np.sqrt(np.dot(MFine * (uFine - uLODcoarse), (uFine - uLODcoarse).conj()))
        coarseerrorL2Aweighted = np.sqrt(np.dot(MFineL2Aweighted * (uFine - uLODcoarse), (uFine - uLODcoarse).conj()))

        coarseerrorH1semi = np.sqrt(np.dot(AFine * (uFine - uLODcoarse), (uFine - uLODcoarse).conj()))
        print('error to uLODcoarse in L2: ', coarseerrorL2)
        print('error to uLODcoarse in weighted L2: ', coarseerrorL2Aweighted)
        print('error to uLODcoarse in weighted H1 semi: ', coarseerrorH1semi)'''

        #if N == 128:
        fig = plt.figure()
        ax1 = fig.add_subplot(1,2,1)
        ax2 = fig.add_subplot(1,2,2)

        #ax1.plot(coordsFine, uLODfine.real)

        ax2.plot(coordsFine, uFine.real)
        plt.show()

def test_elliptic_het_lod():
    NFine = np.array([512, 512])
    NpFine = np.prod(NFine + 1)
    NtFine = np.prod(NFine)
    NList = [4, 8, 16, 32, 64, 128]
    #NList = [128]

    #the diffusion parameter
    scatterer_left = np.array([0.25, 0.25])
    scatterer_right = np.array([0.75, 0.75])
    inclusion_left = np.array([0.25, 0.25])
    inclusion_right = np.array([0.75, 0.75])
    delta = 1./128.
    inclusions = np.array((scatterer_right-scatterer_left)/delta, dtype=int)
    except_row = 1000

    eps_matrix = 1
    eps_incl = 1

    aBaseSquare = np.ones(NFine, dtype=complex)
    aBaseSquare[int(scatterer_left[1] * NFine[1]):int(scatterer_right[1] * NFine[1]),
          int(scatterer_left[0] * NFine[0]):int(scatterer_right[0] * NFine[0])] = eps_matrix
    for ii in range(inclusions[0]):
        for jj in range(inclusions[1]):
            if jj != except_row:
                startindexcols = int((scatterer_left[0] + delta * (ii + inclusion_left[0])) * NFine[0])
                stopindexcols = int((scatterer_left[0] + delta * (ii + inclusion_right[0])) * NFine[0])
                startindexrows = int((scatterer_left[1] + delta * (jj + inclusion_left[1])) * NFine[1])
                stopindexrows = int((scatterer_left[1] + delta * (jj + inclusion_right[1])) * NFine[1])
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = (delta **2) * (eps_incl)
            else:
                startindexcols = int((scatterer_left[0] + delta * (ii + inclusion_left[0])) * NFine[0])
                stopindexcols = int((scatterer_left[0] + delta * (ii + inclusion_right[0])) * NFine[0])
                startindexrows = int((scatterer_left[1] + delta * (jj + 0.375)) * NFine[1])
                stopindexrows = int((scatterer_left[1] + delta * (jj + 0.625)) * NFine[1])
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = (delta ** 2) * (eps_incl)

    aBase = aBaseSquare.ravel()

    #lower order term artifically as complex wavenumber
    wavenumber = 1j
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


    #volume term
    #f = lambda x: (1.0+0j)*np.ones(shape=x.shape())
    ffine = np.ones(NpFine)
    #ffineSquare = np.zeros(NFine+1)
    #ffineSquare[int(0*NFine[1]):int(0.25*NFine[1]), int(0*NFine[0]):int(0.25*NFine[0])]=1.0
    #ffine = ffineSquare.ravel()
    #plt.imshow(ffineSquare, extent=(xfC.min(), xfC.max(), yfC.min(), yfC.max()), cmap=plt.cm.hot)
    #plt.show()
    '''g_fine = np.zeros(NpFine, dtype=complex)
    # bottom boundary (?!)
    g_fine[0:NFine[0] + 1] = -1 * wavenumber * 1j * np.exp(-1 * 1j * wavenumber * xfC[0:NFine[0] + 1])
    # top boundary (?!)
    g_fine[NpFine - NFine[0] - 1:] = -1 * wavenumber * 1j * np.exp(-1 * 1j * wavenumber * xfC[NpFine - NFine[0] - 1:NpFine])
    # right boundary
    g_fine[NFine[0]:NpFine:NFine[0] + 1] = -2 * wavenumber * 1j * np.exp(-1 * 1j * wavenumber * xfC[NFine[0]:NpFine:NFine[0] + 1]) '''

    for N in NList:
        NWorldCoarse = np.array([N,N])
        NpCoarse = np.prod(NWorldCoarse+1)
        NCoarseElement = NFine // NWorldCoarse

        #dirichlet bdry conditions
        boundaryConditions = np.array([[0, 0], [0, 0]])
        #boundaryConditions = np.array([[1, 1],
        #                              [1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase)
        waveCoeff_neg_squared = coef.coefficientFine(NWorldCoarse, NCoarseElement, wavenumberBase_neg_squared)
        waveCoeff_neg_complex = coef.coefficientFine(NWorldCoarse, NCoarseElement, wavenumberBase_neg_complex)

        coordsCoarse = util.pCoordinates(NWorldCoarse)
        xcC = coordsCoarse[:, 0]
        ycC = coordsCoarse[:, 1]

        fcoarse = np.ones(NpCoarse)
        fcoarse = fcoarse / wavenumber**2

        #fcoarseSquare = np.zeros(NWorldCoarse + 1)
        #fcoarseSquare[int(0 * NWorldCoarse[1]):int(0.25 * NWorldCoarse[1]), int(0 * NWorldCoarse[0]):int(0.25 * NWorldCoarse[0])] = 1.0
        #fcoarseSquare = fcoarseSquare / wavenumber**2
        #fcoarse = fcoarseSquare.ravel()
        #plt.imshow(fcoarseSquare.real, extent=(xcC.min(), xcC.max(), ycC.min(), ycC.max()), cmap=plt.cm.hot)
        #plt.show()
        '''g_wo_k = np.zeros(NpCoarse, dtype=complex)
        #bottom boundary (?!)
        g_wo_k[0:NWorldCoarse[0]+1] = -1 * 1j * np.exp(-1 * 1j* wavenumber * xcC[0:NWorldCoarse[0]+1])
        # top boundary (?!)
        g_wo_k[NpCoarse-NWorldCoarse[0]-1:] = -1 * 1j * np.exp(-1 * 1j * wavenumber * xcC[NpCoarse-NWorldCoarse[0]-1:NpCoarse])
        #right boundary
        g_wo_k[NWorldCoarse[0]:NpCoarse:NWorldCoarse[0]+1] = -2 * 1j * np.exp(-1 * 1j* wavenumber * xcC[NWorldCoarse[0]:NpCoarse:NWorldCoarse[0]+1])
        '''

        IPatchGenerator = lambda i, N: interp.weightedL2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, aCoef,
                                                                  boundaryConditions)
        k=2

        #LOD
        pglod = pg_helmholtz.PetrovGalerkinLOD(world, k, IPatchGenerator)
        pglod.updateCorrectors(aCoef, waveCoeff_neg_squared, waveCoeff_neg_complex, clearFineQuantities=True)
        uLODx, _ = pglod.solve(fcoarse, None, boundaryConditions)

        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        #modifiedBasis = basis - pglod.assembleBasisCorrectors()

        uLODcoarse = basis * uLODx
        #uLODfine = modifiedBasis * uLODx
        uLODfine = basis * uLODx - pglod.computeCorrection(basis * uLODx, basis * uLODx, basis * uLODx)

        del basis, pglod

        uLODcoarsegrid = uLODcoarse.reshape(NFine+1)
        uLODfinegrid = uLODfine.reshape(NFine + 1)


        #reference solution
        uFine, AFine, MFine, _ = helmholtz.solveFine(world, aCoef.aFine, waveCoeff_neg_squared.aFine,
                                             waveCoeff_neg_complex.aFine,
                                             ffine, None, None, boundaryConditions)
        uFineGrid = uFine.reshape(NFine + 1, order='C')
        MFine /= wavenumber **2

        MFineL2Aweighted = fem.assemblePatchMatrix(NFine, world.MLocFine, aCoef.aFine)
        errorL2 = np.sqrt(np.dot(MFine*(uFine - uLODfine), (uFine - uLODfine).conj()))
        errorL2Aweighted = np.sqrt(np.dot(MFineL2Aweighted * (uFine - uLODfine), (uFine - uLODfine).conj()))

        #AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aCoef.aFine)
        errorH1semi = np.sqrt(np.dot(AFine * (uFine - uLODfine), (uFine - uLODfine).conj()))
        print('error to uLODfine in L2: ', errorL2)
        print('error to uLODfine in weighted L2: ', errorL2Aweighted)
        print('error to uLODfine in weighted H1 semi: ', errorH1semi)

        coarseerrorL2 = np.sqrt(np.dot(MFine * (uFine - uLODcoarse), (uFine - uLODcoarse).conj()))
        coarseerrorL2Aweighted = np.sqrt(np.dot(MFineL2Aweighted * (uFine - uLODcoarse), (uFine - uLODcoarse).conj()))

        coarseerrorH1semi = np.sqrt(np.dot(AFine * (uFine - uLODcoarse), (uFine - uLODcoarse).conj()))
        print('error to uLODcoarse in L2: ', coarseerrorL2)
        print('error to uLODcoarse in weighted L2: ', coarseerrorL2Aweighted)
        print('error to uLODcoarse in weighted H1 semi: ', coarseerrorH1semi)

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
    #test_elliptic_het_lod()
    test_helmholtz_het_1d_lod()