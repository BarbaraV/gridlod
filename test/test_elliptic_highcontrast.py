import numpy as np
import scipy.sparse as sparse
from itertools import count
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


from gridlod import interp, coef, util, fem, femsolver, pg
from gridlod.world import World

def test_2d_periodic():
    NFine = np.array([1024, 1024])
    NpFine = np.prod(NFine + 1)
    NtFine = np.prod(NFine)
    NList = [4, 8, 16, 32, 64]

    #the diffusion parameter
    scatterer_left = np.array([0.25, 0.25])
    scatterer_right = np.array([0.75, 0.75])
    inclusion_left = np.array([0.25, 0.25])
    inclusion_right = np.array([0.75, 0.75])
    delta = 1./256.
    inclusions = np.array((scatterer_right-scatterer_left)/delta, dtype=int)
    except_row = 1000

    eps_matrix = 1
    eps_incl = 1

    aBaseSquare = np.ones(NFine)
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

    #lower order term
    mass = 1
    massBaseSquare = mass * np.ones(NFine)
    massBase = massBaseSquare.ravel()

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

    for N in NList:
        NWorldCoarse = np.array([N,N])
        NpCoarse = np.prod(NWorldCoarse+1)
        NCoarseElement = NFine // NWorldCoarse

        #dirichlet bdry conditions
        boundaryConditions = np.array([[0, 0], [0, 0]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase)
        massCoeff = coef.coefficientFine(NWorldCoarse, NCoarseElement, massBase)

        coordsCoarse = util.pCoordinates(NWorldCoarse)
        xcC = coordsCoarse[:, 0]
        ycC = coordsCoarse[:, 1]

        fcoarse = np.ones(NpCoarse)

        #fcoarseSquare = np.zeros(NWorldCoarse + 1)
        #fcoarseSquare[int(0 * NWorldCoarse[1]):int(0.25 * NWorldCoarse[1]), int(0 * NWorldCoarse[0]):int(0.25 * NWorldCoarse[0])] = 1.0
        #fcoarseSquare = fcoarseSquare / wavenumber**2
        #fcoarse = fcoarseSquare.ravel()
        #plt.imshow(fcoarseSquare.real, extent=(xcC.min(), xcC.max(), ycC.min(), ycC.max()), cmap=plt.cm.hot)
        #plt.show()

        IPatchGenerator = lambda i, N: interp.weightedL2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, aCoef,
                                                                  boundaryConditions)
        k=2

        #LOD
        pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 1e-6)
        pglod.updateCorrectors(aCoef, True, massCoeff)
        uLODx, _ = pglod.solve(fcoarse, None, boundaryConditions)

        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        #modifiedBasis = basis - pglod.assembleBasisCorrectors()

        uLODcoarse = basis * uLODx
        #uLODfine = modifiedBasis * uLODx
        uLODfine = basis * uLODx - pglod.computeCorrection(basis * uLODx, basis * uLODx)

        del basis, pglod

        uLODcoarsegrid = uLODcoarse.reshape(NFine+1)
        uLODfinegrid = uLODfine.reshape(NFine + 1)

        # reference solution
        uFine, AFine, MFine, _ = femsolver.solveFine(world, aCoef.aFine,
                                                     ffine, None, boundaryConditions, massCoeff.aFine) #MFine is unweighted
        uFineGrid = uFine.reshape(NFine + 1, order='C')

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

        if N == 64:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)

            im1 = ax1.imshow(uLODfinegrid.real, extent=(xcC.min(), xcC.max(), ycC.min(), ycC.max()), cmap=plt.cm.hot)
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(uFineGrid.real, extent=(xcC.min(), xcC.max(), ycC.min(), ycC.max()), cmap=plt.cm.hot)
            fig.colorbar(im2, ax=ax2)
            plt.show()




if __name__ == '__main__':
    test_2d_periodic()
