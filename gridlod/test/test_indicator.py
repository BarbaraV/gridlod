import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from gridlod import pglod, util, lod, interp, coef, fem
from gridlod.world import World, Patch

def test_indicator():
    NFine = np.array([6400])
    NpFine = np.prod(NFine+1)
    NList = [10, 20, 40, 80, 160, 320]
    epsilon = 1./640
    k = 2

    pi = np.pi

    xt = util.tCoordinates(NFine).flatten()
    xp = util.pCoordinates(NFine).flatten()
    aFine = (2 - np.cos(2*pi*xt/epsilon))**(-1)

    uSol  = 4*(xp - xp**2) - 4*epsilon*(1/(4*pi)*np.sin(2*pi*xp/epsilon) -
                                        1/(2*pi)*xp*np.sin(2*pi*xp/epsilon) -
                                        epsilon/(4*pi**2)*np.cos(2*pi*xp/epsilon) +
                                        epsilon/(4*pi**2))

    uSol = uSol/4
    f = np.ones(NpFine)

    errors = np.zeros(np.size(NList))
    indicators = np.zeros(np.size(NList))
    indicators_R = np.zeros(np.size(NList))
    indicators_Q = np.zeros(np.size(NList))
    i = 0

    for N in NList:
        #k = np.log(N)
        NWorldCoarse = np.array([N])
        NCoarseElement = NFine//NWorldCoarse
        boundaryConditions = np.array([[0, 0]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        xpCoarse = util.pCoordinates(NWorldCoarse).flatten()
        MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
        norm_of_f = [np.sqrt(np.dot(f, MFull * f))]

        def computeKmsij(TInd):
            patch = Patch(world, k, TInd)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
            aPatch = lambda: coef.localizeCoefficient(patch, aFine)

            correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
            csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
            return patch, correctorsList, csi.Kmsij, csi

        def computeLMsij(TInd):
            patch = Patch(world, k, TInd)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
            aPatch = lambda: coef.localizeCoefficient(patch, aFine)

            correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)

            lambdasList = list(patch.world.localBasis.T)

            NPatchCoarse = patch.NPatchCoarse
            NTPrime = np.prod(NPatchCoarse)
            numLambdas = len(lambdasList)

            TIndtemp = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse - 1, patch.iElementPatchCoarse)

            LTPrimeij = np.zeros((NTPrime, numLambdas, numLambdas))
            Kij = np.zeros((numLambdas, numLambdas))

            def accumulate(TPrimeInd, TPrimei, P, Q, KTPrime, BTPrimeij, CTPrimeij):
                if TPrimeInd == TIndtemp:
                    Kij[:] = np.dot(P.T, KTPrime * P)
                    LTPrimeij[TPrimeInd] = CTPrimeij \
                                           - BTPrimeij \
                                           - BTPrimeij.T \
                                           + Kij

                else:
                    LTPrimeij[TPrimeInd] = CTPrimeij

            lod.performTPrimeLoop(patch, lambdasList, correctorsList, aPatch, accumulate)

            return patch, correctorsList, LTPrimeij

        def computeRmsi(TInd):
            patch = Patch(world, k, TInd)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
            aPatch = lambda: coef.localizeCoefficient(patch, aFine)
            MRhsList = [f[util.extractElementFine(world.NWorldCoarse,
                                                      world.NCoarseElement,
                                                      patch.iElementWorldCoarse,
                                                      extractElements=False)]]

            correctorRhs = lod.computeElementCorrector(patch, IPatch, aPatch, None, MRhsList)[0]
            Rmsi, cetaTPrime = lod.computeRhsCoarseQuantities(patch, correctorRhs, aPatch, True)

            return patch, correctorRhs, Rmsi, cetaTPrime

        # Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
        patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))
        patchT, correctorRhsT, RmsiT, cetaTPrimeT = zip(*map(computeRmsi, range(world.NtCoarse)))

        KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
        RFull = pglod.assemblePatchFunction(world, patchT, RmsiT)
        #MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
        Rf = pglod.assemblePatchFunction(world, patchT, correctorRhsT)

        free  = util.interiorpIndexMap(NWorldCoarse)
        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)

        bFull = basis.T*MFull*f - RFull
        #norm_of_f = [np.sqrt(np.dot(f, MFull * f))]

        KFree = KFull[free][:,free]
        bFree = bFull[free]

        xFree = sparse.linalg.spsolve(KFree, bFree)

        basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
        modifiedBasis = basis - basisCorrectors
        xFull = np.zeros(world.NpCoarse)
        xFull[free] = xFree
        #uLodCoarse = basis*xFull
        uLodFine = modifiedBasis*xFull
        uLodFine += Rf

        AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aFine)
        #MFine = fem.assemblePatchMatrix(NFine, world.MLocFine)

        #newErrorCoarse = np.sqrt(np.dot(uSol - uLodCoarse, MFine*(uSol - uLodCoarse)))
        errors[i] = np.sqrt(np.dot(uSol - uLodFine, AFine*(uSol - uLodFine)))

        def computeIndicators(TInd):
            aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine)
            TIndinPatch = util.convertpCoordIndexToLinearIndex(patchT[TInd].NPatchCoarse - 1, patchT[TInd].iElementPatchCoarse)
            aT = aFine[util.extractElementFine(world.NWorldCoarse,
                                                        world.NCoarseElement,
                                                        patchT[TInd].iElementWorldCoarse,
                                                        extractElements=True)]

            if aT.ndim == 2:
                amaxT = np.sqrt(np.linalg.norm(aT)) #passt nicht?!
            else:
                amaxT = np.sqrt(1./np.max(np.abs(aT)))

            E_vh = np.sum(csiT[TInd].muTPrime)
            #uTprime = xFull[patchT[TInd]]  #passt nocht nicht
            #_,_,LTPrimeij = computeLMsij(TInd)
            #E_vh = np.sum(np.dot(uTprime, LTPrimeij*uTprime))

            f_patch = f[util.extractElementFine(world.NWorldCoarse,
                                                        world.NCoarseElement,
                                                        patchT[TInd].iElementWorldCoarse,
                                                        extractElements=False)]
            MElement = fem.assemblePatchMatrix(patchT[TInd].world.NCoarseElement, patchT[TInd].world.MLocFine)
            gammaT = np.dot(f_patch, MElement * f_patch)

            E_Rf = np.sum(cetaTPrimeT[TInd])+1./np.max(NWorldCoarse)**2*gammaT*amaxT**2 #weighting in A missing

            return E_vh, E_Rf

        E_vh, E_Rf = zip(*map(computeIndicators, range(world.NtCoarse)))
        indicators_Q[i] = 2*np.sqrt(np.max(E_vh))*norm_of_f[0]
        indicators_R[i] = 2*np.sqrt(np.sum(E_Rf))
        indicators[i] = indicators_Q[i] + indicators_R[i] #not efficient?!
        #indicators[i] = np.max(E_vh) + np.max(E_Rf)

        print(errors[i])
        print(indicators_Q[i]) #np.max(E_vh)
        print(indicators_R[i]) #np.max(E_Rf)
        print(indicators[i])
        print('-------------------------------')

        i+=1

    plt.figure()
    plt.plot(NList, errors, 'r', label= 'errors')
    plt.plot(NList, indicators, 'b', label = 'indicators')
    plt.plot(NList, indicators_Q, 'g', label= 'indicators Q')
    plt.plot(NList, indicators_R, 'k', label = 'indicators R')
    plt.legend()

    plt.figure()
    #plt.plot(NList, indicators/errors, 'b*', label='effectivity')
    plt.plot(NList, indicators_R/errors, 'r*', label='effectivity R')
    #plt.plot(NList, indicators_Q/errors, 'g*', label='effectivity Q')
    plt.legend()


    plt.figure()
    #plt.plot(NList, indicators/errors, 'b*', label='effectivity')
    #plt.plot(NList, indicators_R/errors, 'r*', label='effectivity R')
    plt.plot(NList, indicators_Q/errors, 'g*', label='effectivity Q')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    test_indicator()