import numpy as np
import copy
import time

from gridlod import multiplecoeff, coef, interp, lod, pglod, build_coefficient
from gridlod.world import PatchPeriodic


def computeCSI_offline(world, NepsilonElement, alpha, beta, k, boundaryConditions, model):
    dim = np.size(world.NWorldFine)
    if dim == 2:
        middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
    elif dim == 1:
        middle = world.NWorldCoarse[0] //2
    patch = PatchPeriodic(world, k, middle)

    tic = time.perf_counter()
    assert(model == 'check' or model == 'incl')
    if model == 'check':
        aRefList = build_coefficient.build_checkerboardbasis(patch.NPatchCoarse, NepsilonElement, world.NCoarseElement, alpha, beta)
    else:
        aRefList = build_coefficient.build_inclusionbasis_2d(patch.NPatchCoarse,NepsilonElement, world.NCoarseElement, alpha, beta, left, right)
    toc = time.perf_counter()
    time_basis = toc-tic


    def computeKmsij(TInd, aPatch, k, boundaryConditions):
        #print('.', end='', flush=True)
        tic = time.perf_counter()
        patch = PatchPeriodic(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        toc = time.perf_counter()
        return patch, correctorsList, csi.Kmsij, csi.muTPrime, toc-tic

    computeSingleKms = lambda aRef: computeKmsij(middle, aRef, k, boundaryConditions)
    _, _, KmsijList, muTPrimeList, timeMatrixList = zip(*map(computeSingleKms, aRefList))

    return aRefList, KmsijList, muTPrimeList, time_basis, timeMatrixList

def compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList, KmsijList,muTPrimeList,k,model,compute_indicator=False):
    computePatch = lambda TInd: PatchPeriodic(world, k, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))
    dim = np.size(world.NWorldFine)

    def compute_combinedT(TInd):
        #print('.', end='', flush=True)
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aPert, periodic=True)

        #direct determination of alpha without optimization
        assert (model == 'check' or model == 'incl')

        if model == 'check':
            alphaT = np.zeros(len(aRefList))
            alphaScaled = np.min(aRefList[1])
            betaScaled = np.max(aRefList[1])
            NFineperEpsilon = world.NWorldFine//Nepsilon
            NEpsilonperPatchCoarse = patchT[TInd].NPatchCoarse*(Nepsilon//world.NWorldCoarse)
            if dim == 2:
                tmp_indx =np.array([np.arange(len(aRefList)-1)//NEpsilonperPatchCoarse[0],
                                    np.arange(len(aRefList)-1)%NEpsilonperPatchCoarse[0]])
                indx = tmp_indx[0]*NFineperEpsilon[1]*patchT[TInd].NPatchFine[0]+ tmp_indx[1]*NFineperEpsilon[0]
                alphaT[:len(alphaT)-1] = (rPatch()[indx]-alphaScaled)/(betaScaled-alphaScaled)
            elif dim == 1:
                alphaT[:len(alphaT)-1] = (rPatch()[np.arange(len(aRefList)-1)*np.prod(NFineperEpsilon)]-alphaScaled)/(betaScaled-alphaScaled)
            alphaT[len(alphaT)-1] = 1.-np.sum(alphaT[:len(alphaT)-1])
        else:
            alphaT = np.zeros(len(aRefList))
            NFineperEpsilon = world.NWorldFine // Nepsilon
            NEpsilonperPatchCoarse = patchT[TInd].NPatchCoarse * (Nepsilon // world.NWorldCoarse)
            tmp_indx = np.array([np.arange(len(aRefList) - 1) // NEpsilonperPatchCoarse[0]+0.5,
                                 np.arange(len(aRefList) - 1) % NEpsilonperPatchCoarse[0] + 0.5])
            indx = (tmp_indx[0] * NFineperEpsilon[1] * patchT[TInd].NPatchFine[0]).astype(int) \
                    + (tmp_indx[1] * NFineperEpsilon[0]).astype(int)
            alphaT[:len(alphaT)-1] = (beta - rPatch()[indx])/(beta-alpha)
            alphaT[len(alphaT)-1] = 1. - np.sum(alphaT[:len(alphaT)-1])

        if compute_indicator:
            indicatorT = multiplecoeff.estimatorAlphaTildeA1mod(patchT[TInd],muTPrimeList,aRefList,rPatch,alphaT)
        else:
            indicatorT = None

        KmsijT = np.einsum('i, ijk -> jk', alphaT, KmsijList)

        return KmsijT, indicatorT

    KmsijT_list, error_indicator = zip(*map(compute_combinedT, range(world.NtCoarse)))

    KmsijT = tuple(KmsijT_list)

    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT, periodic=True)

    return KFull, error_indicator

def compute_perturbed_MsStiffness(world,aPert, aRef, KmsijRef, muTPrimeRef,k, update_percentage):
    computePatch = lambda TInd: PatchPeriodic(world, k, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))
    dim = np.size(world.NWorldFine)
    if dim == 2:
        middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2 #2d!!!
    elif dim == 1:
        middle = NCoarse[0] // 2
    patchRef = PatchPeriodic(world, k, middle)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)

    def computeIndicator(TInd):
        aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aPert, periodic=True)  # true coefficient
        E_vh = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], muTPrimeRef, aRef, aPatch)

        return E_vh

    def UpdateCorrectors(TInd):
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aPert, periodic=True)

        correctorsList = lod.computeBasisCorrectors(patchT[TInd], IPatch, rPatch)
        csi = lod.computeBasisCoarseQuantities(patchT[TInd], correctorsList, rPatch)

        return patchT[TInd], correctorsList, csi.Kmsij

    def UpdateElements(tol, E, Kmsij_old):
        #print('apply tolerance')
        Elements_to_be_updated = []
        for (i, eps) in E.items():
            if eps > tol:
                Elements_to_be_updated.append(i)
        if len(E) > 0:
            print('... to be updated: {}%'.format(100 * np.size(Elements_to_be_updated) / len(E)), end='\n', flush=True)

        if np.size(Elements_to_be_updated) != 0:
            #print('... update correctors')
            patchT_irrelevant, correctorsListT_irrelevant, KmsijTNew = zip(*map(UpdateCorrectors, Elements_to_be_updated))

            #print('replace Kmsij')
            KmsijT_list = list(np.copy(Kmsij_old))
            i = 0
            for T in Elements_to_be_updated:
                KmsijT_list[T] = np.copy(KmsijTNew[i])
                i += 1

            KmsijT = tuple(KmsijT_list)
            return KmsijT
        else:
            return Kmsij_old

    #print('computing error indicators', end='', flush=True)
    E_vh = list(map(computeIndicator, range(world.NtCoarse)))
    print()
    print('maximal value error estimator {}'.format(np.max(E_vh)))
    E = {i: E_vh[i] for i in range(np.size(E_vh)) if E_vh[i] > 0}

    # loop over elements with possible recomputation of correctors
    #sortedE = np.sort(E_vh)
    #index = int((1-update_percentage) * (len(E_vh) - 1))
    #tol_relative = sortedE[index]
    tol_relative = np.quantile(E_vh, 1.-update_percentage, interpolation='higher')
    KmsijRefList = [KmsijRef for _ in range(world.NtCoarse)] #tile up the stiffness matrix for one element
    KmsijT = UpdateElements(tol_relative, E, KmsijRefList)

    #assembly of matrix
    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT, periodic=True)

    return KFull, E_vh


#==================================================================================================
#testing

if __name__ == '__main__':

    from gridlod.world import World
    from gridlod import util, fem, coef
    from gridlod.build_coefficient import build_randomcheckerboard
    import scipy.sparse as sparse
    import matplotlib.pyplot as plt

    NFine = np.array([256, 256]) #,64
    NpFine = np.prod(NFine+1)
    Nepsilon = np.array([128, 128]) #,16
    NCoarse = np.array([32, 32]) #,8
    k=4
    NSamples = 1
    dim = np.size(NFine)

    boundaryConditions = None #np.array([[0, 0], [0, 0]])
    alpha = 0.1
    beta = 1.
    p = 0.1
    left = np.array([0.25, 0.25])
    right = np.array([0.75, 0.75])
    percentage = 0.25  # p*((2*k-1)**dim) # for updates in HKM LOD
    #fix random seed
    np.random.seed(123)

    NCoarseElement = NFine // NCoarse
    world = World(NCoarse, NCoarseElement, boundaryConditions)

    xpFine = util.pCoordinates(NFine)
    ffunc = lambda x: 8*np.pi**2*np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
    f = ffunc(xpFine).flatten()

    aRefList, KmsijList,muTPrimeList, timeBasis, timeMatrixList = computeCSI_offline(world, Nepsilon // NCoarse,
                                                                                     alpha, beta, k, boundaryConditions, 'check')

    print('time for setting up of checkerbboard basis {}'.format(timeBasis))
    print('average time for computation of stiffness matrix contribution {}'.format(np.mean(np.array(timeMatrixList))))
    print('variance in stiffness matrix timings {}'.format(np.var(np.array(timeMatrixList))))
    print('time for computation of HKM ref stiffness matrix {}'.format(timeMatrixList[-1]))
    print('total time for computation of ref stiffness matrices {}'.format(np.sum(np.array(timeMatrixList))))

    aRef = np.copy(aRefList[-1])
    KmsijRef = np.copy(KmsijList[-1])
    muTPrimeRef = muTPrimeList[-1]

    mean_error_combined = 0.
    mean_error_perturbed = 0.

    mean_time_true = 0.
    mean_time_perturbed = 0.
    mean_time_combined = 0.

    def computeKmsij(TInd, a, IPatch):
        #print('.', end='', flush=True)
        patch = PatchPeriodic(world, k, TInd)
        aPatch = coef.localizeCoefficient(patch,a, periodic=True)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij, csi

    for N in range(NSamples):
        aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)
        #aPert = build_coefficient.build_inclusions_defect_2d(NFine, Nepsilon, alpha, beta, left, right, p)

        MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
        basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
        bFull = basis.T * MFull * f
        faverage = np.dot(MFull * np.ones(NpFine), f)

        #fine FEM
        '''KFEM = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine,aPert)
        bFine = MFull*f
        #make the matrix periodic
        KFEM.tolil()
        KFEM[np.arange(0, NFine[1]*(NFine[0]+1)+1, NFine[0]+1),:] \
            += KFEM[np.arange(NFine[0], np.prod(NFine+1), NFine[0]+1),:]
        KFEM[:, np.arange(0, NFine[1] * (NFine[0] + 1) + 1, NFine[0] + 1)] \
            += KFEM[:, np.arange(NFine[0], np.prod(NFine + 1), NFine[0] + 1)]
        KFEM[np.arange(NFine[0]+1), :] += KFEM[np.arange(NFine[1]*(NFine[0]+1), np.prod(NFine+1)), :]
        KFEM[:, np.arange(NFine[0] + 1)] += KFEM[:, np.arange(NFine[1] * (NFine[0] + 1), np.prod(NFine + 1))]
        KFEM.tocsc()
        averageVector = MFull * np.ones(np.prod(NFine+1))
        fixed = np.concatenate((np.arange(NFine[1] * (NFine[0] + 1), NpFine),
                                np.arange(NFine[0], NpFine - 1, NFine[0] + 1)))
        free = np.setdiff1d(np.arange(NpFine), fixed)
        bFine[np.arange(0, NFine[1] * (NFine[0] + 1) + 1, NFine[0] + 1)] \
            += bFine[np.arange(NFine[0], NpFine, NFine[0] + 1)]
        bFine[np.arange(NFine[0] + 1)] += bFine[np.arange(NFine[1] * (NFine[0] + 1), NpFine)]
        averageVector[np.arange(0, NFine[1] * (NFine[0] + 1) + 1, NFine[0] + 1)] \
            += averageVector[np.arange(NFine[0], NpFine, NFine[0] + 1)]
        averageVector[np.arange(NFine[0] + 1)] += averageVector[np.arange(NFine[1] * (NFine[0] + 1), NpFine)]
    
        KFemFree = KFEM[free][:, free]
        constraint = averageVector[free].reshape((1, KFemFree.shape[0]))
        K = sparse.bmat([[KFemFree, constraint.T],
                         [constraint, None]], format='csc')
        bFree = bFine[free] - faverage * averageVector[free]  # right-hand side with non-zero average potentially not working correctly yet
        b = np.zeros(K.shape[0])
        b[:np.size(bFree)] = bFree
        x = sparse.linalg.spsolve(K, b)
        uFree = x[:np.size(bFree)]
    
        uFEM = np.zeros(NpFine)
        uFEM[free] = uFree
    
        uFEM[np.arange(NFine[0], NpFine - 1, NFine[0] + 1)] \
            += uFEM[np.arange(0, NFine[1] * (NFine[0] + 1), NFine[0] + 1)]
        uFEM[np.arange(NFine[1] * (NFine[0] + 1), NpFine)] += uFEM[np.arange(NFine[0] + 1)]
        '''

        #true LOD
        if dim == 2:
            middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2 #2d!!!
        elif dim == 1:
            middle = NCoarse[0] // 2
        patchRef = PatchPeriodic(world, k, middle)

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        aPert_Grid = aPert.reshape(world.NWorldFine, order='C')
        im1 = ax1.imshow(aPert_Grid, origin='lower_left', \
                         extent=(xpFine[:, 0].min(), xpFine[:, 0].max(), xpFine[:, 1].min(), xpFine[:, 1].max()),
                         cmap=plt.cm.binary)
        fig.colorbar(im1, ax=ax1)

        fig = plt.figure()
        for ii in range(4):
            ax = fig.add_subplot(2, 2, ii + 1)
            if ii == 0:
                aBase = aRefList[-1].reshape(patchRef.NPatchFine, order='C')
            else:
                aBase = aRefList[ii - 1].reshape(patchRef.NPatchFine, order='C')
            im = ax.imshow(aBase, origin='lower_left',
                           extent=(0, (2 * k + 1) / NCoarse[0], 0, (2 * k + 1) / NCoarse[1]), cmap=plt.cm.binary)
            fig.colorbar(im, ax=ax)

        plt.show()

        IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
        computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
        tic = time.perf_counter()
        patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
        KFulltrue = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)
        toc = time.perf_counter()
        mean_time_true += (toc-tic)

        bFull = basis.T * MFull * f
        uFulltrue, _ = pglod.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
        uLodCoarsetrue = basis * uFulltrue

        #combined LOD
        tic = time.perf_counter()
        KFullcomb, _ = compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,k,'check',compute_indicator=False)
        toc = time.perf_counter()
        mean_time_combined += (toc-tic)
        bFull = basis.T * MFull * f
        uFullcomb, _ = pglod.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
        uLodCoarsecomb = basis * uFullcomb

        L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
        error_combined = np.sqrt(np.dot(uLodCoarsetrue-uLodCoarsecomb, MFull*(uLodCoarsetrue-uLodCoarsecomb)))/L2norm
        print("L2-error in {}th sample for new LOD is: {}".format(N, error_combined))
        mean_error_combined += error_combined

        #perturbedLOD
        tic = time.perf_counter()
        KFullpert, _ = compute_perturbed_MsStiffness(world, aPert, aRef, KmsijRef, muTPrimeRef, k, percentage)
        toc = time.perf_counter()
        mean_time_perturbed += (toc-tic)
        bFull = basis.T * MFull * f
        uFullpert, _ = pglod.solvePeriodic(world, KFullpert, bFull, faverage, boundaryConditions)
        uLodCoarsepert = basis * uFullpert

        error_pert = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsepert, MFull * (uLodCoarsetrue - uLodCoarsepert))) / L2norm
        print("L2-error in {}th sample for HKM LOD is: {}".format(N, error_pert))
        mean_error_perturbed += error_pert


        '''fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)
    
        uLodCoarse_pertGrid = uLodCoarsepert.reshape(world.NWorldFine+1, order='C')
        uLodCoarse_trueGrid = uLodCoarsetrue.reshape(world.NWorldFine+1, order='C')
        uLodCoarse_combGrid = uLodCoarsecomb.reshape(world.NWorldFine+1, order='C')
    
        im4 = ax4.imshow(uLodCoarse_combGrid, origin='lower_left',\
                         extent=(xpFine[:, 0].min(), xpFine[:, 0].max(), xpFine[:, 1].min(), xpFine[:, 1].max()), cmap=plt.cm.hot)
        fig.colorbar(im4, ax=ax4)
    
        im2 = ax2.imshow(uLodCoarse_trueGrid, origin='lower_left',\
                         extent=(xpFine[:, 0].min(), xpFine[:, 0].max(), xpFine[:, 1].min(), xpFine[:, 1].max()), cmap=plt.cm.hot)
        fig.colorbar(im2, ax=ax2)
    
        im3 = ax3.imshow(uLodCoarse_pertGrid, origin='lower_left',\
                         extent=(xpFine[:,0].min(), xpFine[:,0].max(), xpFine[:,1].min(), xpFine[:,1].max()), cmap=plt.cm.hot)
        fig.colorbar(im3, ax=ax3)
    
        im1 = ax1.imshow(uFEM_Grid, origin='lower_left',
                         extent=(xpFine[:,0].min(), xpFine[:,0].max(), xpFine[:,1].min(), xpFine[:,1].max()), cmap=plt.cm.hot)
        fig.colorbar(im1, ax=ax1)
    
        #im3 = ax3.imshow(aPertgrid, origin= 'lower_left',\
        #                 extent=(xp[:, 0].min(), xp[:, 0].max(), xp[:, 1].min(), xp[:, 1].max()), cmap=plt.cm.hot)
        #fig.colorbar(im6, ax=ax6)
        plt.show()'''


    mean_error_combined /= NSamples
    mean_error_perturbed /= NSamples
    print("mean L2-error for HKM-LOD over {} samples is: {}".format(NSamples, mean_error_perturbed))
    print("mean L2-error for new LOD over {} samples is: {}".format(NSamples, mean_error_combined))

    mean_time_true /= NSamples

    mean_time_perturbed /= NSamples
    mean_time_combined /= NSamples
    print('mean time for matrix assembly in true LOD {}'.format(mean_time_true))
    print('mean time for matrix assembly in HKM LOD {}'.format(mean_time_perturbed))
    print('mean time for matrix assembly in new LOD {}'.format(mean_time_combined))