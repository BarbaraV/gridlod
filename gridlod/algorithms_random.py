import numpy as np
import copy
import time

from gridlod import multiplecoeff, coef, interp, lod, pglod
from gridlod.world import PatchPeriodic
from gridlod.build_coefficient import build_checkerboardbasis


def computeCSI_offline(world, NepsilonEelement, alpha, beta, k, boundaryConditions):
    if dim == 2:
        middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2 #2d!!!
    elif dim == 1:
        middle = NCoarse[0] //2
    patch = PatchPeriodic(world, k, middle)

    tic = time.perf_counter()
    aRefList = build_checkerboardbasis(patch.NPatchCoarse, NepsilonEelement, world.NCoarseElement, alpha, beta)
    toc = time.perf_counter()
    time_basis = toc-tic

    muTPrimeList = []
    KmsijList = []
    timeMatrixList = []

    def computeKmsij(TInd, aPatch, k, boundaryConditions):
        #print('.', end='', flush=True)
        patch = PatchPeriodic(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij, csi

    for aRef in aRefList:
        #print('computing correctors', end='', flush=True)
        tic = time.perf_counter()
        _, correctorsRef, KmsijRef, csiTRef = computeKmsij(middle,aRef,k,boundaryConditions)
        toc = time.perf_counter()
        #csiList.append(csiTRef)
        KmsijList.append(KmsijRef)
        muTPrimeList.append(csiTRef.muTPrime)
        timeMatrixList.append(toc-tic)
        #correctorsList.append(correctorsListRef)
        #print()

    return aRefList, KmsijList, muTPrimeList, time_basis,timeMatrixList

def compute_combined_MsStiffness(world,aPert, aRefList, KmsijList,muTPrimeList,k, compute_indicator=False):
    computePatch = lambda TInd: PatchPeriodic(world, k, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))

    def computeAlpha(TInd):
        #print('.', end='', flush=True)
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aPert, periodic=True) # scaleCoefficient
        #aRefListScaled = [aRef for aRef in aRefList]

        #alphaT = multiplecoeff.optimizeAlpha(patchT[TInd], aRefListScaled, rPatch)
        #direct dtermination of alpha without optimization - for randomcheckerboardbasis only
        alphaT = np.zeros(len(aRefList))
        alphaScaled = np.min(aRefList[1])
        betaScaled = np.max(aRefList[1])
        NFineperEpsilon = NFine//Nepsilon
        NEpsilonperPatchCoarse = patchT[TInd].NPatchCoarse*(Nepsilon//NCoarse)
        if dim == 2:
            tmp_indx =np.array([np.arange(len(aRefList)-1)//NEpsilonperPatchCoarse[0],
                                np.arange(len(aRefList)-1)%NEpsilonperPatchCoarse[0]])
            indx = tmp_indx[0]*NFineperEpsilon[1]*patchT[TInd].NPatchFine[0]+ tmp_indx[1]*NFineperEpsilon[0]
            alphaT[1:] = (rPatch()[indx]-alphaScaled)/(betaScaled-alphaScaled)
        elif dim == 1:
            alphaT[1:] = (rPatch()[np.arange(len(aRefList)-1)*np.prod(NFineperEpsilon)]-alphaScaled)/(betaScaled-alphaScaled)
        alphaT[0] = 1.-np.sum(alphaT[1:])

        if compute_indicator:
            indicatorT = multiplecoeff.estimatorAlphaTildeA1mod(patchT[TInd],muTPrimeList,aRefList,rPatch,alphaT)
        else:
            indicatorT = None

        #mu_equivalent = coef.averageCoefficient(coef.localizeCoefficient(patchT[TInd],aPert,periodic=True))\
                        #*np.array([1./coef.averageCoefficient(aRef) for aRef in aRefList])
        #alphaT *= mu_equivalent

        return alphaT, indicatorT

    KmsijT_list = []
    error_indicator = []
    #correctorsListT_list = list(copy.deepcopy(correctors_old[0]))

    for T in range(world.NtCoarse):
        alphaT, indicatorT = computeAlpha(T)
        #correctorsListT_list[T] = list(0 * np.array(correctorsListT_list[T]))
        KmsijT_list.append(np.einsum('i, ijk -> jk', alphaT, KmsijList))
        error_indicator.append(indicatorT)

    KmsijT = tuple(KmsijT_list)
    #correctorsListT = tuple(correctorsListT_list)

    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT, periodic=True)

    return KFull, error_indicator

def compute_perturbed_MsStiffness(world,aPert, aRef, KmsijRef, muTPrimeRef,k, update_percentage):
    computePatch = lambda TInd: PatchPeriodic(world, k, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))

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
                KmsijT_list[T] = KmsijTNew[i]
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
    tol_relative = np.quantile(E_vh, 1.-percentage, interpolation='higher')
    KmsijRefList = [KmsijRef for _ in range(world.NtCoarse)] #tile up the stiffness matrix for one element
    KmsijT = UpdateElements(tol_relative, E, KmsijRefList)

    #assembly of matrix
    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT, periodic=True)

    return KFull, E_vh


#==================================================================================================
#testing

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
NSamples = 100
dim = np.size(NFine)

boundaryConditions = None #np.array([[0, 0], [0, 0]])
alpha = 0.1
beta = 1.
p = 0.01

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8*np.pi**2*np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()

aRefList, KmsijList,muTPrimeList, timeBasis, timeMatrixList = computeCSI_offline(world, Nepsilon // NCoarse,
                                                                                 alpha, beta, k, boundaryConditions)

print('time for setting up of checkerbboard basis {}'.format(timeBasis))
print('average time for computation of stiffness matrix contribution {}'.format(np.mean(np.array(timeMatrixList))))
print('variance in stiffness matrix timings {}'.format(np.var(np.array(timeMatrixList))))
print('time for computation of HKM ref stiffness matrix {}'.format(timeMatrixList[0]))
print('total time for computation of ref stiffness matrices {}'.format(np.sum(np.array(timeMatrixList))))

aRef = aRefList[0]
KmsijRef = KmsijList[0]
muTPrimeRef = muTPrimeList[0]

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
    aPert = build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)
    #aPert = beta*np.ones(world.NtFine)

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
    uFEM[np.arange(NFine[1] * (NFine[0] + 1), NpFine)] += uFEM[np.arange(NFine[0] + 1)]'''


    #true LOD
    if dim == 2:
        middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2 #2d!!!
    elif dim == 1:
        middle = NCoarse[0] // 2
    patchRef = PatchPeriodic(world, k, middle)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
    computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
    tic = time.perf_counter()
    patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
    KFulltrue = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)
    toc = time.perf_counter()
    mean_time_true += (toc-tic)

    uFulltrue, _ = pglod.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
    uLodCoarsetrue = basis * uFulltrue

    #combined LOD
    tic = time.perf_counter()
    KFullcomb, _ = compute_combined_MsStiffness(world,aPert,aRefList,KmsijList,muTPrimeList,k, compute_indicator=False)
    toc = time.perf_counter()
    mean_time_combined += (toc-tic)
    uFullcomb, _ = pglod.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
    uLodCoarsecomb = basis * uFullcomb

    L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
    error_combined = np.sqrt(np.dot(uLodCoarsetrue-uLodCoarsecomb, MFull*(uLodCoarsetrue-uLodCoarsecomb)))/L2norm
    print("L2-error in {}th sample for new LOD is: {}".format(N, error_combined))
    mean_error_combined += error_combined

    #perturbedLOD
    percentage = 0.2#p*((2*k-1)**dim)
    tic = time.perf_counter()
    KFullpert, _ = compute_perturbed_MsStiffness(world, aPert, aRef, KmsijRef, muTPrimeRef, k, percentage)
    toc = time.perf_counter()
    mean_time_perturbed += (toc-tic)
    uFullpert, _ = pglod.solvePeriodic(world, KFullpert, bFull, faverage, boundaryConditions)
    uLodCoarsepert = basis * uFullpert

    error_pert = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsepert, MFull * (uLodCoarsetrue - uLodCoarsepert))) / L2norm
    print("L2-error in {}th sample for HKM LOD is: {}".format(N, error_pert))
    mean_error_perturbed += error_pert

    '''plt.figure('FE part LOD solutions')
    plt.plot(util.pCoordinates(world.NWorldFine), uLodCoarsepert, color='r', label='true')
    plt.plot(util.pCoordinates(world.NWorldFine), uLodCoarse, color='b', label='with reference')
    plt.legend()
    plt.show()'''

    '''plt.figure('error indicators')
    plt.bar(util.tCoordinates(world.NWorldCoarse).flatten(), np.array(error_indicator), width=0.03, color='r')
    plt.show()'''

'''    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    uLodCoarse_pertGrid = uLodCoarsepert.reshape(world.NWorldFine+1, order='C')
    uLodCoarseGrid = uLodCoarse.reshape(world.NWorldFine+1, order='C')
    uFineGrid = uFEM.reshape(world.NWorldFine+1, order='C')

    im3 = ax3.imshow(uFineGrid, origin='lower_left',\
                     extent=(xpFine[:, 0].min(), xpFine[:, 0].max(), xpFine[:, 1].min(), xpFine[:, 1].max()), cmap=plt.cm.hot)
    fig.colorbar(im3, ax=ax3)

    im1 = ax1.imshow(uLodCoarse_pertGrid, origin='lower_left',\
                     extent=(xpFine[:, 0].min(), xpFine[:, 0].max(), xpFine[:, 1].min(), xpFine[:, 1].max()), cmap=plt.cm.hot)
    fig.colorbar(im1, ax=ax1)

    im2 = ax2.imshow(uLodCoarseGrid, origin='lower_left',\
                     extent=(xpFine[:,0].min(), xpFine[:,0].max(), xpFine[:,1].min(), xpFine[:,1].max()), cmap=plt.cm.hot)
    fig.colorbar(im2, ax=ax2)

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