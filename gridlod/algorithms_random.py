import numpy as np
import copy

from gridlod import multiplecoeff, coef, interp, lod, pglod
from gridlod.world import PatchPeriodic
from gridlod.build_coefficient import build_checkerboardbasis


def computeCSI_offline(world, NepsilonEelement, alpha, beta, k, boundaryConditions):
    #middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2 #2d!!!
    middle = NCoarse[0] //2
    patch = PatchPeriodic(world, k, middle)

    aRefList = build_checkerboardbasis(patch.NPatchCoarse, NepsilonEelement, world.NCoarseElement, alpha, beta)

    muTPrimeList = []
    KmsijList = []
    #correctorsList = []

    def computeKmsij(TInd, aPatch, k, boundaryConditions):
        #print('.', end='', flush=True)
        patch = PatchPeriodic(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij, csi

    for aRef in aRefList:
        #print('computing correctors', end='', flush=True)
        _, correctorsRef, KmsijRef, csiTRef = computeKmsij(middle,aRef,k,boundaryConditions)
        #csiList.append(csiTRef)
        KmsijList.append(KmsijRef)
        muTPrimeList.append(csiTRef.muTPrime)
        #correctorsList.append(correctorsListRef)
        #print()

    return aRefList, KmsijList, muTPrimeList

def compute_combined_MsStiffness(world,aPert, aRefList, KmsijList,muTPrimeList,k):
    computePatch = lambda TInd: PatchPeriodic(world, k, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))

    def computeAlpha(TInd):
        print('.', end='', flush=True)
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aPert, periodic=True) # scaleCoefficient
        #aRefListScaled = [aRef for aRef in aRefList]

        #alphaT = multiplecoeff.optimizeAlpha(patchT[TInd], aRefListScaled, rPatch)
        #direct dtermination of alpha without optimization - for randomcheckerboardbasis only
        alphaT = np.zeros(len(aRefList))
        alphaScaled = np.min(aRefList[1])
        betaScaled = np.max(aRefList[1])
        Ntepsilon = np.prod(patchT[TInd].NPatchFine//(patchT[TInd].NPatchCoarse * Nepsilon//NCoarse))#indexing only correct in 1d
        alphaT[1:] = (rPatch()[np.arange(len(aRefList)-1)*Ntepsilon]-alphaScaled)/(betaScaled-alphaScaled)
        alphaT[0] = 1.-np.sum(alphaT[1:])

        indicatorT = multiplecoeff.estimatorAlphaTildeA1mod(patchT[TInd],muTPrimeList,aRefList,rPatch,alphaT)

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


#==================================================================================================
#testing

from gridlod.world import World
from gridlod import util, fem, coef
from gridlod.build_coefficient import build_randomcheckerboard
import scipy.sparse as sparse
import matplotlib.pyplot as plt

NFine = np.array([256]) #,64
NpFine = np.prod(NFine+1)
Nepsilon = np.array([256]) #,16
NCoarse = np.array([32]) #,8
k=4
NSamples = 1000

boundaryConditions = None #np.array([[0, 0], [0, 0]])
alpha = 0.1
beta = 1.

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8*np.pi**2*np.sin(2*np.pi*x)#[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()

aRefList, KmsijList,muTPrimeList = computeCSI_offline(world, Nepsilon // NCoarse, alpha, beta, k, boundaryConditions)

mean_error = 0.

def computeKmsij(TInd, a, IPatch):
    #print('.', end='', flush=True)
    patch = PatchPeriodic(world, k, TInd)
    aPatch = coef.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

for N in range(NSamples):
    aPert = build_randomcheckerboard(Nepsilon,NFine,alpha,beta)
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
    #middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2 #2d!!!
    middle = NCoarse[0] // 2
    patchRef = PatchPeriodic(world, k, middle)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
    computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
    patchT, _, KmsijTpert, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
    KFullpert = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTpert, periodic=True)

    uFullpert, _ = pglod.solvePeriodic(world, KFullpert, bFull, faverage, boundaryConditions)
    uLodCoarsepert = basis * uFullpert

    #combined LOD
    KFull, error_indicator = compute_combined_MsStiffness(world,aPert,aRefList,KmsijList,muTPrimeList,k)
    uFull, _ = pglod.solvePeriodic(world, KFull, bFull, faverage, boundaryConditions)
    uLodCoarse = basis * uFull

    L2norm = np.sqrt(np.dot(uLodCoarsepert, MFull * uLodCoarsepert))
    error = np.sqrt(np.dot(uLodCoarse-uLodCoarsepert, MFull*(uLodCoarse-uLodCoarsepert)))/L2norm
    #L2normref = np.sqrt(np.dot(uFEM, MFull * uFEM))
    #error_ref = np.sqrt(np.dot(uLodCoarse - uFEM, MFull * (uLodCoarse - uFEM))) / L2normref
    print("L2-error in {}th sample between LOD approaches is: {}".format(N, error))
    #print("L2-error in {}th sample to FEM ref sol is: {}".format(N, error_ref))
    mean_error += error

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

mean_error /= NSamples
print("mean L2-error over {} samples is: {}".format(NSamples, mean_error))

