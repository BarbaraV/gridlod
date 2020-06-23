import numpy as np
import copy

from gridlod import multiplecoeff, coef, interp, lod, pglod
from gridlod.world import PatchPeriodic
from gridlod.build_coefficient import build_checkerboardbasis


def computeCSI_offline(world, NepsilonEelement, alpha, beta, k, boundaryConditions):
    middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2 #2d!!!
    patch = PatchPeriodic(world, k, middle)

    aRefList = build_checkerboardbasis(patch.NPatchCoarse, NepsilonEelement, world.NCoarseElement, alpha, beta)

    #csiList = []
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
        #correctorsList.append(correctorsListRef)
        #print()

    return aRefList, KmsijList

def compute_combined_MsStiffness(world,aPert, aRefList, KmsijList,k):
    computePatch = lambda TInd: PatchPeriodic(world, k, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))

    def computeAlpha(TInd):
        print('.', end='', flush=True)
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aPert, periodic=True)
        #muTPrimeList = [csi.muTPrime for csi in csiList]

        alphaT = multiplecoeff.optimizeAlpha(patchT[TInd], aRefList, rPatch)
        #by the above function you can switch the choice of alpha optimization

        return alphaT

    KmsijT_list = []
    #correctorsListT_list = list(copy.deepcopy(correctors_old[0]))

    for T in range(world.NtCoarse):
        alphaT = computeAlpha(T)
        #correctorsListT_list[T] = list(0 * np.array(correctorsListT_list[T]))
        KmsijT_list.append(np.einsum('i, ijk -> jk', alphaT, KmsijList))

    KmsijT = tuple(KmsijT_list)
    #correctorsListT = tuple(correctorsListT_list)

    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT, periodic=True)

    return KFull


#==================================================================================================
#testing

from gridlod.world import World
from gridlod import util, fem, coef
from gridlod.build_coefficient import build_randomcheckerboard

NFine = np.array([256,256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([32,32])
NCoarse = np.array([8,8])
k=2
NSamples = 10

boundaryConditions = np.array([[0, 0], [0, 0]])
alpha = 0.1
beta = 1.

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8*np.pi**2*np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()

aRefList, KmsijList = computeCSI_offline(world, Nepsilon // NCoarse, alpha, beta, k, boundaryConditions)

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

    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    bFull = basis.T * MFull * f
    faverage = np.dot(MFull * np.ones(NpFine), f)

    #true LOD
    middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2 #2d!!!
    patchRef = PatchPeriodic(world, k, middle)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
    computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
    patchT, _, KmsijTpert, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
    KFullpert = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTpert, periodic=True)

    uFullpert, _ = pglod.solvePeriodic(world, KFullpert, bFull, faverage, boundaryConditions)
    uLodCoarsepert = basis * uFullpert

    #combined LOD
    KFull = compute_combined_MsStiffness(world,aPert,aRefList,KmsijList,k)
    uFull, _ = pglod.solvePeriodic(world, KFull, bFull, faverage, boundaryConditions)
    uLodCoarse = basis * uFull

    error = np.sqrt(np.dot(uLodCoarse-uLodCoarsepert, MFull*(uLodCoarse-uLodCoarsepert)))
    print("L2-error in {}th sample is: {}".format(N, error))
    mean_error += error

mean_error /= NSamples
print("mean L2-error over {} samples is: {}".format(NSamples, mean_error))

