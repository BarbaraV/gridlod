import numpy as np
import scipy.io as sio

from gridlod.world import World, PatchPeriodic
from gridlod import util, fem, coef, lod, pglod, interp, build_coefficient
from gridlod.algorithms_random import computeCSI_offline, compute_combined_MsStiffness

NFine = np.array([256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([256])
NList = [8,16,32,64]
k=3
NSamples = 500
dim = np.size(NFine)

boundaryConditions = None
alpha = 0.1
beta = 1.
pList = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
np.random.seed(123)
model ={'name': 'check', 'alpha': alpha, 'beta': beta}

def computeKmsij(TInd, a, IPatch):
    #print('.', end='', flush=True)
    patch = PatchPeriodic(world, k, TInd)
    aPatch = coef.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

def computeAharm(TInd,a):
    assert(dim==1)
    patch = PatchPeriodic(world,0,TInd)
    aPatch = coef.localizeCoefficient(patch, a, periodic=True)
    aPatchHarm = np.sum(1./aPatch)
    return world.NWorldFine/(world.NWorldCoarse * aPatchHarm)

def computeAharm_offline():
    aRefListSingle,_,_,_,_ = computeCSI_offline(world, Nepsilon//NCoarse,0,boundaryConditions,model)
    return [world.NWorldFine/(world.NWorldCoarse * np.sum(1./aRefSingle)) for aRefSingle in aRefListSingle]

def computeAharm_error(aHarmList, aPert):
    computePatch = lambda TInd: PatchPeriodic(world, 0, TInd)
    patchT = list(map(computePatch, range(world.NtCoarse)))

    def compute_errorHarmT(TInd):
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aPert, periodic=True)

        alphaT = np.zeros(len(aHarmList))
        NFineperEpsilon = world.NWorldFine // Nepsilon
        alphaT[:len(alphaT) - 1] = (rPatch()[np.arange(len(aHarmList) - 1) * np.prod(
            NFineperEpsilon)] - alpha) / (beta - alpha)
        alphaT[len(alphaT) - 1] = 1. - np.sum(alphaT[:len(alphaT) - 1])

        aharmT_combined = np.dot(alphaT,aHarmList)
        aharmT = computeAharm(TInd,aPert)

        return np.abs(aharmT-aharmT_combined)

    error_harmList = list(map(compute_errorHarmT, range(world.NtCoarse)))
    return np.max(error_harmList)


for Nc in NList:
    NCoarse = np.array([Nc])
    NCoarseElement = NFine // NCoarse
    world = World(NCoarse, NCoarseElement, boundaryConditions)

    xpFine = util.pCoordinates(NFine)
    ffunc = lambda x: 8 * np.pi ** 2 * np.sin(2 * np.pi * x)
    f = ffunc(xpFine).flatten()

    aRefList, KmsijList, muTPrimeList, _, _ = computeCSI_offline(world, Nepsilon // NCoarse, k, boundaryConditions,model)
    aharmList = computeAharm_offline()

    harmErrorList = np.zeros((len(pList), NSamples))
    absErrorList = np.zeros((len(pList), NSamples))
    relErrorList = np.zeros((len(pList), NSamples))

    ii = 0
    for p in pList:
        for N in range(NSamples):
            aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)

            MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
            basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
            bFull = basis.T * MFull * f
            faverage = np.dot(MFull * np.ones(NpFine), f)

            #true LOD
            middle = NCoarse[0] // 2
            patchRef = PatchPeriodic(world, k, middle)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
            computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
            patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
            KFulltrue = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)

            bFull = basis.T * MFull * f
            uFulltrue, _ = pglod.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
            uLodCoarsetrue = basis * uFulltrue

            #combined LOD
            KFullcomb, _ = compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,k,model,
                                                                          compute_indicator=False)
            bFull = basis.T * MFull * f
            uFullcomb, _ = pglod.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
            uLodCoarsecomb = basis * uFullcomb

            L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
            abserror_combined = np.sqrt(np.dot(uLodCoarsetrue-uLodCoarsecomb, MFull*(uLodCoarsetrue-uLodCoarsecomb)))

            absErrorList[ii, N] = abserror_combined
            relErrorList[ii, N] = abserror_combined/L2norm
            harmErrorList[ii, N] = computeAharm_error(aharmList,aPert)

        print("mean L2-error for new LOD over {} samples for p={} is: {}".format(NSamples, p, np.mean(relErrorList[ii,:])))
        print("mean L2-error for harmonic mean over {} samples for p={} is: {}".format(NSamples, p, np.mean(harmErrorList[ii,:])))
        ii += 1

    sio.savemat('_meanErr_Nc'+str(Nc)+'.mat',
                {'absErr': absErrorList, 'relErr': relErrorList, 'HarmErr': harmErrorList, 'pList': pList})

#plt.plot(pList, mean_error_combined, '*')
#plt.show()