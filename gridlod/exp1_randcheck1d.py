import numpy as np

from gridlod.world import World, PatchPeriodic
from gridlod import util, fem, coef, lod, pglod, interp, build_coefficient
from gridlod.algorithms_random import computeCSI_offline, compute_combined_MsStiffness
import matplotlib.pyplot as plt

NFine = np.array([256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([256])
NCoarse = np.array([64])
k=3
NSamples = 1000
dim = np.size(NFine)

boundaryConditions = None
alpha = 0.1
beta = 1.
pList = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
np.random.seed(123)

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8*np.pi**2*np.sin(2*np.pi*x)
f = ffunc(xpFine).flatten()

aRefList, KmsijList,muTPrimeList, _, _ = computeCSI_offline(world, Nepsilon // NCoarse,
                                                                                 alpha, beta, k, boundaryConditions, 'check')

mean_error_combined = np.zeros(len(pList))

def computeKmsij(TInd, a, IPatch):
    #print('.', end='', flush=True)
    patch = PatchPeriodic(world, k, TInd)
    aPatch = coef.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

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
        KFullcomb, _ = compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,k,'check',
                                                                      compute_indicator=False)
        bFull = basis.T * MFull * f
        uFullcomb, _ = pglod.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
        uLodCoarsecomb = basis * uFullcomb

        L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
        error_combined = np.sqrt(np.dot(uLodCoarsetrue-uLodCoarsecomb, MFull*(uLodCoarsetrue-uLodCoarsecomb)))/L2norm
        #print("L2-error in {}th sample for new LOD is: {}".format(N, error_combined))
        mean_error_combined[ii] += error_combined

    mean_error_combined[ii] /= NSamples
    print("mean L2-error for new LOD over {} samples for p={} is: {}".format(NSamples, p, mean_error_combined[ii]))
    ii += 1

print(mean_error_combined)

#plt.plot(pList, mean_error_combined, '*')
#plt.show()