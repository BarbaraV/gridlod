import numpy as np
import scipy.io as sio
import time

from gridlod.world import World, PatchPeriodic
from gridlod import util, fem, coef, lod, pglod, interp, build_coefficient
from gridlod.algorithms_random import computeCSI_offline, compute_combined_MsStiffness, compute_perturbed_MsStiffness

NFine = np.array([256, 256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([128,128])
NCoarse = np.array([32,32])
k=4
NSamples = 2#350
dim = np.size(NFine)

boundaryConditions = None
alpha = 0.1
beta = 1.
pList = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
percentage_comp = 0.2
model ={'name': 'check', 'alpha': alpha, 'beta': beta}
np.random.seed(123)

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8*np.pi**2*np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()

aRefList, KmsijList,muTPrimeList, timeBasis, timeMatrixList = computeCSI_offline(world, Nepsilon // NCoarse,
                                                                                 k, boundaryConditions, model)
aRef = np.copy(aRefList[-1])
KmsijRef = np.copy(KmsijList[-1])
muTPrimeRef = muTPrimeList[-1]

print('time for setting up of checkerbboard basis {}'.format(timeBasis))
print('average time for computation of stiffness matrix contribution {}'.format(np.mean(np.array(timeMatrixList))))
print('variance in stiffness matrix timings {}'.format(np.var(np.array(timeMatrixList))))
print('time for computation of HKM ref stiffness matrix {}'.format(timeMatrixList[-1]))
print('total time for computation of ref stiffness matrices {}'.format(np.sum(np.array(timeMatrixList))))

print('offline time for new approach {}'.format(timeBasis+np.sum(np.array(timeMatrixList))))
print('offline time for perturbed LOD {}'.format(timeMatrixList[-1]))

abserr_comb= np.zeros((len(pList), NSamples))
relerr_comb= np.zeros((len(pList), NSamples))
abserr_noup= np.zeros((len(pList), NSamples))
relerr_noup= np.zeros((len(pList), NSamples))

def computeKmsij(TInd, a, IPatch):
    #print('.', end='', flush=True)
    patch = PatchPeriodic(world, k, TInd)
    aPatch = coef.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi


# pertrubed LOD
basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
computePatch = lambda TInd: PatchPeriodic(world, k, TInd)
patchT = list(map(computePatch, range(world.NtCoarse)))
KFullpert = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijRef, periodic=True)
bFull = basis.T * MFull * f
faverage = np.dot(MFull * np.ones(NpFine), f)
uFullpert, _ = pglod.solvePeriodic(world, KFullpert, bFull, faverage, boundaryConditions)
uLodCoarsepert = basis * uFullpert

ii = 0
for p in pList:
    if p == 0.1:
        error_samp_hkm = np.zeros(NSamples)
        mean_time_true = 0.
        mean_time_perturbed = 0.
        mean_time_combined = 0.

    for N in range(NSamples):
        aPert = build_coefficient.build_randomcheckerboard(Nepsilon,NFine,alpha,beta,p)

        bFull = basis.T * MFull * f

        #true LOD
        middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
        patchRef = PatchPeriodic(world, k, middle)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
        computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
        if p == 0.1:
            tic = time.perf_counter()
            patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
            KFulltrue = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)
            toc = time.perf_counter()
            mean_time_true += (toc-tic)
        else:
            patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
            KFulltrue = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)

        bFull = basis.T * MFull * f
        uFulltrue, _ = pglod.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
        uLodCoarsetrue = basis * uFulltrue

        #combined LOD
        if p == 0.1:
            tic = time.perf_counter()
            KFullcomb, _ = compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,k,model,
                                                                      compute_indicator=False)
            toc = time.perf_counter()
            mean_time_combined += (toc-tic)
        else:
            KFullcomb, _ = compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,k,model,
                                                                      compute_indicator=False)
        bFull = basis.T * MFull * f
        uFullcomb, _ = pglod.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
        uLodCoarsecomb = basis * uFullcomb

        L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
        abs_error_combined = np.sqrt(np.dot(uLodCoarsetrue-uLodCoarsecomb, MFull*(uLodCoarsetrue-uLodCoarsecomb)))
        abserr_comb[ii, N] = abs_error_combined
        relerr_comb[ii, N] = abs_error_combined/L2norm


        abs_error_pert = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsepert, MFull * (uLodCoarsetrue - uLodCoarsepert)))
        abserr_noup[ii, N] = abs_error_pert
        relerr_noup[ii, N] = abs_error_pert/L2norm
        if p == 0.1:
            tic = time.perf_counter()
            KFullpertup, _ = compute_perturbed_MsStiffness(world, aPert, aRef, KmsijRef, muTPrimeRef, k, percentage_comp)
            toc = time.perf_counter()
            mean_time_perturbed += (toc-tic)
            bFull = basis.T * MFull * f
            uFullpertup, _ = pglod.solvePeriodic(world, KFullpertup, bFull, faverage, boundaryConditions)
            uLodCoarsepertup = basis * uFullpertup
            error_pertup = np.sqrt(
                np.dot(uLodCoarsetrue - uLodCoarsepertup, MFull * (uLodCoarsetrue - uLodCoarsepertup))) / L2norm
            # print("L2-error in {}th sample for HKM LOD is: {}".format(N, error_pert))
            error_samp_hkm[N] = error_pertup

    print("mean L2-error for new LOD over {} samples for p={} is: {}".format(NSamples, p, np.mean(relerr_comb[ii,:])))
    print("mean L2-error for perturbed LOD over {} samples for p={} is: {}".format(NSamples, p, np.mean(relerr_noup[ii,:])))
    ii += 1

    if p == 0.1:
        mean_time_true /= NSamples
        mean_time_perturbed /= NSamples
        mean_time_combined /= NSamples

        sio.savemat('_relErrHKM.mat', {'relErrHKM': error_samp_hkm, 'iiSamp': np.arange(NSamples)})

        print("mean assembly time for standard LOD over {} samples is: {}".format(NSamples, mean_time_true))
        print("mean assembly time for perturbed LOD over {} samples is: {}".format(NSamples, mean_time_perturbed))
        print("mean assembly time for new LOD over {} samples is: {}".format(NSamples, mean_time_combined))

        print("mean L2-error for perturbed LOD over {} samples with {} updates is: {}".format(NSamples, percentage_comp,
                                                                                       np.mean(error_samp_hkm)))

sio.savemat('_meanErr2drandcheck.mat', {'abserrNew': abserr_comb, 'relerrNew': relerr_comb,
                                        'absErrNoup': abserr_noup, 'relerrNoup': abserr_noup, 'pList': pList})
