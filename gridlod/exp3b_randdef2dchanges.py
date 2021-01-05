import numpy as np
import scipy.io as sio

from gridlod.world import World, PatchPeriodic
from gridlod import util, fem, coef, lod, pglod, interp, build_coefficient
from gridlod.algorithms_random import computeCSI_offline, compute_combined_MsStiffness, compute_perturbed_MsStiffness

NFine = np.array([256, 256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([64,64])
NCoarse = np.array([16,16])
k=3
NSamples = 350
dim = np.size(NFine)

boundaryConditions = None
alpha = 1.
beta = 10.
left = np.array([0.25, 0.25])
right = np.array([0.75, 0.75])
pList = [0.01, 0.05, 0.1, 0.15]
modelfill = {'name': 'inclfill', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right}
modelshift = {'name': 'inclshift', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right,
              'def_bl': np.array([0.75, 0.75]), 'def_tr': np.array([1., 1.])}
modelLshape = {'name': 'inclLshape', 'bgval': alpha, 'inclval': beta, 'left': left, 'right': right,
              'def_bl': np.array([0.5, 0.5]), 'def_tr': np.array([0.75, 0.75])}
modelList = [modelfill, modelshift, modelLshape]
np.random.seed(123)

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8*np.pi**2*np.sin(2*np.pi*x[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()

mean_error_combined = np.zeros((len(pList), len(modelList)))
mean_error_perturbed = np.zeros((len(pList), len(modelList)))

def computeKmsij(TInd, a, IPatch):
    #print('.', end='', flush=True)
    patch = PatchPeriodic(world, k, TInd)
    aPatch = coef.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

ii = 0
for p in pList:
    jj = 0
    for model in modelList:
        aRefListdef, KmsijListdef, muTPrimeListdef, _, _ = computeCSI_offline(world, Nepsilon // NCoarse, k,
                                                                              boundaryConditions, model)
        aRef = np.copy(aRefListdef[-1])
        KmsijRef = np.copy(KmsijListdef[-1])
        muTPrimeRef = muTPrimeListdef[-1]

        for N in range(NSamples):
            aPert = build_coefficient.build_inclusions_change_2d(NFine,Nepsilon,alpha,beta,left,right,p,model)

            MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
            basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
            bFull = basis.T * MFull * f
            faverage = np.dot(MFull * np.ones(NpFine), f)

            #true LOD
            middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
            patchRef = PatchPeriodic(world, k, middle)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
            computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
            patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
            KFulltrue = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)

            bFull = basis.T * MFull * f
            uFulltrue, _ = pglod.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
            uLodCoarsetrue = basis * uFulltrue
            L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull*uLodCoarsetrue))

            # combined LOD
            KFullcombdef, _ = compute_combined_MsStiffness(world, Nepsilon, aPert, aRefListdef, KmsijListdef,
                                                            muTPrimeListdef,
                                                            k, model, compute_indicator=False)
            bFull = basis.T * MFull * f
            uFullcombdef, _ = pglod.solvePeriodic(world, KFullcombdef, bFull, faverage, boundaryConditions)
            uLodCoarsecombdef = basis * uFullcombdef

            error_combined = np.sqrt(np.dot(uLodCoarsetrue - uLodCoarsecombdef,
                                                 MFull * (uLodCoarsetrue - uLodCoarsecombdef))) / L2norm
            # print("L2-error in {}th sample for new LOD is: {}".format(N, error_combined))
            mean_error_combined[ii, jj] += error_combined

            #pertrubed LOD
            KFullpert, _ = compute_perturbed_MsStiffness(world, aPert, aRef, KmsijRef, muTPrimeRef, k, 0)
            bFull = basis.T * MFull * f
            uFullpert, _ = pglod.solvePeriodic(world, KFullpert, bFull, faverage, boundaryConditions)
            uLodCoarsepert = basis * uFullpert

            error_pert = np.sqrt(
                np.dot(uLodCoarsetrue - uLodCoarsepert, MFull * (uLodCoarsetrue - uLodCoarsepert))) / L2norm
            #print("L2-error in {}th sample for HKM LOD is: {}".format(N, error_pert))
            mean_error_perturbed[ii, jj] += error_pert
        mean_error_combined[ii,jj] /= NSamples
        mean_error_perturbed[ii,jj] /= NSamples
        print("mean L2-error for new LOD over {} samples for p={} and model {} is: {}".
              format(NSamples, p, model['name'], mean_error_combined[ii, jj]))
        print("mean L2-error for perturbed LOD over {} samples for p={} and model {} is: {}".
              format(NSamples, p, model['name'], mean_error_perturbed[ii,jj]))
        jj += 1

    ii += 1


print("mean error combined {}".format(mean_error_combined))
print("mean error perturbed {}".format(mean_error_perturbed))

sio.savemat('_meanErr2d_defchanges.mat',
                {'relerrL2new': mean_error_combined, 'relerrL2pert': mean_error_perturbed,
                 'pList': pList, 'names': [model['name'] for model in modelList]})