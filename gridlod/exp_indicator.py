import numpy as np
import scipy.io as sio

from gridlod.world import World, PatchPeriodic
from gridlod import util, fem, coef, lod, pglod, interp, build_coefficient
from gridlod.algorithms_random import computeCSI_offline, compute_combined_MsStiffness
from gridlod.multiplecoeff import computeErrorIndicatorFineMultiple

def computeKmsij(TInd, a, IPatch):
    #print('.', end='', flush=True)
    patch = PatchPeriodic(world, k, TInd)
    aPatch = coef.localizeCoefficient(patch,a, periodic=True)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

#2d
NFine = np.array([40, 40])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([20,20])
NCoarse = np.array([5,5])
k=2
NSamples = 500
dim = np.size(NFine)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8 * np.pi ** 2 * np.sin(2 * np.pi * x[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()

boundaryConditions = None
alpha = 0.1
beta = 1.
pList = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11]
np.random.seed(123)
model ={'name': 'check', 'alpha': alpha, 'beta': beta}

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)

aRefList, KmsijList, muTPrimeList, _, _, correctorsList\
    = computeCSI_offline(world, Nepsilon // NCoarse, k, boundaryConditions,model, True)

middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
patchRef = PatchPeriodic(world, k, middle)

for p in pList:
    ETList = []
    ETListmax = []
    ETListmiddle = []
    absErrorList = []
    relErrorList = []
    matrixErrorList = []
    defectsList = []

    for N in range(NSamples):

        aPert = build_coefficient.build_randomcheckerboard(Nepsilon, NFine, alpha, beta, p)

        MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
        basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
        bFull = basis.T * MFull * f
        faverage = np.dot(MFull * np.ones(NpFine), f)

        IPatch = lambda: interp.L2ProjectionPatchMatrix(patchRef)
        computeKmsijT = lambda TInd: computeKmsij(TInd, aPert, IPatch)
        patchT, _, KmsijTtrue, _ = zip(*map(computeKmsijT, range(world.NtCoarse)))
        KFulltrue = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTtrue, periodic=True)

        bFull = basis.T * MFull * f
        uFulltrue, _ = pglod.solvePeriodic(world, KFulltrue, bFull, faverage, boundaryConditions)
        uLodCoarsetrue = basis * uFulltrue

        # combined LOD
        KFullcomb, indic = compute_combined_MsStiffness(world, Nepsilon, aPert, aRefList, KmsijList, muTPrimeList, k,
                                                      model,True,correctorsList)
        ETs = [indic[ii][0] for ii in range(len(indic))]
        defects = indic[middle][1]
        ETs = np.array(ETs)
        ETList.append(ETs)
        ETListmax.append(np.max(ETs))
        ETListmax.append(ETs[middle])
        defectsList.append(defects)

        bFull = basis.T * MFull * f
        uFullcomb, _ = pglod.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
        uLodCoarsecomb = basis * uFullcomb

        L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
        error_combined = np.sqrt(
            np.dot(uLodCoarsetrue - uLodCoarsecomb, MFull * (uLodCoarsetrue - uLodCoarsecomb)))
        absErrorList.append(error_combined)
        relErrorList.append(error_combined/L2norm)
        matrixError=np.linalg.norm((KFulltrue-KFullcomb).todense(),2)
        matrixErrorList.append(matrixError)

    print("mean relative L2-error for p = {} is: {}".format(p, np.mean(relErrorList)))
    print("mean constistency (matrix) error for p = {} is: {}".format(p, np.mean(matrixErrorList)))
    print("mean maximal error indicator for p = {} is: {}".format(p, np.mean(ETListmax)))

    sio.savemat('_ErrIndic2drandcheck_p'+str(p)+'.mat', {'ETListloc': ETList, 'ETListmax': ETListmax,
                                                             'ETListmiddle': ETListmiddle,'defectsList': defectsList,
                                                             'absError': absErrorList, 'relError': relErrorList,
                                                             'matrixError': matrixErrorList})
