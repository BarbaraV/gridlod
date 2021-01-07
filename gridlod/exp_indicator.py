import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

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

#1d
'''NFine = np.array([256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([256])
NList = [32]
k=3
NSamples = 5000
dim = np.size(NFine)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8 * np.pi ** 2 * np.sin(2 * np.pi * x)
f = ffunc(xpFine).flatten()

boundaryConditions = None
alpha = 0.1
beta = 1.
p = 0.1
np.random.seed(123)
model ={'name': 'check', 'alpha': alpha, 'beta': beta}

for Nc in NList:
    NCoarse = np.array([Nc])
    NCoarseElement = NFine // NCoarse
    world = World(NCoarse, NCoarseElement, boundaryConditions)

    aRefList, KmsijList, muTPrimeList, _, _, correctorsList\
        = computeCSI_offline(world, Nepsilon // NCoarse, k, boundaryConditions,model, True)

    middle = NCoarse[0] // 2
    patchRef = PatchPeriodic(world, k, middle)

    ETList = []
    ETListloc = []
    error_combinedList =[]
    #defectsList =[]

    for N in range(NSamples):
        mu = np.zeros(len(aRefList))
        mu[:len(aRefList)-1] = np.random.binomial(1,p,len(aRefList)-1)
        #defects = np.sum(mu[:len(aRefList)-1])
        mu[-1] = 1 - np.sum(mu[:len(aRefList)-1])
        ET = computeErrorIndicatorFineMultiple(patchRef,correctorsList,aRefList,mu)
        #print("maximal local ET in {}th sample is {}".format(N,ET))
        ETListloc.append(ET)
        #defectsList.append(defects)

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

        #combined LOD
        KFullcomb, ETs = compute_combined_MsStiffness(world,Nepsilon,aPert,aRefList,KmsijList,muTPrimeList,k,model,
                                                                      True, correctorsList)
        ETs = np.array(ETs)
        ETList.append(np.max(ETs))
        bFull = basis.T * MFull * f
        uFullcomb, _ = pglod.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
        uLodCoarsecomb = basis * uFullcomb

        L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
        error_combined = np.sqrt(np.dot(uLodCoarsetrue-uLodCoarsecomb, MFull*(uLodCoarsetrue-uLodCoarsecomb)))/L2norm
        error_combinedList.append(error_combined)
        #print("L2-error in {}th sample for new LOD is: {}".format(N, error_combined))
        #print("maximal error indicator in {}th sample for new LOD is: {}".format(N, max(ETs)))
        #print("efficiency index indicator/error in {}th sample is {}".format(N, max(ETs)/error_combined))

        #plt.figure(1)
        #plt.bar(np.arange(NFine[0]), aPert)
        #plt.figure(2)
        #plt.bar(np.arange(Nc), ETs)
        #plt.show()

    #print('maximal ET {} in {}th sample'.format(max(ETList), np.argmax(ETList)))

    samplemax = [np.max(np.array(ETListloc)[:(ii+1)]) for ii in range(NSamples)]
    samplemean = [np.nanmean(np.array(ETList)[:(ii + 1)]) for ii in range(NSamples)]
    samplemeanloc = [np.mean(samplemax[:(ii + 1)]) for ii in range(NSamples)]
    samplemean_error = [np.mean(np.array(error_combinedList)[:(ii+1)]) for ii in range(NSamples)]
    plt.figure(1)
    plt.plot(np.arange(NSamples), samplemeanloc)
    plt.figure(2)
    plt.plot(np.arange(NSamples), samplemax)
    plt.figure(3)
    plt.plot(np.arange(NSamples), samplemean)
    plt.figure(4)
    plt.plot(np.arange(NSamples), samplemean_error)
    plt.show()
    print("mean local maximal ET {}".format(samplemeanloc[-1]))
    print("mean maximal ET {}".format(samplemean[-1]))
    print("mean L2 error {}".format(samplemean_error[-1]))
    #print('maximal L2 error {} in {}th sample'.format(max(error_combinedList), np.argmax(error_combinedList)))

    #mu = np.ones(len(aRefList))
    #mu[-1] = 1 - np.sum(mu[:len(aRefList) - 1])
    #ETone = computeErrorIndicatorFineMultiple(patchRef, correctorsList, aRefList, mu)
    #print("ET with all defects {}".format((ETone)))
'''
#2d
NFine = np.array([256, 256])
NpFine = np.prod(NFine+1)
Nepsilon = np.array([128,128])
NList = [32]
k=4
NSamples = 2000
dim = np.size(NFine)

xpFine = util.pCoordinates(NFine)
ffunc = lambda x: 8 * np.pi ** 2 * np.sin(2 * np.pi * x[:,0])*np.cos(2*np.pi*x[:,1])
f = ffunc(xpFine).flatten()

boundaryConditions = None
alpha = 0.1
beta = 1.
p = 0.1
np.random.seed(123)
model ={'name': 'check', 'alpha': alpha, 'beta': beta}

for Nc in NList:
    NCoarse = np.array([Nc, Nc])
    NCoarseElement = NFine // NCoarse
    world = World(NCoarse, NCoarseElement, boundaryConditions)

    aRefList, KmsijList, muTPrimeList, _, _, correctorsList\
        = computeCSI_offline(world, Nepsilon // NCoarse, k, boundaryConditions,model, True)

    middle = world.NWorldCoarse[1] // 2 * world.NWorldCoarse[0] + world.NWorldCoarse[0] // 2
    patchRef = PatchPeriodic(world, k, middle)

    ETList = []
    ETListloc = []
    error_combinedList = []
    defectsList = []

    for N in range(NSamples):
        mu = np.zeros(len(aRefList))
        mu[:len(aRefList) - 1] = np.random.binomial(1, p, len(aRefList) - 1)
        defects = np.sum(mu[:len(aRefList)-1])
        mu[-1] = 1 - np.sum(mu[:len(aRefList) - 1])
        ET = computeErrorIndicatorFineMultiple(patchRef, correctorsList, aRefList, mu)
        print("maximal local ET in {}th sample is {}".format(N,ET))
        ETListloc.append(ET)
        defectsList.append(defects)

        '''aPert = build_coefficient.build_randomcheckerboard(Nepsilon, NFine, alpha, beta, p)

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
        KFullcomb, ETs = compute_combined_MsStiffness(world, Nepsilon, aPert, aRefList, KmsijList, muTPrimeList, k,
                                                      model,True, correctorsList)
        ETs = np.array(ETs)
        ETList.append(np.max(ETs))
        bFull = basis.T * MFull * f
        uFullcomb, _ = pglod.solvePeriodic(world, KFullcomb, bFull, faverage, boundaryConditions)
        uLodCoarsecomb = basis * uFullcomb

        L2norm = np.sqrt(np.dot(uLodCoarsetrue, MFull * uLodCoarsetrue))
        error_combined = np.sqrt(
            np.dot(uLodCoarsetrue - uLodCoarsecomb, MFull * (uLodCoarsetrue - uLodCoarsecomb))) / L2norm
        error_combinedList.append(error_combined)
        print("L2-error in {}th sample for new LOD is: {}".format(N, error_combined))
        print("maximal error indicator in {}th sample for new LOD is: {}".format(N, max(ETs)))
        # print("efficiency index indicator/error in {}th sample is {}".format(N, max(ETs)/error_combined))'''

    #print('maximal ET {}'.format(max(ETList)))

    #mu = np.ones(len(aRefList))
    #mu[-1] = 1 - np.sum(mu[:len(aRefList) - 1])
    #ETone = computeErrorIndicatorFineMultiple(patchRef, correctorsList, aRefList, mu)
    #print("ET with all defects {}".format((ETone)))

    #samplemax = [np.max(np.array(ETListloc)[:(ii + 1)]) for ii in range(NSamples)]
    #samplemean = [np.nanmean(np.array(ETList)[:(ii + 1)]) for ii in range(NSamples)]
    #samplemeanloc = [np.mean(samplemax[:(ii + 1)]) for ii in range(NSamples)]
    #samplemean_error = [np.mean(np.array(error_combinedList)[:(ii + 1)]) for ii in range(NSamples)]
    #plt.figure(1)
    #plt.scatter(defectsList, ETListloc)
    #plt.figure(2)
    #plt.plot(np.arange(NSamples), samplemax)
    #plt.figure(3)
    #plt.plot(np.arange(NSamples), samplemean)
    #plt.figure(4)
    #plt.plot(np.arange(NSamples), samplemean_error)
    #plt.show()
    #print("mean local maximal ET {}".format(samplemeanloc[-1]))
    #print("mean maximal ET {}".format(samplemean[-1]))
    #print("mean L2 error {}".format(samplemean_error[-1]))

    import scipy.io as sio
    sio.savemat('_ErrIndicLoc2drandcheck.mat', {'ETListloc': ETListloc, 'defectsListloc': defectsList})
