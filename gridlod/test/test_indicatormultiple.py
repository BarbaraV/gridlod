import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, world, coef, interp, lod
from gridlod.world import World, Patch

# this fine resolution should be enough
fine = 512
NFine = np.array([fine])
NpFine = np.prod(NFine + 1)
# list of coarse meshes
NList = [8, 16, 32]


#==================================================================================================================
#Test1
#====================================================================================================================
'''aRef1 = np.ones(fine)
aRef1 /= 10
aRef2 = np.copy(aRef1)

for i in range(int(fine* 2/8.) - 1, int(fine * 3/8.) - 1):
    aRef2[i] = 1

aPert1 = np.copy(aRef1)
for i in range(int(fine* 5/8.) - 1, int(fine * 6/8.) - 1):
    aPert1[i] = 1
aPert2 = np.copy(aRef1)
for i in range(int(fine* 2/8.) - 1, int(fine * 4/8.) - 1):
    aPert2[i] = 1


xpCoarse = util.pCoordinates(NFine).flatten()
xtCoarse = util.tCoordinates(NFine).flatten()

# interior nodes for plotting
xt = util.tCoordinates(NFine).flatten()

# This is the right hand side
f = np.ones(fine + 1)

# plot coefficients and compare them
plt.figure('Coefficient_pert')
plt.plot(xt, aRef1, label='$Aref1$')
plt.plot(xt, aRef2, label='$Aref2$')
plt.plot(xt, aPert1, label='$Apert1$')
plt.plot(xt, aPert2, label='$Apert2$')
plt.grid(True)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=16)
plt.ylabel('$y$', fontsize=16)
plt.xlabel('$x$', fontsize=16)
plt.legend(frameon=False, fontsize=16)

error_indicator_pert1 = []
error_indicator_pert2 = []
x = []
k=2

for N in NList:
    NWorldCoarse = np.array([N])
    boundaryConditions = np.array([[0, 0]])

    NCoarseElement = NFine // NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    # grid nodes
    xtCoarse = util.tCoordinates(NWorldCoarse).flatten()
    x.append(xtCoarse)
    NpCoarse = np.prod(NWorldCoarse + 1)


    def computeKmsij(TInd, a):
        print('.', end='', flush=True)
        patch = Patch(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
        aPatch = lambda: coef.localizeCoefficient(patch, a)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij, csi


    def computeIndicator(TInd, aRefList, aPert):
        print('.', end='', flush=True)
        aPatch = [coef.localizeCoefficient(patchT[TInd], aRef) for aRef in aRefList] #why not working with lambda?
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aPert)
        muTPrimeList = [csi[TInd].muTPrime for csi in csiList]

        E_vh, alphaT = lod.computeErrorIndicatorCoarseMultiple(patchT[TInd], muTPrimeList, aPatch, rPatch, alpha)

        return E_vh, alphaT


    print('computing correctors', end='', flush=True)
    computeKmsij1 = lambda TInd: computeKmsij(TInd, aRef1)
    patchT, _, _, csiT1 = zip(*map(computeKmsij1, range(world.NtCoarse)))
    print()
    print('computing correctors', end='', flush=True)
    computeKmsij2 = lambda TInd: computeKmsij(TInd, aRef2)
    _, _, _, csiT2 = zip(*map(computeKmsij2, range(world.NtCoarse)))
    print()

    csiList =[csiT1, csiT2]
    aRefList = [aRef1, aRef2]

    computeIndicator1 = lambda TInd: computeIndicator(TInd, aRefList, aPert1)
    computeIndicator2 = lambda TInd: computeIndicator(TInd, aRefList, aPert2)

    alpha = None#np.array([0.5, 0.5])
    print('computing error indicators', end='', flush=True)
    E_vh, alphaList = zip(*map(computeIndicator1, range(world.NtCoarse)))
    print()
    print('max error perturbed1  for alpha={}: {}'.format(alphaList[np.argmax(E_vh)],max(E_vh)))
    error_indicator_pert1.append(E_vh)
    print('computing error indicators', end='', flush=True)
    E_vh, alphaList = zip(*map(computeIndicator2, range(world.NtCoarse)))
    print()
    print('max error perturbed2 alpha={}: {}'.format(alphaList[np.argmax(E_vh)],max(E_vh)))
    error_indicator_pert2.append(E_vh)



# plot the indicators
plt.figure('error indicators', figsize=(16, 9))
plt.subplots_adjust(left=0.01, bottom=0.04, right=0.99, top=0.95, wspace=0.1, hspace=0.2)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
for i in range(len(NList)):
    plt.subplot(2,len(NList),i+1)
    plt.bar(x[i], error_indicator_pert1[i], width=0.03, color='r', label='perturbed1')
    #plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
    #                labelleft=False)
    plt.legend(frameon=False, fontsize=16)
    plt.subplot(2,len(NList),len(NList)+i+1)
    plt.bar(x[i], error_indicator_pert2[i], width=0.03, color='b', label='perturbed2')
    plt.legend()

plt.show()'''

#====================================================================================================================
#Test 2

aRef1 = np.ones(fine)
aRef1 /= 10
aRef2 = np.copy(aRef1)

for i in range(int(fine* 2/8.) - 1, int(fine * 3/8.) - 1):
    aRef2[i] = 1

aRef3 = np.copy(aRef1)
for i in range(int(fine* 5/8.) - 1, int(fine * 6/8.) - 1):
    aRef3[i] = 1

aPert = np.copy(aRef2)
for i in range(int(fine* 5/8.) - 1, int(fine * 6/8.) - 1):
    aPert[i] = 1


xpCoarse = util.pCoordinates(NFine).flatten()
xtCoarse = util.tCoordinates(NFine).flatten()

# interior nodes for plotting
xt = util.tCoordinates(NFine).flatten()

# This is the right hand side
f = np.ones(fine + 1)

# plot coefficients and compare them
plt.figure('Coefficient_pert')
plt.plot(xt, aRef1, label='$Aref1$')
plt.plot(xt, aRef2, label='$Aref2$')
plt.plot(xt, aRef3, label='$Aref3$')
plt.plot(xt, aPert, label='$Apert$')
plt.grid(True)
plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=16)
plt.ylabel('$y$', fontsize=16)
plt.xlabel('$x$', fontsize=16)
plt.legend(frameon=False, fontsize=16)

error_indicator_pert = []
x = []
k=2

for N in NList:
    NWorldCoarse = np.array([N])
    boundaryConditions = np.array([[0, 0]])

    NCoarseElement = NFine // NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    # grid nodes
    xtCoarse = util.tCoordinates(NWorldCoarse).flatten()
    x.append(xtCoarse)
    NpCoarse = np.prod(NWorldCoarse + 1)


    def computeKmsij(TInd, a):
        print('.', end='', flush=True)
        patch = Patch(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
        aPatch = lambda: coef.localizeCoefficient(patch, a)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
        return patch, correctorsList, csi.Kmsij, csi


    def computeIndicator(TInd, aRefList, aPert):
        print('.', end='', flush=True)
        aPatch = [coef.localizeCoefficient(patchT[TInd], aRef) for aRef in aRefList]
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd], aPert)
        muTPrimeList = [csi[TInd].muTPrime for csi in csiList]

        E_vh, alphaT = lod.computeErrorIndicatorCoarseMultiple(patchT[TInd], muTPrimeList, aPatch, rPatch, alpha)

        return E_vh, alphaT


    print('computing correctors', end='', flush=True)
    computeKmsij1 = lambda TInd: computeKmsij(TInd, aRef1)
    patchT, _, _, csiT1 = zip(*map(computeKmsij1, range(world.NtCoarse)))
    print()
    print('computing correctors', end='', flush=True)
    computeKmsij2 = lambda TInd: computeKmsij(TInd, aRef2)
    _, _, _, csiT2 = zip(*map(computeKmsij2, range(world.NtCoarse)))
    print()
    print('computing correctors', end='', flush=True)
    computeKmsij3 = lambda TInd: computeKmsij(TInd, aRef3)
    _, _, _, csiT3 = zip(*map(computeKmsij3, range(world.NtCoarse)))
    print()

    csiList =[csiT1, csiT2, csiT3]
    aRefList = [aRef1, aRef2, aRef3]

    computeIndicatorPert = lambda TInd: computeIndicator(TInd, aRefList, aPert)

    alpha = None#np.array([1./3, 1./3., 1./3.])
    print('computing error indicators', end='', flush=True)
    E_vh, alphaList = zip(*map(computeIndicatorPert, range(world.NtCoarse)))
    print()
    print('max error perturbed1  for alpha={}: {}'.format(alphaList[np.argmax(E_vh)], max(E_vh)))
    error_indicator_pert.append(E_vh)


# plot the indicators
plt.figure('error indicators', figsize=(16, 9))
plt.subplots_adjust(left=0.01, bottom=0.04, right=0.99, top=0.95, wspace=0.1, hspace=0.2)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
for i in range(len(NList)):
    plt.subplot(1,len(NList),i+1)
    plt.bar(x[i], error_indicator_pert[i], width=0.03, color='r', label='perturbed1')
    #plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
    #                labelleft=False)
    plt.legend(frameon=False, fontsize=16)

plt.show()