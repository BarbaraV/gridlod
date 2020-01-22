import numpy as np
import matplotlib.pyplot as plt

from gridlod import util, world, coef, interp, lod, pglod, fem
from gridlod.world import World, Patch

# this fine resolution should be enough
fine = 512
NFine = np.array([fine])
NpFine = np.prod(NFine + 1)
# list of coarse meshes
NList = [32]


def computeRmsi(TInd, a):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, a)

    MRhsList = [f[util.extractElementFine(world.NWorldCoarse,
                                            world.NCoarseElement,
                                            patch.iElementWorldCoarse,
                                            extractElements=False)]];

    correctorRhs = lod.computeElementCorrector(patch, IPatch, aPatch, None, MRhsList)[0]
    Rmsi, _ = lod.computeRhsCoarseQuantities(patch, correctorRhs, aPatch, True)

    return patch, correctorRhs, Rmsi

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

def UpdateCorrectors(TInd, aPert):
    # print(" UPDATING {}".format(TInd))
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)

    MRhsList = [f[util.extractElementFine(world.NWorldCoarse,
                                              world.NCoarseElement,
                                              patch.iElementWorldCoarse,
                                              extractElements=False)]];


    rPatch = lambda: coef.localizeCoefficient(patch, aPert)
    correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)

    correctorRhs = lod.computeElementCorrector(patch, IPatch, rPatch, None, MRhsList)[0]
    Rmsij = lod.computeRhsCoarseQuantities(patch, correctorRhs, rPatch)

    return patch, correctorsList, csi.Kmsij, correctorRhs, Rmsij

def UpdateElements(tol, E, Kmsij_old, correctors_old, Rmsij_old, correctorsRhs_old, aPert):
    #_old are treated as lists of quantities for different reference coefficients!
    print('apply tolerance')
    Elements_to_be_updated = []
    alphaTList = []
    for (i,eps) in E.items():
        if eps[0] > tol:# or eps[1] > tol:
            Elements_to_be_updated.append(i)
        else:
            alphaTList.append(eps[1])
    if len(E) > 0:
        print('... to be updated: {}%'.format(100*np.size(Elements_to_be_updated)/len(E)), end='\n', flush=True)

    KmsijT_list = list(np.copy(Kmsij_old[0]))
    RmsijT_list = list(np.copy(Rmsij_old[0]))
    correctorsListT_list = list(np.copy(correctors_old[0]))
    correctorsRhs_list = list(np.copy(correctorsRhs_old[0]))

    j = 0
    for T in np.setdiff1d(range(world.NtCoarse), Elements_to_be_updated):
        KmsijT_list[T] *= 0
        RmsijT_list[T] *= 0
        correctorsListT_list[T] = list(0*np.array(correctorsListT_list[T]))
        correctorsRhs_list[T] = 0 * correctorsRhs_list[T]
        alphaTList[j] /= 1.086*np.linalg.norm(alphaTList[j]) #verbessert die Dinge in Test 2, warum & wie optimaler Wert?!
        for kk in range(len(Kmsij_old)):
            KmsijT_list[T] += alphaTList[j][kk] * Kmsij_old[kk][T]
            correctorsListT_list[T] += alphaTList[j][kk] * np.array(correctors_old[kk][T])
            RmsijT_list[T] += alphaTList[j][kk] * Rmsij_old[kk][T]
            correctorsRhs_list[T] += alphaTList[j][kk] * correctorsRhs_old[kk][T]
        j += 1


    if np.size(Elements_to_be_updated) != 0:
        print('... update correctors')
        UpdateCorrectors1 = lambda TInd: UpdateCorrectors(TInd, aPert)
        patchT_irrelevant, correctorsListTNew, KmsijTNew, correctorsRhsTNew, RmsijTNew = zip(*map(UpdateCorrectors1,
                                                                             Elements_to_be_updated))

        print('replace Kmsij and update correctorsListT')
        i = 0
        for T in Elements_to_be_updated:
            KmsijT_list[T] = KmsijTNew[i]
            correctorsListT_list[T] = correctorsListTNew[i]
            RmsijT_list[T] = RmsijTNew[i]
            correctorsRhs_list[T] = correctorsRhsTNew[i]
            i += 1
    else:
        print('... there is nothing to be updated')

    KmsijT = tuple(KmsijT_list)
    correctorsListT = tuple(correctorsListT_list)
    RmsijT = tuple(RmsijT_list)
    correctorsRhsT = tuple(correctorsRhs_list)

    return KmsijT, correctorsListT, RmsijT, correctorsRhsT

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
error_pert1 = []
error_pert2 = []
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

    print('computing correctors', end='', flush=True)
    computeKmsij1 = lambda TInd: computeKmsij(TInd, aRef1)
    computeRmsi1 = lambda TInd: computeRmsi(TInd, aRef1)
    patchT, correctorsList1, KmsijT1, csiT1 = zip(*map(computeKmsij1, range(world.NtCoarse)))
    _, correctorsRhsList1, RmsijT1 = zip(*map(computeRmsi1, range(world.NtCoarse)))
    print()
    print('computing correctors', end='', flush=True)
    computeKmsij2 = lambda TInd: computeKmsij(TInd, aRef2)
    computeRmsi2 = lambda TInd: computeRmsi(TInd, aRef2)
    _, correctorsList2, KmsijT2, csiT2 = zip(*map(computeKmsij2, range(world.NtCoarse)))
    _, correctorsRhsList2, RmsijT2 = zip(*map(computeRmsi2, range(world.NtCoarse)))
    print()
    print()

    csiList =[csiT1, csiT2]
    aRefList = [aRef1, aRef2]
    KmsijList = [KmsijT1, KmsijT2]
    correctorsList = [correctorsList1, correctorsList2]
    RmsiList = [RmsijT1, RmsijT2]
    correctorsRhsList = [correctorsRhsList1, correctorsRhsList2]

    computeIndicator1 = lambda TInd: computeIndicator(TInd, aRefList, aPert1)
    computeIndicator2 = lambda TInd: computeIndicator(TInd, aRefList, aPert2)

    alpha = None#np.array([0.5, 0.5])
    print('computing error indicators', end='', flush=True)
    E_vh1, alphaList1 = zip(*map(computeIndicator1, range(world.NtCoarse)))
    print()
    print('max error perturbed1  for alpha={}: {}'.format(alphaList1[np.argmax(E_vh1)],max(E_vh1)))
    error_indicator_pert1.append(E_vh1)
    print('computing error indicators', end='', flush=True)
    E_vh2, alphaList2 = zip(*map(computeIndicator2, range(world.NtCoarse)))
    print()
    print('max error perturbed2 alpha={}: {}'.format(alphaList2[np.argmax(E_vh2)],max(E_vh2)))
    error_indicator_pert2.append(E_vh2)
    E1 = {i: [E_vh1[i], alphaList1[i]] for i in range(np.size(E_vh1))}
    E2 = {i: [E_vh2[i], alphaList2[i]] for i in range(np.size(E_vh2))}

    print('compute true LOD solution for perturbed coefficient 1')
    computeKmsijP1 = lambda TInd: computeKmsij(TInd, aPert1)
    computeRmsi1 = lambda TInd: computeRmsi(TInd, aPert1)
    print('computing real correctors', end='', flush=True)
    _, correctorsListT, KmsijT, _ = zip(*map(computeKmsijP1, range(world.NtCoarse)))
    print()
    print('computing real right hand side correctors', end='', flush=True)
    _, correctorRhsT, RmsiT = zip(*map(computeRmsi1, range(world.NtCoarse)))
    print()
    #
    KFull1 = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
    RFull1 = pglod.assemblePatchFunction(world, patchT, RmsiT)
    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
    Rf1 = pglod.assemblePatchFunction(world, patchT, correctorRhsT)
    basis1 = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    bFull1 = basis1.T * MFull * f - RFull1
    basisCorrectors1 = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    modifiedBasis1 = basis1 - basisCorrectors1
    uFull1, _ = pglod.solve(world, KFull1, bFull1, boundaryConditions)
    uLodFine1 = modifiedBasis1 * uFull1
    uLodFine1 += Rf1

    print('compute true LOD solution for perturbed coefficient 2')
    computeKmsijP2 = lambda TInd: computeKmsij(TInd, aPert2)
    computeRmsi2 = lambda TInd: computeRmsi(TInd, aPert2)
    print('computing real correctors', end='', flush=True)
    _, correctorsListT, KmsijT, _ = zip(*map(computeKmsijP2, range(world.NtCoarse)))
    print()
    print('computing real right hand side correctors', end='', flush=True)
    _, correctorRhsT, RmsiT = zip(*map(computeRmsi2, range(world.NtCoarse)))
    print()
    #
    KFull2 = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
    RFull2 = pglod.assemblePatchFunction(world, patchT, RmsiT)
    Rf2 = pglod.assemblePatchFunction(world, patchT, correctorRhsT)
    basis2 = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    bFull2 = basis2.T * MFull * f - RFull2
    basisCorrectors2 = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    modifiedBasis2 = basis2 - basisCorrectors2
    uFull2, _ = pglod.solve(world, KFull2, bFull2, boundaryConditions)
    uLodFine2 = modifiedBasis2 * uFull2
    uLodFine2 += Rf2

    tol= np.inf
    print('compute new LOD for perturbed coefficient 1')
    KmsijT, correctorsListT, RmsijT, correctorsRhsT = UpdateElements(tol, E1, KmsijList,
                                                                             correctorsList, RmsiList, correctorsRhsList, aPert1)
    # LOD solve
    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
    RFull = pglod.assemblePatchFunction(world, patchT, RmsijT)
    Rf = pglod.assemblePatchFunction(world, patchT, correctorsRhsT)
    bFull = basis1.T * MFull * f - RFull
    basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    modifiedBasis = basis1 - basisCorrectors
    uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)
    uLodFine_pert1 = modifiedBasis * uFull
    uLodFine_pert1 += Rf
    AFine_pert1 = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine, aPert1)
    energy_error = np.sqrt(np.dot((uLodFine_pert1 - uLodFine1), AFine_pert1 * (uLodFine_pert1 - uLodFine1)))
    print("Energy norm  error {}".format(energy_error))
    error_pert1.append(energy_error)

    print('compute new LOD for perturbed coefficient 2')
    KmsijT, correctorsListT, RmsijT, correctorsRhsT = UpdateElements(tol, E2, KmsijList,
                                                                     correctorsList, RmsiList, correctorsRhsList, aPert2)
    # LOD solve
    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
    RFull = pglod.assemblePatchFunction(world, patchT, RmsijT)
    Rf = pglod.assemblePatchFunction(world, patchT, correctorsRhsT)
    bFull = basis2.T * MFull * f - RFull
    basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    modifiedBasis = basis2 - basisCorrectors
    uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)
    uLodFine_pert2 = modifiedBasis * uFull
    uLodFine_pert2 += Rf
    AFine_pert2 = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine, aPert2)
    energy_error = np.sqrt(np.dot((uLodFine_pert2 - uLodFine2), AFine_pert2 * (uLodFine_pert2 - uLodFine2)))
    print("Energy norm  error {}".format(energy_error))
    error_pert2.append(energy_error)


# plot the indicators
plt.figure('error indicators')
#plt.subplots_adjust(left=0.01, bottom=0.04, right=0.99, top=0.95, wspace=0.1, hspace=0.2)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
for i in range(len(NList)):
    plt.subplot(1,2,i+1)
    plt.bar(x[i], error_indicator_pert1[i], width=0.03, color='r', label='perturbed1')
    #plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
    #               labelleft=False)
    plt.legend(frameon=False, fontsize=16)
    plt.subplot(1,2,len(NList)+i+1)
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
error_pert = []
k=4

for N in NList:
    NWorldCoarse = np.array([N])
    boundaryConditions = np.array([[0, 0]])

    NCoarseElement = NFine // NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    # grid nodes
    xtCoarse = util.tCoordinates(NWorldCoarse).flatten()
    x.append(xtCoarse)
    NpCoarse = np.prod(NWorldCoarse + 1)

    print('computing correctors', end='', flush=True)
    computeKmsij1 = lambda TInd: computeKmsij(TInd, aRef1)
    computeRmsi1 = lambda TInd: computeRmsi(TInd, aRef1)
    _, correctorsRhsList1, RmsijT1 = zip(*map(computeRmsi1, range(world.NtCoarse)))
    patchT, correctorsList1, KmsijT1, csiT1 = zip(*map(computeKmsij1, range(world.NtCoarse)))
    print()
    print('computing correctors', end='', flush=True)
    computeKmsij2 = lambda TInd: computeKmsij(TInd, aRef2)
    computeRmsi2 = lambda TInd: computeRmsi(TInd, aRef2)
    _, correctorsRhsList2, RmsijT2 = zip(*map(computeRmsi2, range(world.NtCoarse)))
    _, correctorsList2, KmsijT2, csiT2 = zip(*map(computeKmsij2, range(world.NtCoarse)))
    print()
    print('computing correctors', end='', flush=True)
    computeKmsij3 = lambda TInd: computeKmsij(TInd, aRef3)
    computeRmsi3 = lambda TInd: computeRmsi(TInd, aRef3)
    _, correctorsRhsList3, RmsijT3 = zip(*map(computeRmsi3, range(world.NtCoarse)))
    _, correctorsList3, KmsijT3, csiT3 = zip(*map(computeKmsij3, range(world.NtCoarse)))
    print()

    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    bFull = basis.T * MFull * f  # - RFull1
    KFull1 = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT1)
    uFull1, _ = pglod.solve(world, KFull1, bFull, boundaryConditions)
    basisCorrectors1 = pglod.assembleBasisCorrectors(world, patchT, correctorsList1)
    modifiedBasis1 = basis- basisCorrectors1
    uLodFine_ref1 = modifiedBasis1 * uFull1
    #
    KFull2 = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT2)
    uFull2, _ = pglod.solve(world, KFull2, bFull, boundaryConditions)
    basisCorrectors2 = pglod.assembleBasisCorrectors(world, patchT, correctorsList2)
    modifiedBasis2 = basis- basisCorrectors2
    uLodFine_ref2 = modifiedBasis2 * uFull2
    #
    KFull3 = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT3)
    uFull3, _ = pglod.solve(world, KFull3, bFull, boundaryConditions)
    basisCorrectors3 = pglod.assembleBasisCorrectors(world, patchT, correctorsList3)
    modifiedBasis3 = basis- basisCorrectors3
    uLodFine_ref3 = modifiedBasis3 * uFull3

    csiList =[csiT1, csiT2, csiT3]
    aRefList = [aRef1, aRef2, aRef3]
    KmsijList = [KmsijT1, KmsijT2, KmsijT3]
    correctorsList = [correctorsList1, correctorsList2, correctorsList3]
    RmsiList = [RmsijT1, RmsijT2, RmsijT3]
    correctorsRhsList = [correctorsRhsList1, correctorsRhsList2, correctorsRhsList3]

    computeIndicatorPert = lambda TInd: computeIndicator(TInd, aRefList, aPert)

    alpha = None#np.array([1./3, 1./3., 1./3.])
    print('computing error indicators', end='', flush=True)
    E_vh, alphaList = zip(*map(computeIndicatorPert, range(world.NtCoarse)))
    print()
    print('max error perturbed1  for alpha={}: {}'.format(alphaList[np.argmax(E_vh)], max(E_vh)))
    error_indicator_pert.append(E_vh)
    E1 = {i: [E_vh[i], alphaList[i]] for i in range(np.size(E_vh))}

    print('compute true LOD solution for perturbed coefficient 1')
    computeKmsijP1 = lambda TInd: computeKmsij(TInd, aPert)
    computeRmsi1 = lambda TInd: computeRmsi(TInd, aPert)
    print('computing real correctors', end='', flush=True)
    _, correctorsListT, KmsijT, _ = zip(*map(computeKmsijP1, range(world.NtCoarse)))
    print()
    print('computing real right hand side correctors', end='', flush=True)
    _, correctorRhsT, RmsiT = zip(*map(computeRmsi1, range(world.NtCoarse)))
    print()
    #
    KFull1 = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
    RFull1 = pglod.assemblePatchFunction(world, patchT, RmsiT)
    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
    Rf1 = pglod.assemblePatchFunction(world, patchT, correctorRhsT)
    basis1 = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    bFull1 = basis1.T * MFull * f - RFull1
    basisCorrectors1 = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    modifiedBasis1 = basis1 - basisCorrectors1
    uFull1, _ = pglod.solve(world, KFull1, bFull1, boundaryConditions)
    uLodFine1 = modifiedBasis1 * uFull1
    uLodCoarse1 = basis1 * uFull1
    uLodFine1 += Rf1

    tol = np.inf
    print('compute new LOD for perturbed coefficient 1')
    KmsijT, correctorsListT, RmsijT, correctorsRhsT = UpdateElements(tol, E1, KmsijList,
                                                                     correctorsList, RmsiList, correctorsRhsList,
                                                                     aPert)
    # LOD solve
    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
    RFull = pglod.assemblePatchFunction(world, patchT, RmsijT)
    Rf = pglod.assemblePatchFunction(world, patchT, correctorsRhsT)
    bFull = basis1.T * MFull * f - RFull
    basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    modifiedBasis = basis1 - basisCorrectors
    uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)
    uLodFine_pert1 = modifiedBasis * uFull
    uLodCoarse_pert1 = basis1 * uFull
    uLodFine_pert1 += Rf
    AFine_pert1 = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine, aPert)
    energy_norm = np.sqrt(np.dot(uLodFine_pert1, AFine_pert1*uLodFine_pert1))
    energy_error = np.sqrt(np.dot((uLodFine_pert1 - uLodFine1), AFine_pert1 * (uLodFine_pert1 - uLodFine1)))
    print("Energy norm  error {}, relative error {}".format(energy_error, energy_error/energy_norm))
    error_pert.append(energy_error)
    L2norm = np.sqrt(np.dot(uLodCoarse_pert1, MFull*uLodCoarse_pert1))
    L2_error = np.sqrt(np.dot(uLodCoarse1-uLodCoarse_pert1, MFull*(uLodCoarse1-uLodCoarse_pert1)))
    print("L2-norm error {}, relative error {}".format(L2_error, L2_error/L2norm))

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


plt.figure('full LOD solutions')
plt.plot(util.pCoordinates(world.NWorldFine), uLodFine1, color='r', label='true')
plt.plot(util.pCoordinates(world.NWorldFine), uLodFine_pert1, color='b', label='with reference')
#plt.plot(util.pCoordinates(world.NWorldFine), uLodFine_ref1, label='ref1')
#plt.plot(util.pCoordinates(world.NWorldFine), uLodFine_ref2, label='ref2')
#plt.plot(util.pCoordinates(world.NWorldFine), uLodFine_ref3, label='ref3')
plt.legend()

plt.figure('FE part LOD solutions')
plt.plot(util.pCoordinates(world.NWorldFine), uLodCoarse1, color='r', label='true')
plt.plot(util.pCoordinates(world.NWorldFine), uLodCoarse_pert1, color='b', label='with reference')
plt.legend()

plt.show()