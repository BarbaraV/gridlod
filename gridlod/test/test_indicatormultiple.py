import numpy as np
import matplotlib.pyplot as plt
import copy

from gridlod import util, world, coef, interp, lod, pglod, fem
from gridlod.world import World, Patch

# this fine resolution should be enough
fine = 256
NFine = np.array([fine])#, fine
NpFine = np.prod(NFine + 1)
# list of coarse meshes
NList = [8,16,32,64]


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

    E_vh, alphaT = lod.computeErrorIndicatorCoarseMultiple(patchT[TInd], muTPrimeList, aPatch, rPatch, alpha, mu)

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

    KmsijT_list = list(copy.deepcopy(Kmsij_old[0]))
    RmsijT_list = list(copy.deepcopy(Rmsij_old[0]))
    correctorsListT_list = list(copy.deepcopy(correctors_old[0]))
    correctorsRhs_list = list(copy.deepcopy(correctorsRhs_old[0]))

    j = 0
    for T in np.setdiff1d(range(world.NtCoarse), Elements_to_be_updated):
        KmsijT_list[T] *= 0
        RmsijT_list[T] *= 0
        correctorsListT_list[T] = list(0*np.array(correctorsListT_list[T]))
        correctorsRhs_list[T] = 0 * correctorsRhs_list[T]
        #alphaTList[j] /= 1.086*np.linalg.norm(alphaTList[j]) #verbessert die Dinge in Test 2, warum & wie optimaler Wert?!
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

#===============================
# a) 1 D test
#===============================
'''aRef1 = np.ones(fine)
aRef2 = np.copy(aRef1)

for i in range(int(fine* 0/8.) - 1, int(fine * 4/8.) - 1):
    aRef2[i] = 2
for i in range(int(fine*4/8.)-1, int(fine*8/8.)):
    aRef1[i] =2

aPert1 = np.ones(fine)
aPert1 *=1.5'''

#================================
# b) quasi-1D test: 2D, but only changing in one direction
#================================
'''
aRef1 = np.ones(NFine)
aRef2 = np.copy(aRef1)

for i in range(int(fine* 0/8.) - 1, int(fine * 4/8.) - 1):
    aRef2[i,:] = 2
for i in range(int(fine*4/8.)-1, int(fine*8/8.)):
    aRef1[i,:] =2
aRef1=aRef1.flatten(order='F')
aRef2=aRef2.flatten(order='F')

aPert1 = np.ones(NFine).flatten(order='F')
aPert1 *=1.5'''

#==============================================
#c) full 2D
#==============================================
'''aRef1 = np.ones(NFine)
aRef2 = np.copy(aRef1)
aRef2*=2

for i in range(int(fine* 2/8.) - 1, int(fine * 6/8.) - 1):
    for j in range(int(fine* 2/8.) - 1, int(fine * 6/8.) - 1):
        aRef2[i,j] = 1
for i in range(int(fine*2/8.)-1, int(fine*6/8.)):
    for j in range(int(fine * 2 / 8.) - 1, int(fine * 6 / 8.) - 1):
        aRef1[i,j] =2
aRef1=aRef1.flatten(order='F')
aRef2=aRef2.flatten(order='F')

aPert1 = np.ones(NFine).flatten(order='F')
aPert1 *=1.5

xp = util.pCoordinates(NFine)
xt = util.tCoordinates(NFine)

# This is the right hand side
f = np.ones(NFine + 1).flatten(order='F')

# plot coefficients and compare them -1D
#plt.figure('Coefficient_pert')
#plt.plot(xt, aRef1, label='$Aref1$')
#plt.plot(xt, aRef2, label='$Aref2$')
#plt.plot(xt, aPert1, label='$Apert1$')
#plt.grid(True)
#plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], fontsize=16)
#plt.ylabel('$y$', fontsize=16)
#plt.xlabel('$x$', fontsize=16)
#plt.legend(frameon=False, fontsize=16)


#fig = plt.figure()
#ax1 = fig.add_subplot(1, 2, 1)
#ax2 = fig.add_subplot(1, 2, 2)

#aRef1grid = aRef1.reshape(NFine, order='C')
#aRef2grid = aRef2.reshape(NFine, order='C')

#im1 = ax1.imshow(aRef1grid, \
#                 extent=(xp[:, 0].min(), xp[:, 0].max(), xp[:, 1].min(), xp[:, 1].max()), cmap=plt.cm.hot)
#fig.colorbar(im1, ax=ax1)

#im2 = ax2.imshow(aRef2grid, \
#                 extent=(xp[:,0].min(), xp[:,0].max(), xp[:,1].min(), xp[:,1].max()), cmap=plt.cm.hot)
#fig.colorbar(im2, ax=ax2)

error_indicator_pert1 = []
x = []
error_pert1 = []
k=2

for N in NList:
    NWorldCoarse = np.array([N, N])
    boundaryConditions = np.array([[0, 0], [0, 0]])#

    NCoarseElement = NFine // NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    # grid nodes
    xtCoarse = util.tCoordinates(NWorldCoarse).flatten()
    x.append(xtCoarse)
    NpCoarse = np.prod(NWorldCoarse + 1)

    csiList = []
    KmsijList = []
    correctorsList = []
    RmsiList = []
    correctorsRhsList = []
    uLODFineList = []
    
    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    bFull = basis.T * MFull * f  # - RFull1

    for aRef in aRefList:
        print('computing correctors', end='', flush=True)
        computeKmsijRef = lambda TInd: computeKmsij(TInd, aRef)
        computeRmsiRef = lambda TInd: computeRmsi(TInd, aRef)
        _, correctorsRhsListRef, RmsijTRef = zip(*map(computeRmsiRef, range(world.NtCoarse)))
        patchT, correctorsListRef, KmsijRef, csiTRef = zip(*map(computeKmsijRef, range(world.NtCoarse)))
        csiList.append(csiTRef)
        KmsijList.append(KmsijRef)
        correctorsList.append(correctorsListRef)
        RmsiList.append(RmsijTRef)
        correctorsRhsList.append(correctorsRhsListRef)
        print()
        
        KFull_ref = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijRef)
        uFull_ref, _ = pglod.solve(world, KFull_ref, bFull, boundaryConditions)
        basisCorrectors_ref = pglod.assembleBasisCorrectors(world, patchT, correctorsListRef)
        modifiedBasis_ref = basis - basisCorrectors_ref
        uLodFine_ref = modifiedBasis_ref * uFull_ref
        uLODFineList.append(uLodFine_ref)


    computeIndicator1 = lambda TInd: computeIndicator(TInd, aRefList, aPert1)

    alpha = None#np.array([0.5, 0.5])
    mu = np.ones(len(aRefList))
    print('computing error indicators', end='', flush=True)
    E_vh1, alphaList1 = zip(*map(computeIndicator1, range(world.NtCoarse)))
    print()
    print('max error perturbed1  for alpha={}: {}'.format(alphaList1[np.argmax(E_vh1)],max(E_vh1)))
    error_indicator_pert1.append(E_vh1)
    E1 = {i: [E_vh1[i], alphaList1[i]] for i in range(np.size(E_vh1))}

    print('compute true LOD solution for perturbed coefficient 1')
    computeKmsijP1 = lambda TInd: computeKmsij(TInd, aPert1)
    computeRmsi1 = lambda TInd: computeRmsi(TInd, aPert1)
    print('computing real correctors', end='', flush=True)
    _, correctorsListTP1, KmsijTP1, _ = zip(*map(computeKmsijP1, range(world.NtCoarse)))
    print()
    print('computing real right hand side correctors', end='', flush=True)
    _, correctorRhsTP1, RmsiTP1 = zip(*map(computeRmsi1, range(world.NtCoarse)))
    print()

    #
    KFull1 = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTP1)
    RFull1 = pglod.assemblePatchFunction(world, patchT, RmsiTP1)
    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
    Rf1 = pglod.assemblePatchFunction(world, patchT, correctorRhsTP1)
    basis1 = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    bFull1 = basis1.T * MFull * f #- RFull1
    basisCorrectors1 = pglod.assembleBasisCorrectors(world, patchT, correctorsListTP1)
    modifiedBasis1 = basis1 - basisCorrectors1
    uFull1, _ = pglod.solve(world, KFull1, bFull1, boundaryConditions)
    uLodFine1 = modifiedBasis1 * uFull1
    #uLodFine1 += Rf1
    uLodCoarse1 = basis1*uFull1

    tol= np.inf
    print('compute new LOD for perturbed coefficient 1')
    KmsijT, correctorsListT, RmsijT, correctorsRhsT = UpdateElements(tol, E1, KmsijList,
                                                                             correctorsList, RmsiList, correctorsRhsList, aPert1)
    # LOD solve
    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
    RFull = pglod.assemblePatchFunction(world, patchT, RmsijT)
    Rf = pglod.assemblePatchFunction(world, patchT, correctorsRhsT)
    bFull = basis1.T * MFull * f #- RFull
    basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    modifiedBasis = basis1 - basisCorrectors
    uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)
    uLodFine_pert1 = modifiedBasis * uFull
    #uLodFine_pert1 += Rf
    uLodCoarse_pert1 = basis1 * uFull
    AFine_pert1 = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine, aPert1)
    energy_norm = np.sqrt(np.dot(uLodFine_pert1, AFine_pert1*uLodFine_pert1))
    energy_error = np.sqrt(np.dot((uLodFine_pert1 - uLodFine1), AFine_pert1 * (uLodFine_pert1 - uLodFine1)))
    print("Energy norm  error {}, relative error {}".format(energy_error, energy_error/energy_norm))
    error_pert1.append(energy_error)
    L2norm = np.sqrt(np.dot(uLodCoarse_pert1, MFull*uLodCoarse_pert1))
    L2_error = np.sqrt(np.dot(uLodCoarse1-uLodCoarse_pert1, MFull*(uLodCoarse1-uLodCoarse_pert1)))
    print("L2-norm error {}, relative error {}".format(L2_error, L2_error/L2norm))

    fixed = util.boundarypIndexMap(NWorldCoarse, boundaryConditions == 0)
    free = np.setdiff1d(np.arange(NpCoarse), fixed)
    errormatinv=np.linalg.norm(np.linalg.inv(KFull[free][:,free].todense())-np.linalg.inv(KFull1[free][:,free].todense()))
    errormat = np.linalg.norm(KFull[free][:,free].todense()-KFull1[free][:, free].todense())
    print("error in inverse matrices: true LOD vs. re-used {}".format(errormatinv))
    print("error in matrices true LOD vs. re-used {}".format(errormat))

#=======================================
#1D plots
#=======================================
# plot the indicators
#plt.figure('error indicators')
#for i in range(len(NList)):
#    plt.subplot(1,1,i+1)
#    plt.bar(x[i], error_indicator_pert1[i], width=0.03, color='r', label='perturbed1')
#    plt.legend()

#plt.figure('full LOD solutions')
#plt.plot(util.pCoordinates(world.NWorldFine), uLodFine1, color='r', label='true')
#plt.plot(util.pCoordinates(world.NWorldFine), uLodFine_pert1, color='b', label='with reference')
#plt.plot(util.pCoordinates(world.NWorldFine), uLodFine_ref1, label='ref1')
#plt.plot(util.pCoordinates(world.NWorldFine), uLodFine_ref2, label='ref2')
#plt.legend()

#plt.figure('FE part LOD solutions')
#plt.plot(util.pCoordinates(world.NWorldFine), uLodCoarse1, color='r', label='true')
#plt.plot(util.pCoordinates(world.NWorldFine), uLodCoarse_pert1, color='b', label='with reference')
#plt.legend()

#plt.show()


#==============================
#2D plot
#==============================
#fig = plt.figure()
#ax1 = fig.add_subplot(1, 2, 1)
#ax2 = fig.add_subplot(1, 2, 2)

#uLODtruegrid = uLodFine1.reshape(NFine+1, order='C')
#uLODpertgrid = uLodFine_pert1.reshape(NFine+1, order='C')

#im1 = ax1.imshow(uLODtruegrid, \
#                 extent=(xp[:, 0].min(), xp[:, 0].max(), xp[:, 1].min(), xp[:, 1].max()), cmap=plt.cm.hot)
#fig.colorbar(im1, ax=ax1)

#im2 = ax2.imshow(uLODpertgrid, \
#                 extent=(xp[:,0].min(), xp[:,0].max(), xp[:,1].min(), xp[:,1].max()), cmap=plt.cm.hot)
#fig.colorbar(im2, ax=ax2)
#plt.show()
'''
#====================================================================================================================
#Test 2

#===============================
#1D coeff
#================================
aRef1 = np.ones(fine)
aRef1 *= 0.1
bump=1
aRef2 = np.copy(aRef1)

for i in range(int(fine* 15/64.) - 1, int(fine * 25/64.) - 1): #2/8, 3/8
    aRef2[i] = bump

aPert = np.copy(aRef1) #aRef3
for i in range(int(fine* 25/64.) - 1, int(fine * 4/8.) - 1): #3/8, 4/8
    aPert[i] = bump

aRef3 = np.copy(aRef2) #aPert
for i in range(int(fine* 25/64.) - 1, int(fine * 4/8.) - 1): #3/8, 4/8
    aRef3[i] = bump

aRef4 = np.copy(aRef1)
for i in range(int(fine*4./8)-1, int(fine*5./8)-1):
    aRef4[i] = bump

aRefList = [aRef1, aRef2, aRef3, aRef4]

#=========================================================
#quasi 1D coeff (stripes)
#=========================================================
'''aRef1 = np.ones(NFine)
#aRef1 *= 0.1
bump = 10
aRef2 = np.copy(aRef1)

for i in range(int(fine* 2/8.) - 1, int(fine * 3/8.) - 1): #2/8, 3/8
    aRef2[i,:] = bump

aRef3 = np.copy(aRef1) #aRef3
for i in range(int(fine* 3/8.) - 1, int(fine * 4/8.) - 1): #5/8, 6/8
    aRef3[i,:] = bump

aPert = np.copy(aRef2) #aPert
for i in range(int(fine* 3/8.) - 1, int(fine * 4/8.) - 1): #5/8, 6/8
    aPert[i,:] = bump

aRef1=aRef1.flatten(order='F')
aRef2=aRef2.flatten(order='F')
aRef3=aRef3.flatten(order='F')
aPert=aPert.flatten(order='F')

aRefList = [aRef1, aRef2, aRef3]'''

#============================================================
#full 2D coefficient
#============================================================
'''aRef1grid = np.ones(NFine)
aRef1grid *= 0.1
bump = 1
aRef2grid = np.copy(aRef1grid)

for i in range(int(fine* 2/8.) - 1, int(fine * 3/8.) - 1): #2/8, 3/8
    for j in range(int(fine*2./8.)-1, int(fine*3./8.)-1):
        aRef2grid[i, j] = bump

aRef3grid = np.copy(aRef1grid) #aRef3
aPertgrid = np.copy(aRef2grid) #aPert
for i in range(int(fine* 3/8.) - 1, int(fine * 4/8.) - 1): #5/8, 6/8
    for j in range(int(fine* 3/8.) - 1, int(fine * 4/8.) - 1):
        aRef3grid[i,j] = bump
        aPertgrid[i,j] = bump

aRef1=aRef1grid.ravel()
aRef2=aRef2grid.ravel()
aRef3=aRef3grid.ravel()
aPert=aPertgrid.ravel()

aRefList = [aRef1, aRef2, aRef3]'''

xp = util.pCoordinates(NFine)
xtCoarse = util.tCoordinates(NFine).flatten()

# interior nodes for plotting
xt = util.tCoordinates(NFine).flatten()

# This is the right hand side
f = np.ones(NFine + 1).flatten(order='F')

error_indicator_pert = []
x = []
error_pert = []
k=2

for N in NList:
    NWorldCoarse = np.array([N])#, N
    boundaryConditions = np.array([[0, 0]])#, [0, 0]

    NCoarseElement = NFine // NWorldCoarse
    world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

    # grid nodes
    xtCoarse = util.tCoordinates(NWorldCoarse).flatten()
    x.append(xtCoarse)
    NpCoarse = np.prod(NWorldCoarse + 1)

    csiList = []
    KmsijList = []
    correctorsList = []
    RmsiList = []
    correctorsRhsList = []

    for aRef in aRefList:
        print('computing correctors', end='', flush=True)
        computeKmsijRef = lambda TInd: computeKmsij(TInd, aRef)
        computeRmsiRef = lambda TInd: computeRmsi(TInd, aRef)
        _, correctorsRhsListRef, RmsijTRef = zip(*map(computeRmsiRef, range(world.NtCoarse)))
        patchT, correctorsListRef, KmsijRef, csiTRef = zip(*map(computeKmsijRef, range(world.NtCoarse)))
        csiList.append(csiTRef)
        KmsijList.append(KmsijRef)
        correctorsList.append(correctorsListRef)
        RmsiList.append(RmsijTRef)
        correctorsRhsList.append(correctorsRhsListRef)
        print()


    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)

    computeIndicatorPert = lambda TInd: computeIndicator(TInd, aRefList, aPert)

    alpha = None #np.array([1., 0, 0.])
    mu = np.ones(len(aRefList))
    print('computing error indicators', end='', flush=True)
    E_vh, alphaList = zip(*map(computeIndicatorPert, range(world.NtCoarse)))
    print()
    print('max error perturbed1  for alpha={} at element {}: {}'.format(alphaList[np.argmax(E_vh)], np.argmax(E_vh), max(E_vh)))
    error_indicator_pert.append(E_vh)
    E1 = {i: [E_vh[i], alphaList[i]] for i in range(np.size(E_vh))}
    E_vh =np.array(E_vh)

    print('compute true LOD solution for perturbed coefficient 1')
    computeKmsijP1 = lambda TInd: computeKmsij(TInd, aPert)
    computeRmsiP1 = lambda TInd: computeRmsi(TInd, aPert)
    print('computing real correctors', end='', flush=True)
    patchT, correctorsListTP1, KmsijTP1, _ = zip(*map(computeKmsijP1, range(world.NtCoarse)))
    print()
    print('computing real right hand side correctors', end='', flush=True)
    _, correctorRhsTP1, RmsiTP1 = zip(*map(computeRmsiP1, range(world.NtCoarse)))
    print()
    #
    KFullP1 = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijTP1)
    RFullP1 = pglod.assemblePatchFunction(world, patchT, RmsiTP1)
    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)
    RfP1 = pglod.assemblePatchFunction(world, patchT, correctorRhsTP1)
    basis1 = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    bFullP1 = basis1.T * MFull * f #- RFullP1
    basisCorrectorsP1 = pglod.assembleBasisCorrectors(world, patchT, correctorsListTP1)
    modifiedBasisP1 = basis1 - basisCorrectorsP1
    uFullP1, _ = pglod.solve(world, KFullP1, bFullP1, boundaryConditions)
    uLodFineP1 = modifiedBasisP1 * uFullP1
    uLodCoarseP1 = basis1 * uFullP1
    #uLodFineP1 += RfP1

    tol = np.inf
    print('compute new LOD for perturbed coefficient 1')
    KmsijT, correctorsListT, RmsijT, correctorsRhsT = UpdateElements(tol, E1, KmsijList,
                                                                     correctorsList, RmsiList, correctorsRhsList,
                                                                     aPert)
    # LOD solve
    KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
    RFull = pglod.assemblePatchFunction(world, patchT, RmsijT)
    Rf = pglod.assemblePatchFunction(world, patchT, correctorsRhsT)
    bFull = basis1.T * MFull * f #- RFull
    basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    modifiedBasis = basis1 - basisCorrectors
    uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)
    uLodFine_pert1 = modifiedBasis * uFull
    uLodCoarse_pert1 = basis1 * uFull
    #uLodFine_pert1 += Rf
    AFine_pert1 = fem.assemblePatchMatrix(world.NWorldFine, world.ALocFine, aPert)
    energy_norm = np.sqrt(np.dot(uLodFine_pert1, AFine_pert1*uLodFine_pert1))
    energy_error = np.sqrt(np.dot((uLodFine_pert1 - uLodFineP1), AFine_pert1 * (uLodFine_pert1 - uLodFineP1)))
    print("Energy norm  error {}, relative error {}".format(energy_error, energy_error/energy_norm))
    error_pert.append(energy_error)
    L2norm = np.sqrt(np.dot(uLodCoarse_pert1, MFull*uLodCoarse_pert1))
    L2_error = np.sqrt(np.dot(uLodCoarseP1-uLodCoarse_pert1, MFull*(uLodCoarseP1-uLodCoarse_pert1)))
    print("L2-norm error {}, relative error {}".format(L2_error, L2_error/L2norm))


# plot the indicators
plt.figure('error indicators')
#plt.subplots_adjust(left=0.01, bottom=0.04, right=0.99, top=0.95, wspace=0.1, hspace=0.2)
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
for i in range(len(NList)):
    plt.subplot(1,len(NList),i+1)
    plt.bar(x[i], error_indicator_pert[i], width=0.03, color='r', label='perturbed1')
    #plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False,
    #                labelleft=False)
    plt.legend(frameon=False, fontsize=16)


plt.figure('full LOD solutions')
plt.plot(util.pCoordinates(world.NWorldFine), uLodFineP1, color='r', label='true')
plt.plot(util.pCoordinates(world.NWorldFine), uLodFine_pert1, color='b', label='with reference')
plt.legend()

plt.figure('FE part LOD solutions')
plt.plot(util.pCoordinates(world.NWorldFine), uLodCoarseP1, color='r', label='true')
plt.plot(util.pCoordinates(world.NWorldFine), uLodCoarse_pert1, color='b', label='with reference')
plt.legend()

plt.show()

#2D plots
'''fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)

ax5 = fig.add_subplot(2,2,3)
ax6 = fig.add_subplot(2,2,4)

uLodFine_pert1Grid = uLodFine_pert1.reshape(world.NWorldFine+1, order='C')
uLodFineP1Grid = uLodFineP1.reshape(world.NWorldFine+1, order='C')

im1 = ax1.imshow(uLodFine_pert1Grid, origin='lower_left',\
                 extent=(xp[:, 0].min(), xp[:, 0].max(), xp[:, 1].min(), xp[:, 1].max()), cmap=plt.cm.hot)
fig.colorbar(im1, ax=ax1)

im2 = ax2.imshow(uLodFineP1Grid, origin='lower_left',\
                 extent=(xp[:,0].min(), xp[:,0].max(), xp[:,1].min(), xp[:,1].max()), cmap=plt.cm.hot)
fig.colorbar(im2, ax=ax2)


Evhgrid = E_vh.reshape(world.NWorldCoarse, order='C')
im5 = ax5.imshow(Evhgrid, origin='lower_left',\
                 extent=(xp[:, 0].min(), xp[:, 0].max(), xp[:, 1].min(), xp[:, 1].max()), cmap=plt.cm.hot)
fig.colorbar(im5, ax=ax5)

im6 = ax6.imshow(aPertgrid, origin= 'lower_left',\
                 extent=(xp[:, 0].min(), xp[:, 0].max(), xp[:, 1].min(), xp[:, 1].max()), cmap=plt.cm.hot)
fig.colorbar(im6, ax=ax6)
plt.show()'''