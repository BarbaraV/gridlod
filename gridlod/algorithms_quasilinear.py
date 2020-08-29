import numpy as np
import scipy.sparse as sparse
import copy
import matplotlib.pyplot as plt

from gridlod import coef, multiplecoeff, lod, pglod, interp, func, util, fem
from gridlod.world import World, Patch


def computeRefSol(world, Alin, f, u0, tol, maxiter):
    uref = u0
    tcoords = util.tCoordinates(world.NWorldFine)
    pcoords = util.pCoordinates(world.NWorldFine)

    aFine = Alin(tcoords, func.evaluateCQ1D(world.NWorldFine, uref, tcoords))
    assert (aFine.ndim == 1 or aFine.ndim == 3)
    if aFine.ndim == 1:
        Aloc = world.ALocFine
    else:
        Aloc = world.ALocMatrixFine
    AFull = fem.assemblePatchMatrix(world.NWorldFine, Aloc, aFine)
    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)

    fixed = util.boundarypIndexMap(world.NWorldFine, world.boundaryConditions == 0)
    free = np.setdiff1d(np.arange(world.NpFine), fixed)
    resref = np.linalg.norm(AFull[free][:, free] * uref[free] - (MFull * f(pcoords))[free])
    itref = 0

    while resref > tol and itref < maxiter:
        # solve
        bFull = MFull * f(pcoords)
        AFree = AFull[free][:, free]
        bFree = bFull[free]
        uFree = sparse.linalg.spsolve(AFree, bFree)
        uref[free] = uFree

        '''if aFine.ndim == 1:
            aFineGrid = aFine.reshape(world.NWorldFine, order ='C')
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)

            im1 = ax1.imshow(aFineGrid, \
                             extent=(
                             pcoords[:, 0].min(), pcoords[:, 0].max(), pcoords[:, 1].min(), pcoords[:, 1].max()),
                             cmap=plt.cm.hot)
            fig.colorbar(im1, ax=ax1)

            plt.show()

        else:
            aFineGrid1 = aFine[:,0,0].reshape(world.NWorldFine, order='C')
            aFineGrid2 = aFine[:,1,1].reshape(world.NWorldFine, order='C')
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            im1 = ax1.imshow(aFineGrid1, \
                             extent=(
                                 pcoords[:, 0].min(), pcoords[:, 0].max(), pcoords[:, 1].min(), pcoords[:, 1].max()),
                             cmap=plt.cm.hot)
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(aFineGrid2, \
                             extent=(
                                 pcoords[:, 0].min(), pcoords[:, 0].max(), pcoords[:, 1].min(), pcoords[:, 1].max()),
                             cmap=plt.cm.hot)
            fig.colorbar(im2, ax=ax2)

            plt.show()'''

        # update res and it
        aFine = Alin(tcoords, func.evaluateCQ1D(world.NWorldFine, uref, tcoords))
        AFull = fem.assemblePatchMatrix(world.NWorldFine, Aloc, aFine)
        resref = np.linalg.norm(AFull[free][:, free] * uref[free] - (MFull * f(pcoords))[free])
        itref += 1

        print('residual in {}th iteration is {}'.format(itref, resref), end='\n', flush=True)

    return uref

def computeKmsij(TInd, aFine):
    patch = Patch(world, k, TInd)
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    aPatch = lambda: coef.localizeCoefficient(patch, aFine)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
    return patch, correctorsList, csi.Kmsij, csi

def UpdateCorrectors(TInd, aFine, patch):
    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
    rPatch = lambda: coef.localizeCoefficient(patch, aFine)

    correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
    csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)

    return patch, correctorsList, csi.Kmsij, csi

def adaptive_nonlinear_single(world, Alin, f, u0, tolmacro, tolmicro, maxiter):

    def computeIndicator_single(TInd):
        aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine)  # true coefficient of current iteration
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd],
                                                  aFineOld)  # 'reference' coefficient from previous iteration

        E_vh = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime, rPatch, aPatch)

        return E_vh

    def UpdateElements_single(tol, E, Kmsij_old, correctors_old):
        # !! different from adaptive_nonlinear, Kmsij only recomputed where indicator shows this (not using updated a everywhere)
        print('apply tolerance')
        Elements_to_be_updated = []
        for (i, eps) in E.items():
            if eps > tol:
                Elements_to_be_updated.append(i)
        if len(E) > 0:
            print('... to be updated: {}%'.format(100 * np.size(Elements_to_be_updated) / len(E)), end='\n', flush=True)

        if np.size(Elements_to_be_updated) != 0:
            print('... update correctors')
            UpdateCorrectorsa = lambda TInd: UpdateCorrectors(TInd, aFine, patchT[TInd])
            patchT_irrelevant, correctorsListTNew, KmsijTNew, csiT_irrelevant = zip(
                *map(UpdateCorrectorsa,
                     Elements_to_be_updated))

            print('replace Kmsij and update correctorsListT')
            correctorsListT_list = list(np.copy(correctors_old))
            KmsijT_list = list(np.copy(Kmsij_old))
            i = 0
            for T in Elements_to_be_updated:
                KmsijT_list[T] = KmsijTNew[i]
                correctorsListT_list[T] = correctorsListTNew[i]
                i += 1

            KmsijT = tuple(KmsijT_list)
            correctorsListT = tuple(correctorsListT_list)
            return KmsijT, correctorsListT
        else:
            return Kmsij_old, correctors_old

    pcoords = util.pCoordinates(world.NWorldFine)
    tcoords = util.tCoordinates(world.NWorldFine)
    rhs = f(pcoords)

    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)

    # precomputation
    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    aFine = Alin(tcoords, func.evaluateCQ1D(world.NWorldFine, basis * u0, tcoords))
    aFineOld = np.copy(aFine)

    # Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
    print('computing correctors', end='', flush=True)
    computeKmsija = lambda TInd: computeKmsij(TInd,aFine)
    patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsija, range(world.NtCoarse)))
    print()

    uFull = np.copy(u0)

    resmacro = np.inf
    # resmacro = np.linalg.norm((basis.T*Knonlin*modifiedBasis)[free][:,free]*uFull[free] - (basis.T*MFull*rhs)[free])
    it = 0

    while resmacro > tolmacro and it < maxiter:
        print('computing error indicators', end='', flush=True)
        E_vh = list(map(computeIndicator_single, range(world.NtCoarse)))
        print()
        print('maximal value error estimator {}'.format(np.max(E_vh)))
        E = {i: E_vh[i] for i in range(np.size(E_vh)) if E_vh[i] > 0 }

        # loop over elements with possible recomputation of correctors
        #tol_relative = np.min(E_vh) + tolmicro*(np.max(E_vh) - np.min(E_vh))
        sortedE = np.sort(E_vh)
        index = min(int(tolmicro*(len(E_vh)-1)), len(E_vh)-1)
        tol_relative = sortedE[index] # approx. (1-tolmicro)*100% updates
        KmsijT, correctorsListT = UpdateElements_single(tol_relative, E, KmsijT, correctorsListT)

        # LOD solve
        KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
        bFull = basis.T * MFull * rhs
        basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
        modifiedBasis = basis - basisCorrectors
        uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)
        uLodFine = modifiedBasis * uFull

        aFineOld = np.copy(aFine)
        aFine = Alin(tcoords, func.evaluateCQ1D(world.NWorldFine, uLodFine, tcoords))  # basis*uFull

        # update res and it
        assert (aFine.ndim == 1 or aFine.ndim == 3)
        if aFine.ndim == 1:
            Aloc = world.ALocFine
        else:
            Aloc = world.ALocMatrixFine
        AFull = fem.assemblePatchMatrix(world.NWorldFine, Aloc, aFine)
        fixed = util.boundarypIndexMap(world.NWorldCoarse, boundaryConditions == 0)
        free = np.setdiff1d(np.arange(world.NpCoarse), fixed)
        resmacro = np.linalg.norm((basis.T*AFull*modifiedBasis)[free][:,free]*uFull[free] - (basis.T*MFull*rhs)[free])
        it += 1

        print('residual in {}th iteration is {}'.format(it, resmacro),
              end='\n', flush=True)

    return uFull, uLodFine

def adaptive_nonlinear_multiple(world, Alin, f, u0, tolmacro, tolmicro, maxiter, closest = False):

    def computeIndicator_multiple(TInd):
        aPatch = lambda: coef.localizeCoefficient(patchT[TInd], aFine)  # true coefficient of current iteration
        rPatchList = aFinePatchList[TInd]# 'reference' coefficient from previous iteration
        muTPrimeList = [csi.muTPrime for csi in csiList[TInd]]

        if closest:
            E_vhList = []
            alpha = np.zeros(len(rPatchList))
            for ii in range(len(rPatchList)):
                E_vhList.append(lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd],
                                                                                muTPrimeList[ii], rPatchList[ii], aPatch))
            E_vh = min(E_vhList)
            alpha[np.argmin(np.array(E_vhList))] = 1.
        else:
            alpha = multiplecoeff.optimizeAlpha(patchT[TInd], rPatchList, aPatch)
            E_vh = multiplecoeff.estimatorAlphaTildeA1mod(patchT[TInd], muTPrimeList, rPatchList, aPatch, alpha)

        return E_vh, alpha

    def UpdateElements_multiple(tol, E, Kmsij_old, correctors_old, csi_old, alphaList):
        # !! different from adaptive_nonlinear, Kmsij only recomputed where indicator shows this (not using updated a everywhere)
        print('apply tolerance')
        Elements_to_be_updated = []
        alphaTList = []
        for (i, eps) in E.items():
            if eps > tol:  # or eps[1] > tol:
                Elements_to_be_updated.append(i)
            else:
                alphaTList.append(alphaList[i])
        if len(E) > 0:
            print('... to be updated: {}%'.format(100 * np.size(Elements_to_be_updated) / len(E)), end='\n', flush=True)

        KmsijT_list = [copy.deepcopy(Kmsij_old[TInd][0]) for TInd in range(world.NtCoarse)]
        correctorsListT_list = [copy.deepcopy(correctors_old[TInd][0]) for TInd in range(world.NtCoarse)]
        csi_list = [copy.deepcopy(csi_old[TInd][0]) for TInd in range(world.NtCoarse)]
        print('combine stiffness matrices')

        if len(E) > 0:
            j = 0
            for T in np.setdiff1d(range(world.NtCoarse), Elements_to_be_updated):
                assert(len(alphaTList[j]) == len(Kmsij_old[T]))
                assert(len(alphaTList[j]) == len(correctors_old[T]))
                KmsijT_list[T] = np.einsum('i, ijk ->jk', alphaTList[j], Kmsij_old[T])
                correctorsListT_list[T] = np.einsum('i, ijk -> jk', alphaTList[j], np.array(correctors_old[T]))
                #for kk in range(len(Kmsij_old)):
                #    KmsijT_list[T] += alphaTList[j][kk] * Kmsij_old[kk][T]
                #    correctorsListT_list[T] += alphaTList[j][kk] * np.array(correctors_old[kk][T])
                j += 1

            if np.size(Elements_to_be_updated) != 0:
                print('... update correctors')
                UpdateCorrectorsa = lambda TInd: UpdateCorrectors(TInd, aFine, patchT[TInd])
                patchT_irrelevant, correctorsListTNew, KmsijTNew, csiTNew = zip(*map(UpdateCorrectorsa,
                         Elements_to_be_updated))

                print('replace Kmsij and update correctorsListT')
                i = 0
                for T in Elements_to_be_updated:
                    KmsijT_list[T] = KmsijTNew[i]
                    correctorsListT_list[T] = correctorsListTNew[i]
                    csi_list[T] = csiTNew[i]
                    i += 1

            KmsijT = tuple(KmsijT_list)
            correctorsListT = tuple(correctorsListT_list)
            #csi_list = tuple(csi_list)
            return KmsijT, correctorsListT, csi_list, Elements_to_be_updated
        else:
            KmsijT = tuple(KmsijT_list)
            correctorsListT = tuple(correctorsListT_list)
            return KmsijT, correctorsListT, csi_list, Elements_to_be_updated

    pcoords = util.pCoordinates(world.NWorldFine)
    tcoords = util.tCoordinates(world.NWorldFine)
    rhs = f(pcoords)

    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)

    # precomputation
    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    aFine = Alin(tcoords, func.evaluateCQ1D(world.NWorldFine, basis * u0, tcoords))
    aFineOld = np.copy(aFine)

    # Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
    print('computing correctors', end='', flush=True)
    computeKmsija = lambda TInd: computeKmsij(TInd,aFine)
    patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsija, range(world.NtCoarse)))
    print()

    aFinePatchList = [[coef.localizeCoefficient(patchT[TInd],aFine)] for TInd in range(world.NtCoarse)]
    correctorsList = [[correctorsListT[TInd]] for TInd in range(world.NtCoarse)]
    KmsijList = [[KmsijT[TInd]] for TInd in range(world.NtCoarse)]
    csiList = [[csiT[TInd]] for TInd in range(world.NtCoarse)]

    uFull = np.copy(u0)

    resmacro = np.inf
    # resmacro = np.linalg.norm((basis.T*Knonlin*modifiedBasis)[free][:,free]*uFull[free] - (basis.T*MFull*rhs)[free])
    it = 0

    while resmacro > tolmacro and it < maxiter:
        print('computing error indicators', end='', flush=True)
        E_vh, alphaList = zip(*map(computeIndicator_multiple, range(world.NtCoarse)))
        print()
        print('maximal value error estimator {}'.format(np.max(E_vh)))
        E = {i: E_vh[i] for i in range(np.size(E_vh)) if E_vh[i] > 0 }

        # loop over elements with possible recomputation of correctors
        #tol_relative = np.min(E_vh) + tolmicro*(np.max(E_vh) - np.min(E_vh))
        sortedE = np.sort(E_vh)
        index = min(int(tolmicro*(len(E_vh)-1)), len(E_vh)-1)
        tol_relative = sortedE[index] # approx. (1-tolmicro)*100% updates
        KmsijT, correctorsListT, csiT, elementsupdated = UpdateElements_multiple(tol_relative, E,
                                                                           KmsijList, correctorsList, csiList, alphaList)

        # LOD solve
        KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
        bFull = basis.T * MFull * rhs
        basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
        modifiedBasis = basis - basisCorrectors
        uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)
        uLodFine = modifiedBasis * uFull

        if np.size(elementsupdated) != 0:
            aFineOld = np.copy(aFine)
            for T in elementsupdated:
                KmsijList[T].append(KmsijT[T])
                correctorsList[T].append(correctorsListT[T])
                csiList[T].append(csiT[T])
                aFinePatchList[T].append(coef.localizeCoefficient(patchT[T], aFineOld))

        aFine = Alin(tcoords, func.evaluateCQ1D(world.NWorldFine, uLodFine, tcoords))  # basis*uFull

        # update res and it
        assert (aFine.ndim == 1 or aFine.ndim == 3)
        if aFine.ndim == 1:
            Aloc = world.ALocFine
        else:
            Aloc = world.ALocMatrixFine
        AFull = fem.assemblePatchMatrix(world.NWorldFine, Aloc, aFine)
        fixed = util.boundarypIndexMap(world.NWorldCoarse, boundaryConditions == 0)
        free = np.setdiff1d(np.arange(world.NpCoarse), fixed)
        resmacro = np.linalg.norm((basis.T*AFull*modifiedBasis)[free][:,free]*uFull[free] - (basis.T*MFull*rhs)[free])
        it += 1

        print('residual in {}th iteration is {}'.format(it, resmacro),
              end='\n', flush=True)

    return uFull, uLodFine

#=====================================================================================================
# testing

NFine = np.array([256, 256])
NpFine = np.prod(NFine + 1)
NtFine = np.prod(NFine)
NCoarse = np.array([16,16])

maxiter = 100
tolmacro = 1e-11

epsilon = 1/8
boundaryConditions = np.array([[0, 0], [0, 0]])
c = lambda x: 1+x[:,0]*x[:,1] + (1.1+np.pi/3+np.sin(2*np.pi*x[:,0]/epsilon))/(1.1+np.sin(2*np.pi*x[:,1]/epsilon))
#------------------------------------------------------------------------------------------------------------------
#scalar Alin, see Huber
#Anonlin = lambda x, xi: np.array([c(x)*(1+1/np.sqrt(1+np.linalg.norm(xi,2,axis=-1)**2))*xi[:,0], \
#                                  c(x) * (1 + 1 / np.sqrt(1 + np.linalg.norm(xi, 2, axis=-1) ** 2)) * xi[:, 1]])
#Alin = lambda x, xi0: c(x)*(1+1/np.sqrt(1+np.linalg.norm(xi0,2,axis=-1)**2))
#-------------------------------------------------------------------------------------------------------------------
#try Huber with a matrix-valued Alin (artificial)
def Alin_matrix(x,xi0):
    a = np.array([[c(x)*(1+1/np.sqrt(1+np.linalg.norm(xi0,2,axis=-1)**2)), np.zeros(NtFine)],
                  [np.zeros(NtFine), c(x)*(1+10/np.sqrt(1+100*np.linalg.norm(xi0,2,axis=-1)**2))]])
    a = a.swapaxes(0,2)
    return a
Alin = Alin_matrix

f = lambda x: 100*np.ones(np.shape(x)[0])

NCoarseElement = NFine // NCoarse
world = World(NCoarse, NCoarseElement, boundaryConditions)
u0 = np.zeros(NpFine)
u = computeRefSol(world, Alin, f, u0, tolmacro, maxiter)

# LOD
k=2
tolmacro = 1e-5
tolmicro = 0.7# is a kind of factor at the moment
u0LOD = np.zeros(world.NpCoarse)
uLOD, uLODfine = adaptive_nonlinear_multiple(world,Alin,f, u0LOD, tolmacro,tolmicro, maxiter)

SFull = fem.assemblePatchMatrix(NFine, world.ALocFine, np.ones(NtFine))
MFull = fem.assemblePatchMatrix(NFine, world.MLocFine)
error = np.sqrt(np.dot(u-uLODfine, SFull*(u-uLODfine)))/np.sqrt(np.dot(u, SFull*u))
print('relative H1semi error {}'.format(error))

errorL2 = np.sqrt(np.dot(u - uLODfine, MFull * (u - uLODfine))) / np.sqrt(
    np.dot(u, MFull * u))
print('relative L2 error {}'.format(errorL2))
