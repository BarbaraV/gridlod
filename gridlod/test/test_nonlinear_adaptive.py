import unittest
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from gridlod import pglod, util, lod, interp, coef, fem
from gridlod.world import World, Patch


'''def computeGradpCoord(world, u):
    dim = np.size(world.NWorldCoarse)
    assert (dim == 2)
    basisGrad = np.zeros((2 ** dim, 2 ** dim, dim))  # point,basisfct, grad. entry
    basisGrad[0, 0, :] -= 1
    basisGrad[1, 0, 0] -= 1
    basisGrad[2, 0, 1] -= 1
    basisGrad[0, 1, 0] += 1
    basisGrad[1, 1, 0] += 1
    basisGrad[1, 1, 1] -= 1
    basisGrad[3, 1, 1] -= 1
    basisGrad[0, 2, 1] += 1
    basisGrad[2, 2, 1] += 1
    basisGrad[2, 2, 0] -= 1
    basisGrad[3, 2, 0] -= 1
    basisGrad[3, 3, :] += 1
    basisGrad[1, 3, 1] += 1
    basisGrad[2, 3, 0] += 1
    # scaling missing!
    gradu = np.zeros((np.size(u), dim))
    for ii in range(dim):
        gradu[:, ii] = np.sum(fem.assemblePatchMatrix(world.NWorldFine, basisGrad[:, :, ii], u),
                              axis=-1)  # geht so nicht?!
'''

def computeGradtCoord(world, u):
    dim = np.size(world.NWorldCoarse)
    Ntfine = np.prod(world.NWorldFine)
    NFine = world.NWorldFine
    assert (dim == 2)
    basisGrad = 0.5 * np.ones((2 ** dim, dim))  # point,basisfct, grad. entry
    basisGrad[0, :] *= -1
    basisGrad[1, 1] *= -1
    basisGrad[2, 0] *= -1
    basisGrad *=world.NWorldFine  #working correctly?
    gradu = np.zeros((Ntfine, dim))
    for N in range(Ntfine):
        row = N//NFine[0]
        col = N % NFine[0]
        gradu[N,:] = u[row*(NFine[0]+1)+col]*basisGrad[0,:] + u[row*(NFine[0]+1)+col+1]*basisGrad[1,:]\
                     + u[(row+1)*(NFine[0]+1)+col]*basisGrad[2,:] + u[(row+1)*(NFine[0]+1)+col+1]*basisGrad[3,:]
    return gradu

def nonlinear_adaptive(NFine,NCoarse,k,Anonlin,rhs,u0,Amicro0,Alin,tolmacro,tolmicro,tollin,maxit):
    NCoarseElement = NFine // NCoarse
    boundaryConditions = np.array([[0, 0], [0, 0]])
    NpFine = np.prod(NFine+1)
    world = World(NCoarse, NCoarseElement, boundaryConditions)
    pcoords = util.pCoordinates(NFine)
    tcoords = util.tCoordinates(NFine)
    rhs = rhs(pcoords)

    MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)

    def computeKmsij(TInd):
        patch = Patch(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
        amicroPatch = lambda: coef.localizeCoefficient(patch,amicro)
        amacroPatch = lambda: coef.localizeCoefficient(patch,amacro)

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, amicroPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, amacroPatch)
        return patch, correctorsList, csi.Kmsij, csi

    def computeIndicators(TInd):
        aPatch = lambda: coef.localizeCoefficient(patchT[TInd], amicro)
        rPatch = lambda: coef.localizeCoefficient(patchT[TInd], amacro)

        E_vh = lod.computeErrorIndicatorCoarseFromCoefficients(patchT[TInd], csiT[TInd].muTPrime, aPatch, rPatch)

        return E_vh

    def UpdateCorrectors(TInd):
        # print(" UPDATING {}".format(TInd))
        patch = Patch(world, k, TInd)
        IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
        rPatch = lambda: coef.localizeCoefficient(patch, amacro)

        #MRhsList = [rhs[util.extractElementFine(world.NWorldCoarse,
        #                                          world.NCoarseElement,
        #                                          patch.iElementWorldCoarse,
        #                                          extractElements=False)]];

        correctorsList = lod.computeBasisCorrectors(patch, IPatch, rPatch)
        csi = lod.computeBasisCoarseQuantities(patch, correctorsList, rPatch)

        #correctorRhs = lod.computeElementCorrector(patch, IPatch, rPatch, None, MRhsList)[0]
        #Rmsij = lod.computeRhsCoarseQuantities(patch, correctorRhs, rPatch)

        return patch, correctorsList, csi.Kmsij

    def UpdateElements(tol, E_vh, Kmsij_old, correctors_old, amicro_old):
        print('apply tolerance')
        Elements_to_be_updated = []
        for (i,eps) in E_vh.items():
            if eps > tol:
                Elements_to_be_updated.append(i)
        if len(E_vh) > 0:
            print('... to be updated: {}%'.format(100*np.size(Elements_to_be_updated)/len(E_vh)), end='\n', flush=True)


        print('update Kmsij')
        KmsijT_list = list(np.copy(Kmsij_old))
        for T in np.setdiff1d(range(world.NtCoarse), Elements_to_be_updated):
            patch = Patch(world, k, T)
            rPatch = lambda: coef.localizeCoefficient(patch, amacro)
            csi = lod.computeBasisCoarseQuantities(patch, correctors_old[T], rPatch)
            KmsijT_list[T] = csi.Kmsij

        if np.size(Elements_to_be_updated) != 0:
            print('... update correctors')
            patchT_irrelevant, correctorsListTNew, KmsijTNew = zip(*map(UpdateCorrectors,
                                                                                 Elements_to_be_updated))

            print('update correctorsListT')
            correctorsListT_list = list(np.copy(correctors_old))
            amicro_new = np.copy(amicro_old)
            i = 0
            for T in Elements_to_be_updated:
                KmsijT_list[T] = KmsijTNew[i]
                correctorsListT_list[T] = correctorsListTNew[i]
                amicro_new[T] = amacro[T]
                i += 1

            KmsijT = tuple(KmsijT_list)
            correctorsListT = tuple(correctorsListT_list)
            return KmsijT,correctorsListT, amicro_new
        else:
            KmsijT = tuple(KmsijT_list)
            return KmsijT,correctors_old, amicro_old


    def assembleNonlinear(Alin, u): #uses fine mesh atm
        aFine = Alin(tcoords, computeGradtCoord(world, u))
        assert(aFine.ndim == 1 or aFine.ndim == 3)
        if aFine.ndim == 1:
            Aloc = world.ALocFine
        else:
            Aloc = world.ALocMatrixFine
        return fem.assemblePatchMatrix(NFine, Aloc, aFine)


    #precomputation
    basis = fem.assembleProlongationMatrix(world.NWorldCoarse, world.NCoarseElement)
    amacro = Alin(tcoords, computeGradtCoord(world, basis*u0))
    amicro = np.copy(Amicro0)
    # Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
    print('computing correctors', end='', flush=True)
    patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))
    print()

    uFull = np.copy(u0)
    #basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
    #modifiedBasis = basis - basisCorrectors
    #uLodFine =  modifiedBasis * u0
    #Knonlin = assembleNonlinear(Alin, basis*uFull) #assumes Anonlin(x, nabla u) = Alin(x, nabla u)nabla u

    #fixed = util.boundarypIndexMap(world.NWorldCoarse, boundaryConditions == 0)
    #free = np.setdiff1d(np.arange(world.NpCoarse), fixed)

    resmacro = np.inf
    #resmacro = np.linalg.norm((basis.T*Knonlin*modifiedBasis)[free][:,free]*uFull[free] - (basis.T*MFull*rhs)[free])
    it = 0

    while resmacro > tolmacro and it < maxit:
        print('computing error indicators', end='', flush=True)
        E_vh = list(map(computeIndicators, range(world.NtCoarse)))
        print()
        print('maximal value error estimator {}'.format(np.max(E_vh)))
        E_vh = {i: E_vh[i] for i in range(np.size(E_vh)) if E_vh[i] > 0}

        #loop over elements with possible recomputation of correctors
        KmsijT,correctorsListT,amicro = UpdateElements(tolmicro*resmacro,E_vh,KmsijT,correctorsListT,amicro)

        #LOD solve
        KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
        bFull = basis.T * MFull * rhs
        basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
        modifiedBasis = basis - basisCorrectors
        uFull, _ = pglod.solve(world, KFull, bFull, boundaryConditions)
        uLodFine = modifiedBasis * uFull

        errorA = Alin(tcoords, computeGradtCoord(world,uLodFine))-amacro
        assert(amacro.ndim == 1 or amacro.ndim == 3)
        if amacro.ndim == 1:
            errorAmat = fem.assemblePatchMatrix(NFine, world.ALocFine, errorA**2) #correct?!
        else:
            errorAmat = fem.assemblePatchMatrix(NFine, world.ALocMatrixFine, np.einsum('Tij, Tjk->Tik', errorA, errorA))
        resmacro = np.sqrt(np.dot(uLodFine, errorAmat*uLodFine))
        #if amacro.ndim == 1:
        #    amacro_uFull = np.array([amacro, amacro]).T * computeGradtCoord(world, uLodFine)
        #else:
        #    amacro_uFull = np.einsum('Tij,Tj-> Ti', amacro, computeGradtCoord(world,basis * uFull)) #passt das?
        #if np.linalg.norm(np.linalg.norm(Anonlin(tcoords,computeGradtCoord(world,basis*uFull)).T\
        #                                 -amacro_uFull, axis=-1), np.inf) > tollin:
        amacro = Alin(tcoords, computeGradtCoord(world, uLodFine)) #basis*uFull

        #update res and it
        fixed = util.boundarypIndexMap(world.NWorldCoarse, boundaryConditions == 0)
        free = np.setdiff1d(np.arange(world.NpCoarse), fixed)
        Knonlin = assembleNonlinear(Alin, uLodFine) #basis*uFull, must fit to amacro?!
        #resmacro = np.linalg.norm((basis.T*Knonlin*modifiedBasis)[free][:,free]*uFull[free] - (basis.T*MFull*rhs)[free])
        it +=1

        print('residual in {}th iteration is {}'.format(it, np.linalg.norm((basis.T*Knonlin*modifiedBasis)[free][:,free]*uFull[free] - (basis.T*MFull*rhs)[free])), end='\n', flush=True)
        print('linearization error is {}'.format(resmacro))

    return uFull,uLodFine, amacro


def test_nonlinear_adaptive():
    NFine = np.array([256, 256])
    NpFine = np.prod(NFine + 1)
    NtFine = np.prod(NFine)
    # NList = [4, 8, 16, 32, 64]
    NList = [16]

    maxiter = 100
    tolmacro = 1e-11

    epsilon = 1/8
    boundaryConditions = np.array([[0, 0], [0, 0]])
    c = lambda x: 1+x[:,0]*x[:,1] + (1.1+np.pi/3+np.sin(2*np.pi*x[:,0]/epsilon))/(1.1+np.sin(2*np.pi*x[:,1]/epsilon))
    #-------------------------------------------------------------------------------------------------------------------
    # example with matrix-valued Alin, see Patrick !!!does not work yet (no convergence) !!!!
    #Alin = lambda x, xi0: np.array([[c(x)*(1+0.1/3*xi0[:,0]**2), 0], [0, c(x)*(1+10/3*xi0[:,1]**2)]])
    # bzw. Anonlin = lambda x, xi0: np.array([c(x)*(xi[:,0]+0.1/3*xi0[:,0]**3), c(x)*(xi0[:,1]+10/3*xi0[:,1]**3)])
    #def Alin_matrix(x,xi0):
    #    a = np.array([[c(x)*(1+1/3*xi0[:,0]**2), np.zeros(NtFine)], [np.zeros(NtFine), c(x)*(1+1/3*xi0[:,1]**2)]])
    #    a = a.swapaxes(0,2)
    #    return a
    #Alin = Alin_matrix
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
    Anonlin=Alin #util unused Alin removed from the adaptive algo
    f = lambda x: 100*np.ones(np.shape(x)[0])

    for N in NList:
        NCoarse = np.array([N, N])
        NpCoarse = np.prod(NCoarse+1)
        NCoarseElement = NFine // NCoarse
        world = World(NCoarse, NCoarseElement, boundaryConditions)
        pcoords = util.pCoordinates(NFine)
        tcoords = util.tCoordinates(NFine)

        #reference solution -- maybe as own function
        u0ref = np.zeros(NpFine)
        uFullref = np.copy(u0ref)

        aFine = Alin(tcoords, computeGradtCoord(world, uFullref))
        assert (aFine.ndim == 1 or aFine.ndim == 3)
        if aFine.ndim == 1:
            Aloc = world.ALocFine
        else:
            Aloc = world.ALocMatrixFine
        AFull = fem.assemblePatchMatrix(NFine, Aloc, aFine)
        MFull = fem.assemblePatchMatrix(world.NWorldFine, world.MLocFine)

        fixed = util.boundarypIndexMap(NFine, boundaryConditions==0)
        free = np.setdiff1d(np.arange(NpFine), fixed)
        resref = np.linalg.norm(AFull[free][:,free] * uFullref[free] - (MFull* f(pcoords))[free])
        itref = 0

        while resref > tolmacro and itref < maxiter:
            #solve
            bFull = MFull*f(pcoords)
            AFree = AFull[free][:, free]
            bFree = bFull[free]
            uFree = sparse.linalg.spsolve(AFree, bFree)
            uFullref[free] = uFree

            # update res and it
            aFine = Alin(tcoords, computeGradtCoord(world, uFullref))
            AFull = fem.assemblePatchMatrix(NFine, Aloc, aFine)
            resref = np.linalg.norm(AFull[free][:,free] * uFullref[free] - (MFull* f(pcoords))[free])
            itref += 1

            print('residual in {}th iteration is {}'.format(itref, resref), end='\n', flush=True)

        uFineGrid = uFullref.reshape(NFine + 1, order='C')

        #normf = np.sqrt(np.dot(f(pcoords), MFull * f(pcoords)))

        #LOD
        k = 2
        uLOD = np.zeros(NpCoarse)
        Amicro0 = Alin(tcoords, np.zeros_like(tcoords))
        tolmacro = 1e-4 #overwrites tolmacro!
        tollin = 1e-6
        tolmicro = 100 #used as factor atm
        uLOD, uLODfine,amacro = nonlinear_adaptive(NFine,NCoarse,k,Anonlin,f,uLOD,Amicro0,Alin,tolmacro,tolmicro, tollin, maxiter)
        uLODfinegrid = uLODfine.reshape(NFine+1, order='C')

        SFull = fem.assemblePatchMatrix(NFine, world.ALocFine, np.ones(NtFine))
        error = np.sqrt(np.dot(uFullref-uLODfine, SFull*(uFullref-uLODfine)))/np.sqrt(np.dot(uLODfine, SFull*uLODfine))
        print('relative H1semi error {}'.format(error))


        if N == 16:
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax2 = fig.add_subplot(1, 2, 2)

            im1 = ax1.imshow(uLODfinegrid, \
                             extent=(pcoords[:, 0].min(), pcoords[:, 0].max(), pcoords[:, 1].min(), pcoords[:, 1].max()), cmap=plt.cm.hot)
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(uFineGrid, \
                             extent=(pcoords[:,0].min(), pcoords[:,0].max(), pcoords[:,1].min(), pcoords[:,1].max()), cmap=plt.cm.hot)
            fig.colorbar(im2, ax=ax2)
            plt.show()

if __name__ == '__main__':
    test_nonlinear_adaptive()