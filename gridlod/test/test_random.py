import numpy as np
import scipy.stats as stats
import scipy.sparse as sparse
import scipy.sparse.linalg
import scipy.optimize as opti
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

from gridlod import pglod, util, lod, interp, coef, fem, func
from gridlod.world import World, Patch

def test_random():
    fineLevel = 11
    NFine = np.array([2**fineLevel])
    NtFine = np.prod(NFine)
    NList = [2**2]
    epsilonlevel = 9
    NEpsilon = np.array([2**epsilonlevel])
    Samples = 5000
    alpha = 0.5
    beta = 10
    k = 2

    xt = util.tCoordinates(NFine).flatten()
    xp = util.pCoordinates(NFine).flatten()


    for N in NList:
        NWorldCoarse = np.array([N])
        NCoarseElement = NFine//NWorldCoarse
        boundaryConditions = np.array([[0, 0]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NFree = world.NpCoarse-2

        expKFree = np.zeros((NFree, NFree))
        covarKFree = np.zeros((NFree**2, NFree**2))
        allKFree = np.zeros((Samples, NFree**2))

        def computeKmsij(TInd):
            patch = Patch(world, k, TInd)
            IPatch = lambda: interp.L2ProjectionPatchMatrix(patch, boundaryConditions)
            aPatch = lambda: coef.localizeCoefficient(patch, aFine)

            correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
            csi = lod.computeBasisCoarseQuantities(patch, correctorsList, aPatch)
            return patch, correctorsList, csi.Kmsij, csi

        for S in range(Samples):
            if S%50 == 0:
                print('.', flush=True, end='')
            aFineEps = alpha + (beta - alpha) * np.random.rand(2 ** epsilonlevel)
            assert (aFineEps.shape[0] == np.prod(NEpsilon))
            coordIndices, _ = func._computeCoordinateIndexParts(NEpsilon, xt)
            aFine = aFineEps[coordIndices,...]

            # Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
            patchT, correctorsListT, KmsijT, csiT = zip(*map(computeKmsij, range(world.NtCoarse)))

            KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
            free = util.interiorpIndexMap(NWorldCoarse)
            KFree = KFull[free][:, free]

            sparse.linalg.inv(KFree)
            allKFree[S] = KFree.toarray().flatten()
            expKFree += KFree
            covarKFree += np.outer(KFree.toarray().flatten(), KFree.toarray().flatten())


        print()
        expKFree *= 1./Samples
        covarKFree -= Samples*np.outer(expKFree.flatten(), expKFree.flatten())
        covarKFree *= 1./Samples
        #print('sample mean')
        #print(expKFull)
        #print('sample mean - numpy')
        #print(np.mean(allKFree, axis=0))
        #print('sample covariance')
        #print(covarKFull)
        #print('sample covariance - numpy')
        #print(np.cov(allKFree, rowvar=0))

        '''print('skewness (should be 0 for normal distribution)')
        skew = 0
        invcovar = np.linalg.inv(covarKFree)
        for i in range(Samples):
            for j in range(Samples):
                skew +=np.dot(allKFree[i,:]-expKFree.flatten(), invcovar*(allKFree[j,:]-expKFree.flatten()).T)**3
        skew *= 1./(Samples**2)
        print(skew)

        print('kurtosis')
        kurt = 0
        invcovar = np.linalg.inv(covarKFree)
        for i in range(Samples):
            kurt += np.dot(allKFree[i, :] - expKFree.flatten(), invcovar * (allKFree[i, :] - expKFree.flatten()).T) ** 2
        kurt *= 1./Samples
        print(kurt)'''


        #normalize allKFree (via mean)?
        #allKFree = allKFree/np.mean(allKFree, axis=0)

        for n in range(NFree**2):
            print(stats.kstest(allKFree[:,n], 'norm', args=(np.mean(allKFree[:,n]), np.sqrt(np.var(allKFree[:,n])))))
            #print(stats.shapiro(allKFree[:,n]))

            nbins = 50
            ns, bins, patches = plt.hist(allKFree[:,n], nbins, density=True, facecolor='blue')
            midbins = 0.5*(bins[0:nbins] + bins[1:nbins+1])


            def gaus(x, a, x0, sigma):
                return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

            popt, pcov = opti.curve_fit(gaus, midbins, ns, p0=[1, expKFree.flatten()[0,n], covarKFree[n,n]])
            print(*popt)
            print(expKFree.flatten()[0,n], np.sqrt(covarKFree[n,n]))
            plt.plot(midbins, gaus(midbins, *popt), 'r*:', label='fit')
            plt.xlabel('Value')
            plt.ylabel('Probability')
            plt.title('Histogram of LOD stiffness matrix entry {}'.format(n))

            # Tweak spacing to prevent clipping of ylabel
            plt.subplots_adjust(left=0.15)
            plt.show()




if __name__ == '__main__':
    test_random()