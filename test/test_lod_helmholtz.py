import unittest
import numpy as np
import itertools as it
import functools

from gridlod import lod, fem, interp, util, coef, lod_helmholtz
from gridlod.world import World

class corrector_TestCase(unittest.TestCase):
    def test_init(self):
        NWorldCoarse = np.array([4, 4])
        NCoarseElement = np.array([2, 2])
        world = World(NWorldCoarse, NCoarseElement)

        k = 1

        iElementWorldCoarse = np.array([0, 0])
        ec = lod_helmholtz.elementCorrectorHelmholtz(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 2]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 0]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 0]))

        iElementWorldCoarse = np.array([0, 3])
        ec = lod_helmholtz.elementCorrectorHelmholtz(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 2]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 2]))

        iElementWorldCoarse = np.array([0, 2])
        ec = lod_helmholtz.elementCorrectorHelmholtz(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 3]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 1]))

        iElementWorldCoarse = np.array([1, 2])
        ec = lod_helmholtz.elementCorrectorHelmholtz(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [3, 3]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [1, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 1]))

    def test_testCsi(self):
        NWorldCoarse = np.array([4, 5, 6])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)

        k = 1
        iElementWorldCoarse = np.array([2, 1, 2])
        ec = lod_helmholtz.elementCorrectorHelmholtz(world, k, iElementWorldCoarse)
        IPatch = interp.L2ProjectionPatchMatrix(ec.iPatchWorldCoarse, ec.NPatchCoarse, NWorldCoarse, NCoarseElement)

        NtPatch = np.prod(ec.NPatchCoarse * NCoarseElement)

        np.random.seed(1)

        aPatch = np.random.rand(NtPatch)
        wavenumber = 10
        wn_neg_squaredPatch = -1 * wavenumber**2 * np.ones(NtPatch)
        wn_neg_complPatch = -1 * 1j * wavenumber * np.ones(NtPatch)
        coefficientPatch = coef.coefficientFine(ec.NPatchCoarse, NCoarseElement, aPatch)
        k_squared_negPatch = coef.coefficientFine(ec.NPatchCoarse, NCoarseElement, wn_neg_squaredPatch)
        k_neg_complexPatch = coef.coefficientFine(ec.NPatchCoarse, NCoarseElement, wn_neg_complPatch)
        ec.computeCorrectors(coefficientPatch, k_squared_negPatch, k_neg_complexPatch, IPatch)
        ec.computeCoarseQuantities()

        TFinetIndexMap = util.extractElementFine(ec.NPatchCoarse,
                                                 NCoarseElement,
                                                 ec.iElementPatchCoarse,
                                                 extractElements=True)
        TFinepIndexMap = util.extractElementFine(ec.NPatchCoarse,
                                                 NCoarseElement,
                                                 ec.iElementPatchCoarse,
                                                 extractElements=False)
        TCoarsepIndexMap = util.extractElementFine(ec.NPatchCoarse,
                                                   np.ones_like(NCoarseElement),
                                                   ec.iElementPatchCoarse,
                                                   extractElements=False)
        #assemble boundary matrix everywhere
        boundaryMap = np.ones([d, 2], dtype='bool')

        APatchFine = fem.assemblePatchMatrix(ec.NPatchCoarse * NCoarseElement, world.ALocFine, aPatch)
        AElementFine = fem.assemblePatchMatrix(NCoarseElement, world.ALocFine, aPatch[TFinetIndexMap])
        MPatchFine = fem.assemblePatchMatrix(ec.NPatchCoarse * NCoarseElement, world.MLocFine, wn_neg_squaredPatch)
        MElementFine = fem.assemblePatchMatrix(NCoarseElement, world.MLocFine, wn_neg_squaredPatch[TFinetIndexMap])
        BPatchFine = fem.assemblePatchBoundaryMatrix(ec.NPatchCoarse * NCoarseElement, world.BLocGetterFine, wn_neg_complPatch)
        BElementFine = fem.assemblePatchBoundaryMatrix(NCoarseElement, world.BLocGetterFine, wn_neg_complPatch[TFinetIndexMap])
        basisPatch = fem.assembleProlongationMatrix(ec.NPatchCoarse, NCoarseElement)
        correctorsPatch = np.column_stack(ec.fsi.correctorsList)

        localBasis = world.localBasis

        KmsijShouldBe = -basisPatch.T * (APatchFine * (correctorsPatch))
        KmsijShouldBe[TCoarsepIndexMap, :] += np.dot(localBasis.T, AElementFine * localBasis)

        MmsijShouldBe = -basisPatch.T * (MPatchFine * (correctorsPatch))
        MmsijShouldBe[TCoarsepIndexMap, :] += np.dot(localBasis.T, MElementFine * localBasis)

        BmsijShouldBe = -basisPatch.T * (BPatchFine * (correctorsPatch))
        BmsijShouldBe[TCoarsepIndexMap, :] += np.dot(localBasis.T, BElementFine * localBasis)

        self.assertTrue(np.isclose(np.max(np.abs(ec.csi.Kmsij - KmsijShouldBe)), 0))
        self.assertTrue(np.isclose(np.max(np.abs(ec.csi.Mmsij - MmsijShouldBe)), 0))
        #self.assertTrue(np.isclose(np.max(np.abs(ec.csi.Bmsij - BmsijShouldBe)), 0))
        #print(np.max(np.abs(ec.csi.Bmsij - BmsijShouldBe)))

    '''def test_computeSingleT(self):
        NWorldCoarse = np.array([4, 5, 6])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)

        k = 1
        iElementWorldCoarse = np.array([2, 1, 2])
        ec = lod_helmholtz.elementCorrectorHelmholtz(world, k, iElementWorldCoarse)
        IPatch = interp.nodalPatchMatrix(ec.iPatchWorldCoarse, ec.NPatchCoarse, NWorldCoarse, NCoarseElement)

        NtPatch = np.prod(ec.NPatchCoarse * NCoarseElement)
        coefficientPatch = coef.coefficientFine(ec.NPatchCoarse, NCoarseElement, np.ones(NtPatch))
        wavenumber = 6
        k_neg_squaredPatch = coef.coefficientFine(ec.NPatchCoarse, NCoarseElement, -1* wavenumber**2 * np.ones(NtPatch))
        k_neg_complPatch = coef.coefficientFine(ec.NPatchCoarse, NCoarseElement,
                                                  -1 * 1j * wavenumber * np.ones(NtPatch))
        ec.computeCorrectors(coefficientPatch, k_neg_squaredPatch, k_neg_complPatch, IPatch)

        correctorSum = functools.reduce(np.add, ec.fsi.correctorsList)
        #self.assertTrue(np.allclose(correctorSum, 0))

        ec.computeCoarseQuantities()
        # Test that the matrices have the constants in their null space
        # self.assertTrue(np.allclose(np.sum(ec.csi.LTPrimeij, axis=1), 0))
        # self.assertTrue(np.allclose(np.sum(ec.csi.LTPrimeij, axis=2), 0))

        self.assertTrue(np.allclose(np.sum(ec.csi.Kij, axis=0), 0))
        self.assertTrue(np.allclose(np.sum(ec.csi.Kij, axis=1), 0))
        self.assertTrue(np.allclose(np.sum(ec.csi.Kmsij, axis=0), 0))
        self.assertTrue(np.allclose(np.sum(ec.csi.Kmsij, axis=1), 0))

        ec.clearFineQuantities()
    '''

    def test_computeFullDomain(self):
        NWorldCoarse = np.array([2, 3, 4], dtype='int64')
        NWorldCoarse = np.array([1, 1, 1], dtype='int64')
        NCoarseElement = np.array([4, 2, 3], dtype='int64')
        NWorldFine = NWorldCoarse * NCoarseElement
        NpWorldFine = np.prod(NWorldFine + 1)
        NpWorldCoarse = np.prod(NWorldCoarse + 1)
        NtWorldFine = np.prod(NWorldCoarse * NCoarseElement)

        np.random.seed(0)

        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        IWorld = interp.nodalPatchMatrix(0 * NWorldCoarse, NWorldCoarse, NWorldCoarse, NCoarseElement)
        aWorld = np.exp(np.random.rand(NtWorldFine))
        wavenumber = 6
        coefficientWorld = coef.coefficientFine(NWorldCoarse, NCoarseElement, aWorld)
        k2_negWorld = coef.coefficientFine(NWorldCoarse, NCoarseElement, -1*wavenumber**2*np.ones(NtWorldFine))
        kcompl_negWorld = coef.coefficientFine(NWorldCoarse, NCoarseElement, -1*1j*wavenumber*np.ones(NtWorldFine))
        k = np.max(NWorldCoarse)

        elementpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        elementpIndexMapFine = util.lowerLeftpIndexMap(NCoarseElement, NWorldFine)

        coarsepBasis = util.linearpIndexBasis(NWorldCoarse)
        finepBasis = util.linearpIndexBasis(NWorldFine)

        correctors = np.zeros((NpWorldFine, NpWorldCoarse), dtype=complex)
        basis = np.zeros((NpWorldFine, NpWorldCoarse))

        for iElementWorldCoarse in it.product(*[np.arange(n, dtype='int64') for n in NWorldCoarse]):
            iElementWorldCoarse = np.array(iElementWorldCoarse)
            ec = lod_helmholtz.elementCorrectorHelmholtz(world, k, iElementWorldCoarse)
            ec.computeCorrectors(coefficientWorld, k2_negWorld, kcompl_negWorld, IWorld)

            worldpIndices = np.dot(coarsepBasis, iElementWorldCoarse) + elementpIndexMap
            correctors[:, worldpIndices] += np.column_stack(ec.fsi.correctorsList)

            worldpFineIndices = np.dot(finepBasis, iElementWorldCoarse * NCoarseElement) + elementpIndexMapFine
            basis[np.ix_(worldpFineIndices, worldpIndices)] = world.localBasis

        AGlob = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aWorld)
        MGlob = fem.assemblePatchMatrix(NWorldFine, world.MLocFine, -1*wavenumber**2*np.ones(NtWorldFine))
        boundaryMapRobin = world.boundaryConditions == 1
        BdryGlob = fem.assemblePatchBoundaryMatrix(NWorldFine, world.BLocGetterFine,
                                                   -1 * 1j * wavenumber * np.ones(NtWorldFine), boundaryMapRobin)
        BGlob = AGlob + MGlob + BdryGlob

        alpha = np.random.rand(NpWorldCoarse)
        vH = np.dot(basis, alpha)
        QvH = np.dot(correctors, alpha)

        # Check norm inequality
        self.assertTrue(np.dot(QvH.T, AGlob * QvH) <= np.dot(vH.T, AGlob * vH))
        #what checks are meaningful here???
        #print( -1* np.dot(QvH.conj().T, MGlob * QvH))
        #print( -1* np.dot(vH.T, MGlob*vH))

        # Check that correctors are really fine functions
        self.assertTrue(np.isclose(np.linalg.norm(IWorld * correctors, ord=np.inf), 0))

        v = np.random.rand(NpWorldFine, NpWorldCoarse)
        v[util.boundarypIndexMap(NWorldFine)] = 0
        # The chosen interpolation operator doesn't ruin the boundary conditions.
        vf = v - np.dot(basis, IWorld * v)
        vf = vf / np.sqrt(np.sum(vf * (AGlob * vf), axis=0))
        # Check orthogonality
        self.assertTrue(np.isclose(np.linalg.norm(np.dot(vf.T, BGlob * (correctors - basis)), ord=np.inf), 0))

if __name__ == '__main__':
    unittest.main()