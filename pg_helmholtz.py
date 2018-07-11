import numpy as np
import scipy.sparse as sparse
from copy import deepcopy

import lod
import lod_helmholtz
import util
import fem
import ecworker_helmholtz


class PetrovGalerkinLOD:
    def __init__(self, world, k, IPatchGenerator, printLevel=0):
        self.world = world
        NtCoarse = np.prod(world.NWorldCoarse)
        self.k = k
        self.IPatchGenerator = IPatchGenerator
        self.printLevel = printLevel

        self.ecList = None
        self.Kms = None
        self.K = None
        self.Mms = None
        self.M = None
        self.Bms = None
        self.B = None
        self.basisCorrectors = None
        self.coefficient = None
        self.coeff_mass = None
        self.coeff_bdry = None

    def updateCorrectors(self, coefficient, coeff_mass, coeff_bdry, clearFineQuantities=True):
        world = self.world
        k = self.k
        IPatchGenerator = self.IPatchGenerator

        NtCoarse = np.prod(world.NWorldCoarse)

        self.coefficient = deepcopy(coefficient)
        self.coeff_mass = deepcopy(coeff_mass)
        self.coeff_bdry = deepcopy(coeff_bdry)

        # Reset all caches
        self.Kms = None
        self.K = None
        self.Mms = None
        self.M = None
        self.Bms = None
        self.B = None
        self.basisCorrectors = None

        if self.printLevel >= 2:
            print('Setting up workers')
        ecworker_helmholtz.setupWorker(world, coefficient, coeff_mass, coeff_bdry,
                                       IPatchGenerator, k, clearFineQuantities)
        if self.printLevel >= 2:
            print('Done')

        ecList = []

        for TInd in range(NtCoarse):
            iElement = util.convertpIndexToCoordinate(world.NWorldCoarse - 1, TInd)
            ecT = ecworker_helmholtz.computeElementCorrector(iElement)
            ecList.append(ecT)
        self.ecList = ecList


    def clearCorrectors(self):
        NtCoarse = np.prod(self.world.NWorldCoarse)
        self.ecList = None
        self.coefficient = None

    def computeCorrection(self, ARhsFull=None, MRhsFull=None, BRhsFull=None):
        assert (self.ecList is not None)
        assert (self.coefficient is not None)
        assert (self.coeff_mass is not None)
        assert (self.coeff_bdry is not None)

        world = self.world
        NCoarseElement = world.NCoarseElement
        NWorldCoarse = world.NWorldCoarse
        NWorldFine = NWorldCoarse * NCoarseElement

        NpFine = np.prod(NWorldFine + 1)

        coefficient = self.coefficient
        coeff_mass = self.coeff_mass
        coeff_bdry = self.coeff_bdry
        IPatchGenerator = self.IPatchGenerator

        localBasis = world.localBasis

        TpIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NWorldFine)
        TpStartIndices = util.pIndexMap(NWorldCoarse - 1, NWorldFine, NCoarseElement)

        uFine = np.zeros(NpFine, dtype=complex)

        NtCoarse = np.prod(world.NWorldCoarse)
        for TInd in range(NtCoarse):
            if self.printLevel > 0:
                print(str(TInd) + ' / ' + str(NtCoarse))

            ecT = self.ecList[TInd]

            coefficientPatch = coefficient.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            coeff_massPatch = coeff_mass.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            coeff_bdryPatch = coeff_bdry.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

            if ARhsFull is not None:
                ARhsList = [ARhsFull[TpStartIndices[TInd] + TpIndexMap]]
            else:
                ARhsList = None

            if MRhsFull is not None:
                MRhsList = [MRhsFull[TpStartIndices[TInd] + TpIndexMap]]
            else:
                MRhsList = None

            if BRhsFull is not None:
                BRhsList = [BRhsFull[TpStartIndices[TInd] + TpIndexMap]]
            else:
                BRhsList = None

            correctorT = ecT.computeElementCorrector(coefficientPatch, coeff_massPatch, coeff_bdryPatch, IPatch, ARhsList, MRhsList, BRhsList)[0]

            NPatchFine = ecT.NPatchCoarse * NCoarseElement
            iPatchWorldFine = ecT.iPatchWorldCoarse * NCoarseElement
            patchpIndexMap = util.lowerLeftpIndexMap(NPatchFine, NWorldFine)
            patchpStartIndex = util.convertpCoordinateToIndex(NWorldFine, iPatchWorldFine)

            uFine[patchpStartIndex + patchpIndexMap] += correctorT

        return uFine

    def assembleBasisCorrectors(self):
        if self.basisCorrectors is not None:
            return self.basisCorrectors

        assert (self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse
        NCoarseElement = world.NCoarseElement
        NWorldFine = NWorldCoarse * NCoarseElement

        NtCoarse = np.prod(NWorldCoarse)
        NpCoarse = np.prod(NWorldCoarse + 1)
        NpFine = np.prod(NWorldFine + 1)

        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse - 1, NWorldCoarse)

        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert (ecT is not None)
            assert (hasattr(ecT, 'fsi'))

            NPatchFine = ecT.NPatchCoarse * NCoarseElement
            iPatchWorldFine = ecT.iPatchWorldCoarse * NCoarseElement

            patchpIndexMap = util.lowerLeftpIndexMap(NPatchFine, NWorldFine)
            patchpStartIndex = util.convertpCoordinateToIndex(NWorldFine, iPatchWorldFine)

            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = patchpStartIndex + patchpIndexMap
            dataT = np.hstack(ecT.fsi.correctorsList)

            cols.extend(np.repeat(colsT, np.size(rowsT)))
            rows.extend(np.tile(rowsT, np.size(colsT)))
            data.extend(dataT)

        basisCorrectors = sparse.csc_matrix((data, (rows, cols)), shape=(NpFine, NpCoarse))

        self.basisCorrectors = basisCorrectors
        return basisCorrectors

    def assembleMsStiffnessMatrix(self):
        if self.Kms is not None:
            return self.Kms

        assert (self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse

        NtCoarse = np.prod(world.NWorldCoarse)
        NpCoarse = np.prod(world.NWorldCoarse + 1)

        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse - 1, NWorldCoarse)

        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert (ecT is not None)

            NPatchCoarse = ecT.NPatchCoarse

            patchpIndexMap = util.lowerLeftpIndexMap(NPatchCoarse, NWorldCoarse)
            patchpStartIndex = util.convertpCoordinateToIndex(NWorldCoarse, ecT.iPatchWorldCoarse)

            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = patchpStartIndex + patchpIndexMap
            dataT = ecT.csi.Kmsij.flatten()

            cols.extend(np.tile(colsT, np.size(rowsT)))
            rows.extend(np.repeat(rowsT, np.size(colsT)))
            data.extend(dataT)

        Kms = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

        self.Kms = Kms
        return Kms

    def assembleStiffnessMatrix(self):
        if self.K is not None:
            return self.K

        assert (self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse

        NtCoarse = np.prod(world.NWorldCoarse)
        NpCoarse = np.prod(world.NWorldCoarse + 1)

        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse - 1, NWorldCoarse)

        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert (ecT is not None)

            NPatchCoarse = ecT.NPatchCoarse

            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = TpStartIndices[TInd] + TpIndexMap
            dataT = ecT.csi.Kij.flatten()

            cols.extend(np.tile(colsT, np.size(rowsT)))
            rows.extend(np.repeat(rowsT, np.size(colsT)))
            data.extend(dataT)

        K = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

        self.K = K
        return K

    def assembleMsMassMatrix(self):
        if self.Mms is not None:
            return self.Mms

        assert (self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse

        NtCoarse = np.prod(world.NWorldCoarse)
        NpCoarse = np.prod(world.NWorldCoarse + 1)

        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse - 1, NWorldCoarse)

        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert (ecT is not None)

            NPatchCoarse = ecT.NPatchCoarse

            patchpIndexMap = util.lowerLeftpIndexMap(NPatchCoarse, NWorldCoarse)
            patchpStartIndex = util.convertpCoordinateToIndex(NWorldCoarse, ecT.iPatchWorldCoarse)

            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = patchpStartIndex + patchpIndexMap
            dataT = ecT.csi.Mmsij.flatten()

            cols.extend(np.tile(colsT, np.size(rowsT)))
            rows.extend(np.repeat(rowsT, np.size(colsT)))
            data.extend(dataT)

        Mms = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

        self.Mms = Mms
        return Mms

    def assembleMassMatrix(self):
        if self.M is not None:
            return self.M

        assert (self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse

        NtCoarse = np.prod(world.NWorldCoarse)
        NpCoarse = np.prod(world.NWorldCoarse + 1)

        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse - 1, NWorldCoarse)

        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert (ecT is not None)

            NPatchCoarse = ecT.NPatchCoarse

            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = TpStartIndices[TInd] + TpIndexMap
            dataT = ecT.csi.Mij.flatten()

            cols.extend(np.tile(colsT, np.size(rowsT)))
            rows.extend(np.repeat(rowsT, np.size(colsT)))
            data.extend(dataT)

        M = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

        self.M = M
        return M

    def assembleMsBoundaryMatrix(self):
        if self.Bms is not None:
            return self.Bms

        assert (self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse

        NtCoarse = np.prod(world.NWorldCoarse)
        NpCoarse = np.prod(world.NWorldCoarse + 1)

        boundaryMapRobin = self.world.boundaryConditions == 1

        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse - 1, NWorldCoarse)

        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert (ecT is not None)

            #TpIndex = TpStartIndices[TInd] + TpIndexMap
            #for ii in range(TpIndex.size):
                #if TpIndex[ii] in util.boundarypIndexMap(NWorldCoarse, boundaryMapRobin):
            NPatchCoarse = ecT.NPatchCoarse

            patchpIndexMap = util.lowerLeftpIndexMap(NPatchCoarse, NWorldCoarse)
            patchpStartIndex = util.convertpCoordinateToIndex(NWorldCoarse, ecT.iPatchWorldCoarse)

            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = patchpStartIndex + patchpIndexMap
            dataT = ecT.csi.Bmsij.flatten()

            cols.extend(np.tile(colsT, np.size(rowsT)))
            rows.extend(np.repeat(rowsT, np.size(colsT)))
            data.extend(dataT)

        Bms = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

        self.Bms = Bms
        return Bms

    def assembleBoundaryMatrix(self):
        if self.B is not None:
            return self.B

        assert (self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse

        NtCoarse = np.prod(world.NWorldCoarse)
        NpCoarse = np.prod(world.NWorldCoarse + 1)

        boundaryMapRobin = self.world.boundaryConditions == 1

        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse - 1, NWorldCoarse)

        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert (ecT is not None)

            #TpIndex = TpStartIndices[TInd] + TpIndexMap
            #assemble = False
            #for ii in range(TpIndex.size):
            #    if TpIndex[ii] in util.boundarypIndexMap(NWorldCoarse, boundaryMapRobin):
            #        assemble = True
            #if assemble:
            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = TpStartIndices[TInd] + TpIndexMap
            dataT = ecT.csi.Bij.flatten()

            cols.extend(np.tile(colsT, np.size(rowsT)))
            rows.extend(np.repeat(rowsT, np.size(colsT)))
            data.extend(dataT)

        B = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

        self.B = B
        return B

    def solve(self, f, g, boundaryConditions):
        assert (f is None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse
        NpCoarse = np.prod(NWorldCoarse + 1)

        BFull = self.assembleBoundaryMatrix()

        fixed = util.boundarypIndexMap(NWorldCoarse, boundaryConditions == 0)
        free = np.setdiff1d(np.arange(NpCoarse), fixed)
        bFull = 1j * BFull * g    #to account for wrong sign in BFull; attention: BFull contains wavenumber!

        AmsFree = self.assembleMsStiffnessMatrix()[free][:, free] \
                  + self.assembleMsMassMatrix()[free][:, free]\
                  + self.assembleMsBoundaryMatrix()[free][:, free]
        bFree = bFull[free]

        uFree = sparse.linalg.spsolve(AmsFree, bFree)

        uFull = np.zeros(NpCoarse, dtype=complex)
        uFull[free] = uFree

        return uFull, uFree


