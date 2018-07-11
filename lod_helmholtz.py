import numpy as np

import fem
import util
import coef
import lod


class FineScaleInformation:
    def __init__(self, coefficientPatch, k_neg_squaredPatch, k_neg_complexPatch, correctorsList):
        self.coefficient = coefficientPatch
        self.k_neg_squared = k_neg_squaredPatch
        self.k_neg_complex = k_neg_complexPatch
        self.correctorsList = correctorsList


class CoarseScaleInformation:
    def __init__(self, Kij, Kmsij, Mij, Mmsij, Bij, Bmsij, rCoarse=None):
        self.Kij = Kij
        self.Kmsij = Kmsij
        self.Mij = Mij
        self.Mmsij = Mmsij
        self.Bij = Bij
        self.Bmsij = Bmsij


class elementCorrectorHelmholtz:
    def __init__(self, world, k, iElementWorldCoarse, saddleSolver=None):
        self.k = k
        self.iElementWorldCoarse = iElementWorldCoarse[:]
        self.world = world

        # Compute (NPatchCoarse, iElementPatchCoarse) from (k, iElementWorldCoarse, NWorldCoarse)
        d = np.size(iElementWorldCoarse)
        NWorldCoarse = world.NWorldCoarse
        iPatchWorldCoarse = np.maximum(0, iElementWorldCoarse - k).astype('int64')
        iEndPatchWorldCoarse = np.minimum(NWorldCoarse - 1, iElementWorldCoarse + k).astype('int64') + 1
        self.NPatchCoarse = iEndPatchWorldCoarse - iPatchWorldCoarse
        self.iElementPatchCoarse = iElementWorldCoarse - iPatchWorldCoarse
        self.iPatchWorldCoarse = iPatchWorldCoarse

        if saddleSolver == None:
            self._saddleSolver = lod.schurComplementSolverLU() #not run-time optimal!!!!
        else:
            self._saddleSolver = saddleSolver

    @property
    def saddleSolver(self):
        return self._saddleSolver

    @saddleSolver.setter
    def saddleSolver(self, value):
        self._saddleSolver = value

    def computeElementCorrector(self, coefficientPatch, wavenumber_neg_squaredPatch,
                                wavenumber_neg_complexPatch, IPatch, ARhsList=None, MRhsList=None, BRhsList=None):
        '''Compute the fine correctors over the patch.
        '''

        assert (ARhsList is not None or MRhsList is not None)
        numRhs = None

        if ARhsList is not None:
            assert (numRhs is None or numRhs == len(ARhsList))
            numRhs = len(ARhsList)

        if MRhsList is not None:
            assert (numRhs is None or numRhs == len(MRhsList))
            numRhs = len(MRhsList)


        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchCoarse = self.NPatchCoarse
        d = np.size(NCoarseElement)

        NPatchFine = NPatchCoarse * NCoarseElement
        NtFine = np.prod(NPatchFine)
        NpFine = np.prod(NPatchFine + 1)

        aPatch = coefficientPatch.aFine
        k_neg_squaredPatch = wavenumber_neg_squaredPatch.aFine
        k_neg_complexPatch = wavenumber_neg_complexPatch.aFine

        # inherit Robin boundary conditions from the world
        boundaryMapWorld = world.boundaryConditions == 1

        inherit0 = self.iPatchWorldCoarse == 0
        inherit1 = (self.iPatchWorldCoarse + NPatchCoarse) == world.NWorldCoarse

        boundaryMapPatch = np.zeros([d, 2], dtype='bool')
        boundaryMapPatch[inherit0, 0] = boundaryMapWorld[inherit0, 0]
        boundaryMapPatch[inherit1, 1] = boundaryMapWorld[inherit1, 1]

        assert (np.size(aPatch) == NtFine)
        assert (np.size(k_neg_squaredPatch) == NtFine)
        assert (np.size(k_neg_complexPatch) == NtFine)

        ALocFine = world.ALocFine
        MLocFine = world.MLocFine
        BLocGetterFine = world.BLocGetterFine

        iElementPatchCoarse = self.iElementPatchCoarse
        elementFinetIndexMap = util.extractElementFine(NPatchCoarse,
                                                       NCoarseElement,
                                                       iElementPatchCoarse,
                                                       extractElements=True)
        elementFinepIndexMap = util.extractElementFine(NPatchCoarse,
                                                       NCoarseElement,
                                                       iElementPatchCoarse,
                                                       extractElements=False)

        if ARhsList is not None:
            AElementFull = fem.assemblePatchMatrix(NCoarseElement, ALocFine, aPatch[elementFinetIndexMap])
        if MRhsList is not None:
            MkElementFull = fem.assemblePatchMatrix(NCoarseElement, MLocFine, k_neg_squaredPatch[elementFinetIndexMap])

        if BRhsList is not None:
            # inherit Robin-type boundary conditions from the world (for the element)
            boundaryMapElement = np.zeros([d, 2], dtype='bool')

            inheritElement0 = self.iElementWorldCoarse == 0
            inheritElement1 = (self.iElementWorldCoarse + np.ones(d)) == world.NWorldCoarse

            boundaryMapElement[inheritElement0, 0] = boundaryMapWorld[inheritElement0, 0]
            boundaryMapElement[inheritElement1, 1] = boundaryMapWorld[inheritElement1, 1]
            BkElementFull = fem.assemblePatchBoundaryMatrix(NCoarseElement, BLocGetterFine,
                                                            k_neg_complexPatch[elementFinetIndexMap], boundaryMapElement)


        APatchFull = fem.assemblePatchMatrix(NPatchFine, ALocFine, aPatch)
        APatchFull += fem.assemblePatchMatrix(NPatchFine, MLocFine, k_neg_squaredPatch)
        APatchFull += fem.assemblePatchBoundaryMatrix(NPatchFine, BLocGetterFine, k_neg_complexPatch, boundaryMapPatch)

        bPatchFullList = []
        for rhsIndex in range(numRhs):
            bPatchFull = np.zeros(NpFine, dtype=complex) #fix correctly!
            #if BkElementFull.nnz > 0 | aPatch.dtype == complex:
            #    bPatchFull = np.zeros(NpFine, dtype=complex)
            if ARhsList is not None:
                bPatchFull[elementFinepIndexMap] += AElementFull * ARhsList[rhsIndex]
            if MRhsList is not None:
                bPatchFull[elementFinepIndexMap] += MkElementFull * MRhsList[rhsIndex]
            if BRhsList is not None:
                bPatchFull[elementFinepIndexMap] += BkElementFull * BRhsList[rhsIndex]
            bPatchFullList.append(bPatchFull)

        correctorsList = lod.ritzProjectionToFinePatchWithGivenSaddleSolver(world,
                                                                            self.iPatchWorldCoarse,
                                                                            NPatchCoarse,
                                                                            APatchFull,
                                                                            bPatchFullList,
                                                                            IPatch,
                                                                            self.saddleSolver)
        return correctorsList

    def computeCorrectors(self, coefficientPatch, k_neg_squaredPatch, k_neg_complexPatch, IPatch):
        '''Compute the fine correctors over the patch.

        Compute the correctors Q_T\lambda_i (T is given by the class instance):

        and store them in the self.fsi object, together with the extracted A|_{U_k(T)}
        '''
        d = np.size(self.NPatchCoarse)
        ARhsList = list(map(np.squeeze, np.hsplit(self.world.localBasis, 2 ** d)))

        correctorsList = self.computeElementCorrector(coefficientPatch, k_neg_squaredPatch, k_neg_complexPatch,
                                                      IPatch, ARhsList, ARhsList, ARhsList)

        self.fsi = FineScaleInformation(coefficientPatch, k_neg_squaredPatch, k_neg_complexPatch, correctorsList)

    def computeCoarseQuantities(self):
        '''Compute the coarse quantities K and L for this element corrector

        Compute the tensors (T is given by the class instance):

        KTij   = (A \nabla lambda_j, \nabla lambda_i)_{T}
        KmsTij = (A \nabla (lambda_j - Q_T lambda_j), \nabla lambda_i)_{U_k(T)}
        MTij   = -k^2( lambda_j, lambda_i)_{T}
        MmsTij = -k^2( (lambda_j - Q_T lambda_j), lambda_i)_{U_k(T)}
        BTij   = -ik(lambda_j,  lambda_i)_{\partial T\cap \partial D}
        BmsTij = -ik((lambda_j - Q_T lambda_j), lambda_i)_{\partial U_k(T)\cap \partial D}

        and store them in the self.csi object.

        '''
        assert (hasattr(self, 'fsi'))

        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchCoarse = self.NPatchCoarse
        NPatchFine = NPatchCoarse * NCoarseElement

        NTPrime = np.prod(NPatchCoarse)
        NpPatchCoarse = np.prod(NPatchCoarse + 1)

        d = np.size(NPatchCoarse)

        correctorsList = self.fsi.correctorsList
        aPatch = self.fsi.coefficient.aFine
        k_neg_squaredPatch = self.fsi.k_neg_squared.aFine
        k_neg_complexPatch = self.fsi.k_neg_complex.aFine

        ALocFine = world.ALocFine
        MLocFine = world.MLocFine
        BLocGetterFine = world.BLocGetterFine
        localBasis = world.localBasis

        TPrimeCoarsepStartIndices = util.lowerLeftpIndexMap(NPatchCoarse - 1, NPatchCoarse)
        TPrimeCoarsepIndexMap = util.lowerLeftpIndexMap(np.ones_like(NPatchCoarse), NPatchCoarse)

        TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse - 1, NPatchFine - 1, NCoarseElement)
        TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement - 1, NPatchFine - 1)

        TPrimeFinepStartIndices = util.pIndexMap(NPatchCoarse - 1, NPatchFine, NCoarseElement)
        TPrimeFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)

        TInd = util.convertpCoordinateToIndex(NPatchCoarse - 1, self.iElementPatchCoarse)

        QPatch = np.column_stack(correctorsList)

        # This loop can probably be done faster than this. If a bottle-neck, fix!
        Kmsij = np.zeros((NpPatchCoarse, 2 ** d), dtype=complex)
        Mmsij = np.zeros((NpPatchCoarse, 2 ** d), dtype=complex)
        Bmsij = np.zeros((NpPatchCoarse, 2 ** d), dtype=complex)
        for (TPrimeInd,
             TPrimeCoarsepStartIndex,
             TPrimeFinetStartIndex,
             TPrimeFinepStartIndex) \
                in zip(np.arange(NTPrime),
                       TPrimeCoarsepStartIndices,
                       TPrimeFinetStartIndices,
                       TPrimeFinepStartIndices):

            aTPrime = aPatch[TPrimeFinetStartIndex + TPrimeFinetIndexMap]
            KTPrime = fem.assemblePatchMatrix(NCoarseElement, ALocFine, aTPrime)
            k_sqTPrime = k_neg_squaredPatch[TPrimeFinetStartIndex + TPrimeFinetIndexMap]
            MTPrime = fem.assemblePatchMatrix(NCoarseElement, MLocFine, k_sqTPrime)

            # inherit Robin-type boundary conditions from the world
            boundaryMapTPrime = np.zeros([d, 2], dtype='bool')
            boundaryMapWorld = world.boundaryConditions == 1

            iTPrimeCoarse = self.iPatchWorldCoarse + util.convertpIndexToCoordinate(NPatchCoarse-1, TPrimeInd)
            inherit0 = iTPrimeCoarse == 0
            inherit1 = (iTPrimeCoarse + np.ones(d)) == world.NWorldCoarse

            boundaryMapTPrime[inherit0, 0] = boundaryMapWorld[inherit0, 0]
            boundaryMapTPrime[inherit1, 1] = boundaryMapWorld[inherit1, 1]

            k_negiTPrime = k_neg_complexPatch[TPrimeFinetStartIndex + TPrimeFinetIndexMap]
            BTPrime = fem.assemblePatchBoundaryMatrix(NCoarseElement, BLocGetterFine, k_negiTPrime, boundaryMapTPrime)

            P = localBasis
            Q = QPatch[TPrimeFinepStartIndex + TPrimeFinepIndexMap, :]
            KQTPrimeij = np.dot(P.T, KTPrime * Q)
            MQTPrimeij = np.dot(P.T, MTPrime * Q)
            BQTPrimeij = np.dot(P.T, BTPrime * Q)
            sigma = TPrimeCoarsepStartIndex + TPrimeCoarsepIndexMap
            if TPrimeInd == TInd:
                Kij = np.dot(P.T, KTPrime * P)
                Mij = np.dot(P.T, MTPrime * P)
                Bij = np.dot(P.T, BTPrime * P)
                Kmsij[sigma, :] += Kij - KQTPrimeij
                Mmsij[sigma, :] += Mij - MQTPrimeij
                Bmsij[sigma, :] += Bij - BQTPrimeij

            else:
                Kmsij[sigma, :] += -KQTPrimeij
                Mmsij[sigma, :] += -MQTPrimeij
                Bmsij[sigma, :] += -BQTPrimeij


        if isinstance(self.fsi.coefficient, coef.coefficientCoarseFactorAbstract):
            rCoarse = self.fsi.coefficient.rCoarse
        else:
            rCoarse = None
        self.csi = CoarseScaleInformation(Kij, Kmsij, Mij, Mmsij, Bij, Bmsij, rCoarse)

    def clearFineQuantities(self):
        assert (hasattr(self, 'fsi'))
        del self.fsi
