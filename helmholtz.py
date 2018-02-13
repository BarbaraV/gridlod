import numpy as np
import scipy.sparse as sparse

from world import World
import util
import fem
import linalg


def solveFine(world, aFine, wavenumberFine_neg_squared, wavenumberFine_neg_complex, MbFine,
              AbFine, BbFine, boundaryConditions):
    NWorldFine = world.NWorldCoarse * world.NCoarseElement
    NpFine = np.prod(NWorldFine + 1)

    if MbFine is None:
        MbFine = np.zeros(NpFine)

    if AbFine is None:
        AbFine = np.zeros(NpFine)

    boundaryMapDirichlet = boundaryConditions == 0
    boundaryMapRobin = (boundaryConditions == 1)
    fixedFine = util.boundarypIndexMap(NWorldFine, boundaryMap=boundaryMapDirichlet)
    freeFine = np.setdiff1d(np.arange(NpFine), fixedFine)
    AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine, wavenumberFine_neg_squared)
    MFineMass = -1 * MFine
    BdryFine = fem.assemblePatchBoundaryMatrix(NWorldFine, world.BLocGetterFine,
                                               wavenumberFine_neg_complex, boundaryMapRobin)
    BdryFineMass = fem.assemblePatchBoundaryMatrix(NWorldFine, world.BLocGetterFine,
                                                   np.ones(NpFine), boundaryMapRobin)

    bFine = MFine * MbFine + AFine * AbFine + BdryFineMass * BbFine

    BFineFree = AFine[freeFine][:, freeFine] + MFine[freeFine][:, freeFine] + BdryFine[freeFine][:, freeFine]
    bFineFree = bFine[freeFine]

    uFineFree = linalg.linSolve(BFineFree, bFineFree)
    uFineFull = np.zeros(NpFine, dtype=complex)
    uFineFull[freeFine] = uFineFree
    #uFineFull = uFineFull

    return uFineFull, AFine, MFineMass, BdryFineMass


def solveCoarse(world, aFine, wavenumberFine_neg_squared, wavenumberFine_neg_complex, MbFine,
                AbFine, BbFine,  boundaryConditions):
    NWorldCoarse = world.NWorldCoarse
    NWorldFine = world.NWorldCoarse * world.NCoarseElement
    NCoarseElement = world.NCoarseElement

    NpFine = np.prod(NWorldFine + 1)
    NpCoarse = np.prod(NWorldCoarse + 1)

    if MbFine is None:
        MbFine = np.zeros(NpFine)

    if AbFine is None:
        AbFine = np.zeros(NpFine)

    boundaryMap = boundaryConditions == 0
    fixedCoarse = util.boundarypIndexMap(NWorldCoarse, boundaryMap=boundaryMap)
    freeCoarse = np.setdiff1d(np.arange(NpCoarse), fixedCoarse)

    AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine, wavenumberFine_neg_squared)
    BdryFine = fem.assemblePatchBoundaryMatrix(NWorldFine, world._BLocGetterFine,
                                               wavenumberFine_neg_complex, boundaryMap)
    BdryFineMass = fem.assemblePatchBoundaryMatrix(NWorldFine, world._BLocGetterFine,
                                                   np.ones(NpFine), boundaryMap)

    bFine = MFine * MbFine + AFine * AbFine + BdryFineMass * BbFine
    BFine = AFine + MFine + BdryFine

    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    bCoarse = basis.T * bFine
    BCoarse = basis.T * (BFine * basis)

    BCoarseFree = BCoarse[freeCoarse][:, freeCoarse]
    bCoarseFree = bCoarse[freeCoarse]

    uCoarseFree = linalg.linSolve(BCoarseFree, bCoarseFree)
    uCoarseFull = np.zeros(NpCoarse)
    uCoarseFull[freeCoarse] = uCoarseFree
    uCoarseFull = uCoarseFull

    uFineFull = basis * uCoarseFull

    return uCoarseFull, uFineFull