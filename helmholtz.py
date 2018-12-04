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


    if BbFine is None:
        BbFine = np.zeros(NpFine)

    boundaryMapDirichlet = boundaryConditions == 0
    boundaryMapRobin = boundaryConditions == 1
    fixedFine = util.boundarypIndexMap(NWorldFine, boundaryMap=boundaryMapDirichlet)
    freeFine = np.setdiff1d(np.arange(NpFine), fixedFine)
    AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine, wavenumberFine_neg_squared)
    MFineMass = -1 * MFine
    BdryFineMass = fem.assemblePatchBoundaryMatrix(NWorldFine, world.BLocGetterFine,
                                                   np.ones(NpFine), boundaryMapRobin)

    bFine = MFine * MbFine\
            + fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine) * AbFine\
            + fem.assemblePatchBoundaryMatrix(NWorldFine, world.BLocGetterFine,
                                                   np.ones(NpFine), boundaryMapRobin) * BbFine

    BFineFree = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)[freeFine][:, freeFine]\
                + fem.assemblePatchMatrix(NWorldFine, world.MLocFine, wavenumberFine_neg_squared)[freeFine][:, freeFine]\
                + fem.assemblePatchBoundaryMatrix(NWorldFine, world.BLocGetterFine,
                                               wavenumberFine_neg_complex, boundaryMapRobin)[freeFine][:, freeFine]
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

    boundaryMapDirichlet = boundaryConditions == 0
    boundaryMapRobin = boundaryConditions == 1
    fixedCoarse = util.boundarypIndexMap(NWorldCoarse, boundaryMap=boundaryMapDirichlet)
    freeCoarse = np.setdiff1d(np.arange(NpCoarse), fixedCoarse)

    AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine, wavenumberFine_neg_squared)
    BdryFine = fem.assemblePatchBoundaryMatrix(NWorldFine, world.BLocGetterFine,
                                               wavenumberFine_neg_complex, boundaryMapRobin)
    BdryFineMass = fem.assemblePatchBoundaryMatrix(NWorldFine, world.BLocGetterFine,
                                                   np.ones(NpFine), boundaryMapRobin)

    bFine = MFine * MbFine + AFine * AbFine + BdryFineMass * BbFine
    BFine = AFine + MFine + BdryFine

    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    bCoarse = basis.T * bFine
    BCoarse = basis.T * (BFine * basis)

    BCoarseFree = BCoarse[freeCoarse][:, freeCoarse]
    bCoarseFree = bCoarse[freeCoarse]

    uCoarseFree = linalg.linSolve(BCoarseFree, bCoarseFree)
    uCoarseFull = np.zeros(NpCoarse, dtype=complex)
    uCoarseFull[freeCoarse] = uCoarseFree
    #uCoarseFull = uCoarseFull

    uFineFull = basis * uCoarseFull

    return uCoarseFull, uFineFull