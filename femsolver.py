import numpy as np
import scipy.sparse as sparse

from world import World
import util
import fem
import linalg

def solveFine(world, aFine, MbFine, AbFine, boundaryConditions, massFine=None):
    NWorldCoarse = world.NWorldCoarse
    NWorldFine = world.NWorldCoarse*world.NCoarseElement
    NpFine = np.prod(NWorldFine+1)
    
    if MbFine is None:
        MbFine = np.zeros(NpFine)

    if AbFine is None:
        AbFine = np.zeros(NpFine)
        
    boundaryMap = boundaryConditions==0
    fixedFine = util.boundarypIndexMap(NWorldFine, boundaryMap=boundaryMap)
    freeFine  = np.setdiff1d(np.arange(NpFine), fixedFine)
    AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)
    SFine = AFine.copy()

    if massFine is not None:
        SFine +=fem.assemblePatchMatrix(NWorldFine, world.MLocFine, massFine)
    bFine = MFine*MbFine + SFine*AbFine
    
    SFineFree = SFine[freeFine][:,freeFine]
    bFineFree = bFine[freeFine]

    uFineFree = linalg.linSolve(SFineFree, bFineFree)
    uFineFull = np.zeros(NpFine)
    uFineFull[freeFine] = uFineFree
    uFineFull = uFineFull

    if massFine is None:
        return uFineFull, AFine, MFine
    else:
        return uFineFull, AFine, MFine, SFine

def solveCoarse(world, aFine, MbFine, AbFine, boundaryConditions, massFine=None):
    NWorldCoarse = world.NWorldCoarse
    NWorldFine = world.NWorldCoarse*world.NCoarseElement
    NCoarseElement = world.NCoarseElement
    
    NpFine = np.prod(NWorldFine+1)
    NpCoarse = np.prod(NWorldCoarse+1)
    
    if MbFine is None:
        MbFine = np.zeros(NpFine)

    if AbFine is None:
        AbFine = np.zeros(NpFine)
        
    boundaryMap = boundaryConditions==0
    fixedCoarse = util.boundarypIndexMap(NWorldCoarse, boundaryMap=boundaryMap)
    freeCoarse  = np.setdiff1d(np.arange(NpCoarse), fixedCoarse)
    
    AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aFine)
    MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)
    SFine = AFine.copy()

    if massFine is not None:
        SFine +=fem.assemblePatchMatrix(NWorldFine, world.MLocFine, massFine)
    bFine = MFine*MbFine + SFine*AbFine

    basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
    bCoarse = basis.T*bFine
    SCoarse = basis.T*(SFine*basis)

    SCoarseFree = SCoarse[freeCoarse][:,freeCoarse]
    bCoarseFree = bCoarse[freeCoarse]

    uCoarseFree = linalg.linSolve(SCoarseFree, bCoarseFree)
    uCoarseFull = np.zeros(NpCoarse)
    uCoarseFull[freeCoarse] = uCoarseFree
    uCoarseFull = uCoarseFull

    uFineFull = basis*uCoarseFull
    
    return uCoarseFull, uFineFull
