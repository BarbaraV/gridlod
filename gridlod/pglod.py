import numpy as np
import scipy.sparse as sparse
from copy import deepcopy

from . import util
from . import fem

def assembleBasisCorrectors(world, patchT, basisCorrectorsListT, periodic=False):
    '''Compute the basis correctors given the elementwise basis
    correctors for each coarse element.

    '''
    NWorldCoarse = world.NWorldCoarse
    NCoarseElement = world.NCoarseElement
    NWorldFine = NWorldCoarse*NCoarseElement

    NtCoarse = np.prod(NWorldCoarse)
    NpCoarse = np.prod(NWorldCoarse+1)
    NpFine = np.prod(NWorldFine+1)

    TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
    TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)

    cols = []
    rows = []
    data = []
    for TInd in range(NtCoarse):
        if periodic:
            basisCorrectorsList = basisCorrectorsListT # periodic case: same corrector everywhere
        else:
            basisCorrectorsList = basisCorrectorsListT[TInd]
        patch = patchT[TInd]
        
        NPatchFine = patch.NPatchCoarse*NCoarseElement
        iPatchWorldFine = patch.iPatchWorldCoarse*NCoarseElement

        patchpIndexMap = util.lowerLeftpIndexMap(NPatchFine, NWorldFine)
        patchpStartIndex = util.convertpCoordIndexToLinearIndex(NWorldFine, iPatchWorldFine)

        if periodic:
            rowsT = (patchpStartIndex + patchpIndexMap) % (NpCoarse-1)
            colsT = (TpStartIndices[TInd] + TpIndexMap) % (NpCoarse-1)
        else:
            rowsT = patchpStartIndex + patchpIndexMap
            colsT = TpStartIndices[TInd] + TpIndexMap
        dataT = np.hstack(basisCorrectorsList)

        cols.extend(np.repeat(colsT, np.size(rowsT)))
        rows.extend(np.tile(rowsT, np.size(colsT)))
        data.extend(dataT)

    basisCorrectors = sparse.csc_matrix((data, (rows, cols)), shape=(NpFine, NpCoarse))

    return basisCorrectors
        
def assemblePatchFunction(world, patchT, funcT):
    numPatches = len(patchT)
    assert(numPatches == len(funcT))
    assert(numPatches >= 1)

    if funcT[0].size == patchT[0].NpCoarse:
        NWorld = world.NWorldCoarse
        NElement = np.ones_like(world.NCoarseElement)
    elif funcT[0].size == patchT[0].NpFine: 
        NWorld = world.NWorldFine
        NElement = world.NCoarseElement
   
    NpWorld = np.prod(NWorld+1)

    cols = []
    rows = []
    data = []
    for patchInd in range(numPatches):
        func = funcT[patchInd]
        patch = patchT[patchInd]
        
        NPatch = patch.NPatchCoarse*NElement
        iPatchWorld = patch.iPatchWorldCoarse*NElement

        patchpIndexMap = util.lowerLeftpIndexMap(NPatch, NWorld)
        patchpStartIndex = util.convertpCoordIndexToLinearIndex(NWorld, iPatchWorld)

        rowsPatch = patchpStartIndex + patchpIndexMap
        colsPatch = 0*rowsPatch
        dataPatch = func

        cols.extend(colsPatch)
        rows.extend(rowsPatch)
        data.extend(dataPatch)

    funcFullSparse = sparse.csc_matrix((data, (rows, cols)), shape=(NpWorld, 1))
    funcFull = np.squeeze(np.array(funcFullSparse.todense()))
    
    return funcFull
    

def assembleMsStiffnessMatrix(world, patchT, KmsijT, periodic=False):
    '''Compute the multiscale Petrov-Galerkin stiffness matrix given
    Kmsij for each coarse element.

    '''
    NWorldCoarse = world.NWorldCoarse

    NtCoarse = np.prod(world.NWorldCoarse)
    NpCoarse = np.prod(world.NWorldCoarse+1)

    TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
    TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)

    cols = []
    rows = []
    data = []
    for TInd in range(NtCoarse):
        if periodic and not (isinstance(KmsijT, tuple) or isinstance(KmsijT,list)):  # if only one matrix is given in periodic case
            Kmsij = KmsijT
        else:
            Kmsij = KmsijT[TInd]
        patch = patchT[TInd]
        
        NPatchCoarse = patch.NPatchCoarse

        patchpIndexMap = util.lowerLeftpIndexMap(NPatchCoarse, NWorldCoarse)
        patchpStartIndex = util.convertpCoordIndexToLinearIndex(NWorldCoarse, patch.iPatchWorldCoarse)

        if periodic:
            rowsTpCoord = (patch.iPatchWorldCoarse.T + util.convertpLinearIndexToCoordIndex(NWorldCoarse,patchpIndexMap).T)\
                          % NWorldCoarse
            rowsT = util.convertpCoordIndexToLinearIndex(NWorldCoarse,rowsTpCoord)
            colsTbase = TpStartIndices[TInd] + TpIndexMap
            colsTpCoord = util.convertpLinearIndexToCoordIndex(NWorldCoarse,colsTbase).T % NWorldCoarse
            colsT = util.convertpCoordIndexToLinearIndex(NWorldCoarse,colsTpCoord)
        else:
            rowsT = patchpStartIndex + patchpIndexMap
            colsT = TpStartIndices[TInd] + TpIndexMap
        dataT = Kmsij.flatten()

        cols.extend(np.tile(colsT, np.size(rowsT)))
        rows.extend(np.repeat(rowsT, np.size(colsT)))
        data.extend(dataT)

    Kms = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

    return Kms

def assembleStiffnessMatrix(world, KijT):
    '''Compute the standard coarse stiffness matrix given Kij for each
    coarse element.

    '''
    world = self.world
    NWorldCoarse = world.NWorldCoarse

    NtCoarse = np.prod(world.NWorldCoarse)
    NpCoarse = np.prod(world.NWorldCoarse+1)

    TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
    TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)

    cols = []
    rows = []
    data = []
    for TInd in range(NtCoarse):
        Kij = KijT[TInd]

        NPatchCoarse = ecT.NPatchCoarse

        colsT = TpStartIndices[TInd] + TpIndexMap
        rowsT = TpStartIndices[TInd] + TpIndexMap
        dataT = Kij.flatten()

        cols.extend(np.tile(colsT, np.size(rowsT)))
        rows.extend(np.repeat(rowsT, np.size(colsT)))
        data.extend(dataT)

    K = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

    return K
    
def solve(world, KmsFull, bFull, boundaryConditions):
    NWorldCoarse = world.NWorldCoarse
    NpCoarse = np.prod(NWorldCoarse+1)
        
    fixed = util.boundarypIndexMap(NWorldCoarse, boundaryConditions==0)
    free  = np.setdiff1d(np.arange(NpCoarse), fixed)
        
    KmsFree = KmsFull[free][:,free]
    bFree = bFull[free]

    uFree = sparse.linalg.spsolve(KmsFree, bFree)

    uFull = np.zeros(NpCoarse)
    uFull[free] = uFree

    return uFull, uFree


def solvePeriodic(world, KmsFull, bFull, faverage, boundaryConditions=None):
    NWorldCoarse = world.NWorldCoarse
    NpCoarse = np.prod(NWorldCoarse + 1)
    d = np.size(NWorldCoarse)

    MCoarse = fem.assemblePatchMatrix(world.NWorldCoarse, world.MLocCoarse)
    averageVector = MCoarse * np.ones(NpCoarse)

    if d == 1:
        free = np.arange(1, NpCoarse-1)
    elif d == 2:
        fixed = np.concatenate((np.arange(NWorldCoarse[1]*(NWorldCoarse[0]+1), NpCoarse),
                                np.arange(NWorldCoarse[0], NpCoarse-1, NWorldCoarse[0]+1)))
        free = np.setdiff1d(np.arange(NpCoarse), fixed)

        bFull[np.arange(0, NWorldCoarse[1] * (NWorldCoarse[0] + 1)+1, NWorldCoarse[0] + 1)] \
            += bFull[np.arange(NWorldCoarse[0], NpCoarse, NWorldCoarse[0]+1)]
        bFull[np.arange(NWorldCoarse[0] + 1)] += bFull[np.arange(NWorldCoarse[1]*(NWorldCoarse[0]+1), NpCoarse)]
        averageVector[np.arange(0, NWorldCoarse[1] * (NWorldCoarse[0] + 1)+1, NWorldCoarse[0] + 1)] \
            += averageVector[np.arange(NWorldCoarse[0], NpCoarse, NWorldCoarse[0]+1)]
        averageVector[np.arange(NWorldCoarse[0] + 1)] += averageVector[np.arange(NWorldCoarse[1]*(NWorldCoarse[0]+1), NpCoarse)]
    else:
        NotImplementedError('higher dimensions not yet implemented')

    KmsFree = KmsFull[free][:, free]
    constraint = averageVector[free].reshape((1,KmsFree.shape[0]))
    K = sparse.bmat([[KmsFree, constraint.T],
                     [constraint, None]], format='csc')
    bFree = bFull[free] - faverage * averageVector[free]  #right-hand side with non-zero average potentially not working correctly yet
    b = np.zeros(K.shape[0])
    b[:np.size(bFree)] = bFree
    x = sparse.linalg.spsolve(K,b)
    uFree = x[:np.size(bFree)]

    uFull = np.zeros(NpCoarse)
    uFull[free] = uFree


    if d == 1:
        uFull[NpCoarse-1] = uFull[0] #not relevant in 1d
    elif d == 2:
        uFull[np.arange(NWorldCoarse[0], NpCoarse-1, NWorldCoarse[0]+1)] \
            += uFull[np.arange(0, NWorldCoarse[1]*(NWorldCoarse[0]+1),NWorldCoarse[0]+1)]
        uFull[np.arange(NWorldCoarse[1]*(NWorldCoarse[0]+1), NpCoarse)] += uFull[np.arange(NWorldCoarse[0]+1)]
    else:
        NotImplementedError('higher dimensiona not yet implemented')


    return uFull, uFree

# The computeFaceFluxTF function has been temporarily removed since
# there were no tests for it.  Readd if needed. Find it in git
# history.

#def computeFaceFluxTF(self, u, f=None):
