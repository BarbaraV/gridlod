import numpy as np

from . import util

def localizeCoefficient(patch, aFine, periodic = False):
    iPatchWorldCoarse = patch.iPatchWorldCoarse
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = patch.world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement
    iPatchWorldFine = iPatchWorldCoarse*NCoarseElement
    NWorldFine = patch.world.NWorldFine
    NtPatchFine = np.prod(NPatchFine)
    d = np.size(iPatchWorldCoarse)
    
    # a
    coarsetIndexMap = util.lowerLeftpIndexMap(NPatchFine-1, NWorldFine-1)
    coarsetStartIndex = util.convertpCoordIndexToLinearIndex(NWorldFine-1, iPatchWorldFine)
    if periodic:
        coarsetIndCoord = (iPatchWorldFine.T + util.convertpLinearIndexToCoordIndex(NWorldFine-1, coarsetIndexMap).T) \
                      % NWorldFine
        coarsetIndices = util.convertpCoordIndexToLinearIndex(NWorldFine-1, coarsetIndCoord)
        aFineLocalized = aFine[coarsetIndices]
    else:
        aFineLocalized = aFine[coarsetStartIndex + coarsetIndexMap]
    return aFineLocalized


def scaleCoefficient(aFine):

    aMean = None

    if aFine.ndim == 1:
        aMean = np.mean(aFine, axis=0)
    elif aFine.ndim == 3:
        aMean = np.mean(np.trace(aFine, axis1=1, axis2=2))
    else:
        NotImplementedError('only scalar- and matrix-valued coefficients supported')

    return aFine/aMean

def averageCoefficient(aFine):

    aMean = None

    if aFine.ndim == 1:
        aMean = np.mean(aFine, axis=0)
    elif aFine.ndim == 3:
        aMean = np.mean(np.trace(aFine, axis1=1, axis2=2))
    else:
        NotImplementedError('only scalar- and matrix-valued coefficients supported')

    return aMean

