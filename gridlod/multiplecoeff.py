import numpy as np
from scipy import optimize

from . import util
from . import lod

def optimizeAlpha(patch, aPatchRefList, aPatchNew):
    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    world = patch.world
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
    elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse - 1, patch.iElementPatchCoarse)
    TInd = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse - 1, patch.iElementPatchCoarse)

    def approxA(beta):
        assert (len(beta) == len(aPatchRefList))
        aNew = aPatchNew
        aTPrime = aNew[TPrimeIndices]
        scaledAT = np.zeros_like(aTPrime)
        for i in range(len(aPatchRefList)):
            aOld = aPatchRefList[i]
            ### In case aNew and aOld dimensions do not match ###
            if aNew.ndim == 3 and aOld.ndim == 1:
                aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
                aOld = np.einsum('tji, t-> tji', aEye, aOld)
            if aNew.ndim == 1 and aOld.ndim == 3:
                aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
                aNew = np.einsum('tji, t -> tji', aEye, aNew)

            aOldTPrime = aOld[TPrimeIndices]
            scaledAT += beta[i]*(aTPrime-aOldTPrime)
        return np.sqrt(np.sum(np.max(np.abs(scaledAT), axis=1) ** 2))

    betastart = np.zeros(len(aPatchRefList))
    betastart[:]= 1./(len(betastart))
    constrone = lambda beta: 1-np.sum(beta)
    beta = optimize.minimize(approxA, betastart, constraints={'type': 'eq', 'fun':constrone})
    alpha = beta.x
    if not beta.success:
        print('optimization failed')
    return alpha

def optimizeAlpha_indic1(patch, aPatchRefList, aPatchNew,muTPrimeList):
    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    world = patch.world
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
    elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse - 1, patch.iElementPatchCoarse)
    TInd = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse - 1, patch.iElementPatchCoarse)

    def approxA(beta):
        assert (len(beta) == len(aPatchRefList))
        aNew = aPatchNew
        aTPrime = aNew[TPrimeIndices]
        tildeA = np.zeros_like(aTPrime)
        for i in range(len(aPatchRefList)):
            aOld = aPatchRefList[i]
            aOldTPrime = aOld[TPrimeIndices]
            tildeA += beta[i] * aOldTPrime

        indic1 = np.zeros(len(aPatchRefList))

        for i in range(len(aPatchRefList)):
            aOld = aPatchRefList[i]
            ### In case aNew and aOld dimensions do not match ###
            if aNew.ndim == 3 and aOld.ndim == 1:
                aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
                aOld = np.einsum('tji, t-> tji', aEye, aOld)
            if aNew.ndim == 1 and aOld.ndim == 3:
                aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
                aNew = np.einsum('tji, t -> tji', aEye, aNew)

            aOldTPrime = aOld[TPrimeIndices]
            aTPrime = aNew[TPrimeIndices]
            deltaTildeAAOld = np.max(np.abs((aTPrime - tildeA) / np.sqrt(aTPrime * aOldTPrime)), axis=1)
            kappaMaxAAold = np.max(np.abs(aOldTPrime[elementCoarseIndex] / aTPrime[elementCoarseIndex]))
            indic1[i] = np.sqrt(kappaMaxAAold * np.sum(deltaTildeAAOld ** 2 * muTPrimeList[i]))
        return np.min(indic1)

    betastart = np.zeros(len(aPatchRefList))
    betastart[:]= 1./(len(betastart))
    constrone = lambda beta: 1-np.sum(beta)
    beta = optimize.minimize(approxA, betastart, constraints={'type': 'eq', 'fun':constrone})
    alpha = beta.x
    if not beta.success:
        print('Optimization failed')
    return alpha

def optimizeAlpha_indic(patch, aPatchRefList, aPatchNew, muTPrimeList):

    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    def approxA(beta):
        return estimatorAlphaTildeA1mod(patch,muTPrimeList,aPatchRefList,aPatchNew, beta)

    betastart = np.zeros(len(aPatchRefList))
    betastart[:]= 1./(len(betastart))
    constrone = lambda beta: 1-np.sum(beta)
    beta = optimize.minimize(approxA, betastart, constraints={'type': 'eq', 'fun':constrone})
    alpha = beta.x
    return alpha # maybe return also function value of beta because this avoids second evaluation of estimator

def optimizeAlphaMu(patch, aPatchRefList, aPatchNew):
    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    aNew = aPatchNew

    world = patch.world
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
    elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse - 1, patch.iElementPatchCoarse)
    TInd = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse - 1, patch.iElementPatchCoarse)

    aTPrime = aNew[TPrimeIndices] #maybe not working if dimensions of aNew and aOld below are not matching

    #optimization-based choice of alpha
    def approxA(beta):
        assert (len(beta) == 2*len(aPatchRefList))
        alpha= beta[:len(beta)//2]
        mu=beta[len(beta)//2:]
        aNew = aPatchNew
        aTPrime = aNew[TPrimeIndices]
        scaledAT = np.zeros_like(aTPrime)
        for i in range(len(aPatchRefList)):
            aOld = aPatchRefList[i]
            ### In case aNew and aOld dimensions do not match ###
            if aNew.ndim == 3 and aOld.ndim == 1:
                aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
                aOld = np.einsum('tji, t-> tji', aEye, aOld)
            if aNew.ndim == 1 and aOld.ndim == 3:
                aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
                aNew = np.einsum('tji, t -> tji', aEye, aNew)

            aOldTPrime = aOld[TPrimeIndices]
            scaledAT += alpha[i]*(aTPrime-mu[i]*aOldTPrime)
        return np.sqrt(np.sum(np.max(np.abs(scaledAT), axis=1) ** 2))

    betastart = np.zeros(2*len(aPatchRefList))
    betastart[:len(betastart)//2] = 1./(len(betastart)//2)
    betastart[len(betastart)//2:]=1.
    constrone = lambda beta: 1-np.sum(beta[:len(beta)//2])#
    beta = optimize.minimize(approxA, betastart, constraints={'type': 'eq', 'fun':constrone})
    alpha = beta.x[:len(beta.x)//2]
    mu = beta.x[len(beta.x)//2:]
    return alpha, mu

def estimatorAlphaTildeA1(patch, muTPrimeList, aPatchRefList, aPatchNew, alpha):
    for i in range(len(muTPrimeList)):
        while callable(muTPrimeList[i]):
           muTPrimeList[i] = muTPrimeList[i]()
    muTPrimeList = np.array(muTPrimeList)

    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    assert(len(muTPrimeList) == len(aPatchRefList))

    aNew = aPatchNew

    world = patch.world
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
    elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse - 1, patch.iElementPatchCoarse)
    TInd = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse - 1, patch.iElementPatchCoarse)

    aTPrime = aNew[TPrimeIndices] #maybe not working if dimensions of aNew and aOld below are not matching

    tildeA = np.zeros_like(aNew)
    for i in range(len(aPatchRefList)):
        aOld = aPatchRefList[i]
        tildeA += alpha[i]*aOld

    EQTtildeA = np.zeros(len(aPatchRefList))
    indic1 = np.zeros(len(aPatchRefList))

    for i in range(len(aPatchRefList)):
        aOld = aPatchRefList[i]
        ### In case aNew and aOld dimensions do not match ###
        if aNew.ndim == 3 and aOld.ndim ==1:
            aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
            aOld = np.einsum('tji, t-> tji', aEye, aOld)
        if aNew.ndim == 1 and aOld.ndim ==3:
            aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
            aNew = np.einsum('tji, t -> tji', aEye, aNew)

        aOldTPrime = aOld[TPrimeIndices]
        aTPrime = aNew[TPrimeIndices]
        tildeATPrime = tildeA[TPrimeIndices]
        deltaTildeAAOld = np.max(np.abs((aTPrime-tildeATPrime)/np.sqrt(aTPrime * aOldTPrime)), axis=1)
        kappaMaxAAold = np.max(np.abs(aOldTPrime[elementCoarseIndex]/aTPrime[elementCoarseIndex]))
        indic1[i] = np.sqrt(kappaMaxAAold * np.sum(deltaTildeAAOld**2 * muTPrimeList[i]))
        EQTtildeA[i] = lod.computeErrorIndicatorCoarseFromCoefficients(patch,muTPrimeList[i],aOld,tildeA)

    deltaTildeAANew = np.max(np.max(np.abs((aTPrime-tildeATPrime)/np.sqrt(np.abs(aTPrime * tildeATPrime))), axis=1))
    kappaMaxAtildeA = np.max(np.max(np.abs(np.sqrt(aTPrime/np.abs(tildeATPrime))), axis=1))
    indicMin = np.min(indic1 + deltaTildeAANew * EQTtildeA)
    indicSum = np.sum(np.abs(alpha) * kappaMaxAtildeA * EQTtildeA)

    EQT = indicMin + indicSum

    return EQT

def estimatorAlphaTildeA1mod(patch, muTPrimeList, aPatchRefList, aPatchNew, alpha):
    for i in range(len(muTPrimeList)):
        while callable(muTPrimeList[i]):
           muTPrimeList[i] = muTPrimeList[i]()
    muTPrimeList = np.array(muTPrimeList)

    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    assert(len(muTPrimeList) == len(aPatchRefList))

    aNew = aPatchNew

    world = patch.world
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
    elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse - 1, patch.iElementPatchCoarse)
    TInd = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse - 1, patch.iElementPatchCoarse)

    aTPrime = aNew[TPrimeIndices] #maybe not working if dimensions of aNew and aOld below are not matching

    tildeA = np.zeros_like(aNew)
    for i in range(len(aPatchRefList)):
        aOld = aPatchRefList[i]
        tildeA += alpha[i]*aOld

    EQTtildeA = np.zeros(len(aPatchRefList))
    indic1 = np.zeros(len(aPatchRefList))
    kappaMaxA = np.zeros(len(aPatchRefList))
    tildeATPrime = tildeA[TPrimeIndices]
    kappaMaxAtildeA = np.max(np.max(np.abs(np.sqrt(aTPrime/np.abs(tildeATPrime))), axis=1))

    for i in range(len(aPatchRefList)):
        aOld = aPatchRefList[i]
        ### In case aNew and aOld dimensions do not match ###
        if aNew.ndim == 3 and aOld.ndim ==1:
            aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
            aOld = np.einsum('tji, t-> tji', aEye, aOld)
        if aNew.ndim == 1 and aOld.ndim ==3:
            aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
            aNew = np.einsum('tji, t -> tji', aEye, aNew)

        aOldTPrime = aOld[TPrimeIndices]
        aTPrime = aNew[TPrimeIndices]
        deltaTildeAAOld = np.max(np.abs((aTPrime-tildeATPrime)/np.sqrt(aTPrime * aOldTPrime)), axis=1)
        kappaMaxAAold = np.max(np.abs(aOldTPrime[elementCoarseIndex]/aTPrime[elementCoarseIndex]))
        kappaMaxADiff = np.max(np.max(np.abs((aTPrime - aOldTPrime) / np.sqrt(np.abs(aTPrime * tildeATPrime)))))
        kappaMaxA[i] = np.min([kappaMaxAtildeA, kappaMaxADiff], axis=0)
        indic1[i] = np.sqrt(kappaMaxAAold * np.sum(deltaTildeAAOld**2 * muTPrimeList[i]))
        EQTtildeA[i] = lod.computeErrorIndicatorCoarseFromCoefficients(patch,muTPrimeList[i],aOld,tildeA)

    deltaTildeAANew = np.max(np.max(np.abs((aTPrime-tildeATPrime)/np.sqrt(np.abs(aTPrime * tildeATPrime))), axis=1))
    indicMin = np.min(indic1 + deltaTildeAANew * EQTtildeA)
    indicSum = np.sum(np.abs(alpha) * kappaMaxA * EQTtildeA)

    EQT = indicMin + indicSum

    return EQT


def estimatorAlphaTildeA2(patch, muTPrimeList, aPatchRefList, aPatchNew, alpha):
    for i in range(len(muTPrimeList)):
        while callable(muTPrimeList[i]):
           muTPrimeList[i] = muTPrimeList[i]()
    muTPrimeList = np.array(muTPrimeList)

    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    assert(len(muTPrimeList) == len(aPatchRefList))

    aNew = aPatchNew

    world = patch.world
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
    elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse - 1, patch.iElementPatchCoarse)
    TInd = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse - 1, patch.iElementPatchCoarse)

    aTPrime = aNew[TPrimeIndices] #maybe not working if dimensions of aNew and aOld below are not matching

    tildeA = np.zeros_like(aNew)
    for i in range(len(aPatchRefList)):
        aOld = aPatchRefList[i]
        tildeA += alpha[i]*aOld

    EQTtildeA = np.zeros(len(aPatchRefList))
    indic1 = np.zeros(len(aPatchRefList))
    kappaMaxA = np.zeros(len(aPatchRefList))

    for i in range(len(aPatchRefList)):
        aOld = aPatchRefList[i]
        ### In case aNew and aOld dimensions do not match ###
        if aNew.ndim == 3 and aOld.ndim ==1:
            aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
            aOld = np.einsum('tji, t-> tji', aEye, aOld)
        if aNew.ndim == 1 and aOld.ndim ==3:
            aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
            aNew = np.einsum('tji, t -> tji', aEye, aNew)

        aOldTPrime = aOld[TPrimeIndices]
        aTPrime = aNew[TPrimeIndices]
        tildeATprime = tildeA[TPrimeIndices]
        deltaTildeAAOld = np.max(np.abs((aTPrime-tildeATprime)/np.sqrt(aTPrime * aOldTPrime)), axis=1)
        kappaMaxAAold = np.max(np.abs(aOldTPrime[elementCoarseIndex]/aTPrime[elementCoarseIndex]))
        kappaMaxA[i] = np.max(np.max(np.abs((aTPrime-aOldTPrime)/np.sqrt(np.abs(aTPrime * tildeATprime)))))
        indic1[i] = np.sqrt(kappaMaxAAold * np.sum(deltaTildeAAOld**2 * muTPrimeList[i]))
        EQTtildeA[i] = lod.computeErrorIndicatorCoarseFromCoefficients(patch,muTPrimeList[i],aOld,tildeA)

    deltaTildeAANew = np.max(np.max(np.abs((aTPrime-tildeATprime)/np.sqrt(np.abs(aTPrime * tildeATprime))), axis=1))
    indicMin = np.min(indic1 + deltaTildeAANew * EQTtildeA)
    indicSum = np.sum(np.abs(alpha) * kappaMaxA * EQTtildeA)

    EQT = indicMin + indicSum

    return EQT

def estimatorAlphaA(patch, muTPrimeList, aPatchRefList, aPatchNew, alpha):
    for i in range(len(muTPrimeList)):
        while callable(muTPrimeList[i]):
           muTPrimeList[i] = muTPrimeList[i]()
    muTPrimeList = np.array(muTPrimeList)

    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    assert(len(muTPrimeList) == len(aPatchRefList))

    aNew = aPatchNew

    world = patch.world
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
    elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse - 1, patch.iElementPatchCoarse)
    TInd = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse - 1, patch.iElementPatchCoarse)

    aTPrime = aNew[TPrimeIndices] #maybe not working if dimensions of aNew and aOld below are not matching

    tildeA = np.zeros_like(aTPrime)
    for i in range(len(aPatchRefList)):
        aOld = aPatchRefList[i]
        aOldTPrime = aOld[TPrimeIndices]
        tildeA += alpha[i]*aOldTPrime

    EQTA = np.zeros(len(aPatchRefList))
    indic1 = np.zeros(len(aPatchRefList))
    kappaMaxA = np.zeros(len(aPatchRefList))

    for i in range(len(aPatchRefList)):
        aOld = aPatchRefList[i]
        ### In case aNew and aOld dimensions do not match ###
        if aNew.ndim == 3 and aOld.ndim ==1:
            aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
            aOld = np.einsum('tji, t-> tji', aEye, aOld)
        if aNew.ndim == 1 and aOld.ndim ==3:
            aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
            aNew = np.einsum('tji, t -> tji', aEye, aNew)

        aOldTPrime = aOld[TPrimeIndices]
        aTPrime = aNew[TPrimeIndices]
        deltaTildeAAOld = np.max(np.abs((aTPrime-tildeA)/np.sqrt(aTPrime * aOldTPrime)), axis=1)
        kappaMaxAAold = np.max(np.abs(aOldTPrime[elementCoarseIndex]/aTPrime[elementCoarseIndex]))
        kappaMaxA[i] = np.max(np.max(np.abs((aTPrime-aOldTPrime)/np.sqrt(np.abs(aTPrime * aTPrime)))))
        indic1[i] = np.sqrt(kappaMaxAAold * np.sum(deltaTildeAAOld**2 * muTPrimeList[i]))
        EQTA[i] = lod.computeErrorIndicatorCoarseFromCoefficients(patch,muTPrimeList[i],aOld,aPatchNew)

    deltaTildeAANew = np.max(np.max(np.abs((aTPrime-tildeA)/np.sqrt(np.abs(aTPrime * aTPrime))), axis=1))
    indicMin = np.min(indic1 + deltaTildeAANew * EQTA)
    indicSum = np.sum(np.abs(alpha) * kappaMaxA * EQTA)

    EQT = indicMin + indicSum

    return EQT

#def estimatorAlphaMu: