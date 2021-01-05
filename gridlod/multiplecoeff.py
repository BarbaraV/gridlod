import numpy as np
from scipy import optimize
import scipy.linalg

from . import util
from . import lod
from . import fem

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

    aNew = aPatchNew
    aTPrime = aNew[TPrimeIndices]
    for i in range(len(aPatchRefList)):
        ### In case aNew and aOld dimensions do not match ###
        if aNew.ndim == 3 and aPatchRefList[i].ndim == 1:
            aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
            aPatchRefList[i] = np.einsum('tji, t-> tji', aEye, aPatchRefList[i])
        if aNew.ndim == 1 and aPatchRefList[i].ndim == 3:
            aEye = np.tile(np.eye(2), [np.prod(NPatchFine), 1, 1])
            aNew = np.einsum('tji, t -> tji', aEye, aNew)
    aOldTPrime = [aOld[TPrimeIndices] for aOld in aPatchRefList]
    #aTPrimeList = [aTPrime for _ in aPatchRefList]
    #aDiffTPrime = aTPrimeList - aOldTPrime

    def approxA(beta):
        assert (len(beta) == len(aPatchRefList))
        if aNew.ndim == 1:
            scaledAT = aTPrime - np.einsum('i, itj -> tj', beta, np.array(aOldTPrime))
            #scaledAT = np.einsum('i, ij->j', beta, np.array(aDiffTPrime))
        else:
            scaledAT = aTPrime - np.einsum('i, itsjk -> tsjk', beta, np.array(aOldTPrime))
            # scaledAT = np.einsum('i, itsjk->tsjk', beta, np.array(aDiffTPrime))
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
        if aNew.ndim == 3:
            aInvTPrime = np.linalg.inv(aTPrime)
            aOldInvTPrime = np.linalg.inv(aOldTPrime)
            tildeAInvTPrime = np.linalg.inv(tildeATPrime)
            aDiffTPrime = aTPrime - aOldTPrime
            tildeaDiffTPrime = aTPrime - tildeATPrime
            deltaTildeAAOld = np.sqrt(np.max(np.linalg.norm(np.einsum('Ttij, Ttjk, Ttkl, Ttlm -> Ttim',
                                                                     aInvTPrime, tildeaDiffTPrime, tildeaDiffTPrime,
                                                                     aOldInvTPrime),
                                                           axis=(2, 3), ord=2),
                                            axis=1))
            kappaMaxAAold = np.max(np.linalg.norm(np.einsum('tij, tjk -> tik',
                                                                aOldTPrime[elementCoarseIndex],
                                                                aInvTPrime[elementCoarseIndex]),
                                                      axis=(1, 2), ord=2))
            kappaMaxADiff = np.max(np.max(np.sqrt(np.linalg.norm(np.einsum('Ttij, Ttjk, Ttkl, Ttlm -> Ttim',
                                                                     aInvTPrime, aDiffTPrime, aDiffTPrime,
                                                                     tildeAInvTPrime),
                                                           axis=(2, 3), ord=2)),
                                            axis=1))
            kappaMaxAtildeA = np.max(np.max(np.sqrt(np.linalg.norm(np.einsum('Ttij, Ttjk -> Ttik',
                                                                aTPrime,
                                                                tildeAInvTPrime),
                                                      axis=(2, 3), ord=2)), axis=1))
            deltaTildeAANew = np.max(np.max(np.sqrt(np.linalg.norm(np.einsum('Ttij, Ttjk, Ttkl, Ttlm -> Ttim',
                                                                      aInvTPrime, tildeaDiffTPrime, tildeaDiffTPrime,
                                                                      tildeAInvTPrime),
                                                            axis=(2, 3), ord=2)),
                                             axis=1))
        else:
            deltaTildeAAOld = np.max(np.abs((aTPrime-tildeATPrime)/np.sqrt(aTPrime * aOldTPrime)), axis=1)
            kappaMaxAAold = np.max(np.abs(aOldTPrime[elementCoarseIndex]/aTPrime[elementCoarseIndex]))
            kappaMaxADiff = np.max(np.max(np.abs((aTPrime - aOldTPrime) / np.sqrt(np.abs(aTPrime * tildeATPrime)))))
            kappaMaxAtildeA = np.max(np.max(np.abs(np.sqrt(aTPrime / np.abs(tildeATPrime))), axis=1))
            deltaTildeAANew = np.max(
                np.max(np.abs((aTPrime - tildeATPrime) / np.sqrt(np.abs(aTPrime * tildeATPrime))), axis=1))
        kappaMaxA[i] = np.min([kappaMaxAtildeA, kappaMaxADiff], axis=0)
        indic1[i] = np.sqrt(kappaMaxAAold * np.sum(deltaTildeAAOld**2 * muTPrimeList[i]))
        EQTtildeA[i] = lod.computeErrorIndicatorCoarseFromCoefficients(patch,muTPrimeList[i],aOld,tildeA)

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


def computeErrorIndicatorFineMultiple(patch, correctorsList, aRefList, mu, aPatchNew=None):
    ''' Compute the fine error idicator e(T) for given vector mu.

    This requires reference coefficients (already localized) and their correctors. New coefficient is optimal, otherwise
    assumed to be weighetd sume of mu and reference coefficients.
    '''

    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    if aRefList[0].ndim != 1:
        NotImplementedError("matrix-valued coefficient not yet supported")
    if aPatchNew is None:
        NotImplementedError("error indicator with pertrubed coeff not yet implemented correctly")

    lambdasList = list(patch.world.localBasis.T)

    NPatchCoarse = patch.NPatchCoarse
    world = patch.world
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse * NCoarseElement

    nref = len(aRefList)
    a = aPatchNew

    ALocFine = world.ALocFine
    P = np.column_stack(lambdasList)

    TFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement - 1, NPatchFine - 1)
    iElementPatchFine = patch.iElementPatchCoarse * NCoarseElement
    TFinetStartIndex = util.convertpCoordIndexToLinearIndex(NPatchFine - 1, iElementPatchFine)
    TFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)
    TFinepStartIndex = util.convertpCoordIndexToLinearIndex(NPatchFine, iElementPatchFine)

    A = np.zeros_like(world.ALocCoarse)
    aBar = np.einsum('i, ij->j', mu, aRefList)
    if aPatchNew is None:
        a = aBar
    else:
        a = aPatchNew
        bTcoeff = np.sqrt(a)*(1-aBar/a)
        bT = bTcoeff[TFinetStartIndex + TFinetIndexMap]
        TNorm = fem.assemblePatchMatrix(NCoarseElement, ALocFine, bT**2)

    nnz = np.where(mu != 0)[0]
    #b = [mu[ii]*np.sqrt(a)*(1-aRefList[ii]/a) for ii in range(nref)]

    for ii in nnz:
        for jj in nnz:
            bij = (mu[ii]*np.sqrt(a)*(1-aRefList[ii]/a))*(mu[jj]*np.sqrt(a)*(1-aRefList[jj]/a))
            PatchNorm = fem.assemblePatchMatrix(NPatchFine, ALocFine,bij)
            Q1 = np.column_stack(correctorsList[ii])
            Q2 = np.column_stack(correctorsList[jj])
            A += np.dot(Q1.T, PatchNorm*Q2)
            if aPatchNew is not None:
                bii = mu[ii]*np.sqrt(a)*(1-aRefList[ii]/a)
                bjj = mu[jj] * np.sqrt(a) * (1 - aRefList[jj] / a)
                bTii = bT * bii[TFinetStartIndex + TFinetIndexMap]
                bTjj = bT * bjj[TFinetStartIndex + TFinetIndexMap]
                TNormPQ = fem.assemblePatchMatrix(NPatchFine, ALocFine, bTjj)
                TNormQP = fem.assemblePatchMatrix(NPatchFine, ALocFine, bTii)
                QT1 = Q1[TFinepStartIndex + TFinepIndexMap, :]
                QT2 = Q2[TFinepStartIndex + TFinepIndexMap, :]
                A -= np.dot(P.T, TNormPQ*QT2)
                A -= np.dot(QT1.T, TNormQP*P)

    if aPatchNew is not None:
        A += np.dot(P.T, TNorm*P)

    BNorm = fem.assemblePatchMatrix(NCoarseElement, ALocFine, a[TFinetStartIndex + TFinetIndexMap])
    B = np.dot(P.T, BNorm * P)

    eigenvalues = scipy.linalg.eigvals(A[:-1, :-1], B[:-1, :-1])
    epsilonTSquare = np.max(np.real(eigenvalues))

    return np.sqrt(epsilonTSquare)