import lod
import lod_helmholtz
import interp
import coef


world = None
coefficient = None
saddleSolver = None
k = None
IPatchGenerator = None
clearFineQuantities = None
coeff_mass = None
coeff_bdry = None

# for rCoarse
aBase = None


def clearWorker():
    global world, coefficient, saddleSolver, IPatchGenerator, k, clearFineQuantities, coeff_mass, coeff_bdry, aBase
    world = None
    coefficient = None
    saddleSolver = None
    k = None
    IPatchGenerator = None
    clearFineQuantities = None
    coeff_mass = None
    coeff_bdry = None
    aBase = None


def hasaBase():
    return aBase is not None


def sendar(aBaseIn, rCoarseIn):
    global aBase, coefficient
    if aBaseIn is not None:
        aBase = aBaseIn
    coefficient = coef.coefficientCoarseFactor(world.NWorldCoarse, world.NCoarseElement, aBase, rCoarseIn)

def setupWorker(worldIn, coefficientIn, coeff_massIn, coeff_bdryIn,
                IPatchGeneratorIn, kIn, clearFineQuantitiesIn):
    global world, coefficient, saddleSolver, IPatchGenerator, k, clearFineQuantities, coeff_mass, coeff_bdry

    world = worldIn
    if coefficientIn is not None:
        coefficient = coefficientIn
    if coeff_massIn is not None:
        coeff_mass = coeff_massIn
    if coeff_bdryIn is not None:
        coeff_bdry = coeff_bdryIn
    saddleSolver = lod.directSolver()
    k = kIn
    if IPatchGeneratorIn is not None:
        IPatchGenerator = IPatchGeneratorIn
    else:
        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, world.NWorldCoarse,
                                                                      world.NCoarseElement,
                                                                      world.boundaryConditions)
    clearFineQuantities = clearFineQuantitiesIn


def computeElementCorrector(iElement):
    ecT = lod_helmholtz.elementCorrectorHelmholtz(world, k, iElement, saddleSolver)
    coefficientPatch = coefficient.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
    coeff_massPatch = coeff_mass.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
    coeff_bdryPatch = coeff_bdry.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
    IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

    ecT.computeCorrectors(coefficientPatch, coeff_massPatch, coeff_bdryPatch, IPatch)
    ecT.computeCoarseQuantities()
    if clearFineQuantities:
        ecT.clearFineQuantities()

    return ecT
