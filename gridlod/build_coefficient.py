import numpy as np
from gridlod import util

def build_randomcheckerboard(Nepsilon, NFine, alpha, beta, p):
    # builds a random checkerboard coefficient with spectral bounds alpha and beta,
    # piece-wise constant on mesh with Nepsilon blocks
    # returns a fine coefficient on mesh with NFine blocks
    Ntepsilon = np.prod(Nepsilon)
    c = np.random.binomial(1,p,Ntepsilon)
    #c1 = np.random.rand(Ntepsilon)
    values = alpha + (beta-alpha) * c

    def randomcheckerboard(x):
        index = (x*Nepsilon).astype(int)
        d = np.shape(index)[1]

        if d == 1:
            flatindex = index[:]
        if d == 2:
            flatindex = index[:,1]*Nepsilon[0]+index[:,0]
        if d == 3:
            flatindex = index[:,2]*(Nepsilon[0]*Nepsilon[1]) + index[:,1]*Nepsilon[0] + index[:,0]
        else:
            NotImplementedError('other dimensions not available')

        return values[flatindex]

    xFine = util.tCoordinates(NFine)

    return randomcheckerboard(xFine).flatten()

def build_checkerboardbasis(NPatch, NepsilonElement, NFineElement, alpha, beta):
    # builds a list of coeeficients to combine any checkerboard coefficient
    # input: NPatch is number of coarse elements, NepsilonElement and NFineElement the number of cells (per dimension)
    # per coarse element for the epsilon and the fine mesh, respectively; alpha and beta are the spectral bounds of the coefficient

    Nepsilon = NPatch * NepsilonElement
    Ntepsilon = np.prod(Nepsilon)
    NFine = NPatch*NFineElement
    NtFine = np.prod(NFine)

    def checkerboardI(ii):
        coeff = alpha * np.ones(NtFine)
        #find out which indices on fine grid correspond to element ii on epsilon grid
        elementIndex = util.convertpLinearIndexToCoordIndex(Nepsilon-1, ii)[:]
        indices = util.extractElementFine(Nepsilon, NFineElement//NepsilonElement, elementIndex)
        coeff[indices] = beta
        return coeff

    checkerboardbasis = list(map(checkerboardI, range(Ntepsilon)))
    checkerboardbasis.append(alpha*np.ones(NtFine))

    return checkerboardbasis

def build_inclusions_defect_2d(NFine, Nepsilon, bg, val, incl_bl, incl_tr, p_defect, def_val=None):
    # builds a fine coefficient which is periodic with periodicity length 1/epsilon.
    # On the unit cell, the coefficient takes the value val inside a rectangle described by  incl_bl (bottom left) and
    # incl_tr (top right), otherwise the value is bg
    # with a probability of p_defect the inclusion 'vanishes', i.e. the value is set to def_val (default: bg)

    assert(np.all(incl_bl) >= 0.)
    assert(np.all(incl_tr) <= 1.)
    assert(p_defect < 1.)

    if def_val is None:
        def_val = bg

    #include fixed percentage of defects
    #N_defect = int(p_defect*np.prod(Nepsilon))
    #defect_indices = np.random.choice(np.prod(Nepsilon), N_defect, replace=False)
    #c = np.zeros(np.prod(Nepsilon))
    #c[defect_indices] = 1.

    #probability of defect is p_defect
    c = np.random.binomial(1, p_defect, np.prod(Nepsilon))

    aBaseSquare = bg*np.ones(NFine)
    flatidx = 0
    for ii in range(Nepsilon[0]):
        for jj in range(Nepsilon[1]):
            startindexcols = int((ii + incl_bl[0]) * (NFine/Nepsilon)[0])
            stopindexcols = int((ii + incl_tr[0]) * (NFine/Nepsilon)[0])
            startindexrows = int((jj + incl_bl[1]) * (NFine/Nepsilon)[1])
            stopindexrows = int((jj + incl_tr[1]) * (NFine/Nepsilon)[1])
            if c[flatidx] == 0: #not flatidx in defect_indices:
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val
            else:
                aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = def_val
            flatidx += 1

    return aBaseSquare.flatten()

def build_inclusionbasis_2d(NPatch, NEpsilonElement, NFineElement, bg, val, incl_bl, incl_tr):
    Nepsilon = NPatch * NEpsilonElement
    NFine = NPatch * NFineElement

    assert (np.all(incl_bl) >= 0.)
    assert (np.all(incl_tr) <= 1.)

    aBaseSquare = bg * np.ones(NFine)
    for ii in range(Nepsilon[0]):
        for jj in range(Nepsilon[1]):
            startindexcols = int((ii + incl_bl[0]) * (NFine / Nepsilon)[0])
            stopindexcols = int((ii + incl_tr[0]) * (NFine / Nepsilon)[0])
            startindexrows = int((jj + incl_bl[1]) * (NFine / Nepsilon)[1])
            stopindexrows = int((jj + incl_tr[1]) * (NFine / Nepsilon)[1])
            aBaseSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = val

    #aBase = aBaseSquare.flatten()

    def inclusion_defectI(ii):
        aSquare = np.copy(aBaseSquare)
        tmp_indx = np.array([ii % Nepsilon[1], ii // Nepsilon[1]])
        startindexcols = int((tmp_indx[0] + incl_bl[0]) * (NFine / Nepsilon)[0])
        stopindexcols = int((tmp_indx[0] + incl_tr[0]) * (NFine / Nepsilon)[0])
        startindexrows = int((tmp_indx[1] + incl_bl[1]) * (NFine / Nepsilon)[1])
        stopindexrows = int((tmp_indx[1] + incl_tr[1]) * (NFine / Nepsilon)[1])
        aSquare[startindexrows:stopindexrows, startindexcols:stopindexcols] = bg
        return aSquare.flatten()

    coeffList = list(map(inclusion_defectI, range(np.prod(Nepsilon))))
    coeffList.append(aBaseSquare.flatten())

    return coeffList