import numpy as np
from gridlod import util

def build_randomcheckerboard(Nepsilon, NFine, alpha, beta):
    # builds a random checkerboard coefficient with spectral bounds alpha and beta,
    # piece-wise constant on mesh with Nepsilon blocks
    # returns a fine coefficient on mesh with NFine blocks
    Ntepsilon = np.prod(Nepsilon)
    values = alpha + (beta-alpha) * np.random.randint(0,2,Ntepsilon)

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

    checkerboardbasis = [alpha*np.ones(NtFine)]

    for ii in range(Ntepsilon):
        coeff = alpha * np.ones(NtFine)
        #find out which indices on fine grid correspond to element ii on epsilon grid
        elementIndex = util.convertpLinearIndexToCoordIndex(Nepsilon-1, ii)[:]
        indices = util.extractElementFine(Nepsilon, NFineElement//NepsilonElement, elementIndex)
        coeff[indices] = beta
        checkerboardbasis.append(coeff)

    return checkerboardbasis