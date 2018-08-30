import numpy as np
import scipy.sparse as sparse

import util
import fem


def localMatrix(d, matrixValuesTakesBinaryIndices):
    def convertToBinaryIndices(f):
        return lambda *ind: f(np.array(ind[:d], dtype='bool'),
                              np.array(ind[d:], dtype='bool'))

    ABin = np.fromfunction(convertToBinaryIndices(matrixValuesTakesBinaryIndices), shape=[2] * (2 * d))
    AFlat = ABin.flatten('F')
    A = AFlat.reshape(2 ** d, 2 ** d, order='F')
    return A

#TODO: check correct scaling!!!!!
def localMassMatrix(N):
    d = np.size(N)
    assert(d == 3)
    Myz = fem.localMassMatrix(N[1:])
    Mxy = fem.localMassMatrix(N[:2])
    Mxz = fem.localMassMatrix([N[0], N[2]])

    return np.block([[(N[0]**2)*Myz, np.zeros(2**(d-1), 2**(d-1)), np.zeros(2**(d-1), 2**(d-1))],
                     [np.zeros(2**(d-1), 2**(d-1)),(N[1]**2)*Mxz, np.zeros(2**(d-1), 2**(d-1))],
                     [np.zeros(2**(d-1), 2**(d-1)), np.zeros(2**(d-1), 2**(d-1)), (N[2]**2)*Mxy]])


#TODO: check, add scaling
def localStiffnessMatrix(N):
    d = np.size(N)
    assert(d == 3)

    stiffnessy = fem.localStiffnessMatrix(N[1])
    stiffnessz = fem.localStiffnessMatrix(N[2])

    stiffnessxy = fem.localStiffnessMatrix(N[:2])
    stiffnessyz = fem.localStiffnessMatrix(N[1:])
    stiffnessxz = fem.localStiffnessMatrix([N[0], N[2]])

    massx = fem.localMassMatrix(N[0])
    massy = fem.localMassMatrix(N[1])
    massz = fem.localMassMatrix(N[2])

    return np.block([[stiffnessyz, -1./N[0]*np.tensordot(massy, stiffnessz, axes=0),
                        -1./N[0]*np.tensordot(massz, stiffnessy, axes=0)],
                     [-1./N[0]*np.tensordot(massy, stiffnessz, axes=0), stiffnessxz,
                        -1./N[1]*np.tensordot(massx, stiffnessz, axes=0)],
                     [-1. / N[0] * np.tensordot(massz, stiffnessy, axes=0),
                        -1./N[1]*np.tensordot(massx, stiffnessz, axes=0), stiffnessxy]])


#TODO: implement!
def localBoundaryMassMatrix(N, k=0, neg=False):
    d = np.size(N)
    notk = np.ones_like(N, dtype='bool')
    notk[k] = False
    detJk = np.prod(1. / N[notk])

    return

#TODO: does localToPatchSparsityPattern need modification?

#TODO: localBasis

#TODO: assembleProlongation Matrix!