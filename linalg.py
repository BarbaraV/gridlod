import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

import sys
import time

def saddle(A, B, rhsList):
    ''' saddle

    Solve (A B'; B 0) = rhs
    '''
    xList = []

    # Pre-processing (page 12, Engwer, Henning, Malqvist)
    # Rename for brevity
    from scikits.sparse.cholmod import cholesky
            
    # Compute Y
    print "A"
    sys.stdout.flush()
    
    cholA = cholesky(A)
    Y = np.zeros((B.shape[1], B.shape[0]))
    for y, b in zip(Y.T, B):
        print ".",
        sys.stdout.flush()
        
        y[:] = cholA.solve_A(np.array(b.todense()).T.squeeze())

    print "B"
    sys.stdout.flush()
        
    S = B*Y
    invS = np.linalg.inv(S)

    print "C"
    sys.stdout.flush()

    # Post-processing
    for rhs in rhsList:
        q   = cholA.solve_A(rhs)
        lam = np.dot(invS, B*q)
        x   = q - np.dot(Y,lam)
        xList.append(x)

    print "D"
    sys.stdout.flush()
        
    return xList

class FailedToConverge(Exception):
    pass

def saddleNullSpaceHierarchicalBasis(A, B, P, rhsList, coarseNodes):
    '''Solve ( S'*A*S  S'*B' ) ( y  )   ( S'b )
             (    B*S  0     ) ( mu ) = ( 0   )

    and compute x = S*y where

        ( |  0 )
    S = ( P    ) 
        ( |  I )

    if the nodes are reordered so that coarseNodes comes first.
    '''
    Np = A.shape[0]
    Nc = np.size(coarseNodes)

    coarseNodesMask = np.zeros(Np, dtype='bool')
    coarseNodesMask[coarseNodes] = True
    notCoarseNodesMask = np.logical_not(coarseNodesMask)
    notCoarseNodes = np.where(notCoarseNodesMask)[0]
    nodePermutation = np.hstack([coarseNodes, notCoarseNodes])
    
    PSub = P[notCoarseNodes,:]
    I1 = sparse.identity(Nc, format='csc')
    I2 = sparse.identity(Np-Nc, format='csc')
    S = sparse.bmat([[I1,   None],
                     [PSub, I2]], format='csc')

    Bn = B[:,notCoarseNodesMask]
    Z = sparse.bmat([[-Bn],
                     [I2]], format='csc')

    APerm = A[nodePermutation][:,nodePermutation]
    #Z = sparse.coo_matrix((Np,Np-Nc))
    #Z[notCoarseNodesMask,:] = sparse.identity(Np-Nc, format='coo')
    #Z[coarseNodesMask,:] = -B[:,notCoarseNodesMask]
    #Z = Z.tocsc()
    
    class mutableClosure:
        timer = 0
        counter = 0
        
    def Ax(x):
        start = time.time()
        y = Z.T*(S.T*(APerm*(S*(Z*x))))
        end = time.time()
        mutableClosure.timer += end-start
        mutableClosure.counter += 1
        return  y

    ALinearOperator = sparse.linalg.LinearOperator(dtype='float64', shape=(Np-Nc, Np-Nc), matvec=Ax)
    
    correctorList = []
    for rhs in rhsList:
        #print '.',
        b = Z.T*(S.T*rhs[nodePermutation])

        mutableClosure.counter = 0
        mutableClosure.timer = 0
        xPerm,info = sparse.linalg.cg(ALinearOperator, b, tol=1e-9)
        print mutableClosure.counter, mutableClosure.timer
        
        if info != 0:
            raise(FailedToConverge('CG failed to converge, info={}'.format(info)))

        totalDofs = A.shape[0]
        corrector = np.zeros(Np)
        corrector[nodePermutation] = S*(Z*xPerm)
        correctorList.append(corrector)
        
    return correctorList
    
    # We have that (B*S)[coarseNodes,coarseNodes] is identity.  A
    # "nice basis" is found for all projections.
    
def saddleNullSpace(A, B, rhsList, coarseNodes):
    '''lodSaddle

    Solve (A B'; B 0) = rhs

    Use a null space method. We assume the columns of B can be
    permuted with P so that
    
    BP = [D Bn]

    where D is diagonal. This is the case when the interpolation
    operator includes no other coarse node than its own in its nodal
    variable definition.

    It is also possible (see e.g. M. Benzi, G. H. Golub and J. Liesen)
    to make a permutation

    BP = [Bb Bn]

    where Bb is invertible. However, this requires to compute
    Bb^(-1)*Bn...

    '''
    
    ## Find cols with one non-zero only
    Bcsr = B.tocsr()

    # Eliminate zero or almost-zero rows
    Bcsr.data[np.abs(Bcsr.data)<1e-12] = 0
    Bcsr.eliminate_zeros()
    Bcsr  = Bcsr[np.diff(Bcsr.indptr) != 0,:]

    Bcsc = Bcsr.tocsc()
    
    coarseNodesMask = np.zeros(Bcsc.shape[1], dtype='bool')
    coarseNodesMask[coarseNodes] = True
    notCoarseNodesMask = np.logical_not(coarseNodesMask)
    Bb = Bcsc[:,coarseNodesMask]
    Bn = Bcsc[:,notCoarseNodesMask]

    def diagonalCsc(A):
        n = A.shape[0]
        if A.shape[1] != n:
            return None
        if np.all(A.indptr != np.arange(n+1)):
            return None
        if np.all(A.indices != np.arange(n)):
            return None
        else:
            return sparse.dia_matrix((A.data, 0), shape=(n,n))
    
    Bbdiag = diagonalCsc(Bb)
    if Bbdiag is None:
        raise(NotImplementedError('Can''t handle general interpolation ' +
                                  'operators. Needs to be easy to find its null space...'))
        
    BbInv = Bbdiag.copy()
    BbInv.data = 1./BbInv.data

    Btildecsc = -BbInv*Bn
    Btildecsc.sort_indices()
    
    # For faster MV-multiplication
    Btilde = Btildecsc.tocsr()
    Btilde.sort_indices()
    
    A11 = A[coarseNodesMask][:,coarseNodesMask].tocsr()
    A11.sort_indices()
    A12 = A[coarseNodesMask][:,notCoarseNodesMask].tocsr()
    A12.sort_indices()
    A22 = A[notCoarseNodesMask][:,notCoarseNodesMask].tocsr()
    A22.sort_indices()
    A21 = A12.T.tocsr()
    A21.sort_indices()

    class mutableClosure:
        timer = 0
        counter = 0
        
    def Ax(x):
        start = time.time()
        y = A21*(Btilde*x) + A22*x + Btildecsc.T*(A11*(Btilde*x)) + Btildecsc.T*(A12*x)
        end = time.time()
        mutableClosure.timer += end-start
        mutableClosure.counter += 1
        return  y

    ALinearOperator = sparse.linalg.LinearOperator(dtype='float64', shape=A22.shape, matvec=Ax)
    
    correctorList = []
    for rhs in rhsList:
        #print '.',
        b = rhs[notCoarseNodesMask] + Btildecsc.T*rhs[coarseNodesMask]

        mutableClosure.counter = 0
        mutableClosure.timer = 0
        x,info = sparse.linalg.cg(ALinearOperator, b, tol=1e-9)
        print mutableClosure.counter, mutableClosure.timer
        
        if info != 0:
            raise(FailedToConverge('CG failed to converge, info={}'.format(info)))

        totalDofs = A.shape[0]
        corrector = np.zeros(totalDofs)
        corrector[notCoarseNodesMask] = x
        corrector[coarseNodesMask] = Btilde*x
        correctorList.append(corrector)
        
    return correctorList


from scikits.sparse.cholmod import cholesky
            
def solveWithBlockDiagonalPreconditioner(A, B, bList):
    """Solve saddle point problem with block diagonal preconditioner

    / A  B.T \   / r \   / b \
    |        | * |   | = |   |
    \ B   0  /   \ s /   \ 0 /

    Section 10.1.1 in "Numerical solution of saddle point problems",
    Benzi, Golub and Liesen.
    """

    n = np.size(A,0)
    m = np.size(B,0)

    cholA = cholesky(A)
    S = np.zeros((B.shape[0], B.shape[0]))
    for s, Brow in zip(S.T, B):
        y = cholA.solve_A(np.array(Brow.todense()).T.squeeze())
        s[:] = -B*y
        
    SInv = np.linalg.inv(S)

    def solveP(x):
        r = x[:n]
        s = x[-m:]
        rSol = cholA.solve_A(r)
        sSol = -np.dot(SInv, s)
        return np.hstack([rSol, sSol])

    M = sparse.linalg.LinearOperator((n+m,n+m), solveP)

    K = sparse.bmat([[A, B.T],
                     [B, None]], format='csc');
    c = np.zeros(n+m)

    numIter = [0]
    def cgCallback(x):
        numIter[0] +=  1

    rList = []
    xList = []
    infoList = []
    numIterList = []
    for b in bList:
        numIter = [0]
        c[:n] = b
        x, info = sparse.linalg.cg(K, c, tol=1e-9, M=M, callback=cgCallback)
        r = x[:n]
        rList.append(r)
        xList.append(x)
        infoList.append(info)
        numIterList.append(numIter[0])

    return rList

def schurComplementSolve(A, B, bList):
    correctorFreeList = []

    # Pre-processing (page 12, Engwer, Henning, Malqvist)
    # Rename for brevity
            
    # Compute Y
    #luA = sparse.linalg.splu(A)
    #luA_approxpprox = sparse.linalg.spilu(A)
    cholA = cholesky(A)
    Y = np.zeros((B.shape[1], B.shape[0]))
    for y, c in zip(Y.T, B):
        #y[:] = luA.solve(np.array(c.todense()).T.squeeze())
        #y[:] = luA_approx.solve(np.array(c.todense()).T.squeeze())
        y[:] = cholA.solve_A(np.array(c.todense()).T.squeeze())
                
    S = B*Y
    invS = np.linalg.inv(S)

    # Post-processing
    for b in bList:
        r = b
        #q = luA.solve(r)
        #q = luA_approx.solve(r)
        q = cholA.solve_A(r)
        lam = np.dot(invS, B*q)
        correctorFree = q - np.dot(Y,lam)
        correctorFreeList.append(correctorFree)
    return correctorFreeList
