import numpy as np
from numpy.linalg import eigvals, matrix_rank

def is_pos_def(M):
    return np.allclose(M, M.T) and np.all(eigvals(M) > 1e-12)

def stabilisable(A, B, tol=1e-9):
    """Kalman stabilisability test for discrete-time pair (A,B)."""
    n = A.shape[0]
    eigs = eigvals(A)
    for lam in eigs:
        if abs(lam) >= 1:                  # unstable or marginal
            # rank( [lam*I-A  B] ) == n ?
            test = np.block([[lam*np.eye(n)-A, B]])
            if matrix_rank(test, tol) < n:
                return False
    return True

def detectable(A, C, tol=1e-9):
    """ Detectability of (A,C) ↔ stabilisability of (Aᵀ,Cᵀ). """
    return stabilisable(A.T, C.T, tol)