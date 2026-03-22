import numpy as np
import numpy.linalg as LA

"""
Arguments
L Real valued NxN symmetric matrix
s vector of size N
M natural number indicating basis size

Returns
-------
V : ndarray
    M-dimensional vector with orthonormal columns.
alp : ndarray
    M-dimensional array of scalars.
beta : ndarray
    M-dimensional array of scalars.
"""


def lanczos(L, s, M):
    N = len(s)
    alp = np.zeros(M)
    beta = np.zeros(M - 1)
    V = np.zeros((N, M))
    V[:, 0] = s / LA.norm(s)

    for j in range(M):
        w = L @ V[:, j]
        alp[j] = np.dot(V[:, j], w)

        v_tilde = w - V[:, j] * alp[j]
        if j > 0:
            v_tilde = v_tilde - V[:, j - 1] * beta[j - 1]

        if j < M - 1:
            beta[j] = LA.norm(v_tilde)
            V[:, j + 1] = v_tilde / beta[j]

    return [V, alp, beta]
