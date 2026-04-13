import numpy as np
import numpy.linalg as LA

"""
Classic Lanczos method (without re-orthogonalization)

Arguments
L : Real valued NxN symmetric matrix
s : vector of size N
M : natural number indicating basis size

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

        w = w - alp[j] * V[:, j]
        if j > 0:
            w = w - beta[j - 1] * V[:, j - 1]

        if j < M - 1:
            beta[j] = LA.norm(w)
            if beta[j] == 0:
                break
            V[:, j + 1] = w / beta[j]

    return [V, alp, beta]
