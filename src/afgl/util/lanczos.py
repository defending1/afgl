import numpy as np
import numpy.linalg as LA


def double_orthogonalization(V, w, j):
    # Why it works well until j+1? it should be until j-1 from Demmel
    for _ in range(2):
        w -= V[:, : j + 1] @ (V[:, : j + 1].T @ w)
    return w


def no_orthogonalization(V, w, alp, beta, j):
    w = w - alp[j] * V[:, j]
    if j > 0:
        w = w - beta[j - 1] * V[:, j - 1]
    return w


def lanczos(L, s, M):
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
    N = len(s)
    alp = np.zeros(M)
    beta = np.zeros(M - 1)
    V = np.zeros((N, M))
    V[:, 0] = s / LA.norm(s)

    for j in range(M):
        w = L @ V[:, j]
        alp[j] = np.dot(V[:, j], w)

        w = no_orthogonalization(V, w, alp, beta, j)

        if j < M - 1:
            beta[j] = LA.norm(w)
            if beta[j] < 1e-14:
                print("BREAKDOWN")
                return V[:, : j + 1], alp[: j + 1], beta[:j]
            V[:, j + 1] = w / beta[j]

    return V, alp, beta
