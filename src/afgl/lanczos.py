import numpy as np

"""
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
    alp = np.zeros(M)
    beta = np.zeros(M)
    V = np.zeros(M)
    V[0] = s / np.norm(s)

    for j in range(0, M):
        w = L * V
        alp[j] = V[j] @ w
        Vtmp = w - V[j] * alp[j]
        if j > 1:
            Vtmp = Vtmp - V[j - 1] * beta[j - 1]
        beta[j] = np.norm(Vtmp)
        V[j + 1] = Vtmp / beta[j]
