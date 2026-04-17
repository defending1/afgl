import numpy as np
import numpy.linalg as LA
from afgl.util.build_T_matrix import build_T_matrix


def full_orthogonalization(V, w, j):
    # Why it works well until j+1? it should be until j-1 from Demmel
    for _ in range(2):
        w -= V[:, : j + 1] @ (V[:, : j + 1].T @ w)
    return w


def no_orthogonalization(V, w, alp, beta, j):
    w = w - alp[j] * V[:, j]
    if j > 0:
        w = w - beta[j - 1] * V[:, j - 1]
    return w


def FOM_reminder(alp, beta, j, s):
    e_1 = np.zeros(j)
    e_1[0] = 1

    T_j = build_T_matrix(alp[:j], beta[: j - 1])
    y_j = LA.solve(T_j, LA.norm(s) * e_1)

    return abs(beta[j - 1]) * abs(y_j[-1])


def lanczos_iteration(L, s, M: int, full_ortho: bool, threshold: float):
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

    early_stop = "No"

    for j in range(M):
        w = L @ V[:, j]
        alp[j] = np.dot(V[:, j], w)

        if full_ortho:
            w = full_orthogonalization(V, w, j)
        else:
            w = no_orthogonalization(V, w, alp, beta, j)

        if j < M - 1:
            beta[j] = LA.norm(w)

            if beta[j] < 1e-14:
                early_stop = f"$\\beta[{j}]=0$"
                return V[:, : j + 1], alp[: j + 1], beta[:j], early_stop
            else:
                r_j_norm = -1
                if j > 2:
                    r_j_norm = FOM_reminder(alp, beta, j + 1, s)
                if r_j_norm < threshold and r_j_norm > 0:
                    early_stop = (
                        f"$r_j^{{(FOM)}}  < \\varepsilon \\text{{ at step j}} = {j}$"
                    )
                    return V[:, : j + 1], alp[: j + 1], beta[:j], early_stop
                else:
                    V[:, j + 1] = w / beta[j]

    return V, alp, beta, early_stop


def lanczos(L, s, M, threshold=10e-33):
    full_ortho = False

    ORTHO_EPS = 10e-2
    V, alp, beta, early_stop = lanczos_iteration(L, s, M, False, threshold)

    # Check basis orthogonality
    Id = np.eye(len(alp))
    if LA.norm(V.T @ V - Id) > ORTHO_EPS:
        V, alp, beta, early_stop = lanczos_iteration(L, s, M, True, threshold)
        full_ortho = True

    return V, alp, beta, full_ortho, early_stop
