import numpy as np
import numpy.linalg as LA

from afgl.util.g_function import compute_g_M
from afgl.util.T_tridiag import T_tridiag


def full_orthogonalization(V, w, j):
    for _ in range(2):
        # Remember that numpy slicing is not inclusive, thus mathematically we are
        # considering the submatrix V(:,0:j)
        w = w - V[:, : j + 1] @ (V[:, : j + 1].T @ w)

    return w


def classic_orthogonalization(V, w, alp, beta, j):
    w = w - alp[j] * V[:, j]
    if j > 0:
        w = w - beta[j - 1] * V[:, j - 1]
    return w


def lanczos_iteration(L, s, M, g, full_ortho, eps_STOP=None):
    N = len(s)
    alp = np.zeros(M)
    beta = np.zeros(M - 1)
    V = np.zeros((N, M))
    V[:, 0] = s / LA.norm(s)

    break_reason = "No"

    for j in range(M):
        w = L @ V[:, j]
        alp[j] = np.dot(V[:, j], w)

        if full_ortho:
            w = full_orthogonalization(V, w, j)
        else:
            w = classic_orthogonalization(V, w, alp, beta, j)

        if j < M - 1:
            beta[j] = LA.norm(w)

            if beta[j] < 1e-14:
                break_reason = "$\\beta_j=0$"
                return V[:, : j + 1], alp[: j + 1], beta[:j], break_reason, j
            else:
                if eps_STOP is not None and g is not None and (j + 1) - 3 > 0:
                    T = T_tridiag(alp[: j + 1], beta[:j])
                    g_j3 = compute_g_M(
                        V[:, : (j + 1) - 3], T[: (j + 1) - 3, : (j + 1) - 3], s, g
                    )
                    g_j = compute_g_M(V[:, : j + 1], T[: j + 1, : j + 1], s, g)
                    r_j = LA.norm(g_j - g_j3)

                    if r_j < eps_STOP * LA.norm(s):
                        break_reason = "Bound"
                        return V[:, : j + 1], alp[: j + 1], beta[:j], break_reason, j

                V[:, j + 1] = w / beta[j]

    return V, alp, beta, break_reason, j


def Arnoldi(A, s, m):
    v = s
    beta = LA.norm(v)
    v = v / beta
    H = np.zeros((m + 1, m))
    V = np.zeros((A.shape[0], m + 1))
    V[:, 0] = v

    for j in range(m):
        w1 = A @ V[:, j]

        # First orthogonalization pass
        h1 = V[:, : j + 1].T @ w1
        w1 = w1 - V[:, : j + 1] @ h1
        norm_w1 = LA.norm(w1)

        # Temporary normalization
        v_tmp = w1 / norm_w1

        # Second orthogonalization pass
        h2 = V[:, : j + 1].T @ v_tmp
        w2 = v_tmp - V[:, : j + 1] @ h2
        norm_w2 = LA.norm(w2)

        # Matrix H coefficients
        H[: j + 1, j] = h1 + h2
        H[j + 1, j] = norm_w1 * norm_w2

        if H[j + 1, j] == 0:
            print("Arnoldi breakdown")
            m = j
            break
        else:
            if j < m - 1:
                V[:, j + 1] = w2 / norm_w2

    epsilon = np.finfo(H.dtype).eps
    print(epsilon)
    H[np.abs(H) < epsilon] = 0

    return V[:, : j + 1], H[: j + 1, : j + 1]


def lanczos(L, s, M, g=None, eps_STOP=None):
    """
    Lanczos extended method that calls re-orthogonalization when || V^T*V - I||
    < EPS_ORTHO.

    Arguments
    L : Real valued NxN symmetric matrix
    s : vector of size N
    M : natural number indicating basis size
    g : itersine function to evaluate for stopping
    eps_STOP: When passed float value, Lanczos method stops when \|g_{M+j} -
    g_{M}\| < eps_STOP.

    Returns
    -------
    V : ndarray
        M-dimensional vector with orthonormal columns.
    alp : ndarray
        M-dimensional array of scalars.
    beta : ndarray
        M-dimensional array of scalars.
    """
    full_ortho = False

    EPS_ORTHO = 10e-10
    V, alp, beta, break_reason, j = lanczos_iteration(L, s, M, g, False, eps_STOP)

    # Check basis orthogonality
    Id = np.eye(len(alp))
    if LA.norm(V.T @ V - Id) > EPS_ORTHO:
        V, alp, beta, break_reason, j = lanczos_iteration(L, s, M, g, True, eps_STOP)
        full_ortho = True

    T = T_tridiag(alp, beta)
    return V, T, [full_ortho, break_reason, j]
