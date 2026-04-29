import numpy as np
import numpy.linalg as LA

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


def FOM_reminder(alp, beta, j, s):
    """Computes r_j^{(FOM)} = |b_j||y_j^T e_j|"""
    e_1 = np.zeros(j)
    e_1[0] = 1

    T_j = T_tridiag(alp[:j], beta[: j - 1])
    y_j = LA.solve(T_j, LA.norm(s) * e_1)

    return abs(beta[j - 1]) * abs(y_j[-1])


def lanczos_iteration(L, s, M, full_ortho, eps_FOM=None):
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
            w = classic_orthogonalization(V, w, alp, beta, j)

        if j < M - 1:
            beta[j] = LA.norm(w)

            if beta[j] < 1e-14:
                early_stop = f"$\\beta[{j}]=0$"
                return V[:, : j + 1], alp[: j + 1], beta[:j], early_stop
            else:
                if eps_FOM is not None:
                    r_j_FOM_norm2 = FOM_reminder(alp, beta, j + 1, s)
                    if r_j_FOM_norm2 < eps_FOM * LA.norm(s):
                        early_stop = (
                            f"$\\|r_j^{{FOM}}\\|_2  < \\varepsilon \\|s\\|_2 \n"
                            f"\\text{{ at iteration }} = {j+1}$"
                        )
                        return V[:, : j + 1], alp[: j + 1], beta[:j], early_stop

                V[:, j + 1] = w / beta[j]

    return V, alp, beta, early_stop


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


def lanczos(L, s, M, eps_FOM=None):
    """
    Lanczos extended method that calls re-orthogonalization when || V^T*V - I||
    < EPS_ORTHO.

    Arguments
    L : Real valued NxN symmetric matrix
    s : vector of size N
    M : natural number indicating basis size
    eps_FOM: When passed float value, Lanczos method stops using FOM method,
    that is when r_j_FOM_norm2 < eps_FOM.

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
    V, alp, beta, early_stop = lanczos_iteration(L, s, M, False, eps_FOM)

    # Check basis orthogonality
    Id = np.eye(len(alp))
    if LA.norm(V.T @ V - Id) > EPS_ORTHO:
        V, alp, beta, early_stop = lanczos_iteration(L, s, M, True, eps_FOM)
        full_ortho = True

    T = T_tridiag(alp, beta)
    return V, T, [full_ortho, early_stop]
