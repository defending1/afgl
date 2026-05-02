from typing import Tuple, Union

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp

from afgl.util.g_function import compute_g_M


def arnoldi(
    A: Union[np.ndarray, sp.spmatrix], s: np.ndarray, M: int, g, eps_STOP
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes an orthonormal basis for the Krylov subspace using the Arnoldi iteration.

    Args:
        A: The input matrix (square), can be dense or sparse.
        s: The starting vector.
        M: The number of iterations/dimension of the Krylov subspace.

    Returns:
        V: Matrix whose columns form an orthonormal basis for the Krylov subspace.
        H: Upper Hessenberg matrix.
    """
    if sp.issparse(A):
        A = A.tocsr()
    v = s
    beta = LA.norm(v)
    v = v / beta
    H = np.zeros((M + 1, M))
    V = np.zeros((A.shape[0], M + 1))
    V[:, 0] = v

    j = 0
    for j in range(M):
        w1 = A @ V[:, j]

        # First orthogonalization pass
        h1 = V[:, : j + 1].T @ w1
        w1 = w1 - V[:, : j + 1] @ h1
        norm_w1 = LA.norm(w1)

        # Temporary normalization
        v_tmp = w1 / norm_w1

        h2 = V[:, : j + 1].T @ v_tmp
        w2 = v_tmp - V[:, : j + 1] @ h2
        norm_w2 = LA.norm(w2)

        H[: j + 1, j] = h1 + h2
        H[j + 1, j] = norm_w1 * norm_w2

        if H[j + 1, j] == 0:
            print("Arnoldi breakdown")
            break
        else:
            if j < M - 1:
                if j > 2:
                    g_j3 = compute_g_M(
                        V[:, : (j + 1) - 3], H[: (j + 1) - 3, : (j + 1) - 3], s, g
                    )
                    g_j = compute_g_M(V[:, : j + 1], H[: j + 1, : j + 1], s, g)
                    r_j = LA.norm(g_j - g_j3)
                    if r_j < eps_STOP:
                        return V[:, : j + 1], H[: j + 1, : j + 1], j

                V[:, j + 1] = w2 / norm_w2

    epsilon = np.finfo(H.dtype).eps
    H[np.abs(H) < epsilon] = 0

    return V[:, : j + 1], H[: j + 1, : j + 1], j
