from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp

from afgl.util.g_function import compute_g_M
from afgl.util.T_tridiag import T_tridiag


def full_orthogonalization(
    V: np.ndarray, w: np.ndarray, j: int, gs_iterations=2
) -> np.ndarray:
    """
    Performs full re-orthogonalization using the double Gram-Schmidt process.

    Args:
        V: Matrix of basis vectors.
        w: Vector to orthogonalize.
        j: Current iteration index.
        gs_iterations: Number of grahm-schmidt iterations (2 for safety)

    Returns:
        Orthogonalized vector w.
    """
    for _ in range(gs_iterations):
        # Remember that numpy slicing is not inclusive, thus mathematically we are
        # considering the submatrix V(:,0:j)
        w = w - V[:, : j + 1] @ (V[:, : j + 1].T @ w)

    return w


def classic_orthogonalization(
    V: np.ndarray, w: np.ndarray, alp: np.ndarray, beta: np.ndarray, j: int
) -> np.ndarray:
    """
    Performs classic Lanczos orthogonalization against the last two basis vectors.

    Args:
        V: Matrix of basis vectors.
        w: Vector to orthogonalize.
        alp: Array of diagonal elements alpha.
        beta: Array of off-diagonal elements beta.
        j: Current iteration index.

    Returns:
        Orthogonalized vector w.
    """
    w = w - alp[j] * V[:, j]
    if j > 0:
        w = w - beta[j - 1] * V[:, j - 1]
    return w


def lanczos_iteration(
    L: Union[np.ndarray, sp.spmatrix],
    s: np.ndarray,
    M: int,
    g: Optional[Any],
    full_ortho: bool,
    eps_STOP: Optional[float] = None,
    gs_iterations=2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    """
    Performs the Lanczos iteration process.

    Args:
        L: Symmetric input matrix (dense or sparse).
        s: Starting vector.
        M: Number of iterations.
        g: Function to evaluate for stopping criteria.
        full_ortho: Whether to use full re-orthogonalization.
        eps_STOP: Tolerance for the stopping criterion.

    Returns:
        V: Matrix of basis vectors.
        alp: Diagonal elements of the tridiagonal matrix.
        beta: Off-diagonal elements of the tridiagonal matrix.
        break_reason: String describing why the iteration stopped.
        j: Number of iterations completed.
    """
    if sp.issparse(L):
        L = L.tocsr()
    N = len(s)
    alp = np.zeros(M)
    beta = np.zeros(M - 1)
    V = np.zeros((N, M))
    V[:, 0] = s / LA.norm(s)

    break_reason = "No"

    j = 0
    for j in range(M):
        w = L @ V[:, j]
        alp[j] = np.dot(V[:, j], w)

        if full_ortho:
            w = full_orthogonalization(V, w, j, gs_iterations)
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

                    if r_j < eps_STOP:
                        break_reason = "Bound"
                        return V[:, : j + 1], alp[: j + 1], beta[:j], break_reason, j

                V[:, j + 1] = w / beta[j]

    return V, alp, beta, break_reason, j


def lanczos(
    L: Union[np.ndarray, sp.spmatrix],
    s: np.ndarray,
    M: int,
    g: Optional[Any] = None,
    eps_STOP: Optional[float] = None,
    full_orthogonalization=False,
    gs_iterations=2,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    r"""
    Lanczos extended method with optional re-orthogonalization.

    Args:
        L: Real valued NxN symmetric matrix (dense or sparse).
        s: Starting vector of size N.
        M: Target basis size.
        g: Optional function to evaluate for stopping.
        eps_STOP: Optional tolerance for stopping when ||g_{M+j} - g_{M}|| < eps_STOP.

    Returns:
        V: Matrix with orthonormal columns forming the basis.
        T: Tridiagonal matrix.
        info: List containing [full_ortho_used, break_reason, iterations_completed].
    """
    full_ortho_triggered = False
    EPS_ORTHO = 10e-10

    if not full_orthogonalization:
        V, alp, beta, break_reason, j = lanczos_iteration(
            L, s, M, g, False, eps_STOP, gs_iterations
        )
        # Check basis orthogonality
        Id = np.eye(len(alp))

        if LA.norm(V.T @ V - Id) > EPS_ORTHO:
            V, alp, beta, break_reason, j = lanczos_iteration(
                L, s, M, g, True, eps_STOP, gs_iterations
            )
            full_ortho_triggered = True
    else:
        V, alp, beta, break_reason, j = lanczos_iteration(
            L, s, M, g, True, eps_STOP, gs_iterations
        )
        full_ortho_triggered = True

    T = T_tridiag(alp, beta)
    return V, T, [full_ortho_triggered, break_reason, j]
