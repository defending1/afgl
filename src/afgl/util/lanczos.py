from typing import Any, List, Optional, Tuple, Union

import numpy as np
import numpy.linalg as LA
import scipy.sparse as sp

from afgl.util.g_function import compute_g_M
from afgl.util.T_tridiag import T_tridiag


def full_orthogonalization(
    V: np.ndarray, w: np.ndarray, j: int, gs_iterations=2
) -> np.ndarray:
    """Re-orthogonalize a vector against the full current basis.

    Uses (by default) a double Gram-Schmidt pass for numerical stability.

    Args:
        V: Basis matrix whose columns are the current Krylov vectors.
        w: Vector to orthogonalize.
        j: Current iteration index (orthogonalize against ``V[:, : j+1]``).
        gs_iterations: Number of Gram-Schmidt passes (2 is a common safe choice).

    Returns:
        The orthogonalized vector ``w``.
    """
    for _ in range(gs_iterations):
        # Remember that numpy slicing is not inclusive, thus mathematically we are
        # considering the submatrix V(:,0:j)
        w = w - V[:, : j + 1] @ (V[:, : j + 1].T @ w)

    return w


def classic_orthogonalization(
    V: np.ndarray, w: np.ndarray, alp: np.ndarray, beta: np.ndarray, j: int
) -> np.ndarray:
    """Orthogonalize against the last one or two Lanczos vectors.

    This is the classic three-term Lanczos recurrence (cheap but less stable than
    full re-orthogonalization when orthogonality is lost).

    Args:
        V: Basis matrix.
        w: Vector to orthogonalize.
        alp: Diagonal coefficients (alpha).
        beta: Off-diagonal coefficients (beta).
        j: Current iteration index.

    Returns:
        The orthogonalized vector ``w``.
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
    tau: Optional[float] = None,
    gs_iterations=2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]:
    """Run the core Lanczos iteration.

    Optionally uses a simple stopping criterion based on the bound
    ``||g_{j} - g_{j-3}|| < tau`` (computed via the projected problem).

    Args:
        L: Symmetric input matrix (dense or sparse).
        s: Starting vector.
        M: Maximum number of Lanczos steps.
        g: Optional filter/function used for the stopping criterion.
        full_ortho: If True, use full re-orthogonalization.
        tau: Optional tolerance for early stopping.
        gs_iterations: Gram-Schmidt passes for full re-orthogonalization.

    Returns:
        V: Orthonormal basis vectors (columns).
        alp: Diagonal of the projected tridiagonal matrix.
        beta: Off-diagonal of the projected tridiagonal matrix.
        break_reason: Why the loop stopped ("No", "$\\beta_j=0$", or "Bound").
        j: Last completed iteration index.
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
                if tau is not None and g is not None and (j + 1) - 3 > 0:
                    T = T_tridiag(alp[: j + 1], beta[:j])
                    g_j3 = compute_g_M(
                        V[:, : (j + 1) - 3],
                        T[:, : (j + 1) - 3],
                        s,
                        g,
                    )
                    g_j = compute_g_M(V[:, : j + 1], T[:, : j + 1], s, g)
                    r_j = LA.norm(g_j - g_j3)

                    if r_j < tau:
                        break_reason = "Bound"
                        return V[:, : j + 1], alp[: j + 1], beta[:j], break_reason, j

                V[:, j + 1] = w / beta[j]

    return V, alp, beta, break_reason, j


def lanczos(
    L: Union[np.ndarray, sp.spmatrix],
    s: np.ndarray,
    M: int,
    g: Optional[Any] = None,
    tau: Optional[float] = None,
    ortho_type: Optional[str] = "auto",
    gs_iterations: Optional[int] = 2,
) -> Tuple[np.ndarray, np.ndarray, List[Any]]:
    r"""
    Compute a Lanczos basis (with optional re-orthogonalization).

    Args:
        L: Real valued NxN symmetric matrix (dense or sparse).
        s: Starting vector of size N.
        M: Target basis size.
        g: Optional function to evaluate for stopping.
        tau: Optional tolerance for early stopping.
            The stopping check uses a lag-3 bound: ``||g_j - g_{j-3}|| < tau``.

    Returns:
        V: Matrix with orthonormal columns forming the basis.
        T: Tridiagonal matrix.
        info: List containing [full_ortho_used, break_reason, iterations_completed].
    """
    full_ortho_triggered = False
    delta = 1e-9

    if ortho_type == "auto":
        V, alp, beta, break_reason, j = lanczos_iteration(
            L, s, M, g, False, tau, gs_iterations
        )
        # Check basis orthogonality
        Id = np.eye(len(alp))

        if LA.norm(V.T @ V - Id) > delta:
            V, alp, beta, break_reason, j = lanczos_iteration(
                L, s, M, g, True, tau, gs_iterations
            )
            full_ortho_triggered = True
    elif ortho_type == "full":
        V, alp, beta, break_reason, j = lanczos_iteration(
            L, s, M, g, True, tau, gs_iterations
        )
        full_ortho_triggered = True
    elif ortho_type == "partial":
        V, alp, beta, break_reason, j = lanczos_iteration(
            L, s, M, g, False, tau, gs_iterations
        )
        full_ortho_triggered = False
    else:
        raise ValueError("Orthogonalization type not supported")

    T = T_tridiag(alp, beta)
    return V, T, [full_ortho_triggered, break_reason, j]
