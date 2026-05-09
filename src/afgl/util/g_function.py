import numpy as np
import numpy.linalg as LA
import scipy.linalg as SCLA
from pygsp import filters


def compute_g_itersine(G, Nf=7):
    """Construct an itersine filter bank for a graph.

    Args:
        G: Graph.
        Nf: Number of filters in the bank.

    Returns:
        A :class:`pygsp.filters.Itersine` filter bank instance.
    """
    G.estimate_lmax()
    g = filters.Itersine(G, Nf=Nf)
    return g


def compute_g_M(V: np.ndarray, T, s: np.ndarray, g, ch=0) -> np.ndarray:
    """
    Compute the projected approximation ``g_M`` (see [1]) via Lanczos.

    Args:
        V: Lanczos basis
        T: Tridiagonal matrix
        s: signal
        g: Itersine
        ch: Itersine selected channel

    Returns:
        The vector ``g_M`` approximating ``g(L)s``.

    Notes:
        ``T`` can be passed either as a symmetric banded representation with
        shape ``(2, n)`` (as returned by :func:`afgl.util.T_tridiag.T_tridiag`)
        or as a full square matrix.

    """
    T_arr = np.asarray(T)
    if T_arr.ndim == 2 and T_arr.shape[0] == 2:
        # Uses divide and conquer.
        alp = T_arr[0]
        n = len(alp)
        beta = T_arr[1, : n - 1]

        eigvals, U = SCLA.eigh_tridiagonal(alp, beta, lapack_driver="stevd")
    elif T_arr.ndim == 2 and T_arr.shape[0] == T_arr.shape[1]:
        if SCLA.issymmetric(T):
            eigvals, U = SCLA.eigh(T_arr)
        else:
            # Use Schur-Parlett for the general case
            g_ch = lambda x: g.evaluate(x)[ch]  # noqa: E731
            gT = SCLA.funm(T, g_ch)
            e_1 = np.zeros(T.shape[0])
            e_1[0] = 1
            y = LA.norm(s) * gT @ e_1
            return V @ y
    else:
        raise ValueError(
            "T must be either a 2xN tridiagonal representation or a square matrix."
        )
    g_lambda = g.evaluate(eigvals)[ch]
    u_1 = U.T[:, 0]
    # Computing Ug(eigvals)U*e_1
    y = LA.norm(s) * U @ (g_lambda * u_1)
    return V @ y


def filter_signal_with_fourier(G, s: np.ndarray, g, ch=0) -> np.ndarray:
    """Evaluate ``g(L)s`` using the full Fourier basis.

    Args:
        G: Graph
        s: Signal
        g: Itersine
        ch: Itersine selected channel

    Returns:
        The filtered signal ``g(L)s``.
    """
    G.compute_fourier_basis()
    U = G.U
    ev = g.evaluate(G.e)
    g_e = ev[ch]
    return (U @ np.diag(g_e) @ U.T) @ s
