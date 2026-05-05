import numpy as np
import numpy.linalg as LA
import scipy.linalg as SCLA
from pygsp import filters


def compute_g_itersine(G, Nf=7):
    G.estimate_lmax()
    g = filters.Itersine(G, Nf=Nf)
    return g


def compute_g_M(V: np.ndarray, T, s: np.ndarray, g, ch=0) -> np.ndarray:
    """
    Computes the approximation g_M (see [1]) using Lanczos

    Args:
        V: Lanczos basis
        T: Tridiagonal matrix
        s: signal
        g: Itersine
        ch: Itersine selected channel

    """
    if isinstance(T, np.ndarray) and T.ndim == 2 and T.shape[0] == 2:
        eigvals, U = SCLA.eig_banded(T, lower=True)
        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        U = U[:, order]
    else:
        eigvals, U = LA.eigh(T)
    g_lambda = g.evaluate(eigvals)[ch]
    u_1 = U.T[:, 0]
    # Computing Ug(eigvals)U*e_1
    y = LA.norm(s) * U @ (g_lambda * u_1)
    return V @ y


def filter_signal_with_fourier(G, s: np.ndarray, g, ch=0) -> np.ndarray:
    """Returns evaluation g(L)=Ug(Λ)U*s, which is the filtered signal using the
    fourier basis.

    Args:
        G: Graph
        s: Signal
        g: Itersine
        ch: Itersine selected channel
    """
    G.compute_fourier_basis()
    U = G.U
    ev = g.evaluate(G.e)
    g_e = ev[ch]
    return (U @ np.diag(g_e) @ U.T) @ s
