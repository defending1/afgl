import numpy as np
import numpy.linalg as LA
from pygsp import filters


def compute_g_itersine(G, Nf=7):
    G.estimate_lmax()
    g = filters.Itersine(G, Nf=Nf)
    return g


def compute_g_M(V: np.ndarray, T, s: np.ndarray, g) -> np.ndarray:
    """
    Computes the approximation g_M (see [1]) using Lanczos
    """
    M = T.shape[0]
    e_1 = np.zeros(M)
    e_1[0] = 1

    eigvals, U = LA.eigh(T)
    ev = g.evaluate(eigvals)
    g_e = ev[0]

    g_T = U @ np.diag(g_e) @ U.T

    y = LA.norm(s) * (g_T @ e_1)
    return V @ y


def filter_signal_with_fourier(G, s: np.ndarray, g) -> np.ndarray:
    """Returns evaluation g(L)=Ug(Λ)U*s, which is the filtered signal using the
    fourier basis.
    """
    G.compute_fourier_basis()
    U = G.U
    ev = g.evaluate(G.e)
    g_e = ev[0]
    return (U @ np.diag(g_e) @ U.T) @ s
