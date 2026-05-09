import numpy as np
import numpy.linalg as LA

# Requires latex installed
import scienceplots  # noqa: F401
from pygsp import graphs

from afgl.plot.ex1 import plot_error_comparison, plot_graphs
from afgl.plot.itersine import plot_itersine
from afgl.util.g_function import (
    compute_g_itersine,
    compute_g_M,
    filter_signal_with_fourier,
)
from afgl.util.lanczos import lanczos


def run_comparison_1_for_graph(
    G, s: np.ndarray, M_MAX: int, ch
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare a Lanczos error estimator with the true filtering error.

    Args:
        G: Graph
        s: Signal vector
        ch: Itersine selected channel

    Returns:
        A tuple ``(lanczos_err, true_err, GLs)``:
            ``lanczos_err``: The estimator ``||g_{M+3} - g_M||`` for each ``M``.
            ``true_err``: The true error ``||g(L)s - g_M||`` for each ``M``.
            ``GLs``: The reference filtered signal ``g(L)s`` computed via Fourier.
    """
    G.compute_laplacian("combinatorial")
    L = G.L

    j = 3
    g = compute_g_itersine(G)

    (
        V,
        T,
        _,
    ) = lanczos(L, s, M_MAX + j)

    lanczos_err = np.zeros(M_MAX)
    true_err = np.zeros(M_MAX)

    GLs = filter_signal_with_fourier(G, s, g)

    for M in range(1, M_MAX + 1):
        g_M = compute_g_M(V[:, :M], T[:, :M], s, g, ch)
        g_Mj = compute_g_M(V[:, : M + j], T[:, : M + j], s, g, ch)

        lanczos_err[M - 1] = LA.norm(g_Mj - g_M)
        true_err[M - 1] = LA.norm(GLs - g_M)

    return lanczos_err, true_err, GLs


def run() -> None:
    """Reproduce Example 1 from the reference paper using only Lanczos.

    The filter is the itersine kernel
    ``g(t) = sin(0.5π cos(π t)^2) * χ_{[-0.5, 0.5]}`` (no Chebyshev baseline).
    """
    N = 500
    M = 200
    p = 0.04

    s = np.random.rand(N).astype(float)
    # Normalize s as in request
    s /= LA.norm(s)

    G_ER = graphs.ErdosRenyi(N, p)
    G_S = graphs.Sensor(N)

    l_err_ER, t_err_ER, signal_ER = run_comparison_1_for_graph(G_ER, s, M, 0)
    l_err_S, t_err_S, signal_S = run_comparison_1_for_graph(G_S, s, M, 0)

    plot_graphs(G_ER, G_S, s, signal_ER, signal_S, N, p)
    plot_error_comparison(l_err_ER, t_err_ER, l_err_S, t_err_S)
    plot_itersine(G_S, G_ER)
