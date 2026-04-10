# src/afgl/main.py
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.linalg as LA
import scienceplots  # noqa: F401
import scipy

# Requires latex installed
from pygsp import graphs, plotting

from afgl.util.build_T_matrix import build_T_matrix
from afgl.util.lanczos import lanczos


def plot_setup():
    plotting.BACKEND = "matplotlib"
    plt.style.use(["science"])
    # TODO match font with document


def test_plot():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    # G = graphs.ErdosRenyi()
    G = graphs.Sensor()

    G.set_coordinates()

    signal = np.sin(G.coords[:, 0] * 10)

    G.plot(signal, ax=ax, vertex_size=15, edge_width=0.5, edge_color="gray")
    ax.set_title(r"Sensor Network $\mathcal{G} = (\mathcal{V}, \mathcal{E})$")
    ax.set_axis_off()
    plt.savefig("./out/test.pdf", bbox_inches="tight")


def g_extended(t):
    return np.sin(0.5 * np.pi * np.cos(np.pi * t) ** 2)


"""
Evaluates the function sin(0.5*pi*cos(pi*t)^2)chi_[-1/2,1/2] where chi_I is the
characteristic function of I, as defined in example 1 (see [1]).
"""


def g(T):
    if scipy.sparse.issparse(T):
        # Operator & not supporting sparse matrix
        T = T.toarray()
    Chi = ((T >= -1 / 2) & (T <= 1 / 2)).astype(int)
    # Apply g_extended where Chi is True, else output 0
    return np.where(Chi, g_extended(T), 0)


""" 
Computes the approximation g_M (see [1]) using Lanczos
"""


def compute_g_M(L, s, M):
    [V, alp, beta] = lanczos(L, s, M)
    e_1 = np.zeros(M)
    e_1[0] = 1
    T = build_T_matrix(alp, beta)
    y = LA.norm(s) * (g(T) @ e_1)
    return V @ y


def latex_log_formatter(y, pos):
    """Custom formatter to render tick labels as LaTeX 10^{n}."""
    if y <= 0:
        return ""
    # Extract the exponent using log10
    n = int(np.round(np.log10(y)))
    return f"$10^{{{n}}}$"


def example_1():
    N = 500
    M_MAX = 200
    p = 0.04

    G = graphs.ErdosRenyi(N, p)
    # G = graphs.Sensor(N)

    G.compute_laplacian("combinatorial")
    L = G.L

    s = np.random.randint(1, 10000, N)
    # Normalize s as in request
    s = s / LA.norm(s)

    lanczos_err = np.zeros(M_MAX)
    true_err = np.zeros(M_MAX)

    GLs = g(L) @ s
    for M in range(1, M_MAX + 1):
        g_M = compute_g_M(L, s, M)
        j = 3
        g_Mj = compute_g_M(L, s, M + j)

        lanczos_err[M - 1] = LA.norm(g_Mj - g_M)
        true_err[M - 1] = LA.norm(GLs - g_M)

    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    ax.plot(lanczos_err, label="Error estimate")
    ax.plot(true_err, label="Error")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(latex_log_formatter))

    ax.legend()
    plt.savefig("./out/erdos_estimate.pdf", bbox_inches="tight")


def run():
    plot_setup()
    example_1()


if __name__ == "__main__":
    # This block runs only if you call: python -m my_package.main
    try:
        run()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
        sys.exit(0)
