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


def latex_sci(val, decimals=2):
    """Converts a value to LaTeX scientific notation A x 10^{B}."""
    if val == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(val))))
    mantissa = val / 10**exponent
    return rf"{mantissa:.{decimals}f} \times 10^{{{exponent}}}"


def plot_graphs(G_ER, G_Sensor, s, N, p):
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5))

    # Set coordinates
    G_ER.set_coordinates()
    G_Sensor.set_coordinates()

    signal_ER = filter_signal_with_fourier(G_ER, s)
    signal_S = filter_signal_with_fourier(G_Sensor, s)

    # TOP LEFT
    G_ER.plot(s, ax=axs[0, 0], vertex_size=15, edge_width=0.5, edge_color="gray")
    axs[0, 0].set_title(rf"Erdős-Rényi Graph $(N = {N}, p = {p})$", pad=20)
    axs[0, 0].set_axis_off()

    # BOTTOM LEFT
    G_ER.plot(
        signal_ER, ax=axs[1, 0], vertex_size=15, edge_width=0.5, edge_color="gray"
    )
    axs[1, 0].set_title("", pad=20)
    axs[1, 0].set_axis_off()

    # TOP RIGHT
    G_Sensor.plot(s, ax=axs[0, 1], vertex_size=15, edge_width=0.5, edge_color="gray")
    axs[0, 1].set_title(rf"Sensor Network $(N = {N})$", pad=20)
    axs[0, 1].set_axis_off()

    # BOTTOM RIGHT
    G_Sensor.plot(
        signal_S, ax=axs[1, 1], vertex_size=15, edge_width=0.5, edge_color="gray"
    )
    axs[1, 1].set_title("", pad=20)
    axs[1, 1].set_axis_off()

    # Prevent label/title overlap

    plt.savefig("./out/printed_graphs.pdf", bbox_inches="tight")


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


def compute_g_M(V, alp, beta, s):
    M = len(alp)
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


def filter_signal_with_fourier(G, s):
    G.compute_fourier_basis()
    U = G.U
    GLs = (U @ np.diag(g(G.e)) @ U.T) @ s
    # GLs = g(L) @ s
    return GLs


def run_comparison_1_for_graph(G, s, M_MAX):
    G.compute_laplacian("combinatorial")
    L = G.L

    j = 3
    [V, alp, beta] = lanczos(L, s, M_MAX + j)

    lanczos_err = np.zeros(M_MAX + j)
    true_err = np.zeros(M_MAX + j)

    GLs = filter_signal_with_fourier(G, s)
    # GLs = g(L) @ s
    for M in range(2, M_MAX + j):
        g_M = compute_g_M(V[:, 0:M], alp[0:M], beta[0 : M - 1], s)
        g_Mj = compute_g_M(V[:, 0 : M + j], alp[0 : M + j], beta[0 : M + j - 1], s)

        lanczos_err[M - 1] = LA.norm(g_Mj - g_M)
        true_err[M - 1] = LA.norm(GLs - g_M)

    return [lanczos_err, true_err]


def example_1():
    N = 500
    M_MAX = 200
    p = 0.04

    s = np.random.randint(1, 10000, N)
    # Normalize s as in request
    s = s / LA.norm(s)

    G_ER = graphs.ErdosRenyi(N, p)
    G_S = graphs.Sensor(N)

    [l_err_ER, t_err_ER] = run_comparison_1_for_graph(G_ER, s, M_MAX)
    [l_err_S, t_err_S] = run_comparison_1_for_graph(G_S, s, M_MAX)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))

    # Left plot (Erdos-Renyi)
    ax1.plot(l_err_ER, label=r"$\left\lVert g_{M+3} - g_M \right\rVert_2$")
    ax1.plot(t_err_ER, label=r"$\left\lVert e_M \right\rVert_2$")
    ax1.set_title("Erdős-Rényi graph")

    # Right plot (Sensor)
    ax2.plot(l_err_S, label=r"$\left\lVert g_{M+3} - g_M \right\rVert_2$")
    ax2.plot(t_err_S, label=r"$\left\lVert e_M \right\rVert_2$")
    ax2.set_title("Sensor graph")

    # Apply identical formatting to both subplots
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(latex_log_formatter))
        ax.legend()

    # Prevents overlapping of labels between the subplots
    plt.tight_layout()

    plt.savefig("./out/ex1_estimate.pdf", bbox_inches="tight")

    plot_graphs(G_ER, G_S, s, N, p)


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
