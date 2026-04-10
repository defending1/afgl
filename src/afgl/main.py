# src/afgl/main.py
import sys

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA

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
characteristic function of I, as defined in example 1 of the paper.
"""


def g(T):
    Ind = ((T >= -1 / 2) & (T <= 1 / 2)).astype(int)
    Val = g_extended(T)
    # Perform element wise multiplication to resemble applying indicating
    # function to all matrix.
    return np.multiply(Val, Ind)


def example_1():
    N = 500
    M = 300
    p = 0.04

    G = graphs.ErdosRenyi(N, p)

    G.compute_laplacian("combinatorial")
    L = G.L

    s = np.random.randint(1, 10000, N)

    [V, alp, beta] = lanczos(L, s, M)

    T = build_T_matrix(alp, beta)

    e_1 = np.zeros(M)
    e_1[0] = 1
    y = LA.norm(s) * (g(T) * e_1)
    G_L_s = V @ y
    return G_L_s


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
