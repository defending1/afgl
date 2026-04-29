from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from afgl.plot.setup import latex_log_formatter

if TYPE_CHECKING:
    from pygsp.graphs import Graph


def plot_graphs(
    G_ER: "Graph", G_Sensor: "Graph", s, signal_ER, signal_S, N: int, p: float
) -> None:
    """Visualization of signal being filtered on two different types of graphs.

    Args:
        G_ER: Erdős-Rényi graph.
        G_Sensor: Sensor network graph.
        s: Input signal vector.
        N: Number of nodes.
        p: Probability for Erdős-Rényi graph.
    """
    fig, axs = plt.subplots(2, 2, figsize=(6.6, 5))

    # Set coordinates
    G_ER.set_coordinates()
    G_Sensor.set_coordinates()

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


def plot_error_comparison(
    l_err_ER: np.ndarray, t_err_ER: np.ndarray, l_err_S: np.ndarray, t_err_S: np.ndarray
) -> None:
    """Plots the error comparison between Lanczos approximation and true error.

    Args:
        l_err_ER: Lanczos error estimate for Erdős-Rényi graph.
        t_err_ER: True error for Erdős-Rényi graph.
        l_err_S: Lanczos error estimate for Sensor graph.
        t_err_S: True error for Sensor graph.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.5))

    ax1.plot(l_err_S, label=r"$\left\lVert g_{M+3} - g_M \right\rVert_2$")
    ax1.plot(t_err_S, label=r"$\left\lVert e_M \right\rVert_2$")
    ax1.set_title("Sensor graph")

    ax2.plot(l_err_ER, label=r"$\left\lVert g_{M+3} - g_M \right\rVert_2$")
    ax2.plot(t_err_ER, label=r"$\left\lVert e_M \right\rVert_2$")
    ax2.set_title("Erdős-Rényi graph")

    # Apply formatting to both subplots
    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(latex_log_formatter))
        ax.legend()

    plt.tight_layout()

    plt.savefig("./out/ex1_estimate.pdf", bbox_inches="tight")
