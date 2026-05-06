import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
from pygsp import graphs

from afgl.util.arnoldi import arnoldi
from afgl.util.g_function import compute_g_itersine
from afgl.util.lanczos import lanczos


class LanczosVsArnoldi:
    def __init__(self, out_dir: str = "./out"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.out_dir / "plots"
        self.tables_dir = self.out_dir / "tables"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        results_ex2 = self.run_exp()
        df = pd.DataFrame(results_ex2)
        stats_cols = [
            col for col in df.columns if "time" in col.lower() or "index" in col.lower()
        ]
        stats_df = pd.DataFrame(
            {
                "mean": df[stats_cols].mean(),
                "variance": df[stats_cols].var(),
            }
        )
        stats_df.to_latex(self.tables_dir / "lanczos_vs_arnoldi_stats.tex")
        plt.figure(figsize=(10, 6))  # set the figure size
        plt.plot(
            df["$N$"],
            df["Time Arnoldi"],
            label="Time Arnoldi",
            color="blue",
            linestyle="-",
            marker=".",
        )
        plt.plot(
            df["$N$"],
            df["Time Lanczos (full ortho)"],
            label="Time Lanczos (full ortho)",
            color="red",
            linestyle="--",
            marker=".",
        )

        plt.plot(
            df["$N$"],
            df["Time Lanczos (no ortho)"],
            label="Time Lanczos (no ortho)",
            color="green",
            linestyle="dashdot",
            marker=".",
        )

        plt.xlabel("N", fontsize=16)
        plt.ylabel("Time", fontsize=16)
        plt.yscale("log")
        # plt.title("Comparison ")

        plt.legend(fontsize=16)
        plt.grid(True)
        plt.savefig(self.plots_dir / "lanczos_vs_arnoldi_plot.pdf", bbox_inches="tight")
        plt.close()

        return df

    def _perform_iteration(self, N: int, M: int) -> dict[str, Any]:
        """Perform a single Lanczos iteration on a graph.

        Args:
            N: Number of nodes in the graph.
            M: Number of Lanczos iterations.

        Returns:
            Dictionary containing iteration results and metadata.
        """
        # Signal vector s
        p = 0.04
        s = np.random.randint(1, 10000, N).astype(float)
        s /= LA.norm(s)

        G = graphs.ErdosRenyi(N, p)
        G.compute_laplacian("combinatorial")
        L = G.L

        g = compute_g_itersine(G)

        eps = 10e-6

        start_Lno = time.perf_counter()
        _, _, debug_Lno = lanczos(L, s, M, g, False, eps)
        end_Lno = time.perf_counter()

        start_L = time.perf_counter()
        _, _, debug = lanczos(L, s, M, g, True, eps)
        end_L = time.perf_counter()

        start_A = time.perf_counter()
        _, _, j_A = arnoldi(L, s, M, g, eps)
        end_A = time.perf_counter()

        return {
            "$N$": N,
            "$p$": p,
            "$M$": M,
            "Time Lanczos (no ortho)": end_Lno - start_Lno,
            "Time Lanczos (full ortho)": end_L - start_L,
            "Time Arnoldi": end_A - start_A,
            "Stopping index Lanczos (no ortho)": debug_Lno[2],
            "Stopping index Lanczos (full ortho)": debug[2],
            "Stopping index Arnoldi": j_A,
            "$\\varepsilon$": eps,
        }

    def run_exp(self) -> list[dict[str, Any]]:
        """Experiment 2: Increasing graph size N, fixed p and M."""
        M = 200
        N_values = list(range(300, 3000, 100))

        results = []
        for N in N_values:
            results.append(self._perform_iteration(N, M))
        return results
