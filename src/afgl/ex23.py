import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.linalg as LA
import pandas as pd
from pygsp import graphs

from afgl.util.g_function import compute_g_itersine
from afgl.util.lanczos import lanczos


class Ex23:
    def __init__(self, times: int, out_dir: str = "./out"):
        self.n = 6
        self.times = times
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        palette = [
            "#000000",  # black
            "#ff0000",  # red
            "#0000ff",  # blue
            "#00aa00",  # green
            "#ff00ff",  # magenta
            "#00ffff",  # cyan
        ]
        self.colors = [palette[i % len(palette)] for i in range(self.n)]

    def run(self) -> pd.DataFrame:
        results_ex2 = self.run_ex_2()
        results_ex3 = self.run_ex_3()

        df2 = pd.DataFrame(results_ex2)
        df3 = pd.DataFrame(results_ex3)

        self.save_to_latex(df2, 2)
        self.save_to_latex(df3, 3)

        print(df2)

        df2.attrs["n"] = self.n
        df2.attrs["times"] = self.times
        df2.to_pickle(self.out_dir / "df2.pkl")

        df3.attrs["n"] = self.n
        df3.attrs["times"] = self.times
        df3.to_pickle(self.out_dir / "df3.pkl")

    def _perform_iteration(
        self, N: int, M: int, p: float, index: int, ex_num: int
    ) -> dict[str, Any]:
        """Perform a single Lanczos iteration on an Erdős-Rényi graph.

        Args:
            N: Number of nodes in the graph.
            M: Number of Lanczos iterations.
            p: Edge probability for the Erdős-Rényi graph.
            index: current index for color.
            ex_num: Exercise number for result labeling.

        Returns:
            Dictionary containing iteration results and metadata.
        """
        # Signal vector s
        s = np.random.randint(1, 10000, N).astype(float)
        s /= LA.norm(s)

        G = graphs.ErdosRenyi(N, p)
        G.compute_laplacian("combinatorial")
        L = G.L

        g = compute_g_itersine(G)

        # Lanczos method
        eps = 10e-6

        start = time.perf_counter()
        _, _, debug = lanczos(L, s, M, g, eps)
        end = time.perf_counter()

        return {
            "$N$": N,
            "$M$": M,
            "$p$": p,
            "Time": end - start,
            "Orthogonalization": "Full" if debug[0] else "Partial",
            "Breakdown": debug[1],
            "Stopping index": debug[2],
            "$\\varepsilon$": eps,
            "color": self.colors[index],
        }

    def print_group_table(self):
        index = ["$N$", "$p$", "Time", "Stopping index"]
        df2 = pd.read_pickle("./out/df2.pkl")
        df3 = pd.read_pickle("./out/df3.pkl")
        df2.drop(columns=["color"])
        df3.drop(columns=["color"])
        df2 = df2[index]
        df3 = df3[index]

        df2.groupby(["$N$"]).agg(["mean", "var"]).to_latex(
            self.out_dir / "group_table_2.tex"
        )
        df3.groupby(["$p$"]).agg(["mean", "var"]).to_latex(
            self.out_dir / "group_table_3.tex"
        )

    def save_to_latex(self, df: pd.DataFrame, ex_num: int) -> None:
        """Save DataFrame to a LaTeX table."""
        output_path = self.out_dir / f"table_ex_{ex_num}.tex"
        df.to_latex(output_path, index=False)

    def run_ex_2(self) -> list[dict[str, Any]]:
        """Experiment 2: Increasing graph size N, fixed p and M."""
        p = 0.04
        M = 200
        N_values = 50 * (2 ** np.arange(self.n))

        results = []
        for _ in range(self.times):
            for N in N_values:
                results.append(
                    self._perform_iteration(N, M, p, list(N_values).index(N), ex_num=2)
                )
        return results

    def run_ex_3(self) -> list[dict[str, Any]]:
        """Experiment 3: Fixed N and M, increasing p."""
        N = 1000
        M = 200
        p_values = 0.01 * (2 ** np.arange(self.n))

        results = []
        for i in range(self.times):
            for p in p_values:
                results.append(
                    self._perform_iteration(N, M, p, list(p_values).index(p), ex_num=3)
                )
        return results
