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
    def __init__(self, n: int, times: int, out_dir: str = "./out"):
        self.n = n
        self.times = times
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> pd.DataFrame:
        results_ex2 = self.run_ex_2()
        results_ex3 = self.run_ex_3()

        df_ex2 = pd.DataFrame(results_ex2)
        df_ex3 = pd.DataFrame(results_ex3)

        self.save_to_latex(df_ex2, 2)
        self.save_to_latex(df_ex3, 3)

        combined_df = pd.concat([df_ex2["Time 2"], df_ex3["Time 3"]], axis=1)

        print(combined_df)

        combined_df.attrs["n"] = self.n
        combined_df.attrs["times"] = self.times
        combined_df.to_pickle(self.out_dir / "ex_23.pkl")

        return combined_df

    def _perform_iteration(
        self, N: int, M: int, p: float, ex_num: int
    ) -> dict[str, Any]:
        """Perform a single Lanczos iteration on an Erdős-Rényi graph.

        Args:
            N: Number of nodes in the graph.
            M: Number of Lanczos iterations.
            p: Edge probability for the Erdős-Rényi graph.
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
        eps = 1e-3

        start = time.perf_counter()
        _, _, debug = lanczos(L, s, M, g, eps)
        end = time.perf_counter()

        return {
            "$N$": N,
            "$M$": M,
            "$p$": p,
            f"Time {ex_num}": end - start,
            "Orthogonalization": "Full" if debug[0] else "Partial",
            "Breakdown": debug[1],
            "Stopping index": debug[2],
            "$\\varepsilon$": eps,
        }

    def save_to_latex(self, df: pd.DataFrame, ex_num: int) -> None:
        """Save DataFrame to a LaTeX table."""
        output_path = self.out_dir / f"table_ex_{ex_num}.tex"
        df.to_latex(output_path, index=False)

    def run_ex_2(self) -> list[dict[str, Any]]:
        """Experiment 2: Increasing graph size N, fixed p and M."""
        p = 0.04
        M = 200
        N_values = 250 * (2 ** np.arange(self.n))

        results = []
        for _ in range(self.times):
            for N in N_values:
                results.append(self._perform_iteration(N, M, p, ex_num=2))
        return results

    def run_ex_3(self) -> list[dict[str, Any]]:
        """Experiment 3: Fixed N and M, increasing p."""
        N = 1000
        M = 200
        p_values = 0.01 * (2 ** np.arange(self.n))

        results = []
        for _ in range(self.times):
            for p in p_values:
                results.append(self._perform_iteration(N, M, p, ex_num=3))
        return results
