import time

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA
import pandas as pd
from pygsp import graphs

from afgl.util.lanczos import lanczos


class Ex_23:
    def __init__(self, n):
        self.n = n
        x = self.run_ex_2()
        y = self.run_ex_3()
        self.plot(x, y)

    def plot(self, x, y):
        plt.figure(figsize=(8, 6))
        plt.loglog(x, y, "o", color="blue", label="Raw Data")
        plt.xlabel("Duration [s] ($N$ varies)")
        plt.ylabel("Duration [s] ($p$ varies)")
        plt.title("Performance correlation between $N$ and $p$.")

        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.savefig("./out/correlation_23.pdf", bbox_inches="tight")

    def iteration(self, N, M, p, results):
        s = np.random.randint(1, 10000, N).astype(float)
        # Normalize s as in request
        s /= LA.norm(s)

        G = graphs.ErdosRenyi(N, p)
        G.compute_laplacian("combinatorial")
        L = G.L

        start = time.perf_counter()
        eps = 10e-4
        V, alp, beta, debug = lanczos(L, s, M, eps)
        end = time.perf_counter()

        results.append(
            {
                "$N$": N,
                "$M$": M,
                "$p$": p,
                "Duration (s)": end - start,
                "Orthogonalization": "Full" if debug[0] else "Partial",
                "Breakdown": debug[1],
                "$\\varepsilon$": eps,
            }
        )

    def to_latex(self, df, ex):
        df.to_latex(f"./report/assets/table_ex_{ex}.tex", index=False)

    def run_ex_2(self):
        """Genera i grafi di di Erdos-Reny di grandezza crescente 250,
        500,..., 250*2^n) e parametro p = 0.04 e misura il tempo
        computazionale del metodo di Lanczos utilizzando come soglia per il criterio
        d'arresto FOM con epsilon = 10^-2 (o una soglia a scelta).

         Args:
             self

         Returns:
             None
        """
        p = 0.04
        M = 200

        N_VALUES = 250 * (2 ** np.arange(self.n))

        results = []

        for N in N_VALUES:
            self.iteration(N, M, p, results)

        df = pd.DataFrame(results)
        self.to_latex(df, 2)
        return df["Duration (s)"]

    def run_ex_3(self):
        """Genera i grafi di di Erdos-Reny di grandezza fissata n=1000 e testa p =
        0.01, 0.02,..., 0.01 * 2^n, e misura il tempo
        computazionale del metodo di Lanczos utilizzando come soglia per il criterio
        d'arresto FOM con epsilon = 10^-2 (o una soglia a scelta).

         Args:
             self

         Returns:
             None
        """
        M = 200
        N = 1000
        p_values = 0.01 * (2 ** np.arange(self.n))

        results = []

        for p in p_values:
            self.iteration(N, M, p, results)

        df = pd.DataFrame(results)
        self.to_latex(df, 3)
        return df["Duration (s)"]
