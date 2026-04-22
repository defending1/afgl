import time

import numpy as np
import numpy.linalg as LA
import pandas as pd
from pygsp import graphs

from afgl.util.lanczos import lanczos


class Ex_23:
    def __init__(self, n, times):
        self.n = n
        self.times = times
        x = self.to_df(self.run_ex_2(), "ex 2")
        y = self.to_df(self.run_ex_3(), "ex 3")
        df = pd.concat([x, y], axis=1)
        print(df)
        df.attrs["n"] = n
        df.attrs["times"] = times
        df.to_pickle("./out/ex_23.pkl")

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
        df.to_latex(f"./out/table_ex_{ex}.tex", index=False)

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

        for _ in range(self.times):
            for N in N_VALUES:
                self.iteration(N, M, p, results)

        return results

    def to_df(self, results, name, to_latex=False):
        df = pd.DataFrame(results)
        self.to_latex(df, 3)
        df.rename(columns={"Duration (s)": "Duration (s) " + name}, inplace=True)
        return df["Duration (s) " + name]

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

        for _ in range(self.times):
            for p in p_values:
                self.iteration(N, M, p, results)

        return results
