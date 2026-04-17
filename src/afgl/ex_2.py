import time

import numpy as np
import numpy.linalg as LA
import pandas as pd
from pygsp import graphs

from afgl.util.lanczos import lanczos


def run() -> None:
    """Genera i grafi di di Erdos-Reny di grandezza crescente (ad esempio 250,
    500,1000, 2000, 4000) e parametro p = 0.04 e misura il tempo
    computazionale del metodo di Lanczos utilizzando come soglia per il criterio
    d'arresto epsilon = 10^-2 (o una soglia a scelta).

     Args:
         None

     Returns:
         None
    """
    n = 6
    p = 0.04
    M = 200

    N_VALUES = 250 * (2 ** np.arange(n))

    results = []

    for N in N_VALUES:
        s = np.random.randint(1, 10000, N).astype(float)
        # Normalize s as in request
        s /= LA.norm(s)

        G = graphs.ErdosRenyi(N, p)
        G.compute_laplacian("combinatorial")
        L = G.L

        start = time.perf_counter()
        eps = 10e-4
        V, alp, beta, full_ortho, early_stop = lanczos(L, s, M, eps)
        end = time.perf_counter()

        results.append(
            {
                "$N$": N,
                "$M$": M,
                "$p$": p,
                "Duration (s)": end - start,
                "Orthogonalization": "Full" if full_ortho else "Partial",
                "Breakdown": early_stop,
                "$\\varepsilon$": eps,
            }
        )
    pd.DataFrame(results).to_latex("./report/assets/table_ex_2.tex", index=False)
