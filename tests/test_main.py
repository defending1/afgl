import numpy as np
import numpy.linalg as LA
from afgl.lanczos import lanczos

"""
Todo: better test case
"""


def test_lanczos_return_correct_solution():
    N = 1000
    M = 999

    eigvals = np.random.uniform(10000, 100000, N)
    Q, _ = LA.qr(np.random.randn(N, N))
    L = Q @ np.diag(eigvals) @ Q.T

    s = np.random.randint(1, 10, N)
    [V, alp, beta] = lanczos(L, s, M)

    T = np.diag(alp) + np.diag(beta, -1) + np.diag(beta, 1)

    x = LA.solve(L, s)
    e_1 = np.zeros(M)
    e_1[0] = 1
    y = (LA.inv(T) @ e_1) * LA.norm(s)
    x_lanczos = V @ y

    assert LA.norm(x - x_lanczos) < 1e-10
