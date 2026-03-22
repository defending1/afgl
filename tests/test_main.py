import numpy as np
import numpy.linalg as LA
from afgl.lanczos import lanczos

"""
Todo: better test case
"""


def test_lanczos_return_correct_solution():
    N = 6
    M = 4

    A = np.random.randint(1, 10, size=(N, N))
    L = (A + A.T) / 2
    s = np.random.randint(1, 10, N)
    [V, alp, beta] = lanczos(L, s, M)

    T = np.diag(alp) + np.diag(beta, -1) + np.diag(beta, 1)

    x = LA.solve(A, s)
    e_1 = np.zeros(M)
    e_1[0] = 1
    y = (LA.inv(T) @ e_1) * LA.norm(s)
    x_lanczos = V @ y

    assert LA.norm(x - x_lanczos) < 1e-3
