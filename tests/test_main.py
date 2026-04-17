import numpy as np
import numpy.linalg as LA
from afgl.ex_1 import compute_g_M, filter_signal_with_fourier, g
from afgl.util.build_T_matrix import build_T_matrix
from afgl.util.lanczos import lanczos
from pygsp import graphs


def g_evaluation_should_respect_chi():
    A = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 0]])
    expected_gA = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]])
    assert (expected_gA == g(A).astype(int)).all()


def g_evaluation_should_return_matrix_of_zeros():
    A = 1 / 2 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    assert (g(A).astype(int) == np.zeros((3, 3))).all()


def test_lanczos_return_correct_solution_with_dense():
    """Tests correctness of solution of Lx=s comparing Lanczos projected
    solution with numpy solve function.
    """
    N = 1000
    M = 999

    # Generate a good conditioned matrix
    eigvals = np.random.uniform(10000, 100000, N)
    Q, _ = LA.qr(np.random.randn(N, N))
    L = Q @ np.diag(eigvals) @ Q.T

    s = np.random.randint(1, 10, N)
    (
        V,
        alp,
        beta,
        _,
    ) = lanczos(L, s, M)

    T = build_T_matrix(alp, beta)

    x = LA.solve(L, s)
    e_1 = np.zeros(len(alp))
    e_1[0] = 1
    y = (LA.inv(T) @ e_1) * LA.norm(s)
    x_lanczos = V @ y

    assert LA.norm(x - x_lanczos) < 1e-10


def test_function_g_with_graph_laplacian():
    N = 1000
    p = 0.04
    j = 3

    n = 5
    M_VALS = 25 * (2 ** np.arange(n))
    M_VALS = [200]

    for M in M_VALS:
        s = np.random.randint(1, 10000, N).astype(float)
        # Normalize s as in request
        s /= LA.norm(s)

        GRAPHS = [graphs.ErdosRenyi(N, p), graphs.Sensor(N)]

        for G in GRAPHS:
            G.compute_laplacian()
            L = G.L
            (
                V,
                alp,
                beta,
                _,
            ) = lanczos(L, s, M + j)

            GLs = filter_signal_with_fourier(G, s)

            g_M = compute_g_M(V[:, 0:M], alp[0:M], beta[0 : M - 1], s)
            g_Mj = compute_g_M(V[:, 0 : M + j], alp[0 : M + j], beta[0 : M + j - 1], s)

            diff = LA.norm(g_Mj - g_M)
            e_M = LA.norm(GLs - g_M)

            assert abs(diff - e_M) < 1e-2
