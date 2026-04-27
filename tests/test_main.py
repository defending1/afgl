import numpy as np
import numpy.linalg as LA
import scipy.linalg as SCLA
from afgl.ex_1 import compute_g_M, filter_signal_with_fourier, g
from afgl.util.lanczos import lanczos
from pygsp import graphs


def is_sorted(a):
    return np.all(a[:-1] <= a[1:])


def test_if_T_is_symmetric():
    N = 1000
    M = 200
    p = 0.04
    s = np.random.rand(N).astype(float)

    G = graphs.ErdosRenyi(N, p)
    G.compute_laplacian("combinatorial")
    G.estimate_lmax()
    G.L = G.L / (2 * G.lmax)

    _, T, debug = lanczos(G.L, s, M)

    assert SCLA.issymmetric(T)


def test_rescaling_L_should_give_T_with_eigvals_in_range():
    p = 0.04
    M = 200
    N = 1000
    s = np.random.rand(N).astype(float)

    G = graphs.ErdosRenyi(N, p)
    G.compute_laplacian("combinatorial")
    G.estimate_lmax()
    G.L = G.L / (2 * G.lmax)
    G.compute_fourier_basis()
    assert min(G.e) > -1 / 2 and max(G.e) < 1 / 2

    (
        V,
        T,
        _,
    ) = lanczos(G.L, s, M)

    eigvals, _ = LA.eigh(T)
    assert -1 / 2 < min(eigvals) and max(eigvals) < 1 / 2

    assert is_sorted(eigvals)


def test_g_evaluation_should_respect_chi():
    A = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 0]])
    expected_gA = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 1]])
    assert (expected_gA == g(A).astype(int)).all()


def test_g_evaluation_should_return_matrix_of_zeros():
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
        T,
        _,
    ) = lanczos(L, s, M)

    x = LA.solve(L, s)
    e_1 = np.zeros(T.shape[0])
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
                T,
                _,
            ) = lanczos(L, s, M + j)

            GLs = filter_signal_with_fourier(G, s)

            g_M = compute_g_M(V[:, 0:M], T[:M, :M], s)
            g_Mj = compute_g_M(V[:, 0 : M + j], T[: M + j, : M + j], s)

            diff = LA.norm(g_Mj - g_M)
            e_M = LA.norm(GLs - g_M)

            assert abs(diff - e_M) < 1e-2
