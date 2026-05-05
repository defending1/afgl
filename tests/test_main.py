import numpy as np
import numpy.linalg as LA
import scipy.linalg as SCLA
from afgl.util.arnoldi import arnoldi
from afgl.util.g_function import (
    compute_g_itersine,
    compute_g_M,
    filter_signal_with_fourier,
)
from afgl.util.lanczos import lanczos
from pygsp import graphs


def is_sorted(a):
    return np.all(a[:-1] <= a[1:])


def test_check_orthogonalization_trigger():
    N = 1000
    M = 200
    p = 0.04
    s = np.random.rand(N).astype(float)

    G = graphs.ErdosRenyi(N, p)
    G.compute_laplacian("combinatorial")
    G.estimate_lmax()
    L = G.L

    print(f"$k_2(L)={LA.cond(L.toarray(), 2)}")
    V, T, debug = lanczos(G.L, s, M, full_orthogonalization=False)
    Id = np.eye(V.shape[1])
    print(f"Partial orthogonalization {LA.norm(V.T @ V - Id)}")

    V, T, debug = lanczos(G.L, s, M, full_orthogonalization=False, gs_iterations=1)
    Id = np.eye(V.shape[1])
    print(f"Single Gram-Schmidt orthogonalization {LA.norm(V.T @ V - Id)}")

    V, T, debug = lanczos(G.L, s, M, full_orthogonalization=True, gs_iterations=2)
    Id = np.eye(V.shape[1])
    print(f"Double Gram-Schmidt orthogonalization {LA.norm(V.T @ V - Id)}")


def test_if_H_is_symmetric():
    N = 1000
    M = 200
    p = 0.04
    s = np.random.rand(N).astype(float)

    G = graphs.ErdosRenyi(N, p)
    G.compute_laplacian("combinatorial")
    G.estimate_lmax()
    g = compute_g_itersine(G)
    _, T, _ = arnoldi(G.L, s, M, g, 10e-20)

    assert LA.norm(T - T.T) < 10e-12


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

    eigvals, _ = SCLA.eig_banded(T, lower=True)
    eigvals = np.sort(eigvals)
    assert -1 / 2 < min(eigvals) and max(eigvals) < 1 / 2

    assert is_sorted(eigvals)


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
    e_1 = np.zeros(T.shape[1])
    e_1[0] = 1
    y = (SCLA.solveh_banded(T, e_1, lower=True)) * LA.norm(s)
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

            g = compute_g_itersine(G)

            (
                V,
                T,
                _,
            ) = lanczos(L, s, M + j)

            GLs = filter_signal_with_fourier(G, s, g)

            g_M = compute_g_M(V[:, 0:M], T[:, :M], s, g)
            g_Mj = compute_g_M(V[:, 0 : M + j], T[:, : M + j], s, g)

            diff = LA.norm(g_Mj - g_M)
            e_M = LA.norm(GLs - g_M)

            assert abs(diff - e_M) < 1e-2
