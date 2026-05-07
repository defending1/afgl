import numpy as np


def T_tridiag(alp, beta):
    """Constructs Lanczos T tridiagonal matrix in banded form.

    Args:
        alp: Vector of alphas of size N
        beta: Vector of betas of size N-1

    Returns:
        T: Symmetric banded matrix with shape (2, N), lower form
    """
    alp = np.asarray(alp)
    beta = np.asarray(beta)
    n = len(alp)
    T = np.zeros((2, n), dtype=np.result_type(alp, beta))
    T[0] = alp
    if n > 1:
        T[1, : n - 1] = beta
    return T
