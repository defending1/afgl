import numpy as np


def T_tridiag(alp, beta):
    """Constructs Lanczos T tridiagonal matrix.

    Args:
        alp: Vector of alphas of size N
        beta: Vector of betas of size N-1

    Returns:
        T : Matrix T
    """

    return np.diag(alp) + np.diag(beta, -1) + np.diag(beta, 1)
