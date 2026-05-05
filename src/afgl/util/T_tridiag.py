import numpy as np


def T_tridiag(alp, beta):
    """Constructs Lanczos T tridiagonal matrix in banded form.

    Args:
        alp: Vector of alphas of size N
        beta: Vector of betas of size N-1

    Returns:
        T : Banded matrix with shape (2, N), lower form
    """
    n = len(alp)
    T = np.zeros((2, n), dtype=np.asarray(alp).dtype)
    T[0, :] = alp
    if n > 1:
        T[1, :-1] = beta
    return T
