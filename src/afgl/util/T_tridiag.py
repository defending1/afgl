import numpy as np


def T_tridiag(alp, beta):
    """Construct a tridiagonal matrix in symmetric banded form.

    Args:
        alp: Diagonal entries (alpha), length ``n``.
        beta: Subdiagonal entries (beta), length ``n-1``.

    Returns:
        Symmetric banded matrix with shape ``(2, n)`` in SciPy lower-banded form.
    """
    alp = np.asarray(alp)
    beta = np.asarray(beta)
    n = len(alp)
    T = np.zeros((2, n), dtype=np.result_type(alp, beta))
    T[0] = alp
    if n > 1:
        T[1, : n - 1] = beta
    return T
