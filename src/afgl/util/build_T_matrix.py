import numpy as np


def build_T_matrix(alp, beta):
    return np.diag(alp) + np.diag(beta, -1) + np.diag(beta, 1)
