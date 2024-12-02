from params import *
import numpy as np
from scipy.sparse.linalg import inv
import regularizers

def gcv_param_selection_tkv(alpha, Ax, Ay, A, b):
    print(f"gcv start, alpha = {alpha}")
    x, center = regularizers.tikhonov_regularization(b, Ax, Ay, alpha)

    influence = A @ inv(center) @ A.T
    trce = np.power((1/Ax.shape[0]) * np.trace(np.eye(influence.shape[0]) - influence), 2)
    numer = np.linalg.norm(A@x - b)/Ax.shape[0]
    return numer/trce