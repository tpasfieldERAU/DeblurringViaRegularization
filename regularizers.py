import numpy as np
from scipy.sparse import csr_array, kron, identity
from scipy.sparse.linalg import spsolve, LinearOperator, spilu, cg
import scipy.linalg as la
from scipy.fft import fft, ifft


def tikhonov_regularization(b, Ax, Ay, alpha):
    alpha = alpha[0]
    # print("tkv: Create identity matrix")
    L = alpha * identity(b.shape[0])
    # print("tkv: Calculate LHS")
    A_transpose_product = kron(Ax@Ax, Ay@Ay)
    # print("tkv: Transpose product")
    lhs = csr_array(A_transpose_product + L)
    # print("tkv: Calculate RHS")
    rhs = csr_array(kron(Ax,Ay) @ b)
    print("Solve Inverse")
    ilu = spilu(lhs)
    M = LinearOperator(lhs.shape, ilu.solve)
    x, _ = cg(lhs, rhs.A.ravel(), M=M)
    return x, lhs