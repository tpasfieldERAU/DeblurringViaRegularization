import numpy as np
from scipy.sparse import csr_array, kron
import scipy.linalg as la
from params import *

def one_dim_blur_kernel(psf):
    # Consider immediately compressing the array if size becomes an issue.s
    # Ensure PSF is normalized
    psf /= sum(psf) * 2
    return csr_array(la.toeplitz(psf))

def two_dim_blur_kernel(Ax, Ay):
    return kron(Ax, Ay)
    