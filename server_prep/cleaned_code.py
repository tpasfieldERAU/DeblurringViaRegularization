import numpy as np
from mpi4py import MPI  # Correct import of mpi4py
from pylops import Gradient, Identity
from pylops.optimization.leastsquares import RegularizedInversion
from pylops.optimization.sparsity import SplitBregman
from scipy.optimize import curve_fit
from scipy.linalg import toeplitz
from scipy.sparse import kron
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt

# Setup MPI environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Logistic functions for calibrator modeling
def logistic_function(xs, k, x0):
    """
    Logistic function with scaling for image calibration.
    """
    out = 255 / (1 + np.exp(-k * (xs - x0))) + 127.0
    out -= np.min(out)
    out = 255 * out / np.max(out)
    return out

def logistic_derivative(xs, k):
    """
    Derivative of the logistic function for matrix generation.
    """
    numerator = 255 * k * np.exp(k * xs)
    denominator = (np.exp(k * xs) + 1)**2
    return numerator / denominator

# Parameter estimation using curve fitting
def eval_fit(index_range, calibrator):
    """
    Fit logistic function to a range of lines in the calibrator image.
    """
    num_lines = index_range[1] - index_range[0]
    lines = calibrator[index_range[0]:index_range[1], :]
    xs0 = np.linspace(0, calibrator.shape[0], calibrator.shape[0])
    xs = np.tile(xs0, num_lines)  # Faster equivalent of concatenating xs0
    p0 = (1.0, calibrator.shape[0] // 2)

    means, covs = curve_fit(logistic_function, xs, lines.ravel(), p0)
    return means, covs

# Sampling from a multivariate normal distribution
def weighted_sampling(means, covs, n_samples):
    """
    Sample weighted parameters from a multivariate normal distribution.
    """
    samples = multivariate_normal(mean=means, cov=covs, size=n_samples)
    return samples

# Generate the operator matrix A from logistic derivative
def A_from_sample(xs, k):
    """
    Generate operator matrix A based on logistic derivative.
    """
    curve = logistic_derivative(xs, k)
    a = toeplitz(curve)
    A = kron(a, a)
    return A

# GCV for Tikhonov regularization
def gcv_tikhonov(A, b, lambdas):
    """
    Generalized Cross-Validation for Tikhonov regularization.
    """
    n = len(b)
    gcv_scores = []
    for lam in lambdas:
        x_reg = RegularizedInversion(A, b, [Identity(n)], [lam])
        residual = np.linalg.norm(A @ x_reg - b)
        AtA = A.T @ A if isinstance(A, np.ndarray) else A.T * A
        trace = np.trace(np.linalg.inv(AtA + lam * np.eye(n)) @ AtA)
        gcv_score = (residual**2) / (n - trace)**2
        gcv_scores.append(gcv_score)

    best_idx = np.argmin(gcv_scores)
    best_lambda = lambdas[best_idx]
    best_gcv = gcv_scores[best_idx]
    best_sol = RegularizedInversion(A, b, [Identity(n)], [best_lambda])
    return best_lambda, best_gcv, best_sol

# GCV for TV regularization
def gcv_tv(A, b, G, lambdas, niter_inner=5, niter_outer=20, mu=1.0):
    """
    Generalized Cross-Validation for Total Variation (TV) regularization.
    """
    n = len(b)
    gcv_scores = []
    solutions = []

    for lam in lambdas:
        x_tv, _ = SplitBregman(A, b, G, niter_inner=niter_inner, niter_outer=niter_outer, mu=mu, epsRL1s=[lam])
        solutions.append(x_tv)
        residual = np.linalg.norm(A @ x_tv - b)
        trace_approx = n - np.linalg.norm(G @ x_tv, ord=1)
        gcv_score = (residual**2) / (n - trace_approx)**2
        gcv_scores.append(gcv_score)

    best_idx = np.argmin(gcv_scores)
    best_lambda = lambdas[best_idx]
    best_gcv = gcv_scores[best_idx]
    best_solution = solutions[best_idx]
    return best_lambda, best_gcv, best_solution

if __name__ == "__main__":
    # Load input data
    calibrators = np.load("./server_prep/calib_images.npz")
    samples = np.load("./server_prep/kodim02_images.npz")
    true_image = samples['base']
    calibrator = calibrators['2_25']
    image = samples['2_25']

    # Fit logistic parameters
    means, covs = eval_fit((140, 250), calibrator)
    new_means = weighted_sampling(means, covs, 50)

    # Distribute parameters across MPI processes
    params_split = np.array_split(new_means, size)
    local_params = params_split[rank]

    for param in local_params:
        A = A_from_sample(np.linspace(0, 255, 256), param[0])
        G = Gradient(dims=(256, 256), kind='forward')
        tik_lambdas = np.logspace(-6, 2, 200)
        tv_lambdas = np.logspace(-4, 1, 60)
        
        # Compute optimal solutions
        tik_best_lam, tik_best_gcv, tik_best_sol = gcv_tikhonov(A, image.ravel(), tik_lambdas)
        tv_best_lam, tv_best_gcv, tv_best_sol = gcv_tv(A, image.ravel(), G, tv_lambdas)
        
        # Save results
        np.savez(
            f'{param[0]}_sols.npz',
            params=param,
            tik=[tik_best_lam, tik_best_gcv, tik_best_sol],
            tv=[tv_best_lam, tv_best_gcv, tv_best_sol]
        )
        print(f"Rank {rank}: Processed parameter {param}")
