import numpy as np
from mpi4py import MPI  # Correct import of mpi4py
from pylops import Gradient, Identity, MatrixMult
from pylops.optimization.leastsquares import regularized_inversion
from pylops.optimization.sparsity import splitbregman
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
    A = np.kron(a, a)
    return A

# GCV for Tikhonov regularization
def gcv_tikhonov(A, b, lambdas):
    """
    Generalized Cross-Validation for Tikhonov regularization.
    """
    n = len(b)
    gcv_scores = []
    for lam in lambdas:
        print(f"Running tik lam = {lam}")
        x_reg, _, _, _, _ = regularized_inversion(A, b, [lam*Identity(n)])
        residual = np.linalg.norm(A @ x_reg - b)
        AtA = A.T @ A if isinstance(A, np.ndarray) else A.T * A
        trace = np.trace(np.linalg.inv(AtA + lam * np.eye(n)) @ AtA)
        gcv_score = (residual**2) / (n - trace)**2
        gcv_scores.append(gcv_score)

    best_idx = np.argmin(gcv_scores)
    best_lambda = lambdas[best_idx]
    best_gcv = gcv_scores[best_idx]
    best_sol, _, _, _, _ = regularized_inversion(A, b, [best_lambda*Identity(n)])
    return best_lambda, best_gcv, best_sol

# GCV for TV regularization
def gcv_tv(A, b, G, lambdas, niter_inner=5, niter_outer=10):
    """
    Generalized Cross-Validation for Total Variation (TV) regularization.
    """
    n = len(b)
    gcv_scores = []
    solutions = []

    for lam in lambdas:
        print(f"Running tv lam = {lam}")
        x_tv, _, _ = splitbregman(A, b, [G], niter_inner=niter_inner, niter_outer=niter_outer, mu=1.0, epsRL1s=[lam])
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
    calibrators = np.load("./calib_images.npz")
    samples = np.load("./kodim02_images.npz")
    true_image = samples['base']
    calibrator = calibrators['2_25']
    image = samples['2_25']

    print(f"Image shape: {image.shape}")
    print(f"calib shape: {calibrator.shape}")

    # Fit logistic parameters (140,250)
    means, covs = eval_fit((140, 250), calibrator)
    new_means = weighted_sampling(means, covs, 50)

    # Distribute parameters across MPI processes
    params_split = np.array_split(new_means, size)
    local_params = params_split[rank]

    for param in local_params: #(0, 255, 256)
        print(f"param : {param}")
        A = A_from_sample(np.linspace(0, 255, 256), param[0])
        A_wrapped = MatrixMult(A)
        G = Gradient(dims=(image.shape[0], image.shape[0]), kind='forward')
        tik_lambdas = np.logspace(-6, 2, 100)
        tv_lambdas = np.logspace(-4, 1, 45)
        
        # Compute optimal solutions
        tik_best_lam, tik_best_gcv, tik_best_sol = gcv_tikhonov(A, image.ravel(), tik_lambdas)
        tv_best_lam, tv_best_gcv, tv_best_sol = gcv_tv(A_wrapped, image.ravel(), G, tv_lambdas)
        
        print(f"tk_best_lam:{tik_best_lam}")
        print(f"tk_best_gcv:{tik_best_gcv}")
        print(f"tv_best_lam:{tv_best_lam}")
        print(f"tv_best_gcv:{tv_best_gcv}")

        # Save results
        np.savez(
            f'/scratch/pasfielt/deblur/{param[0]}_sols.npz',
            params=param,
            tik_lam=tik_best_lam, 
            tik_gcv=tik_best_gcv, 
            tik_sol=tik_best_sol,
            tv_lam=tv_best_lam, 
            tv_gcv=tv_best_gcv,
            tv_sol=tv_best_sol
        )
        print(f"Rank {rank}: Processed parameter {param}")
