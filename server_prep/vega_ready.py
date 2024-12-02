import numpy as np
import mpi4py as MPI
from pylops import Gradient, Identity
from pylops.optimization.leastsquares import RegularizedInversion
from pylops.optimization.basic import cg
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


def logistic_function(xs, k, x0):
    #k, x0, y0, L = params
    out = 255/(1 + np.exp(-k * (xs-x0))) + 127.0
    out = out - np.min(out)
    out = 255*out/np.max(out)
    return out


def logistic_derivative(xs, k):
    numerator = 255*k*np.exp(k*(xs))
    denominator = np.power(np.exp(k*(xs)) + 1, 2)
    return numerator/denominator


def eval_fit(index_range, calibrator):
    num_lines = index_range[1] - index_range[0]
    lines = calibrator[index_range[0]:index_range[1], :]
    xs0 = np.linspace(0, calibrator.shape[0], calibrator.shape[0])
    xs = xs0
    for _ in range(num_lines-1):
        xs = np.concatenate((xs,xs0))
    p0 = (1.0, calibrator.shape[0]//2)

    means, covs = curve_fit(logistic_function, xs, lines.ravel(), p0)
    return means, covs


def weighted_sampling(means, covs, n_samples):
    samples = multivariate_normal(mean=means, cov=covs, size=n_samples)
    return samples


def A_from_sample(xs, k):
    curve = logistic_derivative(xs, k)
    a = toeplitz(curve)
    A = kron(a,a)
    return A


def gcv_tikhonov(A, b, lambdas):
    """
    Perform Generalized Cross-Validation (GCV) to find the optimal lambda for Tikhonov regularization.

    Parameters:
    - A (numpy.ndarray or pylops.LinearOperator): Forward operator (e.g., blurring matrix).
    - b (numpy.ndarray): Observed data (e.g., noisy blurred image).
    - lambdas (numpy.ndarray): Array of lambda values to test.

    Returns:
    - best_lambda (float): Optimal lambda value minimizing the GCV score.
    - best_gcv (float): GCV score corresponding to the best lambda.
    """
    n = len(b)
    gcv_scores = []
    for lam in lambdas:
        # Regularized solution
        x_reg = RegularizedInversion(A, b, [Identity(n)], [lam])
        
        # Compute residual norm
        residual = np.linalg.norm(A @ x_reg - b)
        
        # Compute effective degrees of freedom: tr((A^T A + λI)^(-1) A^T A)
        AtA = A.T @ A if isinstance(A, np.ndarray) else A.T * A
        trace = np.trace(np.linalg.inv(AtA + lam * np.eye(n)) @ AtA)
        
        # GCV score
        gcv_score = (residual**2) / (n - trace)**2
        gcv_scores.append(gcv_score)

    # Find the lambda with the minimum GCV score
    best_idx = np.argmin(gcv_scores)
    best_lambda = lambdas[best_idx]
    best_gcv = gcv_scores[best_idx]

    best_sol = RegularizedInversion(A, b, [Identity(n)], [best_lambda])
    return best_lambda, best_gcv, best_sol


def gcv_tv(A, b, G, lambdas, niter_inner=5, niter_outer=20, mu=1.0):
    """
    Perform Generalized Cross-Validation (GCV) to find the optimal lambda for TV regularization.
    
    Parameters:
    - A (numpy.ndarray or pylops.LinearOperator): Forward operator (e.g., blurring matrix).
    - b (numpy.ndarray): Observed data (e.g., noisy blurred image).
    - G (pylops.LinearOperator): Gradient operator for TV regularization.
    - lambdas (numpy.ndarray): Array of lambda values to test.
    - niter_inner, niter_outer: Split Bregman iteration parameters.
    - mu (float): Penalty weight in Split Bregman method.
    
    Returns:
    - best_lambda (float): Optimal lambda value minimizing the GCV score.
    - best_gcv (float): GCV score corresponding to the best lambda.
    - best_solution (numpy.ndarray): Solution corresponding to the best lambda.
    """
    n = len(b)
    gcv_scores = []
    solutions = []

    for lam in lambdas:
        # Solve TV-regularized problem using Split Bregman
        x_tv, _ = SplitBregman(A, b, G, niter_inner=niter_inner, niter_outer=niter_outer, mu=mu, epsRL1s=[lam])
        solutions.append(x_tv)

        # Compute residual norm
        residual = np.linalg.norm(A @ x_tv - b)

        # Approximate effective degrees of freedom (trace of projection matrix)
        # For TV, an approximation is often used since explicit trace computation is hard
        trace_approx = n - np.linalg.norm(G @ x_tv, ord=1)

        # GCV score
        gcv_score = (residual**2) / (n - trace_approx)**2
        gcv_scores.append(gcv_score)

    # Find the lambda with the minimum GCV score
    best_idx = np.argmin(gcv_scores)
    best_lambda = lambdas[best_idx]
    best_gcv = gcv_scores[best_idx]
    best_solution = solutions[best_idx]

    return best_lambda, best_gcv, best_solution


if __name__ == "__main__":
    calibrators = np.load("./server_prep/calib_images.npz")
    samples = np.load("./server_prep/kodim02_images.npz")

    true_image = samples['base']

    calibrator = calibrators['2_25']
    image = samples['2_25']
    
    n = 256
    means, covs = eval_fit((140,250), calibrator)

    new_means = weighted_sampling(means, covs, 50)
    print(means)
    print(covs)
    
    print(new_means)

    plt.imshow(calibrator[140:250,:])
    plt.show()

    num_lines = 250-140
    xs0 = np.linspace(0, calibrator.shape[0], calibrator.shape[0])
    xs = xs0
    for _ in range(num_lines-1):
        xs = np.concatenate((xs,xs0))
    
    params_split = np.array_split(new_means, size)
    local_params = params_split[rank]

    for param in local_params:
        A = A_from_sample(np.linspace(0,255,256), param[0])
        G = Gradient(dims=(n,n), kind='forward')
        tik_lambdas = np.logspace(-6, 2, 200)
        tv_lambdas = np.logspace(-4,1,60)
        tik_best_lam, tik_best_gcv, tik_best_sol = gcv_tikhonov(A,image.ravel(),tik_lambdas)
        tv_best_lam, tv_best_gcv, tv_best_sol = gcv_tv(A, image.ravel(), G, tv_lambdas)
        np.savez(f'{param[0]}_sols.npz', {"params": param, "tik":[tik_best_lam, tik_best_gcv, tik_best_sol], "tv":[tv_best_lam, tv_best_gcv, tv_best_sol]})
        
