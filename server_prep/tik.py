import numpy as np
from pylops.signalprocessing import Convolve2D
from pylops import Identity
from pylops.optimization.basic import cg
from scipy.optimize import minimize_scalar
from scipy.signal.windows import gaussian as siggaussian
from scipy.ndimage import uniform_filter

from mpi4py import MPI

def gaussian_kernel(n, std, normalized=True):
    gaussian1d = siggaussian(n, std)
    gaussian2d = np.outer(gaussian1d, gaussian1d)
    if normalized:
        gaussian2d /= (2*np.pi*(std**2))
    return gaussian2d

def cgtik(blur_op, id_op, tols=1e-4, maxiters=50):
    test, _, _ = cg(blur_op.T * blur_op + id_op, blur_op.T*blurred_image.ravel(), tol=tols, niter=maxiters, x0=blurred_image.ravel())
    return test.reshape(256,256)

def gcv(lambd, blur_op, tols=1e-4, maxiters=50):
    id_op = lambd * Identity(256**2, 256**2, dtype='float32')
    reg_op = blur_op.T * blur_op + id_op
    rhs = blur_op.T*blurred_image.ravel()
    deblurred_image,_,_ = cg(reg_op, rhs, tol=tols, niter=maxiters, x0=blurred_image.ravel())
    # deblurred_image = np.clip(deblurred_image, 0, 255).astype(np.uint8)
    residual = blur_op@deblurred_image - blurred_image.ravel()
    residual_norm = np.linalg.norm(residual)**2

    def trace_estimation():
        trace_est = 0.0
        for _ in range(10):
            v = np.random.choice([-1, 1], size=256*256).astype(np.float32)
            Av = reg_op @ v
            trace_est += np.dot(v, Av)
        return trace_est / 10

        
    trace_term = trace_estimation()
    return -residual_norm / (2*trace_term)**2


def residual(lambd, blur_op, tols=1e-4, maxiters=50):
    id_op = lambd * Identity(256**2, 256**2, dtype='float32')
    reg_op = blur_op.T * blur_op + id_op
    rhs = blur_op.T*blurred_image.ravel()
    deblurred_image,_,_ = cg(reg_op, rhs, tol=tols, niter=maxiters, x0=blurred_image.ravel())
    # deblurred_image = np.clip(deblurred_image, 0, 255).astype(np.uint8)
    residual = blur_op@deblurred_image - blurred_image.ravel()
    residual_norm = np.linalg.norm(residual)**2
    return residual_norm


def mse(lambd, blur_op, tols=1e-4, maxiters=50):
    id_op = lambd * Identity(256**2, 256**2, dtype='float32')
    reg_op = blur_op.T * blur_op + id_op
    rhs = blur_op.T*blurred_image.ravel()
    deblurred_image,_,_ = cg(reg_op, rhs, tol=tols, niter=maxiters, x0=blurred_image.ravel())
    # deblurred_image = np.clip(deblurred_image, 0, 255).astype(np.uint8)
    residual = deblurred_image.reshape(256,256) - true_image
    residual = np.square(residual)
    return np.sum(residual)

def ssim(lambd, blur_op, tols=1e-4, maxiters=50, data_range=255, window_size=7):
    id_op = lambd * Identity(256**2, 256**2, dtype='float32')
    reg_op = blur_op.T * blur_op + id_op
    rhs = blur_op.T*blurred_image.ravel()
    deblurred_image,_,_ = cg(reg_op, rhs, tol=tols, niter=maxiters, x0=blurred_image.ravel())
    deblurred_image = deblurred_image.reshape(256,256)

    # Constants for numerical stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Ensure inputs are float arrays

    # Sliding window mean
    mu1 = uniform_filter(deblurred_image, size=window_size)
    mu2 = uniform_filter(true_image, size=window_size)

    # Sliding window variance and covariance
    sigma1_sq = uniform_filter(deblurred_image ** 2, size=window_size) - mu1 ** 2
    sigma2_sq = uniform_filter(true_image ** 2, size=window_size) - mu2 ** 2
    sigma12 = uniform_filter(deblurred_image * true_image, size=window_size) - mu1 * mu2

    # SSIM formula
    numerator1 = 2 * mu1 * mu2 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1 ** 2 + mu2 ** 2 + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
    return -np.mean(ssim_map)  # Return the mean SSIM over all windows


def tik_deblur(sigma, tols=1e-4, maxiters=50, kernel_size=64, param_selector='mse'):
    blur_op = Convolve2D(image_shape, gaussian_kernel(kernel_size, sigma, normalized=True), offset=(kernel_size//2, kernel_size//2), method='fft', dtype='float32')
    match param_selector:
        case 'mse':
            best_lambd = minimize_scalar(lambda l: mse(l, blur_op), bounds=(1e-6, 10), method='bounded')
        case 'gcv':
            best_lambd = minimize_scalar(lambda l: gcv(l, blur_op), bounds=(1e-6, 10), method='bounded')
        case 'ssim':
            best_lambd = minimize_scalar(lambda l: ssim(l, blur_op), bounds=(1e-6, 10), method='bounded')
        case 'residual':
            best_lambd = minimize_scalar(lambda l: residual(l, blur_op), bounds=(1e-6, 10), method='bounded')
        case _:
            raise(IndexError("Invalid match case."))
        
    soln_id_op = best_lambd.x * Identity(256**2, dtype='float32')
    best_soln = cgtik(blur_op, soln_id_op, tols=tols, maxiters=maxiters)
    best_soln = np.clip(best_soln, 0, 255).astype(np.uint8)
    return best_soln

if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    height, width = 256, 256
    image_shape = (height, width)
    N = height*width

    images = np.load('kodim02_images.npz')
    true_image = images['base'].astype(np.float32)
    blurred_image = images['2_5'].astype(np.float32)
    
    samples = None
    values_split = None

    if rank == 0:
        samples = np.load('samples.npz')['samples'][:2]
        values_split = np.array_split(samples, size)

    local_values = comm.scatter(values_split, root=0)
    local_results = {str(val):tik_deblur(val, param_selector='mse') for val in local_values}

    gathered_results = comm.gather(local_results, root=0)

    if rank==0:
        results = {}
        for local_dict in gathered_results:
            results.update(local_dict)
        np.savez("TIK_Batch.npz", **{str(k): v for k,v in results.items()})