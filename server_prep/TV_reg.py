import numpy as np
from pylops.signalprocessing import Convolve2D
from pylops import Gradient
from pylops.optimization.sparsity import splitbregman
from scipy.signal.windows import gaussian as siggaussian
from scipy.ndimage import uniform_filter
from mpi4py import MPI

def gaussian_kernel(n, std, normalized=False):
    gaussian1D = siggaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalized:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D

def mse(true_img, sol_img):
    diff = true_img - sol_img
    sdiff = np.square(diff)
    return np.sum(sdiff)

def ssim(image1, image2, data_range=1.0, window_size=7):
    # Constants for numerical stability
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Ensure inputs are float arrays
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    # Sliding window mean
    mu1 = uniform_filter(image1, size=window_size)
    mu2 = uniform_filter(image2, size=window_size)

    # Sliding window variance and covariance
    sigma1_sq = uniform_filter(image1 ** 2, size=window_size) - mu1 ** 2
    sigma2_sq = uniform_filter(image2 ** 2, size=window_size) - mu2 ** 2
    sigma12 = uniform_filter(image1 * image2, size=window_size) - mu1 * mu2

    # SSIM formula
    numerator1 = 2 * mu1 * mu2 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1 ** 2 + mu2 ** 2 + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
    return np.mean(ssim_map)  # Return the mean SSIM over all windows


def SplitBregmanTV(sigma):
    print(f"Rank {rank}: Deblurring with std={sigma}")
    kernel_size = 64
    # Define operators
    blur_op = Convolve2D((shape[0], shape[1]), gaussian_kernel(kernel_size, sigma, normalized=True), offset=(kernel_size//2, kernel_size//2), method='fft', dtype='float32')
    gradient_op = Gradient(shape, edge=True, kind='forward', dtype='float32')
    
    lambdas = np.logspace(-2.5, 1, 42)
    # Run Split Bregman
    # lambd = 0.12  # Regularization parameter
    gamma = 1.0  # Bregman parameter
    tol = 1e-3   # Tolerance for convergence
    max_iter = 4 # Maximum iterations

    best_sol = None
    lowest_mse = float('inf')

    iterator = 0
    for lambd in lambdas:
        if iterator % 6 == 0: print(f"Rank {rank}: {sigma} -- {iterator} values complete.")
        solution, _, _ = splitbregman(
            Op = blur_op,
            y = blurred_image.flatten(),
            RegsL1=[gradient_op],
            epsRL1s=[lambd],
            niter_outer=max_iter,
            tol=tol,
            niter_inner=4,
            mu=gamma
        )

        # Reshape solution to image
        deblurred_image = solution.reshape(shape)

        # Clip to valid range and convert to uint8
        deblurred_image = np.clip(deblurred_image, 0, 255).astype(np.uint8)
        err = mse(input_image, deblurred_image)
        if err < lowest_mse:
            lowest_mse = err
            best_sol = deblurred_image
        iterator += 1

    return best_sol

def SplitBregmanTV_SSIM(sigma):
    print(f"Rank {rank}: Deblurring with std={sigma}")
    kernel_size = 128
    # Define operators
    blur_op = Convolve2D((shape[0], shape[1]), gaussian_kernel(kernel_size, sigma, normalized=True), offset=(kernel_size//2, kernel_size//2), method='fft', dtype='float32')
    gradient_op = Gradient(shape, edge=True, kind='forward', dtype='float32')
    
    lambdas = np.logspace(-2.5, 1, 42)
    # Run Split Bregman
    # lambd = 0.12  # Regularization parameter
    gamma = 1.0  # Bregman parameter
    tol = 1e-3   # Tolerance for convergence
    max_iter = 100 # Maximum iterations

    best_sol = None
    best_ssim = -float('inf')

    iterator = 0
    for lambd in lambdas:
        if iterator % 6 == 0: print(f"Rank {rank}: {sigma} -- {iterator} values complete.")
        solution, _, _ = splitbregman(
            Op = blur_op,
            y = blurred_image.flatten(),
            RegsL1=[gradient_op],
            epsRL1s=[lambd],
            niter_outer=max_iter,
            tol=tol,
            niter_inner=20,
            mu=gamma
        )

        # Reshape solution to image
        deblurred_image = solution.reshape(shape)

        # Clip to valid range and convert to uint8
        deblurred_image = np.clip(deblurred_image, 0, 255).astype(np.uint8)
        local_ssim = ssim(input_image, deblurred_image)
        if local_ssim > best_ssim:
            best_ssim = local_ssim
            best_sol = deblurred_image
        iterator += 1

    print(f"Rank {rank}: {sigma} calculation complete.")
    return best_sol


if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    input_image = np.load("/scratch/pasfielt/deblur/kodim02_images.npz")['base']
    blurred_image = np.load("/scratch/pasfielt/deblur/kodim02_images.npz")['2_5']
    
    samples = None
    values_split = None
    if rank == 0:
        samples = np.load("/scratch/pasfielt/deblur/samples.npz")['samples']
        values_split = np.array_split(samples, size)

    shape = (256, 256)

    local_values = comm.scatter(values_split, root=0)

    local_results = {val: SplitBregmanTV_SSIM(val) for val in local_values}

    gathered_results = comm.gather(local_results, root=0)

    if rank==0:
        results = {}
        for local_dict in gathered_results:
            results.update(local_dict)
        np.savez("/scratch/pasfielt/deblur/VEGA_SSIM_TV_Batch.npz", **{str(k): v for k,v in results.items()})
    