import numpy as np
from pylops.signalprocessing import Convolve2D
from pylops import Gradient, Diagonal
from pylops.optimization.sparsity import splitbregman
from scipy.signal.windows import gaussian as siggaussian
from mpi4py import MPI

def gaussian_kernel(n, std, normalised=False):
    '''
    Generates a n x n matrix with a centered gaussian 
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    gaussian1D = siggaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D




def SplitBregmanTV(lambd):
    # Run Split Bregman
    # lambd = 0.12  # Regularization parameter
    gamma = 1.0  # Bregman parameter
    tol = 1e-3   # Tolerance for convergence
    max_iter = 100 # Maximum iterations

    solution, _, _ = splitbregman(
        Op = blur_op,
        y = blurred_image.flatten(),
        RegsL1=[gradient_op],
        epsRL1s=[lambd],
        niter_outer=max_iter,
        tol=tol,
        niter_inner=25,
        mu=gamma,
        show=True
    )

    # Reshape solution to image
    deblurred_image = solution.reshape(shape)

    # Clip to valid range and convert to uint8
    deblurred_image = np.clip(deblurred_image, 0, 255).astype(np.uint8)

    return deblurred_image


if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    input_image = np.load("kodim02_images.npz")['base']
    blurred_image = np.load("kodim02_images.npz")['2_5']

    shape = (256, 256)

    
    kernel_sigma = 2
    kernel_size = 64
    # Define operators
    blur_op = Convolve2D((shape[0], shape[1]), gaussian_kernel(kernel_size, kernel_sigma, normalised=True), offset=(kernel_size//2, kernel_size//2), method='fft', dtype='float32')
    gradient_op = Gradient(shape, edge=True, kind='forward', dtype='float32')

    solutions = []
    lambdas = np.logspace(-3,1,64)

    values_split = None
    if rank==0:
        values_split = np.array_split(lambdas, size)

    local_values = comm.scatter(values_split, root=0)

    local_results = {val: SplitBregmanTV(val) for val in local_values}

    gathered_results = comm.gather(local_results, root=0)

    if rank==0:
        results = {}
        for local_dict in gathered_results:
            results.update(local_dict)
        np.savez("TV_Batch.npz", **{str(k): v for k,v in results.items()})
