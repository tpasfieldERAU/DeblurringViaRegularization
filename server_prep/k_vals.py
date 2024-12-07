import numpy as np
from scipy.optimize import curve_fit
from numpy.random import multivariate_normal
import matplotlib.pyplot as plt

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

calibrator = np.load("server_prep/calib_images.npz")
calib = calibrator['2_25']

means, covs = eval_fit((140, 250), calib)
new_means = weighted_sampling(means, covs, 25)

test = np.random.normal(means[0], np.sqrt(covs[0,0])*5, 256)
np.savez('server_prep/samples.npz', samples=np.array(test))