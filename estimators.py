import numpy as np
from scipy.stats import norm
import scipy.linalg as la
from scipy.optimize import curve_fit
from params import *
import cv2

def logistic_function(xs, k, x0, y0, L):
    #k, x0, y0, L = params
    return L/(1 + np.exp(-k * (xs-x0))) + y0

def logistic_derivative(xs, k, L):
    numerator = L*k*np.exp(k*(xs))
    denominator = np.power(np.exp(k*(xs)) + 1, 2)
    return numerator/denominator

def line_inference(image_params, line_index, axis=0):
    std, snr = image_params
    image = cv2.imread(f"prepared_images/blur_{std}/{snr}/calibrator.png", cv2.IMREAD_GRAYSCALE)
    shape = image.shape
    line=[]
    if axis:
        line=image[:, line_index]
    else:
        line=image[line_index, :]

    try:
        assert len(line) == shape[axis]
    except AssertionError:
        print("estimators.py : Error encountered in dimension assertion.")
        return []
    
    xs = np.linspace(0, shape[axis], shape[axis])
    p0 = (1.0, shape[axis]//2, 127.0, 255.0)

    params = curve_fit(logistic_function, xs, line, p0)

    return params[0]



