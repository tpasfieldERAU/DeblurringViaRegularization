import numpy as np
from scipy.stats import norm
import scipy.linalg as la
from scipy.optimize import curve_fit
import cv2

def logistic_function(xs, params):
    k, x0, y0, L = params
    return L/(1 + np.exp(-k * (xs-x0))) + y0

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



