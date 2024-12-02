import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import minimize
from params import *
import estimators
import kernels
import regularizers
import gcv

print("Import Image")
image_params = (blur_standard_deviation, SNR)

image = cv2.imread(f"prepared_images/blur_{image_params[0]}/{image_params[1]}/kodim02.png", cv2.IMREAD_GRAYSCALE)
shape = image.shape
image = image.reshape((-1,1), order='F')


# Collect curves from calibration.
print("Fit X-Axis Curve")
axis0_params = estimators.line_inference(image_params, 80, axis=0)
axis0_domain = np.linspace(0, shape[0], shape[0])
axis0_range = estimators.logistic_derivative(axis0_domain, axis0_params[0], axis0_params[3]).astype(np.float32)

print("Fit Y-Axis Curve")
axis1_params = estimators.line_inference(image_params, 80, axis=1)
axis1_domain = np.linspace(0, shape[1], shape[1])
axis1_range = estimators.logistic_derivative(axis1_domain, axis1_params[0], axis1_params[3]).astype(np.float32)

print("Create X Kernel")
Ax = kernels.one_dim_blur_kernel(axis0_range)
print("Create Y Kernel")
Ay = kernels.one_dim_blur_kernel(axis1_range)

print("Start 2D Kernel")
A = kernels.two_dim_blur_kernel(Ax, Ay)
print("2D Kernel Created")

# print("Deleting old matrices")
# del Ax
# del Ay

print("GCV START AAA")
best_alpha = minimize(gcv.gcv_param_selection_tkv, 0.3, (Ax, Ay, A, image))

print("Tikhonov Regularizer Start")
deblurred = regularizers.tikhonov_regularization(image, Ax, Ay, best_alpha.x)
print("Tikhonov Regularizer End")

deblurred = deblurred.reshape(shape, order='F')

plt.imshow(deblurred)
plt.show()

