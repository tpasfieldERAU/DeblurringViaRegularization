import numpy as np
import matplotlib.pyplot as plt
from params import *
import estimators

image_params = (blur_standard_deviation, SNR)
k, x0, y0, L = estimators.line_inference(image_params, 80, axis=0)

xs = np.linspace(0,128,128)
ys = estimators.logistic_function(xs, k,x0,y0,L)


plt.plot(xs,ys)
plt.plot(xs, estimators.logistic_derivate(xs, k, L))
plt.show()