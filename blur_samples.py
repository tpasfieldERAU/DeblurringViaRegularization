import numpy as np
from scipy.sparse import kron
from scipy.stats import norm
import scipy.linalg as la
import cv2
import os

from params import *

std_devs = [1.0, 2.0, 4.0]
snrs = [5.0, 10.0, 15.0]

kodim02 = cv2.imread('sample_images/kodim02.png', cv2.IMREAD_GRAYSCALE)
# kodim08 = cv2.imread('sample_images/kodim08.png', cv2.IMREAD_GRAYSCALE)
print("IM READ")


if rescale_image:
    divisor = 1/image_scale
    kodim02 = cv2.resize(kodim02, dsize = ( int(kodim02.shape[1]//divisor), int(kodim02.shape[0]//divisor) ) )
    # kodim08 = cv2.resize(kodim08, dsize = ( int(kodim08.shape[1]//divisor), int(kodim08.shape[0]//divisor) ) )
    # print("IM SCALE")

print("RESCALED")

shape = kodim02.shape[0]
if force_square:
    kodim02 = kodim02[:, 0:shape]
    # kodim08 = kodim08[:, kodim08.shape[1]-shape:]
    # print("IM SQUARE")

print("RESHAPED")

calibrator = np.zeros((shape, shape), dtype=np.uint8)
calibrator[shape//2:, shape//2:] = 255
calibrator[:shape//2, :shape//2] = 255
#print("CALIBRATOR INIT")

print("CALIBRATOR GENERATED")

kodim02 = kodim02.reshape((-1,1), order='F')
# kodim08 = kodim08.reshape((-1,1), order='F')
calibrator = calibrator.reshape((-1,1), order='F')
# print("RESHAPING")

print("VECTORS CREATED")

for stddev in std_devs:
    x = np.linspace(0, shape, shape)
    psf = norm.pdf(x, loc=0, scale=stddev)
    psf /= np.sum(psf) * 2
    A = la.toeplitz(psf).astype(np.float32)
    print("1D KERNEL CREATED")
    A = kron(A,A).astype(np.float32)
    print("2D KERNEL CREATED")
    #print("BLURRING MATRIX INIT")

    b_kodim02 = A @ kodim02
    print("IMAGE BLURRED")
    # print("kodim02 BLUR")
    # b_kodim08 = A @ kodim08
    # print("kodim08 BLUR")
    b_calibrator = A @ calibrator
    print("CALIBRATOR BLURRED")
    # print("calibrator BLUR")

    del A
    del psf
    del x
    print("LARGE MATRIX DELETION")

    b_kodim02 = b_kodim02.reshape((shape,shape), order='F')
    # b_kodim08 = b_kodim08.reshape((shape,shape), order='F')
    b_calibrator = b_calibrator.reshape((shape,shape), order='F')
    # print("SHAPE RESTORATION")


    for nr in snrs:
        noise_scale_1 = np.mean(b_kodim02) / nr
        # noise_scale_2 = np.mean(b_kodim08) / nr
        noise_scale_3 = np.mean(b_calibrator) / nr
        n1_generator = np.random.default_rng(seed=2)
        n2_generator = np.random.default_rng(seed=8)
        n3_generator = np.random.default_rng(seed=42)

        noise1 = n1_generator.normal(loc=0, scale=noise_scale_1, size=(shape,shape))
        # noise2 = n2_generator.normal(loc=0, scale=noise_scale_2, size=(shape,shape))
        noise3 = n3_generator.normal(loc=0, scale=noise_scale_3, size=(shape,shape))

        p_kodim02 = b_kodim02 + noise1
        # p_kodim08 = b_kodim08 + noise2
        p_calibrator = b_calibrator + noise3

        p_kodim02[p_kodim02 < 0] = 0
        # p_kodim08[p_kodim08 < 0] = 0
        p_calibrator[p_calibrator < 0] = 0

        p_kodim02[p_kodim02 > 255] = 255
        # p_kodim08[p_kodim08 > 255] = 255
        p_calibrator[p_calibrator > 255] = 255

        p_kodim02 = p_kodim02.astype(np.uint8)
        # p_kodim08 = p_kodim08.astype(np.uint8)
        p_calibrator = p_calibrator.astype(np.uint8)

        try:
            os.mkdir(f"prepared_images/blur_{stddev}")
        except FileExistsError:
            pass

        try:
            os.mkdir(f"prepared_images/blur_{stddev}/{nr}")
        except FileExistsError:
            pass

        cv2.imwrite(f"prepared_images/blur_{stddev}/{nr}/kodim02.png", p_kodim02[14:114, 14:114])
        # cv2.imwrite(f"prepared_images/blur_{stddev}/{nr}/kodim08.png", p_kodim08[14:114, 14:114])
        cv2.imwrite(f"prepared_images/blur_{stddev}/{nr}/calibrator.png", p_calibrator[14:114, 14:114])