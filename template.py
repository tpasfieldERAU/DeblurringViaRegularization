import numpy as np
import scipy.stats as stats
import scipy.linalg as la
from scipy.sparse import csr_array, kron, eye_array
import matplotlib.pyplot as plt
import sys

n = 50

img = np.random.normal(127, 63, (n,n)).astype(np.int8)
img = img.reshape((-1,1))
print("image generated")

x = np.linspace(0, n, n)
y = stats.norm.pdf(x, loc=0, scale=1) 
y /= np.sum(y)
y = y.astype(np.float32)
print("kernel generated")

Ax = la.toeplitz(y)
print("toeplitz generated")
Ax = csr_array(Ax)
print("toeplitz sparsed")
Ay = Ax.copy()
A = kron(Ax,Ay)
A.eliminate_zeros()
A.sum_duplicates()
print("kronecker product complete. Displaying.")

b = A@img
b = b + np.random.normal(0, 25, b.shape)

alpha = 0.6
restored = (A@A + alpha * eye_array(n*n, dtype=np.float32)) @ img

fig, axs = plt.subplots(1, 3)
fig.set_dpi(300)
fig.set_size_inches(7,2)
axs[0].imshow(img.reshape((n,n)))
axs[1].imshow(b.reshape((n,n)))
axs[2].imshow(restored.reshape((n,n)))
for ax in axs:
    ax.axis(False)
fig.tight_layout()
plt.show()

# plt.imshow(img.reshape((n,n)))
# plt.show()

# plt.imshow(b.reshape((n,n)))
# plt.show()

# plt.imshow(restored.reshape(n,n))
# plt.show()
