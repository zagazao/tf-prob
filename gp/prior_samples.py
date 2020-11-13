import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_kernels

from gp.kernel import SquaredExponentialKernel

x = np.random.uniform(low=-5, high=5, size=100)  # .reshape(-1, 1)
k = SquaredExponentialKernel()

k_mat = pairwise_kernels(x.reshape(-1, 1), metric=k)

n_samples = 100

print(k_mat)

mv_normal = np.random.multivariate_normal(mean=np.zeros(100), cov=k_mat, size=n_samples)

sort_idx = np.argsort(x)
x_sorted = x.flatten()[sort_idx]

for i in range(n_samples):
    y = mv_normal[i].flatten()[sort_idx]
    plt.plot(x_sorted, y)
plt.show()

print(mv_normal)
