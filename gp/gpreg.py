import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import pairwise_kernels

from gp.kernel import SquaredExponentialKernel


class GaussianProcessRegressor(object):

    def __init__(self, kernel_fn, noise_level):
        self.kernel_fn = kernel_fn
        self.noise_level = noise_level

    def fit(self, x_train, y_train):
        # Compute kernel mat, invert kernel mat

        self.x = x_train

        self.n_samples = x_train.shape[0]

        self.k_mat = pairwise_kernels(x_train, metric=self.kernel_fn) + self.noise_level * np.eye(self.n_samples)
        self.k_inv = np.linalg.inv(self.k_mat)

        self.alpha = self.k_inv @ y_train

        self.L = np.linalg.cholesky(self.k_mat)  # Apply on k_mat or k_inv?

        self.f = y_train

    def predict(self, x_test):
        # Can i compute those better?
        kxxstar = pairwise_kernels(x_test.reshape(-1, 1), self.x, metric=self.kernel_fn)
        kxstarx = kxxstar.T
        kxsxs = pairwise_kernels(x_test.reshape(-1, 1), metric=self.kernel_fn)

        # Great
        pred_mean = self.alpha @ kxxstar.T

        # pred_mean = kxxstar.dot(self.k_inv).dot(self.f)
        pred_cov = kxsxs - kxxstar.dot(self.k_inv).dot(kxstarx)
        # dist = multivariate_normal(mean=pred_mean, cov=pred_cov)

        return pred_mean, np.sqrt(np.diagonal(pred_cov))


# Now let's get

gp = GaussianProcessRegressor(kernel_fn=SquaredExponentialKernel(), noise_level=0.01)

n_samples = 25

x = np.random.uniform(low=-10, high=10, size=n_samples)
y = np.tanh(x)  # + 0.1 * np.random.randn(5)

x_test = np.linspace(start=-10, stop=10, num=500)
y_test = np.tanh(x_test)

print(y_test)

# One dimensional GP
gp.fit(x.reshape(-1, 1), y)
y_hat, y_std = gp.predict(x_test)

plt.scatter(x, y, label='train')
plt.scatter(np.linspace(-10, 10, 500), np.tanh(np.linspace(-10, 10, 500)), label='true')
plt.plot(x_test, y_hat, label='pred')
plt.fill_between(x_test, y_hat, y_hat - y_std)
plt.fill_between(x_test, y_hat, y_hat + y_std)
plt.legend()
plt.show()

fig = plt.figure()

# How can we sample from the posterior distribution?
