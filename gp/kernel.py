import numpy as np


class Kernel(object):

    def __call__(self, x, x_prime, *args, **kwargs):
        raise NotImplementedError()


class SquaredExponentialKernel(Kernel):

    def __call__(self, x, x_prime, *args, **kwargs):
        return np.exp(-0.5 * np.linalg.norm(x - x_prime) ** 2)


x = np.random.randn(5)
y = np.random.randn(5)

kernel = SquaredExponentialKernel()

print(kernel(x, y))
