import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

dtype = tf.float32
shape = (15, 3)

dist = tfp.distributions.Normal(tf.zeros(shape, dtype), scale=dtype.as_numpy_dtype(1))
batch_ndims = tf.size(dist.batch_shape_tensor())

ind = tfp.distributions.Independent(
    dist, reinterpreted_batch_ndims=batch_ndims)

# p(x) = p(x1) * ... p(xn)

print('dbg..')

x = np.ones(10) / 10
y = np.ones(10) / 10
y[5] *= 2
y[6] = 0
# y = np.random.randint(low=10)

assert np.sum(x) == 1.0
assert np.sum(y) == 1.0

z = tf.keras.losses.kl_divergence(x, y)

print(z)
