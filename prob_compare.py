import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from tensorflow_probability.python.layers import DenseVariational


def prior_trainable(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=tf.zeros(n), scale=1),
            reinterpreted_batch_ndims=1)),
    ])


def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    n = kernel_size + bias_size
    c = np.log(np.expm1(1.))
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.Independent(
            tfd.Normal(loc=t[..., :n],
                       scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
            reinterpreted_batch_ndims=1)),
    ])


tfd = tfp.distributions
model = tf.keras.Sequential()
model.add(DenseVariational(15, posterior_mean_field, prior_trainable, activation='relu'))
model.add(DenseVariational(1, posterior_mean_field, prior_trainable, ))


def neg_log_likelihood(y_obs, y_pred, sigma=1.0):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return tf.reduce_sum(-dist.log_prob(y_obs))


x, y = load_boston(return_X_y=True)

x = MinMaxScaler().fit_transform(x)

x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.float32)

tf_data = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(100).batch(32)
model.compile(loss=neg_log_likelihood, optimizer=tf.keras.optimizers.Adam(lr=0.08), metrics=['mse'])
model.fit(tf_data, batch_size=32, epochs=1500, verbose=1)
