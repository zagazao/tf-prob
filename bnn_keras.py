import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras import initializers

tfd = tfp.distributions


class BayesianLinear(keras.layers.Layer):

    def __init__(self, units, activation=keras.activations.linear, sigma1=1.5, sigma2=0.1, kl_weight=1):
        super(BayesianLinear, self).__init__()

        self.units = units

        self.activation = activation

        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.initializer_sigma = 0.6164414

        self.kl_weight = kl_weight

    def build(self, input_shape):
        self.hidden_shape = (input_shape[-1], self.units)
        self.non_batch = input_shape[-1]
        # Define trainable parameter
        self.kernel_mu = self.add_weight(shape=self.hidden_shape,
                                         initializer=initializers.RandomNormal(mean=0.0, stddev=self.initializer_sigma),
                                         trainable=True,
                                         name="kernel_mu")

        self.bias_mu = self.add_weight(shape=[self.units],
                                       initializer=initializers.RandomNormal(mean=0.0, stddev=self.initializer_sigma),
                                       trainable=True,
                                       name="bias_mu")

        self.kernel_rho = self.add_weight(shape=self.hidden_shape,
                                          initializer=initializers.Zeros(),
                                          trainable=True,
                                          name="kernel_rho")

        self.bias_rho = self.add_weight(shape=[self.units],
                                        initializer=initializers.Zeros(),
                                        trainable=True,
                                        name="bias_rho")

        self.prior_bias = tfd.Independent(tfd.Mixture(
            cat=tfd.Categorical(probs=np.tile([0.5, 0.5], (self.units, 1))),
            components=[
                tfd.Normal(loc=0.0, scale=tf.constant(self.sigma1, shape=(self.units))),
                tfd.Normal(loc=0.0, scale=tf.constant(self.sigma2, shape=(self.units))),
            ]), reinterpreted_batch_ndims=1)

        self.prior_kernel = tfd.Independent(tfd.Mixture(
            cat=tfd.Categorical(probs=np.tile([0.5, 0.5], reps=(self.non_batch, self.units, 1))),
            components=[
                tfd.Normal(loc=0.0, scale=tf.constant(self.sigma1, shape=(self.non_batch, self.units))),
                tfd.Normal(loc=0.0, scale=tf.constant(self.sigma2, shape=(self.non_batch, self.units))),
            ]), reinterpreted_batch_ndims=2)

        self.posterior_kernel = tfd.Independent(tfp.distributions.Normal(loc=self.kernel_mu, scale=tf.math.softplus(self.kernel_rho)), reinterpreted_batch_ndims=2)
        self.posterior_bias = tfd.Independent(tfp.distributions.Normal(loc=self.bias_mu, scale=tf.math.softplus(self.bias_rho)), reinterpreted_batch_ndims=1)

    def _reparametrize(self, mu, std, eps):
        return mu + tf.math.softplus(std) * eps

    def call(self, x, training=None):
        if training:
            W = self._reparametrize(self.kernel_mu, self.kernel_rho, tf.random.normal(shape=self.hidden_shape))
            b = self._reparametrize(self.bias_mu, self.bias_rho, tf.random.normal(shape=[self.units]))

            log_prior = self.prior_kernel.log_prob(W) + self.prior_bias.log_prob(b)
            log_posterior = self.posterior_kernel.log_prob(W) + self.posterior_bias.log_prob(b)

            self.add_loss(self.kl_weight * (log_posterior - log_prior))

            return self.activation(tf.matmul(x, W) + b)

        return self.activation(tf.matmul(x, self.kernel_mu) + self.bias_mu)

    def get_config(self):
        config = super(BayesianLinear, self).get_config()
        config.update({"units": self.units, })


def neg_log_likelihood(y_obs, y_pred):
    # if np.isclose(tf.reduce_sum(y_pred, axis=2), 1).all():
    #     dist = tfd.Categorical(probs=y_pred)
    # else:
    # print(y_pred)

    # y_pred is softmax_out
    # y_obs = 5
    # tf.math.log(y_preds[y_obs])

    from tensorflow.keras.losses import SparseCategoricalCrossentropy
    loss_obj = SparseCategoricalCrossentropy()
    result = loss_obj(y_obs, y_pred)

    dist = tfd.Categorical(logits=y_pred)
    return tf.reduce_sum(-dist.log_prob(y_obs))
