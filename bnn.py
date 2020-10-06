from abc import ABC

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class ProbabilisticLayer(ABC):

    def __call__(self, x, inference=False):
        pass


class BayesianLinear(ProbabilisticLayer):

    def __init__(self, input_shape, hidden_units, sigma1=1.5, sigma2=0.1):
        initializer_sigma = 0.6164414

        self.hidden_shape = (input_shape, hidden_units)
        self.hidden_units = hidden_units

        # Define trainable parameter
        self.kernel_mu = tf.Variable(tf.random.normal(shape=(input_shape, hidden_units), mean=0.0, stddev=initializer_sigma), name='kernel_mu')
        self.kernel_rho = tf.Variable(tf.zeros(shape=(input_shape, hidden_units)), name='kernel_rho')

        self.bias_mu = tf.Variable(tf.random.normal(shape=[hidden_units], mean=0.0, stddev=initializer_sigma), name='bias_mu')
        self.bias_rho = tf.Variable(tf.zeros(shape=[hidden_units]), name='bias_rho')

        self.prior_bias = tfd.Independent(tfd.Mixture(
            cat=tfd.Categorical(probs=np.tile([0.5, 0.5], (hidden_units, 1))),
            components=[
                tfd.Normal(loc=0.0, scale=tf.constant(sigma1, shape=(hidden_units))),
                tfd.Normal(loc=0.0, scale=tf.constant(sigma2, shape=(hidden_units))),
            ]), reinterpreted_batch_ndims=1)

        self.prior_kernel = tfd.Independent(tfd.Mixture(
            cat=tfd.Categorical(probs=np.tile([0.5, 0.5], reps=(input_shape, hidden_units, 1))),
            components=[
                tfd.Normal(loc=0.0, scale=tf.constant(sigma1, shape=(input_shape, hidden_units))),
                tfd.Normal(loc=0.0, scale=tf.constant(sigma2, shape=(input_shape, hidden_units))),
            ]), reinterpreted_batch_ndims=2)

        self.posterior_kernel = tfd.Independent(tfp.distributions.Normal(loc=self.kernel_mu, scale=tf.math.softplus(self.kernel_rho)), reinterpreted_batch_ndims=2)
        self.posterior_bias = tfd.Independent(tfp.distributions.Normal(loc=self.bias_mu, scale=tf.math.softplus(self.bias_rho)), reinterpreted_batch_ndims=1)

        self.log_prior = 0
        self.log_posterior = 0

    def get_trainable_vars(self):
        return [self.kernel_mu, self.kernel_rho, self.bias_mu, self.bias_rho]

    def _reparametrize(self, mu, std, eps):
        return mu + tf.math.softplus(std) * eps

    def __call__(self, x, inference=False):
        if inference:
            return tf.matmul(x, self.kernel_mu) + self.bias_mu

        W = self._reparametrize(self.kernel_mu, self.kernel_rho, tf.random.normal(shape=self.hidden_shape))
        b = self._reparametrize(self.bias_mu, self.bias_rho, tf.random.normal(shape=[self.hidden_units]))

        prior_log_prob_w = self.prior_kernel.log_prob(W)
        posterior_log_prob_w = self.posterior_kernel.log_prob(W)

        prior_log_prob_b = self.prior_bias.log_prob(b)
        posterior_log_prob_b = self.posterior_bias.log_prob(b)

        self.log_prior = prior_log_prob_w + prior_log_prob_b
        self.log_posterior = posterior_log_prob_w + posterior_log_prob_b

        return tf.matmul(x, W) + b


class BNN(object):

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x, inference=False):
        z = x
        for layer in self.layers:
            if isinstance(layer, ProbabilisticLayer):
                z = layer(z, inference=inference)
            else:
                z = layer(z)

        return z

    def log_priors(self):
        priors = 0.0
        for layer in self.layers:
            if isinstance(layer, ProbabilisticLayer):
                priors += layer.log_prior

        return priors

    def log_posteriors(self):
        posteriors = 0.0
        for layer in self.layers:
            if isinstance(layer, ProbabilisticLayer):
                posteriors += layer.log_posterior

        return posteriors

    def get_trainable_vars(self):
        trainable_vars = []
        for layer in self.layers:
            if isinstance(layer, ProbabilisticLayer):
                trainable_vars += layer.get_trainable_vars()

        return trainable_vars
