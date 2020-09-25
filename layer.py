import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.losses import mean_squared_error

tfd = tfp.distributions


class BayesianLinear(object):

    def __init__(self, input_shape, hidden_units, sigma1=1.5, sigma2=0.1):
        initializer_sigma = 0.6164414

        # Store shapes
        self.hidden_shape = (input_shape, hidden_units)
        self.hidden_units = hidden_units

        # Define trainable parameter
        self.kernel_mu = tf.Variable(tf.random.normal(shape=(input_shape, hidden_units), mean=0.0, stddev=initializer_sigma), name='kernel_mu')
        self.kernel_rho = tf.Variable(tf.zeros(shape=(input_shape, hidden_units)), name='kernel_rho')

        self.bias_mu = tf.Variable(tf.random.normal(shape=[hidden_units], mean=0.0, stddev=initializer_sigma), name='bias_mu')
        self.bias_rho = tf.Variable(tf.zeros(shape=[hidden_units]), name='bias_rho')

        # Specify prior as independent normal distributions..
        # self.prior_kernel = tfd.Independent(tfp.distributions.Normal(loc=tf.zeros(shape=(input_shape, hidden_units)), scale=1.0), reinterpreted_batch_ndims=2)
        # self.prior_bias = tfd.Independent(tfp.distributions.Normal(loc=tf.zeros(shape=hidden_units), scale=1.0), reinterpreted_batch_ndims=1)

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

        # Specify prior as independent normal distributions given estimated neural net parameters..
        self.posterior_kernel = tfd.Independent(tfp.distributions.Normal(loc=self.kernel_mu, scale=tf.math.softplus(self.kernel_rho)), reinterpreted_batch_ndims=2)
        self.posterior_bias = tfd.Independent(tfp.distributions.Normal(loc=self.bias_mu, scale=tf.math.softplus(self.bias_rho)), reinterpreted_batch_ndims=1)

        # Specify noise distribution...
        # self.dist_eps = tfp.distributions.Normal(loc=0.0, scale=0.1)

        self.kl_div = 0

        self.log_prior = 0
        self.log_posterior = 0

    def get_trainable_vars(self):
        return [self.kernel_mu, self.kernel_rho, self.bias_mu, self.bias_rho]

    def _reparametrize(self, mu, std, eps):
        return mu + tf.math.softplus(std) * eps

    def forward(self, x):
        # Sample W and bias...

        W = self._reparametrize(self.kernel_mu, self.kernel_rho, tf.random.normal(shape=self.hidden_shape))
        b = self._reparametrize(self.bias_mu, self.bias_rho, tf.random.normal(shape=[self.hidden_units]))

        prior_log_prob_w = self.prior_kernel.log_prob(W)
        posterior_log_prob_w = self.posterior_kernel.log_prob(W)

        prior_log_prob_b = self.prior_bias.log_prob(b)
        posterior_log_prob_b = self.posterior_bias.log_prob(b)

        self.log_prior = prior_log_prob_w + prior_log_prob_b
        self.log_posterior = posterior_log_prob_w + posterior_log_prob_b

        #
        # kl_div_kernel = kl_divergence(self.prior_kernel, self.posterior_kernel)
        # kl_div_bias = kl_divergence(self.prior_bias, self.posterior_bias)

        # self.kl_div = kl_div_kernel + kl_div_bias

        # Calculate loss... and store this somewhere???
        return tf.matmul(x, W) + b


x, y = load_boston(return_X_y=True)

x = MinMaxScaler().fit_transform(x)

x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.float32)

tf_data = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(100).batch(32)
n_batches = len(tf_data)

l1 = BayesianLinear(13, 15)
l2 = BayesianLinear(15, 1)

opt = tf.optimizers.Adam(learning_rate=0.08)

W_SAMPLES = 5

for epochs in range(100):
    # Shuffle data after epochs...
    mses = []
    tf_data = tf_data.shuffle(100)
    for batch_id, (x, y) in enumerate(tf_data):
        with tf.GradientTape() as tape:

            # Net outputs...
            # Log priors...
            # Log posteriors..
            outputs, log_priors, log_posteriors = [], [], []

            for wi in range(W_SAMPLES):
                z = l1.forward(x)
                z = tf.nn.relu(z)
                z = l2.forward(z)

                outputs.append(z)
                log_priors.append(l1.log_prior + l2.log_prior)
                log_posteriors.append(l1.log_posterior + l2.log_posterior)

            outputs = tf.convert_to_tensor(outputs)  # Shape w_samples, batch_size, 2
            log_priors = tf.convert_to_tensor(log_priors)  # shape w_samples
            log_posteriors = tf.convert_to_tensor(log_posteriors)  # shape w_samples

            # means = outputs
            means = tf.squeeze(tf.reduce_mean(outputs, axis=0), 1)
            # stddevs = tf.math.softplus(outputs[..., 1])

            mse = mean_squared_error(y, tf.reduce_mean(means, axis=0))
            mses.append(mse)

            likelihood_dist = tfd.Normal(loc=means, scale=1)
            # Vector of (w_samples, batch_size)
            log_likelihood = likelihood_dist.log_prob(y)

            # kl_weights = tf.cast((tf.pow(2, n_batches - batch_id)) / (tf.pow(2, n_batches) - 1), tf.float32)
            kl_weights = 1 / n_batches

            loss = (tf.reduce_sum(log_posteriors) - tf.reduce_sum(log_priors)) * kl_weights - tf.reduce_sum(log_likelihood)

            print('{:10.3f} - {:10.3f} - {:10.3f}'.format(log_likelihood.numpy().mean(), loss.numpy(), mse.numpy()))

        # print(np.mean(mses))
        trainable_vars = l1.get_trainable_vars() + l2.get_trainable_vars()

        grads = tape.gradient(loss, trainable_vars)
        opt.apply_gradients(zip(grads, trainable_vars))
