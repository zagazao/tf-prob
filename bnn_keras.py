import datetime

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import initializers

tfd = tfp.distributions


class BayesianLinear(keras.layers.Layer):

    def __init__(self, units, activation=keras.activations.linear, sigma1=0.75, sigma2=0.1, kl_weight=1):
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
                                          initializer=initializers.Constant(-3),
                                          trainable=True,
                                          name="kernel_rho")

        self.bias_rho = self.add_weight(shape=[self.units],
                                        initializer=initializers.Constant(-3),
                                        trainable=True,
                                        name="bias_rho")

        self.prior_bias = tfd.Independent(tfd.Mixture(
            cat=tfd.Categorical(probs=np.tile([0.25, 0.75], (self.units, 1))),
            components=[
                tfd.Normal(loc=0.0, scale=tf.constant(self.sigma1, shape=(self.units))),
                tfd.Normal(loc=0.0, scale=tf.constant(self.sigma2, shape=(self.units))),
            ]), reinterpreted_batch_ndims=1)

        self.prior_kernel = tfd.Independent(tfd.Mixture(
            cat=tfd.Categorical(probs=np.tile([0.25, 0.75], reps=(self.non_batch, self.units, 1))),
            components=[
                tfd.Normal(loc=0.0, scale=tf.constant(self.sigma1, shape=(self.non_batch, self.units))),
                tfd.Normal(loc=0.0, scale=tf.constant(self.sigma2, shape=(self.non_batch, self.units))),
            ]), reinterpreted_batch_ndims=2)

        self.posterior_kernel = tfd.Independent(tfp.distributions.Normal(loc=self.kernel_mu, scale=tf.math.softplus(self.kernel_rho)), reinterpreted_batch_ndims=2)
        self.posterior_bias = tfd.Independent(tfp.distributions.Normal(loc=self.bias_mu, scale=tf.math.softplus(self.bias_rho)), reinterpreted_batch_ndims=1)

        return self

    def _reparametrize(self, mu, std, eps):
        return mu + tf.math.softplus(std) * eps

    # def _kl_normal(self, W):
    #     # a = variational posterior
    #     # b = prior()
    #     b_scale = tf.convert_to_tensor(self.prior_kernel.scale)  # We'll read it thrice.
    #     diff_log_scale = tf.math.log(a.scale) - tf.math.log(b_scale)
    #     return (
    #             0.5 * tf.math.squared_difference(a.loc / b_scale, b.loc / b_scale) +
    #             0.5 * tf.math.expm1(2. * diff_log_scale) -
    #             diff_log_scale)

    def call(self, x, training=None):
        if training:
            W = self._reparametrize(self.kernel_mu, self.kernel_rho, tf.random.normal(shape=self.hidden_shape))
            b = self._reparametrize(self.bias_mu, self.bias_rho, tf.random.normal(shape=[self.units]))

            log_prior = self.prior_kernel.log_prob(W) + self.prior_bias.log_prob(b)
            log_posterior = self.posterior_kernel.log_prob(W) + self.posterior_bias.log_prob(b)

            # kl = kl_divergence(self.posterior_kernel, self.prior_bias)
            # print(kl)

            self.add_loss(self.kl_weight * (log_posterior - log_prior))

            return self.activation(tf.matmul(x, W) + b)

        return self.activation(tf.matmul(x, self.kernel_mu) + self.bias_mu)

    def get_config(self):
        config = super(BayesianLinear, self).get_config()
        config.update({"units": self.units, })


class VariationalLocalReparam(tf.keras.layers.Layer):

    def __init__(self, units, activation=tf.keras.activations.linear, kl_weight=1, sigma1=1.5, sigma2=0.1, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self.units = units

        self.activation = activation

        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.initializer_sigma = 0.6164414

        self.kl_weight = kl_weight

    def build(self, input_shape):
        self.hidden_shape = (input_shape[-1], self.units)
        self.non_batch = input_shape[-1]

        # Put parameters over x

        # Define trainable parameter
        self.kernel_mu = self.add_weight(shape=self.hidden_shape,
                                         initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.initializer_sigma),
                                         trainable=self.trainable,
                                         name='kernel_mu')

        self.kernel_rho = self.add_weight(shape=self.hidden_shape,
                                          initializer=tf.keras.initializers.Zeros(),
                                          trainable=self.trainable,
                                          name='kernel_rho')

        self.bias_mu = self.add_weight(shape=[self.units],
                                       initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=self.initializer_sigma),
                                       trainable=self.trainable,
                                       name='bias_mu')

        self.bias_rho = self.add_weight(shape=[self.units],
                                        initializer=tf.keras.initializers.Zeros(),
                                        trainable=self.trainable,
                                        name='bias_rho')

        self.prior_activation = tfd.Independent(tfd.Normal(loc=tf.zeros(self.units), scale=1.0), reinterpreted_batch_ndims=1)
        self.posterior_activation = None

        return self

    def call(self, inputs, training=None, **kwargs):
        if training is None:
            return tf.matmul(inputs, self.kernel_mu) + self.bias_mu
        # Different reparametrization...
        # I need gamma_mj
        # I need:
        #  gamma_mj (mean) \sum_{i...hidden_dim} input_m_i mean_i_j
        #  delta_mj (mean) \sum_{i...hidden_dim} input_m_i**2 rho_i_j**2
        # m = x.shape[0]  # batch_size
        # out_dim = self.bias_mu.shape[0]

        # Shape = (batch_size, hidden_dim)
        gamma_l = tf.matmul(inputs, self.kernel_mu) + self.bias_mu
        delta_l = tf.matmul(tf.pow(inputs, 2), tf.pow(tf.math.softplus(self.kernel_rho), 2)) + tf.pow(tf.math.softplus(self.bias_rho), 2)

        self.posterior_activation = tfd.Independent(tfd.Normal(loc=gamma_l, scale=delta_l), reinterpreted_batch_ndims=1)  # or 2 to be complety

        noise_shape = tf.shape(gamma_l)

        b = gamma_l + tf.sqrt(delta_l) * tf.random.normal(shape=noise_shape)

        log_prior = self.prior_activation.log_prob(b)
        log_posterior = self.posterior_activation.log_prob(b)

        kl_div = log_posterior - log_prior
        # TODO: This throws an error but is also extremely high... (seen in debugger before strange crash)

        # kl_exact = kl_divergence(self.posterior_activation, self.prior_activation, b)
        self.add_loss(kl_div)

        return b


def neg_log_likelihood(y_obs, y_pred):
    dist = tfd.Categorical(logits=y_pred)
    return tf.reduce_sum(-dist.log_prob(y_obs))


x, y = fetch_openml('mnist_784', version=1, return_X_y=True)

x = StandardScaler().fit_transform(x)

x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.float32)

subset_size = len(x)
train_size = 0.7
batch_size = 128
tf_data = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(100).take(subset_size)
train_data = tf_data.take(int(train_size * subset_size))
test_data = tf_data.skip(int(train_size * subset_size))

nr_testdatapoints = len(test_data)
print(len(train_data), len(test_data))

train_data = train_data.batch(batch_size)
test_data = test_data.batch(batch_size)

n_batches = len(train_data)
n_testbatches = len(test_data)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=784),
    VariationalLocalReparam(units=128, activation=tf.keras.activations.relu),
    VariationalLocalReparam(units=128, activation=tf.keras.activations.relu),
    VariationalLocalReparam(units=10),
])

model.compile(metrics=['accuracy'])
model.summary()

opt = tf.optimizers.Adam(learning_rate=0.08)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# x -> Conv -> Dense -> mu, sigma (€ R^k) reparam(mu, sigma) -> Dense -> DeConv -> x_out
# Loss = MSE(x, x_out) + KL(q(), N(0,1))
# p(W) = N(w|0,1)

# 2d idx to linear 1 d
# (i,j) i * n_batches + j

# \prod_i q_i(z_i)
#  i = N(i0-i9|mu, SIGMA)

for epochs in range(100):
    # Shuffle data after epochs...

    for batch_id, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:
            z = model(x, training=True)

            likelihood_dist = tfd.Categorical(logits=z)
            log_likelihood = likelihood_dist.log_prob(y)

            # kl_weights = tf.cast((tf.pow(2, n_batches - batch_id)) / (tf.pow(2, n_batches) - 1), tf.float32)
            kl_weights = 1 / n_batches

            model_loss = model.losses
            print(model_loss)

            loss = kl_weights * (tf.reduce_sum(model.losses)) - tf.reduce_sum(log_likelihood)  # * 49000 / 128
            # print('{:10.3f} - {:10.3f} - {:10.3f}'.format(log_likelihood.numpy().mean(), loss.numpy(), mse.numpy()))

        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss', loss, step=epochs * n_batches + batch_id)
        #     tf.summary.scalar('log_likelihood', tf.reduce_sum(log_likelihood), step=epochs * n_batches + batch_id)
        #     tf.summary.scalar('scaled_log_likelihood', tf.reduce_sum(log_likelihood) * 49000 / 128, step=epochs * n_batches + batch_id)
        #     tf.summary.scalar('log_prior', tf.reduce_sum(log_priors), step=epochs * n_batches + batch_id)
        #     tf.summary.scalar('log_posterior', tf.reduce_sum(log_posteriors), step=epochs * n_batches + batch_id)
        #
        #     # Log parameters
        #     tf.summary.histogram('layer0_kernel_mu', l1.kernel_mu, step=epochs * n_batches + batch_id)
        #     tf.summary.histogram('layer0_bias_mu', l1.bias_mu, step=epochs * n_batches + batch_id)
        #
        #     tf.summary.histogram('layer0_kernel_rho', tf.math.softplus(l1.kernel_rho), step=epochs * n_batches + batch_id)
        #     tf.summary.histogram('layer0_bias_rho', tf.math.softplus(l1.bias_rho), step=epochs * n_batches + batch_id)
        #
        #     tf.summary.histogram('layer1_kernel_mu', l2.kernel_mu, step=epochs * n_batches + batch_id)
        #     tf.summary.histogram('layer1_bias_mu', l2.bias_mu, step=epochs * n_batches + batch_id)
        #
        #     tf.summary.histogram('layer1_kernel_rho', tf.math.softplus(l2.kernel_rho), step=epochs * n_batches + batch_id)
        #     tf.summary.histogram('layer1_bias_rho', tf.math.softplus(l2.bias_rho), step=epochs * n_batches + batch_id)

        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

    train_loss, train_acc = model.evaluate(train_data, batch_size=128, verbose=False)
    test_loss, test_acc = model.evaluate(test_data, batch_size=128, verbose=False)

    print('Epoch end')
    print(train_loss, train_acc)
    print(test_loss, test_acc)
