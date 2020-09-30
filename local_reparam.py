import datetime

import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

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

        self.prior_activation = tfd.Independent(tfd.Normal(loc=tf.zeros(self.hidden_units), scale=1.0), reinterpreted_batch_ndims=1)
        self.posterior_activation = None

        # Specify prior as independent normal distributions..
        # self.prior_activation = tfd.Independent(tfd.Mixture(
        #     cat=tfd.Categorical(probs=np.tile([0.5, 0.5], (hidden_units, 1))),
        #     components=[
        #         tfd.Normal(loc=0.0, scale=tf.constant(sigma1, shape=(hidden_units))),
        #         tfd.Normal(loc=0.0, scale=tf.constant(sigma2, shape=(hidden_units))),
        #     ]), reinterpreted_batch_ndims=1)
        #
        # self.prior_kernel = tfd.Independent(tfd.Mixture(
        #     cat=tfd.Categorical(probs=np.tile([0.5, 0.5], reps=(input_shape, hidden_units, 1))),
        #     components=[
        #         tfd.Normal(loc=0.0, scale=tf.constant(sigma1, shape=(input_shape, hidden_units))),
        #         tfd.Normal(loc=0.0, scale=tf.constant(sigma2, shape=(input_shape, hidden_units))),
        #     ]), reinterpreted_batch_ndims=2)
        #
        # # Specify prior as independent normal distributions given estimated neural net parameters..
        # self.posterior_kernel = tfd.Independent(tfp.distributions.Normal(loc=self.kernel_mu, scale=tf.math.softplus(self.kernel_rho)), reinterpreted_batch_ndims=2)
        # self.posterior_bias = tfd.Independent(tfp.distributions.Normal(loc=self.bias_mu, scale=tf.math.softplus(self.bias_rho)), reinterpreted_batch_ndims=1)

        # Specify noise distribution...
        # self.dist_eps = tfp.distributions.Normal(loc=0.0, scale=0.1)
        self.prior_log_prob_b = 0
        self.posterior_log_prob_b = 0

        self.kl_div = 0

        self.log_prior = 0
        self.log_posterior = 0

    def get_trainable_vars(self):
        return [self.kernel_mu, self.kernel_rho, self.bias_mu, self.bias_rho]

    def _reparametrize(self, mu, std, eps):
        return mu + tf.math.softplus(std) * eps

    def forward(self, x, inference=False):
        # Sample W and bias...

        if inference:
            return tf.matmul(x, self.kernel_mu) + self.bias_mu

        # Different reparametrization...
        # I need gamma_mj
        # I need:
        #  gamma_mj (mean) \sum_{i...hidden_dim} input_m_i mean_i_j
        #  delta_mj (mean) \sum_{i...hidden_dim} input_m_i**2 rho_i_j**2
        # m = x.shape[0]  # batch_size
        # out_dim = self.bias_mu.shape[0]

        # Shape = (batch_size, hidden_dim)
        gamma_l = tf.matmul(x, self.kernel_mu) + self.bias_mu
        delta_l = tf.matmul(tf.pow(x, 2), tf.pow(tf.math.softplus(self.kernel_rho), 2)) + tf.pow(tf.math.softplus(self.bias_rho), 2)

        self.posterior_activation = tfd.Independent(tfd.Normal(loc=gamma_l, scale=delta_l), reinterpreted_batch_ndims=1)  # or 2 to be complety

        b = gamma_l + tf.sqrt(delta_l) * tf.random.normal(shape=gamma_l.shape)

        self.prior_log_prob_b = self.prior_activation.log_prob(b)
        self.posterior_log_prob_b = self.posterior_activation.log_prob(b)

        self.log_prior = self.prior_log_prob_b
        self.log_posterior = self.posterior_log_prob_b

        # self.kl_div = self.log_posterior - self.log_prior

        return b


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

l1 = BayesianLinear(784, 1000)
l2 = BayesianLinear(1000, 10)

layers = [l1, l2]

opt = tf.optimizers.Adam(learning_rate=0.008)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# 2d idx to linear 1 d
# (i,j) i * n_batches + j


for epochs in range(50):
    # Shuffle data after epochs...
    running_loss = 0.0

    for batch_id, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:
            z = l1.forward(x)
            z = tf.nn.relu(z)
            z = l2.forward(z)

            outputs = z
            log_priors = l1.log_prior + l2.log_prior
            log_posteriors = l1.log_posterior + l2.log_posterior

            likelihood_dist = tfd.Categorical(logits=outputs)
            # Vector of (w_samples, batch_size)
            log_likelihood = likelihood_dist.log_prob(y)

            # kl_weights = tf.cast((tf.pow(2, n_batches - batch_id)) / (tf.pow(2, n_batches) - 1), tf.float32)
            kl_weights = 1 / n_batches

            loss = kl_weights * (tf.reduce_sum(log_posteriors) - tf.reduce_sum(log_priors)) - tf.reduce_sum(log_likelihood) * 49000 / 128
            running_loss += loss
            # print('{:10.3f} - {:10.3f} - {:10.3f}'.format(log_likelihood.numpy().mean(), loss.numpy(), mse.numpy()))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=epochs * n_batches + batch_id)
            tf.summary.scalar('log_likelihood', tf.reduce_sum(log_likelihood), step=epochs * n_batches + batch_id)
            tf.summary.scalar('scaled_log_likelihood', tf.reduce_sum(log_likelihood) * 49000 / 128, step=epochs * n_batches + batch_id)
            tf.summary.scalar('log_prior', tf.reduce_sum(log_priors), step=epochs * n_batches + batch_id)
            tf.summary.scalar('log_posterior', tf.reduce_sum(log_posteriors), step=epochs * n_batches + batch_id)

            # Log parameters
            tf.summary.histogram('layer0_kernel_mu', l1.kernel_mu, step=epochs * n_batches + batch_id)
            tf.summary.histogram('layer0_bias_mu', l1.bias_mu, step=epochs * n_batches + batch_id)

            tf.summary.histogram('layer0_kernel_rho', tf.math.softplus(l1.kernel_rho), step=epochs * n_batches + batch_id)
            tf.summary.histogram('layer0_bias_rho', tf.math.softplus(l1.bias_rho), step=epochs * n_batches + batch_id)

            tf.summary.histogram('layer1_kernel_mu', l2.kernel_mu, step=epochs * n_batches + batch_id)
            tf.summary.histogram('layer1_bias_mu', l2.bias_mu, step=epochs * n_batches + batch_id)

            tf.summary.histogram('layer1_kernel_rho', tf.math.softplus(l2.kernel_rho), step=epochs * n_batches + batch_id)
            tf.summary.histogram('layer1_bias_rho', tf.math.softplus(l2.bias_rho), step=epochs * n_batches + batch_id)

        # print(np.mean(mses))
        # df/dw, muss man da nicht das gesampelte (gemittelte) w nehmen? also tape.gradient(loss, w_samples) ?
        trainable_vars = l1.get_trainable_vars() + l2.get_trainable_vars()

        grads = tape.gradient(loss, trainable_vars)
        opt.apply_gradients(zip(grads, trainable_vars))

    correct_test = 0.0
    for batch_id, (x, y) in enumerate(test_data):
        z = l1.forward(x, inference=True)
        z = tf.nn.relu(z)
        z = l2.forward(z, inference=True)
        preds = tf.nn.softmax(z)

        binary_comp = tf.cast(tf.argmax(preds, axis=1), tf.float32) == y
        correctly_classified = tf.reduce_sum(tf.cast(binary_comp, tf.float32))
        assert correctly_classified <= batch_size

        correct_test += correctly_classified

    test_accuracy = correct_test / float(nr_testdatapoints)
    correct_test = 0.0
    print("train_loss {} test_accuracy {}".format(running_loss, test_accuracy))

    running_loss = 0.0
