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

            prior_log_prob_w = self.prior_kernel.log_prob(W)
            posterior_log_prob_w = self.posterior_kernel.log_prob(W)

            prior_log_prob_b = self.prior_bias.log_prob(b)
            posterior_log_prob_b = self.posterior_bias.log_prob(b)

            log_prior = prior_log_prob_w + prior_log_prob_b
            log_posterior = posterior_log_prob_w + posterior_log_prob_b

            self.add_loss(log_posterior - log_prior)

            return self.activation(tf.matmul(x, W) + b)

        return self.activation(tf.matmul(x, self.kernel_mu) + self.bias_mu)

    def get_config(self):
        config = super(BayesianLinear, self).get_config()
        config.update({"units": self.units, })


def neg_log_likelihood(y_obs, y_pred):
    dist = tfd.Categorical(logits=y_pred)
    return tf.reduce_sum(-dist.log_prob(y_obs))


model = keras.Sequential(
    [keras.Input(shape=(784,)),
     BayesianLinear(1000, activation=tf.nn.relu),
     BayesianLinear(10, activation=tf.nn.softmax)])

model.compile(optimizer=keras.optimizers.Adam(lr=0.08))  # ,loss=neg_log_likelihood, metrics=['accuracy']

# loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

# Limit to 15k
tf_train = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 784).astype("float32") / 255.0, y_train.astype('int64'))).shuffle(128).batch(128)
tf_test = tf.data.Dataset.from_tensor_slices((x_test.reshape(10000, 784).astype("float32") / 255.0, y_test.astype('int64'))).shuffle(128).batch(128)

optimizer = keras.optimizers.Adam(lr=0.08)

# @tf.function  # Make it fast.
# def train_on_batch(x, y):
#     with tf.GradientTape() as tape:
#         logits = model(x)
#
#         likelihood_dist = tfd.Categorical(logits=logits)
#         log_likelihood = likelihood_dist.log_prob(y)
#
#         loss = tf.reduce_sum(model.loss) - tf.reduce_sum(log_likelihood)
#         gradients = tape.gradient(loss, model.trainable_weights)
#     optimizer.apply_gradients(zip(gradients, model.trainable_weights))
#     return loss

#
# for epoch in range(10):
#     for batch_id, (x, y) in enumerate(train_data):
#         batch_loss = train_on_batch(x, y)
#         print(batch_loss)
#
# exit(0)

# model.fit(train_data, validation_data=test_data, batch_size=32, epochs=10, verbose=1)
# Iterate over the batches of a dataset.
for epochs in range(10):
    # Shuffle data after epochs...
    running_loss = 0.0

    for batch_id, (x, y) in enumerate(tf_train):
        with tf.GradientTape() as tape:
            # Forward pass.
            logits = model(x, training=True)

            likelihood_dist = tfd.Categorical(logits=logits)
            log_likelihood = likelihood_dist.log_prob(y)

            loss_value = tf.reduce_sum(model.losses) - tf.reduce_sum(log_likelihood)
            # bla_loss = tf.reduce_sum(model.losses) + neg_log_likelihood(y, logits)
            running_loss += loss_value

        # Update the weights of the model to minimize the loss value.
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    # correct_test = 0.0
    # cum_acc = 0.0

    metric = tf.keras.metrics.Accuracy(name="accuracy", dtype=None)

    for batch_id, (x, y) in enumerate(tf_test):
        preds = model(x, training=False)
        # preds = z  # tf.nn.softmax(z)

        acc = tf.keras.metrics.sparse_categorical_accuracy(y, preds)
        metric.update_state(y, tf.argmax(preds, axis=1))
        # acc = tf.keras.metrics.binary_accuracy(y, tf.argmax(preds, axis=1))
        #
        # binary_comp = tf.cast(tf.argmax(preds, axis=1), tf.int64) == y
        # correctly_classified = tf.reduce_sum(tf.cast(binary_comp, tf.float32))
        # assert correctly_classified <= 32
        #
        # print(acc, correctly_classified / 32)
        # cum_acc += acc

        # correct_test += correctly_classified

    # test_accuracy = correct_test / len(tf_test)
    # test_accuracy_avg = cum_acc / len(tf_test)
    # correct_test = 0.0
    print("train_loss {} test_accuracy {}".format(running_loss, metric.result().numpy()))
    # print(metric.result().numpy())

    running_loss = 0.0
