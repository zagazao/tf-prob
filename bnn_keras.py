import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
import tensorflow_probability as tfp
from abc import ABC, abstractmethod 
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


tfd = tfp.distributions

class BayesianLinear(keras.layers.Layer):

    def __init__(self, units, activation=keras.activations.linear, sigma1=1.5, sigma2=0.1):
        super(BayesianLinear, self).__init__()

        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.initializer_sigma = 0.6164414
        self.activation = activation
        self.units = units
        self.log_prior = 0
        self.log_posterior = 0

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

    # def get_trainable_vars(self):
    #     return [self.kernel_mu, self.kernel_rho, self.bias_mu, self.bias_rho]

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


x = tf.random.normal(shape=(100, 32))

def neg_log_likelihood(y_obs, y_pred):
    dist = tfd.Categorical(logits=y_pred)
    return tf.reduce_sum(-dist.log_prob(y_obs))

model = keras.Sequential([keras.Input(shape=(784,)), BayesianLinear(1000), BayesianLinear(10)])
model.compile(loss=neg_log_likelihood, optimizer=keras.optimizers.Adam(lr=0.08), metrics=['accuracy'])

#loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)

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


#model.fit(train_data, validation_data=test_data, batch_size=32, epochs=10, verbose=1)
# Iterate over the batches of a dataset.
for epochs in range(10):
    # Shuffle data after epochs...
    running_loss = 0.0

    for batch_id, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:
            # Forward pass.
            logits = model(x, training=True)

            likelihood_dist = tfd.Categorical(logits=logits)
            log_likelihood = likelihood_dist.log_prob(y)

            loss_value = tf.reduce_sum(model.losses) - tf.reduce_sum(log_likelihood)
            bla_loss = tf.reduce_sum(model.losses) + neg_log_likelihood(y, logits)
            running_loss += loss_value

        # Update the weights of the model to minimize the loss value.
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    correct_test = 0.0
    for batch_id, (x, y) in enumerate(test_data):
        z = model(x, training=False)
        preds = tf.nn.softmax(z)

        binary_comp = tf.cast(tf.argmax(preds, axis=1), tf.float32) == y
        correctly_classified = tf.reduce_sum(tf.cast(binary_comp, tf.float32))
        assert correctly_classified <= batch_size

        correct_test += correctly_classified

    test_accuracy = correct_test / float(nr_testdatapoints)
    correct_test = 0.0
    print("train_loss {} test_accuracy {}".format(running_loss, test_accuracy))

    running_loss = 0.0


