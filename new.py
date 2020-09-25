# Regression:
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler
from tensorflow_probability.python.distributions import kl_divergence

TF_D_TYPE = tf.float64
EPOCHS = 500
w_samples = 1

tfd = tfp.distributions

n_samples, n_features = 1000, 13

dim_input = 13
dim_hidden = 2

x, y = load_boston(return_X_y=True)

x = MinMaxScaler().fit_transform(x)

x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.float32)

tf_data = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(100).batch(32)

print(x.shape, y.shape)

hidden1_shape = (dim_input, dim_hidden)

kernel_mu = tf.Variable(tf.random.normal(shape=hidden1_shape, mean=0.0, stddev=1), name='kernel_mu')
kernel_rho = tf.Variable(tf.random.normal(shape=hidden1_shape, mean=0.0, stddev=1), name='kernel_rho')

bias_mu = tf.Variable(tf.random.normal(shape=[dim_hidden], mean=0.0, stddev=1), name='bias_mu')
bias_rho = tf.Variable(tf.random.normal(shape=[dim_hidden], mean=0.0, stddev=1), name='bias_rho')

prior_kernel = tfd.Independent(tfp.distributions.Normal(loc=tf.zeros(shape=hidden1_shape), scale=1), reinterpreted_batch_ndims=2)
prior_bias = tfd.Independent(tfp.distributions.Normal(loc=tf.zeros(shape=dim_hidden), scale=1), reinterpreted_batch_ndims=1)

posterior_kernel = tfd.Independent(tfp.distributions.Normal(loc=kernel_mu, scale=tf.math.softplus(kernel_rho)), reinterpreted_batch_ndims=2)
posterior_bias = tfd.Independent(tfp.distributions.Normal(loc=bias_mu, scale=tf.math.softplus(bias_rho)), reinterpreted_batch_ndims=1)

dist_eps = tfp.distributions.Normal(loc=0.0, scale=1.0)


def reparametrize(mu, std, eps):
    return mu + tf.math.softplus(std) * eps


def forward(X):
    eps_kernel = dist_eps.sample(sample_shape=hidden1_shape)
    eps_bias = dist_eps.sample(sample_shape=dim_hidden)

    W = reparametrize(kernel_mu, kernel_rho, eps_kernel)
    b = reparametrize(bias_mu, bias_rho, eps_bias)
    return tf.matmul(X, W) + b, W, b


opt = tf.optimizers.SGD(1e-4)
lls = []
losses = []

for epoch in range(EPOCHS):
    for x, y in tf_data:
        with tf.GradientTape() as tape:

            # Approximate our target with some samples of the posterior
            for i in range(w_samples):
                eps_kernel = dist_eps.sample(sample_shape=hidden1_shape)
                eps_bias = dist_eps.sample(sample_shape=dim_hidden)

                # W = reparametrize(kernel_mu, kernel_rho, eps_kernel)
                # b = reparametrize(bias_mu, bias_rho, eps_bias)

                out, W, b = forward(x)

                # Evaluate prior and posterior...
                # log_prior_kernel = prior_kernel.log_prob(W)
                # log_prior_bias = prior_bias.log_prob(b)
                #
                # log_posterior_kernel = posterior_kernel.log_prob(W)
                # log_posterior_bias = posterior_bias.log_prob(b)

                kl_div_kernel = kl_divergence(prior_kernel, posterior_kernel)
                kl_div_bias = kl_divergence(prior_bias, posterior_bias)

                kl_div = kl_div_kernel + kl_div_bias

                means = out[:, 0]
                stddevs = tf.math.softplus(out[:, 1])

                dist1 = tfp.distributions.Normal(loc=means, scale=stddevs)
                log_likelihood = dist1.log_prob(y)

                loss = tf.reduce_mean((kl_div - log_likelihood))

                print('Log likelihood:', log_likelihood.numpy().mean())
                print('Loss:', loss.numpy())

                losses.append(loss.numpy())
                lls.append(log_likelihood.numpy().mean())

        grads = tape.gradient(loss, [kernel_mu, kernel_rho, bias_mu, bias_rho])

        # Process gradients in a custom way...

        opt.apply_gradients(zip(grads, [kernel_mu, kernel_rho, bias_mu, bias_rho]))

fig, axs = plt.subplots(2, 1)
axs[0].plot(losses)
axs[1].plot(lls)

plt.show()
