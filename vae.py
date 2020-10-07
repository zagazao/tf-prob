import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

tfd = tfp.distributions

# logp(x,z)−logq(z;λ)

# log p(x|z)*p(z) -log q(z|lambda) = log p(x|z) + log(z) - log q(z|l)
#                                  = log p(x|z) + log(p(z)/log(q(z))

# ELBO = LL + KLDIV(posterior, prior)
# NELBO = NLL - KLDIV <- min

# What do we need?

# Latent dim k
# Encoder maps x^i to z \in R^k
# Decoder maps z -> x^i

# tfd.Independent(tfd.Normal)) MultivariateDiag.

# Prior(z) = N(z|0,1) (Isotropic Gaussian)

# Decoder = p_{\theta}(x|z) is
#   continuous -> multivariate gaussian
#   binary     -> bernoulli
#   discrete   -> categorical

# Decoder = MLP which computes parameter of p(x|z) (single hidden layer)


# Assume true posterior p(z|x) is gaussian with diagonal cov

# Variational posterior is multivariate Gaussian with diagonal cov
# log q(z|x) ) = log N(z|mu, sigma * I)
# Encoder = MLP, which maps x^i -> mu^i, sigma^i

# Sample^l = mu^i + sigma^i * eps^l    eps^l ~ N(0,1)


# Bernoulli:

#     P(X) = p^x * (1-p)^(1-x)
#
# log P(X) = log(p^x * (1-p)^(1-x))
#          = log(p^x) + log( (1-p)^(1-x))
#          = x * log(p) + (1-x) * log(1-p)

# Log-Likelihood: \sum_{x \in D}  x * log(p) + (1-x) * log(1-p)

#     P(X) \in (0,1)
# log P(X) \in (-inf,0)
# \sum log P(X) = negative => Log-Likelihood is always negative
#                          => Neg-LL is always positive

# ELBO = LL - KL
# NELBO = -ELBO
#       = -LL + KL    <- We want to minimize this !!!

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = (mnist_digits.astype("float32") / 255).reshape(mnist_digits.shape[0], 784)


# https://blog.keras.io/building-autoencoders-in-keras.html Here they don't use unit normal but 0.1 stddev...

class SampleLayer(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        z_mu, z_log_var = inputs

        batch = tf.shape(z_mu)[0]
        dim = tf.shape(z_mu)[1]

        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        # kl_div = 1 + tf.math.log(z_sigma ** 2) - z_mu ** 2 - z_sigma ** 2

        # log(sigma * sigma) - mu * mu - sigma * sigma

        # sigma = exp(rho) => log sigma = rho = PARAMETER

        # Assumption: z_sigma is z_log_sigma
        kl_div = 1 + z_log_var - tf.square(z_mu) - tf.math.exp(z_log_var)
        # * -0.5 Why the hell???
        kl_div = -0.5 * tf.reduce_sum(kl_div)

        # kl_div = tf.reduce_sum(1 + tf.math.log(z_sigma ** 2) - z_mu ** 2 - z_sigma ** 2)

        # kl_div = 1 + z_sigma - tf.square(z_mu) - tf.exp(z_sigma)
        # kl_div = tf.reduce_mean(kl_div) * 0.5

        self.add_loss(kl_div)

        return z_mu + tf.math.exp(0.5 * z_log_var) * epsilon


input_dim = 784
latent_dim = 2

# H1 -> hidden_params -> sample -> H2

activation = 'relu'  # 'relu

inputs = keras.Input(shape=(input_dim,))

x = keras.layers.Dense(units=128, activation=activation)(inputs)
x = keras.layers.Dense(units=64, activation=activation)(x)

z_mu = keras.layers.Dense(units=latent_dim, name='z_mean')(x)
z_sigma = keras.layers.Dense(units=latent_dim, name='z_sigma')(x)

# Sample step
sample = SampleLayer()([z_mu, z_sigma])

input_enc = keras.Input(shape=(latent_dim,))
# 2 conv layers
# x = keras.layers.Conv2D(filters=32, kernel_size=(3, 3))

x = keras.layers.Dense(units=64, activation=activation)(input_enc)
x = keras.layers.Dense(units=128, activation=activation)(x)
x = keras.layers.Dense(units=input_dim, activation='sigmoid')(x)

encoder = keras.Model(inputs=inputs, outputs=sample, name='encoder')
decoder = keras.Model(inputs=input_enc, outputs=x, name='decoder')


# model = keras.Model(inputs=inputs, outputs=x, name="vae")
# model.summary()


class VAE(keras.Model):

    def __init__(self, encoder, decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.decoder = decoder
        self.encoder = encoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        # for x_batch, y_batch in data:
        with tf.GradientTape() as tape:
            z = self.encoder(data)
            out = self.decoder(z)

            log_likelihood_dist = tfd.Bernoulli(probs=out)
            log_likelihood = tf.reduce_sum(log_likelihood_dist.log_prob(data))

            # binary_crossentropy

            neg_log_likelihood = tf.reduce_mean(keras.losses.binary_crossentropy(data, out))
            # reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(data, out)) * 784
            # reconstruction_loss = log_likelihood

            # kl_div + reconstruction
            kl_div = sum(self.encoder.losses)

            # ELBO = LL - KL
            # NELBO = -ELBO
            #       = -LL + KL    <- We want to minimize this !!!
            # loss = -(kl_div + neg_log_likelihood)  # WTF WILL HAPPEN
            nelbo_loss = neg_log_likelihood + kl_div

        trainable_weights = self.encoder.trainable_weights + self.decoder.trainable_weights

        grads = tape.gradient(nelbo_loss, trainable_weights)
        self.optimizer.apply_gradients(zip(grads, trainable_weights))

        return {"loss": nelbo_loss,
                "reconstruction_loss": neg_log_likelihood,
                "kl_loss": kl_div}


vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam(), run_eagerly=True)

print('pre_fit()')

vae.fit(mnist_digits, batch_size=128, verbose=1, epochs=25)

print('post_fit()')


def plot_latent(encoder, decoder):
    # display a n*n 2D manifold of digits
    n = 30
    digit_size = 28
    scale = 2.0
    figsize = 15
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
            i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent(encoder, decoder)
