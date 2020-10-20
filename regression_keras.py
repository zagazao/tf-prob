import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow_probability.python.layers import DenseReparameterization

tfd = tfp.distributions

x, y = load_boston(return_X_y=True)

x = MinMaxScaler().fit_transform(x)

x = tf.convert_to_tensor(x, dtype=tf.float32)
y = tf.convert_to_tensor(y, dtype=tf.float32)

tf_data = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(100).batch(32)
n_batches = len(tf_data)

W_SAMPLES = 1

# log q(w) - log p (w) = log (q(w) / p(w)) : Negative if Prior > Posterior (p > q)

model = tf.keras.Sequential([
    DenseReparameterization(15, activation='relu'),
    DenseReparameterization(1),
])

optimizer = tf.optimizers.Adam(learning_rate=0.08)

for epochs in range(10000):
    # Shuffle data after epochs...

    tf_data = tf_data.shuffle(100)

    running_loss = 0.0
    running_likelihood = 0.0
    running_kl_div = 0.0

    running_mse = 0.0

    for batch_id, (x, y) in enumerate(tf_data):
        with tf.GradientTape() as tape:

            outputs, kl_divs = [], []

            for wi in range(W_SAMPLES):
                z = model(x)

                outputs.append(z)
                kl_divs.append(model.losses)

            outputs = tf.convert_to_tensor(outputs)  # Shape w_samples, batch_size, 2
            kl_divs = tf.convert_to_tensor(kl_divs)

            # means = outputs
            means = tf.squeeze(tf.reduce_mean(outputs, axis=0), 1)

            # If more than
            mse = mean_squared_error(y, means)
            running_mse += mse
            # mses.append(mse)

            likelihood_dist = tfd.Normal(loc=means, scale=1)
            # Vector of (w_samples, batch_size)
            log_likelihood = likelihood_dist.log_prob(y)

            kl_weights = 1 / n_batches

            running_likelihood += tf.reduce_sum(log_likelihood)
            running_kl_div += tf.reduce_sum(kl_divs)

            loss = kl_weights * (tf.reduce_sum(kl_divs)) - tf.reduce_sum(log_likelihood)
            running_loss += loss

        # df/dw, muss man da nicht das gesampelte (gemittelte) w nehmen? also tape.gradient(loss, w_samples) ?
        trainable_vars = model.trainable_weights

        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))

    print("total {}, kl-div {} likelihood {} mse {}".format(running_loss, running_kl_div, running_likelihood, mse))

# model.compile(loss='', optimizer=tf.optimizers.Adam(learning_rate=0.08))

