import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

from bnn import BayesianLinear, BNN

tfd = tfp.distributions

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

model = BNN([
    BayesianLinear(784, 1000),
    tf.nn.relu,
    BayesianLinear(1000, 10)
])

opt = tf.optimizers.Adam(learning_rate=0.08)

W_SAMPLES = 2

for epochs in range(10):
    # Shuffle data after epochs...
    running_loss = 0.0

    for batch_id, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:

            outputs, log_priors, log_posteriors = [], 0.0, 0.0

            for wi in range(W_SAMPLES):
                z = model(x)

                outputs.append(z)
                log_priors += model.log_priors()
                log_posteriors += model.log_posteriors()

            outputs = tf.convert_to_tensor(outputs)  # Shape w_samples, batch_size, 2
            log_priors = tf.convert_to_tensor(log_priors)  # shape w_samples
            log_posteriors = tf.convert_to_tensor(log_posteriors)  # shape w_samples

            # log_priors = tf.reduce_sum(log_priors)
            # log_posteriors = tf.reduce_sum(log_posteriors)
            # means = outputs
            # means = tf.reduce_mean(outputs, axis=0)
            # stddevs = tf.math.softplus(outputs[..., 1])

            likelihood_dist = tfd.Categorical(logits=outputs)
            # Vector of (w_samples, batch_size)
            log_likelihood = likelihood_dist.log_prob(y)

            # kl_weights = tf.cast((tf.pow(2, n_batches - batch_id)) / (tf.pow(2, n_batches) - 1), tf.float32)
            kl_weights = 1 / n_batches

            loss = kl_weights * (log_posteriors - log_priors) - tf.reduce_sum(log_likelihood) * 49000 / 128
            running_loss += loss
            # print('{:10.3f} - {:10.3f} - {:10.3f}'.format(log_likelihood.numpy().mean(), loss.numpy(), mse.numpy()))

        # print(np.mean(mses))
        # df/dw, muss man da nicht das gesampelte (gemittelte) w nehmen? also tape.gradient(loss, w_samples) ? 
        trainable_vars = model.get_trainable_vars()

        grads = tape.gradient(loss, trainable_vars)
        opt.apply_gradients(zip(grads, trainable_vars))

    correct_test = 0.0
    for batch_id, (x, y) in enumerate(test_data):
        z = model(x, inference=True)
        preds = tf.nn.softmax(z)

        binary_comp = tf.cast(tf.argmax(preds, axis=1), tf.float32) == y
        correctly_classified = tf.reduce_sum(tf.cast(binary_comp, tf.float32))
        assert correctly_classified <= batch_size

        correct_test += correctly_classified

    test_accuracy = correct_test / float(nr_testdatapoints)
    correct_test = 0.0
    print("train_loss {} test_accuracy {}".format(running_loss, test_accuracy))

    running_loss = 0.0
