import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp

from bnn_keras import BayesianLinear, neg_log_likelihood


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

tf_train = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 784).astype("float32") / 255.0, y_train.astype('int64'))).shuffle(128).batch(128)
tf_test = tf.data.Dataset.from_tensor_slices((x_test.reshape(10000, 784).astype("float32") / 255.0, y_test.astype('int64'))).shuffle(128).batch(128)

bs_train = len(tf_train)
bs_test = len(tf_test)

model = keras.Sequential(
    [keras.Input(shape=(784,)),
     BayesianLinear(1000, activation=tf.nn.relu, kl_weight=1.0/bs_train),
     BayesianLinear(10, activation=tf.nn.softmax, kl_weight=1.0/bs_train)])

model.compile()#optimizer=keras.optimizers.Adam(lr=0.08))  # ,loss=neg_log_likelihood, metrics=['accuracy']


optimizer = keras.optimizers.Adam(lr=0.08)

W_SAMPLES = 2

for epochs in range(10):
    running_loss = 0.0

    for batch_id, (x, y) in enumerate(tf_train):
        with tf.GradientTape() as tape:

            outputs = []
            model_losses = 0.0
            for wi in range(W_SAMPLES):
                logits = model(x, training=True)
                model_losses += tf.reduce_sum(model.losses)
                outputs.append(logits)

            outputs = tf.convert_to_tensor(outputs)  # Shape w_samples, batch_size, 2
            loss_value = model_losses + neg_log_likelihood(y, outputs)
            running_loss += loss_value

        # Update the weights of the model to minimize the loss value.
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    metric = tf.keras.metrics.Accuracy(name="accuracy", dtype=None)

    for batch_id, (x, y) in enumerate(tf_test):
        preds = model(x, training=False)

        acc = tf.keras.metrics.sparse_categorical_accuracy(y, preds)
        metric.update_state(y, tf.argmax(preds, axis=1))

    print("train_loss {} test_accuracy {}".format(running_loss, metric.result().numpy()))

    running_loss = 0.0




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
