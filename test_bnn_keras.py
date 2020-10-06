import tensorflow as tf
import tensorflow.keras as keras

from bnn_keras import BayesianLinear, neg_log_likelihood

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

tf_train = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 784).astype("float32") / 255.0, y_train.astype('int32'))).shuffle(128).batch(128)
tf_test = tf.data.Dataset.from_tensor_slices((x_test.reshape(10000, 784).astype("float32") / 255.0, y_test.astype('int32'))).shuffle(128).batch(128)

bs_train = len(tf_train)
bs_test = len(tf_test)

model = keras.Sequential(
    [keras.Input(shape=(784,)),
     BayesianLinear(1000, activation=tf.nn.relu, kl_weight=1.0 / bs_train),
     BayesianLinear(10, activation=keras.activations.softmax, kl_weight=1.0 / bs_train)])  # activation=tf.nn.softmax,

model.compile(metrics='accuracy')  # optimizer=keras.optimizers.Adam(lr=0.08))  # ,loss=neg_log_likelihood, metrics=['accuracy']

optimizer = keras.optimizers.Adam(lr=0.08)

W_SAMPLES = 1


# @tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = tf.reduce_sum(model.losses) + neg_log_likelihood(y, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


for epochs in range(50):
    running_loss = 0.0

    for batch_id, (x, y) in enumerate(tf_train):
        running_loss += train_step(x, y)

    train_loss, acc = model.evaluate(tf_test, verbose=0)
    print("train_loss {} test_accuracy {}".format(running_loss / len(tf_train), acc))

exit(1)

# model.fit(train_data, validation_data=test_data, batch_size=32, epochs=10, verbose=1)
# Iterate over the batches of a dataset.