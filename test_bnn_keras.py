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
     BayesianLinear(1000, activation=tf.nn.relu, kl_weight=1.0 / 60_000),
     BayesianLinear(128, activation=tf.nn.relu, kl_weight=1.0 / 60_000),
     BayesianLinear(10, kl_weight=1.0 / bs_train)])  # activation=tf.nn.softmax,

model.compile(metrics='accuracy')  # optimizer=keras.optimizers.Adam(lr=0.08))  # ,loss=neg_log_likelihood, metrics=['accuracy']

optimizer = keras.optimizers.Adam(lr=0.0008)

W_SAMPLES = 2


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        kl = tf.reduce_sum(model.losses)
        nll = neg_log_likelihood(y, logits)
        loss_value = kl + nll
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value, kl, nll


for epochs in range(50):
    running_loss = 0.0
    running_kl = 0.0
    running_nll = 0.0

    for batch_id, (x, y) in enumerate(tf_train):
        batch_loss, kl, nll = train_step(x, y)
        running_loss += batch_loss
        running_kl += kl
        running_nll += nll

    test_loss, test_acc = model.evaluate(tf_test, verbose=0)
    train_loss, train_acc = model.evaluate(tf_train, verbose=0)
    print("train_loss {} train_accuracy {} test_accuracy {}".format(running_loss / len(tf_train), train_acc, test_acc))
    print('{}, {}'.format(running_kl / len(tf_train), running_nll / len(tf_train)))
