import tensorflow as tf
import tensorflow_probability as tfp
# from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits

from tensorflow_probability.python.bijectors import BatchNormalization

tfd = tfp.distributions

input_dim = 784
hidden1_units = 200
hidden2_units = 10

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

tf_train = tf.data.Dataset.from_tensor_slices((x_train.reshape(60000, 784).astype("float32") / 255.0, y_train.astype('int32'))).shuffle(128).batch(128)
tf_test = tf.data.Dataset.from_tensor_slices((x_test.reshape(10000, 784).astype("float32") / 255.0, y_test.astype('int32'))).shuffle(128).batch(128)

W1 = tf.Variable(shape=(input_dim, hidden1_units), initial_value=tf.constant(0.5, shape=(input_dim, hidden1_units)))
W2 = tf.Variable(shape=(hidden1_units, hidden2_units), initial_value=tf.constant(0.5, shape=(hidden1_units, hidden2_units)))

b1 = tf.Variable(shape=(hidden1_units), initial_value=tf.constant(0.5, shape=[hidden1_units]))  # tf.random.uniform(shape=[hidden1_units], minval=0, maxval=1))
b2 = tf.Variable(shape=(hidden2_units), initial_value=tf.constant(0.5, shape=[hidden2_units]))

in_data = tf.random.normal(shape=(520, input_dim))
coef = tf.random.normal(shape=(input_dim, 1))

y = tf.matmul(in_data, coef)


@tf.custom_gradient
def binarize(probs):
    shape = tf.shape(probs)
    uniform = tf.random.uniform(shape=shape, minval=0, maxval=1)

    def grad(dy):
        return dy

    a = uniform - probs
    return tf.nn.relu(a) / a, grad


@tf.custom_gradient
def binarize2(t):
    def grad(dy):
        return (tf.nn.relu(dy) + tf.nn.relu(-dy)) / dy

    return (tf.nn.relu(t) + tf.nn.relu(-t)) / t, grad


def _reparametrize(_probs):
    # TODO: We called binaruze2
    return binarize2(_probs)


batch_norm = BatchNormalization()
batch_norm2 = BatchNormalization()


def forward(x):
    # squeeze weights to [0,1] by applying sigmoid
    w1_sample = _reparametrize(tf.sigmoid(W1))
    w2_sample = _reparametrize(tf.sigmoid(W2))

    b1_sample = _reparametrize(tf.sigmoid(b1))
    b2_sample = _reparametrize(tf.sigmoid(b2))

    # Layer1
    x = tf.matmul(x, w1_sample)
    x = tf.add(x, b1_sample)
    x = batch_norm.forward(x)
    x = tf.keras.activations.relu(x)

    # Layer2
    x = tf.matmul(x, w2_sample)
    x = tf.add(x, b2_sample)
    x = batch_norm2.forward(x)
    return x


optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

for i in range(500):
    running_loss = 0.0
    for batch_x, batch_y in tf_train:
        with tf.GradientTape() as tape:
            logits = forward(batch_x)
            loss = sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_y)
            loss = tf.reduce_mean(loss)

        running_loss += loss.numpy()
        grads = tape.gradient(loss, [W1, W2, b1, b2, batch_norm.trainable_variables[0], batch_norm.trainable_variables[1], batch_norm2.trainable_variables[0],
                                     batch_norm2.trainable_variables[1]])
        optimizer.apply_gradients(zip(grads, [W1, W2, b1, b2, batch_norm.trainable_variables[0], batch_norm.trainable_variables[1], batch_norm2.trainable_variables[0],
                                              batch_norm2.trainable_variables[1]]))

        # TODO: Loss went to nan at epoch 23...
        # acc = tf.reduce_mean(accuracy(y_true=tf.cast(y_test, tf.float32), y_pred=tf.argmax(forward(x_test), axis=1))).numpy()

    print(i, running_loss / len(tf_train))

# 0 352.07998379054607
# 1 219.1410643815486
# 2 158.7359290539837
# 3 125.25012032157068
# 4 104.13239512819726
# 5 84.69821023432685
# 6 74.18646333365044
# 7 62.3973411934208
# 8 55.61504479918653
# 9 47.168402523374255
# 10 41.419312357139994
# 11 36.70580397689266
# 12 32.32251742755426
# 13 28.36534303490287
# 14 25.00282331722886
# 15 23.240364619663783
# 16 19.271315838990688
# 17 17.640051992208974
# 18 15.264730886609824
# 19 nan
