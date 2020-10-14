import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

tfpl = tfp.layers
tfd = tfp.distributions

x, y = load_breast_cancer(return_X_y=True)

print(x.max(), x.min())

x = MinMaxScaler().fit_transform(x)

print(x.max(), x.min())

model = tfk.Sequential([
    tfk.layers.Input(shape=(x.shape[1],)),
    tfpl.DenseReparameterization(100, activation='relu'),
    tfpl.DenseReparameterization(1, activation='sigmoid'),
    tfpl.DistributionLambda(lambda t: tfd.Bernoulli(probs=t))
])


def neg_log_lik(y, p_y):
    return -p_y.log_prob(y)


optimizer = tfk.optimizers.Adam(learning_rate=0.1)

tf_data = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(buffer_size=128).batch(32)

for epoch in range(5):
    for b_x, b_y in tf_data:
        with tf.GradientTape() as tape:
            dist = model(b_x)
            ll = -tf.reduce_sum(dist.log_prob(b_y))

            # neg_log_like_sum = tf.reduce_sum(neg_log_lik(b_y, model(b_x)))
            # neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
            #     labels=labels, logits=logits)
            kl = sum(model.losses)
            loss = ll + kl * 0.0001
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        print(loss.numpy(), ll.numpy(), kl.numpy())

...
