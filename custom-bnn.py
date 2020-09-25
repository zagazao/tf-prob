# Regression:
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.datasets import make_classification, make_regression

tfd = tfp.distributions

n_samples = 100

x, y = make_regression(n_samples=n_samples, n_features=10)
# x, y = make_classification(n_samples=n_samples, n_features=10, n_classes=2)

# coef = np.random.randn(10)
# noise = np.random.randn(n_samples) * 2

# x = np.random.normal(loc=0, scale=1, size=(n_samples, 10))
# y = np.dot(x, coef) + noise

# x = x / 100
# y = y / 100

print(x.shape, y.shape)

# 2 Variational Dense


# => Prior für weights

# Wir brauchen irgendwie einen Posterior... (und Likelihood)
# => Posterior = MeanField??

# Variational learning finds the parameters θ of a distribution on the weights q(w|θ)

# θ der Posteriorverteilung
# θ müssen wir lernen (Maximierung ELBO)

# \argmin θ KL[q(w|θ) || P(w)] - \E_{q(w|θ)} [log P(D|w)]
#                                 --- LogLikelihood of data, given weights under posterior ---

# F(D, θ) = ...

# MC of expectation: 1 / n_samples model.logprob(x)

# Weights W1, b1

# f(w,θ) = loq q(w|θ) - log(P(w)) - log(P(D|w)) ###  / loq q(w|θ) - log(P(w) * P(D|w))

# Steps:
# 1. Sample eps ~ N(0,1)
# 2. Let w = mu + log( 1 + exp(rho)) * eps (pointwise) # This is softplus operations
# 3. Let θ = (mu, rho)
# 4. Let f = f(w,θ)
# 5. Calc gradient...


dim_input = 10
dim_hidden = 2

hidden1_shape = (dim_input, dim_hidden)

tf_dtype = tf.float64

# Step 1
dist_eps = tfp.distributions.Normal(loc=0.0, scale=1.0)

mu = tf.Variable(tf.random.normal(hidden1_shape, dtype=tf_dtype), shape=hidden1_shape, dtype=tf_dtype, name='mu')
rho = tf.Variable(tf.random.normal(hidden1_shape, dtype=tf_dtype), shape=hidden1_shape, dtype=tf_dtype, name='rho')

#  f(w,θ) = loq q(w|θ) - log(P(w)) - log(P(D|w))

# Wie sieht Prior, Likelihood und Posterior aus?

# Prior = N(0,1)
prior_normal = tfp.distributions.Normal(loc=tf.zeros(hidden1_shape, dtype=tf_dtype), scale=0.1, name='priorW')
prior_w = tfp.distributions.Independent(prior_normal, reinterpreted_batch_ndims=2)

# prior_w = tfp.distributions.Independent(tfp.distributions.Mixture(cat=tfp.distributions.Categorical([0.7, 0.3]), components=[
#     tfp.distributions.Normal(loc=tf.zeros(hidden1_shape, dtype=tf_dtype), scale=1.5, name='priorW_Comp1'),
#     tfp.distributions.Normal(loc=tf.zeros(hidden1_shape, dtype=tf_dtype), scale=0.3, name='priorW_Comp2')
# ]), reinterpreted_batch_ndims=2)

# tfp.distributions.MultivariateNormalDiag(loc=)

# prior_w = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(
#     probs=[0.7, 0.3]),
#     components_distribution=tfd.Normal(
#         loc=tf.zeros(shape=(*hidden1_shape, 2), dtype=tf_dtype),
#         scale=tf.ones(shape=(*hidden1_shape, 2), dtype=tf_dtype)))

# Likelihood P(D|w))

# mu_out = <x,w> + b
# P(D|w) = N(y|mu_out,1)

# Posterior over weights q(w|θ)
# Jeder ist normalverteilt mit mu, rho
# Alle unabhängig = Mean-Field??


opt = tf.optimizers.SGD(1e-10)

for i in range(5000):
    # Step1:
    eps = tf.cast(dist_eps.sample(sample_shape=hidden1_shape), dtype=tf_dtype)
    # Step2:
    w = mu + tf.math.log(1 + tf.exp(rho)) * eps
    w_var = tf.Variable(w, dtype=tf_dtype, name='w')
    with tf.GradientTape() as tape:
        # Forward pass = sample w
        out = tf.matmul(x, w_var)
        means = out[:, 0]
        stds = tf.nn.softplus(out[:, 1])
        # Evaluate likelihood of the data...

        # pred_dist = tfp.distributions.Normal(loc=means, scale=stds)
        pred_dist = tfp.distributions.Independent(tfp.distributions.Normal(loc=means, scale=stds), reinterpreted_batch_ndims=1)
        log_likelihood = pred_dist.log_prob(y)

        posterior = tfp.distributions.Independent(tfp.distributions.Normal(loc=mu, scale=tf.math.log(1 + tf.exp(rho))), reinterpreted_batch_ndims=2)  # ???

        #  f(w,θ) = loq q(w|θ) - log(P(w)) - log(P(D|w))
        loss = posterior.log_prob(w_var) - prior_w.log_prob(w_var) - log_likelihood  # this is also f

    grads = tape.gradient(loss, [mu, rho, w_var])

    delta_mu = grads[2] + grads[0]
    delta_rho = grads[2] * (eps / 1 + tf.exp(-rho)) + grads[1]

    opt.apply_gradients(zip([delta_mu, delta_rho], [mu, rho]))

    print(loss)

...
