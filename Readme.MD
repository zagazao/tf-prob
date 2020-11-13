# Bayesian Neural Networks

# Implementation notes:
- Tensorflow suggests to use kl_weight = 1 / num_samples_per_epoch

```python
model = tf.keras.Sequential([
    tfp.layers.DenseLocalReparameterization(512, activation=tf.nn.relu),
    tfp.layers.DenseLocalReparameterization(10),
])

# Here is a way to use a tensorflow function as loss...
logits = model(features)
neg_log_likelihood = tf.nn.softmax_cross_entropy_with_logits(
    labels=labels, logits=logits)
kl = sum(model.losses)
loss = neg_log_likelihood + kl
train_op = tf.train.AdamOptimizer().minimize(loss)

```

# Tensorflow Probability Layers

- What are the differences? Local Global and draw sample from prior?

- https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseFlipout
- https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseReparameterization
- https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseLocalReparameterization
- https://www.tensorflow.org/probability/api_docs/python/tfp/layers/DenseVariational

## Paper by year.

### 2011

- [Practical Variational Inference for Neural Networks](https://papers.nips.cc/paper/4329-practical-variational-inference-for-neural-networks)

### 2012

- [Variational Bayesian Inference with Stochastic Search](https://arxiv.org/abs/1206.6430)

### 2014

- [Black box variational inference](https://arxiv.org/abs/1401.0118)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

### 2015

- [Variational Dropout and the Local Reparameterization Trick](https://arxiv.org/abs/1506.02557)
- [Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424)
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142)

### 2016

- [Sticking the Landing: Simple, Lower-Variance Gradient Estimators for Variational Inference](https://arxiv.org/abs/1703.09194)

### 2017 

- [Reducing Reparameterization Gradient Variance](https://arxiv.org/abs/1705.07880)
- [Variational Dropout Sparsifies Deep Neural Networks](https://arxiv.org/abs/1701.05369)

### 2018 

- [Flipout: Efficient Pseudo-Independent Weight Perturbations on Mini-Batches](https://arxiv.org/abs/1803.04386)
- [Good Initializations of Variational Bayes for Deep Models](https://arxiv.org/abs/1810.08083)

# More interesting stuff:
- Deep Gaussian Process (VI?)

## Further todos:

- Implement the "good initialization". Therefore we need an implementation of probabilistic linear regression...

### Further papers:

- [Variational Bayesian dropout: pitfalls and fixes](https://arxiv.org/pdf/1807.01969.pdf)
- [Variational Gaussian Dropout is not Bayesian](https://arxiv.org/pdf/1711.02989.pdf)
- [Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/pdf/1506.02142.pdf)
- [Characteristics of Monte Carlo Dropout in Wide Neural Networks](https://arxiv.org/pdf/2007.05434.pdf)
- [A Systematic Comparison of Bayesian Deep LearningRobustness in Diabetic Retinopathy Tasks](http://bayesiandeeplearning.org/2019/papers/12.pdf)
- [Beyond temperature scaling: Obtaining well-calibrated multiclass probabilities with Dirichlet calibration](https://papers.nips.cc/paper/2019/file/8ca01ea920679a0fe3728441494041b9-Paper.pdf)
- [Bayesian Optimization with Robust Bayesian Neural Networks](https://papers.nips.cc/paper/2016/hash/a96d3afec184766bfeca7a9f989fc7e7-Abstract.html)

- [Bayesian Compression for Deep Learning](https://arxiv.org/abs/1705.08665)
- [Variational Dropout Sparsifies Deep Neural Networks](https://arxiv.org/abs/1701.05369)

### Random resources

- http://krasserm.github.io/2019/03/14/bayesian-neural-networks/
- http://www.tina-vision.net/docs/memos/2003-003.pdf (KL) 
- https://infoscience.epfl.ch/record/174055/files/durrieuThiranKelly_kldiv_icassp2012_R1.pdf