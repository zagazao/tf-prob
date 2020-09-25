import numpy as np
import scipy.stats as stats


import tf.keras.layers

steps = 10000
n_samples = 100
data = np.random.normal(loc=2, size=n_samples)
# print(data)
print(data.mean(), data.std())

prior = stats.norm(loc=2.5, scale=2.5)

x = np.linspace(0, 5, 100)
y = prior.pdf(x)

print(y)

# plt.plot(x, y)
# plt.title('Prior')
# plt.show()

# Now I need the likelihood.
# Log_likelihood = \sum_{x,y} N(y|

initial_guess = 0

proposal_dist = stats.norm(loc=0, scale=1.5)


def eval_prop_posterior(_parameter):
    return prior.pdf(_parameter) * stats.norm(loc=_parameter, scale=1).pdf(x).prod()


analytical_var = 1 / (1 / (2.5 * 2.5) + 1 / (1 / n_samples))
analytical_solutin = analytical_var * (2.5 / 2.5 + data.mean() / (1 / n_samples))

print(analytical_solutin, analytical_var)

for i in range(steps):
    new_guess = initial_guess + proposal_dist.rvs()

    current_ll = eval_prop_posterior(initial_guess)
    new_ll = eval_prop_posterior(new_guess)

    p_accept = min(1, new_ll / current_ll)
    # print(p_accept)

    if stats.bernoulli(p=p_accept).rvs():
        initial_guess = new_guess
        print('Accept:', initial_guess)
