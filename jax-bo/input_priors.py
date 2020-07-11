import jax.numpy as np
from jax import random
from jax.scipy.stats import multivariate_normal, uniform

class uniform_prior:
    def __init__(self, lb=0, ub=1):
        self.lb = lb
        self.ub = ub
        self.dim = lb.shape[0]
    def sample(self, rng_key, N):
        return self.lb + (self.ub-self.lb)*random.uniform(rng_key, (N, self.dim))
    def pdf(self, x):
        return np.sum(uniform.pdf(x, self.lb, self.ub-self.lb), axis=-1)

class gaussian_prior:
    def __init__(self, mu, cov):
        self.mu = mu
        self.cov = cov
        self.dim = mu.shape[0]
    def sample(self, rng_key, N):
        return random.multivariate_normal(rng_key, self.mu, self.cov, (N,))
    def pdf(self, x):
        return multivariate_normal.pdf(x, self.mu, self.cov)
