import jax.numpy as np
import jax.random as random
from jax import vmap, jit
from jax.scipy.linalg import cholesky, solve_triangular
from jax.config import config
config.update("jax_enable_x64", True)

from jaxbo.models import GPmodel
import jaxbo.kernels as kernels

from numpyro import sample, handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from functools import partial
import time

# A minimal Gaussian process regression class (inherits from GPmodel)
class GP(GPmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    def model(self, batch):
        X = batch['X']
        y = batch['y']
        N, D = X.shape
        # set uninformative log-normal priors
        var = sample('kernel_var', dist.LogNormal(0.0, 10.0))
        length = sample('kernel_length', dist.LogNormal(np.zeros(D),
                                                        10.0*np.ones(D)))
        noise = sample('noise_var', dist.LogNormal(0.0, 10.0))
        theta = np.concatenate([np.array([var]), np.array(length)])

        # compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(noise + 1e-8)

        # sample Y according to the standard gaussian process formula
        sample("y", dist.MultivariateNormal(loc=np.zeros(N), covariance_matrix=K), obs=y)

    # helper function for doing hmc inference
    def train(self, batch, rng_key, settings):
        start = time.time()
        kernel = NUTS(self.model,
                      target_accept_prob = settings['target_accept_prob'])
        mcmc = MCMC(kernel,
                    num_warmup = settings['num_warmup'],
                    num_samples = settings['num_samples'],
                    num_chains = settings['num_chains'],
                    progress_bar=True,
                    jit_model_args=True)
        mcmc.run(rng_key, batch)
        mcmc.print_summary()
        elapsed = time.time() - start
        print('\nMCMC elapsed time: %.2f seconds' % (elapsed))
        return mcmc.get_samples()

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        X = batch['X']
        N, D = X.shape
        # Fetch params
        sigma_n = params[-1]
        theta = params[:-1]
        # Compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(sigma_n + 1e-8)
        L = cholesky(K, lower=True)
        return L

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        params = kwargs['params']
        batch = kwargs['batch']
        norm_const = kwargs['norm_const']
        # Normalize
        X_star = (X_star - norm_const['mu_X'])/norm_const['sigma_X']
        # Fetch training data
        X, y = batch['X'], batch['y']
        # Fetch params
        sigma_n = params[-1]
        theta = params[:-1]
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0])*(sigma_n + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        # Denormalize
        mu = mu*norm_const['sigma_y'] + norm_const['mu_y']
        std = std*norm_const['sigma_y']**2
        return mu, std

# A minimal Gaussian process regression class (inherits from GPmodel)
class BayesianMLP(GPmodel):
    # Initialize the class
    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers

    def model(self, X, y):
        H = X
        N, D = H.shape
        # Forward pass
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            D_X, D_H = self.layers[l], self.layers[l+1]
            W = sample('w%d' % (l+1), dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))
            b = sample('b%d' % (l+1), dist.Normal(np.zeros(D_H), np.ones(D_H)))
            H = np.tanh(np.add(np.matmul(H, W), b))
        D_X, D_H = self.layers[-2], self.layers[-1]
        W = sample('w%d' % (num_layers-1), dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))
        b = sample('b%d' % (num_layers-1), dist.Normal(np.zeros(D_H), np.ones(D_H)))
        H = np.add(np.matmul(H, W), b)
        # Noise
        noise_var = sample("noise_var", dist.LogNormal(0.0, 10.0))
        # Likelihood
        sample("y", dist.Normal(H, noise_var), obs=y)

    # helper function for doing hmc inference
    def train(self, batch, rng_key, settings):
        start = time.time()
        kernel = NUTS(self.model,
                      target_accept_prob = settings['target_accept_prob'])
        mcmc = MCMC(kernel,
                    num_warmup = settings['num_warmup'],
                    num_samples = settings['num_samples'],
                    num_chains = settings['num_chains'],
                    progress_bar=True,
                    jit_model_args=True)
        X, y = batch['X'], batch['y']
        mcmc.run(rng_key, X, y)
        # mcmc.print_summary()
        elapsed = time.time() - start
        print('\nMCMC elapsed time: %.2f seconds' % (elapsed))
        return mcmc.get_samples()

    @partial(jit, static_argnums=(0,))
    def forward(self, H, samples, rng_key):
        N = H.shape[0]
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            W = samples['w%d'%(l+1)]
            b = samples['b%d'%(l+1)]
            H = np.tanh(np.add(np.matmul(H, W), b))
        W = samples['w%d'%(num_layers-1)]
        b = samples['b%d'%(num_layers-1)]
        H = np.add(np.matmul(H, W), b)
        noise = samples['noise_var']*random.normal(rng_key, H.shape)
        return H + noise

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        N = X_star.shape[0]
        samples = kwargs['samples']
        rng_keys = kwargs['rng_keys']
        norm_const = kwargs['norm_const']
        X_star = (X_star - norm_const['mu_X'])/norm_const['sigma_X']
        pred_fun = lambda samples, rng_key: self.forward(X_star, samples, rng_key)
        predictions = vmap(pred_fun)(samples, rng_keys)
        mu = np.mean(predictions, axis = 0)
        std = np.std(predictions, axis = 0)
        # Denormalize
        mu = mu*norm_const['sigma_y'] + norm_const['mu_y']
        std = norm_const['sigma_y']**2
        return mu.flatten(), std.flatten()
