import jax.numpy as np
import jax.random as random
from jax import vmap, jit
from jax.scipy.linalg import cholesky, solve_triangular

from jaxbo.models import GPmodel
import jaxbo.kernels as kernels

from numpyro import sample, handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from functools import partial

# A minimal MCMC model class (inherits from GPmodel)
class MCMCmodel(GPmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    # helper function for doing hmc inference
    def train(self, batch, rng_key, settings):
        kernel = NUTS(self.model,
                      target_accept_prob = settings['target_accept_prob'])
        mcmc = MCMC(kernel,
                    num_warmup = settings['num_warmup'],
                    num_samples = settings['num_samples'],
                    num_chains = settings['num_chains'],
                    progress_bar=True,
                    jit_model_args=True)
        mcmc.run(rng_key, batch)
        # mcmc.print_summary()
        return mcmc.get_samples()

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        # Normalize
        norm_const = kwargs['norm_const']
        X_star = (X_star - norm_const['mu_X'])/norm_const['sigma_X']
        # Vectorized predictions
        rng_keys = kwargs['rng_keys']
        samples = kwargs['samples']
        sample_fn = lambda key, sample: self.posterior_sample(key,
                                                              sample,
                                                              X_star,
                                                              **kwargs)
        means, predictions = vmap(sample_fn)(rng_keys, samples)
        mean_prediction = np.mean(means, axis=0)
        std_prediction = np.std(predictions, axis=0)
        return mean_prediction, std_prediction


# A minimal Gaussian process regression class (inherits from MCMCmodel)
class GP(MCMCmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    def model(self, batch):
        X = batch['X']
        y = batch['y']
        N, D = X.shape
        # set uninformative log-normal priors
        var = sample('kernel_var', dist.LogNormal(0.0, 10.0))
        length = sample('kernel_length', dist.LogNormal(np.zeros(D), 10.0*np.ones(D)))
        noise = sample('noise_var', dist.LogNormal(0.0, 10.0))
        theta = np.concatenate([np.array([var]), np.array(length)])
        # compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(noise + 1e-8)
        # sample Y according to the standard gaussian process formula
        sample("y", dist.MultivariateNormal(loc=np.zeros(N), covariance_matrix=K), obs=y)

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
    def posterior_sample(self, key, sample, X_star, **kwargs):
        # Fetch training data
        norm_const = kwargs['norm_const']
        batch = kwargs['batch']
        X, y = batch['X'], batch['y']
        # Fetch params
        var = sample['kernel_var']
        length = sample['kernel_length']
        noise = sample['noise_var']
        params = np.concatenate([np.array([var]), np.array(length), np.array([noise])])
        theta = params[:-1]
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0])*(noise + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        sample = mu + std * random.normal(key, mu.shape)
        mu = mu*norm_const['sigma_y'] + norm_const['mu_y']
        sample = sample*norm_const['sigma_y'] + norm_const['mu_y']
        return mu, sample


# A minimal Gaussian process regression class (inherits from MCMCmodel)
class BayesianMLP(MCMCmodel):
    # Initialize the class
    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers

    def model(self, batch):
        X = batch['X']
        y = batch['y']
        N, D = X.shape
        H = X
        # Forward pass
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            D_X, D_H = self.layers[l], self.layers[l+1]
            W = sample('w%d' % (l+1), dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))
            b = sample('b%d' % (l+1), dist.Normal(np.zeros(D_H), np.ones(D_H)))
            H = np.tanh(np.add(np.matmul(H, W), b))
        D_X, D_H = self.layers[-2], self.layers[-1]
        # Output mean
        W = sample('w%d_mu' % (num_layers-1), dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))
        b = sample('b%d_mu' % (num_layers-1), dist.Normal(np.zeros(D_H), np.ones(D_H)))
        mu = np.add(np.matmul(H, W), b)
        # Output std
        W = sample('w%d_std' % (num_layers-1), dist.Normal(np.zeros((D_X, D_H)), np.ones((D_X, D_H))))
        b = sample('b%d_std' % (num_layers-1), dist.Normal(np.zeros(D_H), np.ones(D_H)))
        std = np.add(np.matmul(H, W), b)
        mu, std = mu.flatten(), std.flatten()
        # Likelihood
        sample("y", dist.Normal(mu, std**2), obs=y)

    @partial(jit, static_argnums=(0,))
    def forward(self, H, sample):
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            W = sample['w%d'%(l+1)]
            b = sample['b%d'%(l+1)]
            H = np.tanh(np.add(np.matmul(H, W), b))
        W = sample['w%d_mu'%(num_layers-1)]
        b = sample['b%d_mu'%(num_layers-1)]
        mu = np.add(np.matmul(H, W), b)
        W = sample['w%d_std'%(num_layers-1)]
        b = sample['b%d_std'%(num_layers-1)]
        std = np.add(np.matmul(H, W), b)
        return mu, std

    @partial(jit, static_argnums=(0,))
    def posterior_sample(self, key, sample, X_star, **kwargs):
        mu, std = self.forward(X_star, sample)
        sample = mu + std * random.normal(key, mu.shape)
        # De-normalize
        norm_const = kwargs['norm_const']
        mu = mu*norm_const['sigma_y'] + norm_const['mu_y']
        sample = sample*norm_const['sigma_y'] + norm_const['mu_y']
        return mu.flatten(), sample.flatten()


# A minimal Gaussian process regression class (inherits from MCMCmodel)
class MissingInputsGP(MCMCmodel):
    # Initialize the class
    def __init__(self, options, layers):
        super().__init__(options)
        self.layers = layers

    def model(self, X, y):
        H = X
        N, D = X.shape
        # Generate latent inputs
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
        # Concatenate true and latent inputs
        X = np.concatenate([X, H], axis = 1)
        # set uninformative log-normal priors on GP hyperparameters
        var = sample('kernel_var', dist.LogNormal(0.0, 10.0))
        length = sample('kernel_length', dist.LogNormal(np.zeros(D), 10.0*np.ones(D)))
        noise = sample('noise_var', dist.LogNormal(0.0, 10.0))
        theta = np.concatenate([np.array([var]), np.array(length)])
        # compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(noise + 1e-8)
        # sample Y according to the GP likelihood
        sample("y", dist.MultivariateNormal(loc=np.zeros(N), covariance_matrix=K), obs=y)

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
    def forward(self, H, sample):
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            W = sample['w%d'%(l+1)]
            b = sample['b%d'%(l+1)]
            H = np.tanh(np.add(np.matmul(H, W), b))
        W = sample['w%d'%(num_layers-1)]
        b = sample['b%d'%(num_layers-1)]
        H = np.add(np.matmul(H, W), b)
        return H

    @partial(jit, static_argnums=(0,))
    def posterior_sample(self, key, sample, X_star, **kwargs):
        # Predict latent inputs at test locations
        H_star = self.forward(X_star, sample)
        X_star = np.concatenate([X_star, H_star], axis=1)
        # Predict latent inputs at training locations
        batch = kwargs['batch']
        X, y = batch['X'], batch['y']
        H = self.forward(X, sample)
        X = np.concatenate([X, H], axis=1)
        # Fetch GP params
        var = sample['kernel_var']
        length = sample['kernel_length']
        noise = sample['noise_var']
        params = np.concatenate([np.array([var]), np.array(length), np.array([noise])])
        theta = params[:-1]
        # Compute kernels
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0])*(noise + 1e-8)
        k_pX = self.kernel(X_star, X, theta)
        L = self.compute_cholesky(params, batch)
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        beta  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean, std
        mu = np.matmul(k_pX, alpha)
        cov = k_pp - np.matmul(k_pX, beta)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        sample = mu + std * random.normal(key, mu.shape)
        # De-normalize
        norm_const = kwargs['norm_const']
        mu = mu*norm_const['sigma_y'] + norm_const['mu_y']
        sample = sample*norm_const['sigma_y'] + norm_const['mu_y']
        return mu, sample
