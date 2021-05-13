import jax.numpy as np
import jax.random as random
from jax import vmap, jit
from jax.scipy.linalg import cholesky, solve_triangular
from jax.scipy.special import expit as sigmoid

from jaxbo.models import GPmodel
import jaxbo.kernels as kernels

from numpyro import sample, deterministic, handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from functools import partial

# A minimal MCMC model class (inherits from GPmodel)
class MCMCmodel(GPmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    # helper function for doing hmc inference
    def train(self, batch, rng_key, settings, verbose = False):
        kernel = NUTS(self.model,
                      target_accept_prob = settings['target_accept_prob'])
        mcmc = MCMC(kernel,
                    num_warmup = settings['num_warmup'],
                    num_samples = settings['num_samples'],
                    num_chains = settings['num_chains'],
                    progress_bar=True,
                    jit_model_args=True)
        mcmc.run(rng_key, batch)
        if verbose:
            mcmc.print_summary()
        return mcmc.get_samples()

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, **kwargs):
        # Normalize to [0,1]
        bounds = kwargs['bounds']
        X_star = (X_star - bounds['lb'])/(bounds['ub'] - bounds['lb'])
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


# A minimal Gaussian process classification class (inherits from MCMCmodel)
class GPclassifier(MCMCmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    def model(self, batch):
        X = batch['X']
        y = batch['y']
        N, D = X.shape
        # set uninformative log-normal priors
        var = sample('kernel_var', dist.LogNormal(0.0, 1.0), sample_shape = (1,))
        length = sample('kernel_length', dist.LogNormal(0.0, 1.0), sample_shape = (D,))
        theta = np.concatenate([var, length])
        # compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*1e-8
        L = cholesky(K, lower=True)
        # Generate latent function
        beta = sample('beta', dist.Normal(0.0, 1.0))
        eta = sample('eta', dist.Normal(0.0, 1.0), sample_shape=(N,))
        f = np.matmul(L, eta) + beta
        # Bernoulli likelihood
        sample('y', dist.Bernoulli(logits=f), obs=y)

    @partial(jit, static_argnums=(0,))
    def posterior_sample(self, key, sample, X_star, **kwargs):
        # Fetch training data
        batch = kwargs['batch']
        X = batch['X']
        # Fetch params
        var = sample['kernel_var']
        length = sample['kernel_length']
        beta = sample['beta']
        eta = sample['eta']
        theta = np.concatenate([var, length])
        # Compute kernels
        K_xx = self.kernel(X, X, theta) + np.eye(X.shape[0])*1e-8
        k_pp = self.kernel(X_star, X_star, theta) + np.eye(X_star.shape[0])*1e-8
        k_pX = self.kernel(X_star, X, theta)
        L = cholesky(K_xx, lower=True)
        f = np.matmul(L, eta) + beta
        tmp_1 = solve_triangular(L.T,solve_triangular(L, f, lower=True))
        tmp_2  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean
        mu = np.matmul(k_pX, tmp_1)
        cov = k_pp - np.matmul(k_pX, tmp_2)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        sample = mu + std * random.normal(key, mu.shape)
        return mu, sample

# A minimal Gaussian process classification class (inherits from MCMCmodel)
class MultifidelityGPclassifier(MCMCmodel):
    # Initialize the class
    def __init__(self, options):
        super().__init__(options)

    def model(self, batch):
        XL, XH = batch['XL'], batch['XH']
        y = batch['y']
        NL, NH = XL.shape[0], XH.shape[0]
        D = XH.shape[1]
        # set uninformative log-normal priors for low-fidelity kernel
        var_L = sample('kernel_var_L', dist.LogNormal(0.0, 1.0), sample_shape = (1,))
        length_L = sample('kernel_length_L', dist.LogNormal(0.0, 1.0), sample_shape = (D,))
        theta_L = np.concatenate([var_L, length_L])
        # set uninformative log-normal priors for high-fidelity kernel
        var_H = sample('kernel_var_H', dist.LogNormal(0.0, 1.0), sample_shape = (1,))
        length_H = sample('kernel_length_H', dist.LogNormal(0.0, 1.0), sample_shape = (D,))
        theta_H = np.concatenate([var_H, length_H])
        # prior for rho
        rho = sample('rho', dist.Normal(0.0, 10.0), sample_shape = (1,))
        # Compute kernels
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL)*1e-8
        K_LH = rho*self.kernel(XL, XH, theta_L)
        K_HH = rho**2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH)*1e-8
        K = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = cholesky(K, lower=True)
        # Generate latent function
        beta_L = sample('beta_L', dist.Normal(0.0, 1.0))
        beta_H = sample('beta_H', dist.Normal(0.0, 1.0))
        eta_L = sample('eta_L', dist.Normal(0.0, 1.0), sample_shape=(NL,))
        eta_H = sample('eta_H', dist.Normal(0.0, 1.0), sample_shape=(NH,))
        beta = np.concatenate([beta_L*np.ones(NL), beta_H*np.ones(NH)])
        eta = np.concatenate([eta_L, eta_H])
        f = np.matmul(L, eta) + beta
        # Bernoulli likelihood
        sample('y', dist.Bernoulli(logits=f), obs=y)

    @partial(jit, static_argnums=(0,))
    def posterior_sample(self, key, sample, X_star, **kwargs):
        # Fetch training data
        batch = kwargs['batch']
        XL, XH = batch['XL'], batch['XH']
        NL, NH = XL.shape[0], XH.shape[0]
        # Fetch params
        var_L = sample['kernel_var_L']
        var_H = sample['kernel_var_H']
        length_L = sample['kernel_length_L']
        length_H = sample['kernel_length_H']
        beta_L = sample['beta_L']
        beta_H = sample['beta_H']
        eta_L = sample['eta_L']
        eta_H = sample['eta_H']
        rho = sample['rho']
        theta_L = np.concatenate([var_L, length_L])
        theta_H = np.concatenate([var_H, length_H])
        beta = np.concatenate([beta_L*np.ones(NL), beta_H*np.ones(NH)])
        eta = np.concatenate([eta_L, eta_H])
        # Compute kernels
        k_pp = rho**2 * self.kernel(X_star, X_star, theta_L) + \
                        self.kernel(X_star, X_star, theta_H) + \
                        np.eye(X_star.shape[0])*1e-8
        psi1 = rho*self.kernel(X_star, XL, theta_L)
        psi2 = rho**2 * self.kernel(X_star, XH, theta_L) + \
                        self.kernel(X_star, XH, theta_H)
        k_pX = np.hstack((psi1,psi2))
        # Compute K_xx
        K_LL = self.kernel(XL, XL, theta_L) + np.eye(NL)*1e-8
        K_LH = rho*self.kernel(XL, XH, theta_L)
        K_HH = rho**2 * self.kernel(XH, XH, theta_L) + \
                        self.kernel(XH, XH, theta_H) + np.eye(NH)*1e-8
        K_xx = np.vstack((np.hstack((K_LL,K_LH)),
                       np.hstack((K_LH.T,K_HH))))
        L = cholesky(K_xx, lower=True)
        # Sample latent function
        f = np.matmul(L, eta) + beta
        tmp_1 = solve_triangular(L.T,solve_triangular(L, f, lower=True))
        tmp_2  = solve_triangular(L.T,solve_triangular(L, k_pX.T, lower=True))
        # Compute predictive mean
        mu = np.matmul(k_pX, tmp_1)
        cov = k_pp - np.matmul(k_pX, tmp_2)
        std = np.sqrt(np.clip(np.diag(cov), a_min=0.))
        sample = mu + std * random.normal(key, mu.shape)
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
        sigma = np.exp(np.add(np.matmul(H, W), b))
        mu, sigma = mu.flatten(), sigma.flatten()
        # Likelihood
        sample("y", dist.Normal(mu, sigma), obs=y)

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
        sigma = np.exp(np.add(np.matmul(H, W), b))
        return mu, sigma

    @partial(jit, static_argnums=(0,))
    def posterior_sample(self, key, sample, X_star, **kwargs):
        mu, sigma = self.forward(X_star, sample)
        sample = mu + np.sqrt(sigma) * random.normal(key, mu.shape)
        # De-normalize
        norm_const = kwargs['norm_const']
        mu = mu*norm_const['sigma_y'] + norm_const['mu_y']
        sample = sample*norm_const['sigma_y'] + norm_const['mu_y']
        return mu.flatten(), sample.flatten()

# A minimal Gaussian process regression class (inherits from MCMCmodel)
# Work in progress..
class MissingInputsGP(MCMCmodel):
    # Initialize the class
    def __init__(self, options, dim_H, latent_bounds):
        super().__init__(options)
        self.dim_H = dim_H
        self.latent_bounds = latent_bounds

    def model(self, batch):
        X = batch['X']
        y = batch['y']
        N = y.shape[0]
        dim_X = X.shape[1]
        dim_H = self.dim_H
        D = dim_X + dim_H
        # Generate latent inputs
        H = sample('H', dist.Normal(np.zeros((N, dim_H)), np.ones((N, dim_H))))
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
    def posterior_sample(self, key, sample, X_star, **kwargs):
        batch = kwargs['batch']
        X, y = batch['X'], batch['y']
        # Fetch missing inputs
        H = sample['H']
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

