import numpy as onp
import jax.numpy as np
import jax.random as random
from jax import vmap, jit, vjp
from jax.scipy.linalg import cholesky, solve_triangular
from jax.scipy.stats import multivariate_normal
from scipy.optimize import minimize

from pyDOE import lhs
from functools import partial
from sklearn import mixture
from KDEpy import FFTKDE
from scipy.interpolate import interp1d

# A minimal Gaussian process class
class GP:
    # Initialize the class
    def __init__(self):
        return

    def random_init(self, rng_key, dim):
        logsigma_f = np.log(2.0*random.uniform(rng_key, (1,)))
        log_ell = np.log(random.uniform(rng_key, (dim,)))
        logsigma_n = np.array([-4.0]) + random.normal(rng_key, (1,))
        hyp = np.concatenate([logsigma_f, log_ell, logsigma_n])
        return hyp

    @partial(jit, static_argnums=(0,))
    def kernel(self, x1, x2, params):
        output_scale = np.exp(params[0])
        lengthscales = np.exp(params[1:])
        diffs = np.expand_dims(x1 / lengthscales, 1) - \
                np.expand_dims(x2 / lengthscales, 0)
        return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

    @partial(jit, static_argnums=(0,))
    def likelihood(self, params, batch):
        X, y = batch['X'], batch['y']
        N, D = X.shape
        # Fetch params
        sigma_n = np.exp(params[-1])
        theta = params[:-1]
        # Compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(sigma_n + 1e-8)
        L = cholesky(K, lower=True)
        # Compute negative log-marginal likelihood
        alpha = solve_triangular(L.T,solve_triangular(L, y, lower=True))
        NLML = 0.5*np.matmul(np.transpose(y),alpha) + \
               np.sum(np.log(np.diag(L))) + 0.5*N*np.log(2.0*np.pi)
        return NLML

    @partial(jit, static_argnums=(0,))
    def likelihood_value_and_grad(self, params, batch):
        fun = lambda params: self.likelihood(params, batch)
        primals, f_vjp = vjp(fun, params)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    #  Prints the negative log-marginal likelihood at each training step
    def callback(self, params):
        print("Log likelihood: {}".format(self.likelihood(params)))

    # Minimizes the negative log-marginal likelihood
    def minimize(self, objective, x0, bnds = None):
        result = minimize(objective, x0, jac=True,
                          method='L-BFGS-B', bounds = bnds,
                          callback=None)
        return result.x, result.fun

    def train(self, batch, rng_key, num_restarts = 10):
        # Define objective that returns NumPy arrays
        def objective(params):
            value, grads = self.likelihood_value_and_grad(params, batch)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        params = []
        likelihood = []
        dim = batch['X'].shape[1]
        rng_key = random.split(rng_key, num_restarts)
        for i in range(num_restarts):
            init = self.random_init(rng_key[i], dim)
            p, val = self.minimize(objective, init)
            params.append(p)
            likelihood.append(val)
        params = np.vstack(params)
        likelihood = np.vstack(likelihood)
        idx_best = np.argmin(likelihood)
        best_params = params[idx_best,:]
        return best_params

    @partial(jit, static_argnums=(0,))
    def compute_cholesky(self, params, batch):
        X, y = batch['X'], batch['y']
        N, D = X.shape
        # Fetch params
        sigma_n = np.exp(params[-1])
        theta = params[:-1]
        # Compute kernel
        K = self.kernel(X, X, theta) + np.eye(N)*(sigma_n + + 1e-8)
        L = cholesky(K, lower=True)
        return L

    @partial(jit, static_argnums=(0,))
    def predict(self, X_star, *args):
        params, batch, norm_const = args
        # Normalize
        X_star = (X_star - norm_const['mu_X'])/norm_const['sigma_X']
        # Fetch training data
        X, y = batch['X'], batch['y']
        # Fetch params
        sigma_n = np.exp(params[-1])
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

    def fit_kernel_density(self, X, xi):
        kde_pdf_x, kde_pdf_y  = FFTKDE().fit(onp.array(X)).evaluate()
        # Define the interpolation function
        interp1d_fun = interp1d(kde_pdf_x,
                                kde_pdf_y,
                                kind = 'linear',
                                fill_value = 'extrapolate')
        # Evaluate the weights on the input data
        pdf = interp1d_fun(xi)
        return np.clip(pdf, a_min=0.0) + 1e-8

    def fit_gmm(self, *args, num_comp = 2, N_samples = 10000):
        params, batch, norm_const, bounds, input_prior, rng_key = args
        lb = bounds['lb']
        ub = bounds['ub']
        dim = batch['X'].shape[1]
        # Sample data across the entire domain
        X = lb + (ub-lb)*lhs(dim, N_samples)
        y = self.predict(X,
                         params,
                         batch,
                         norm_const)[0]
        # Sample data according to prior
        X_samples = input_prior.sample(rng_key, N_samples)
        y_samples = self.predict(X_samples,
                                 params,
                                 batch,
                                 norm_const)[0]
        # Compute p_x and p_y from samples across the entire domain
        p_x = input_prior.pdf(X)
        p_y = self.fit_kernel_density(y_samples, y)
        weights = p_x/p_y
        # Label each input data
        indices = np.arange(N_samples)
        # Scale inputs data to [0, 1]^D
        X = (X - lb) / (ub - lb)
        # rescale weights as probability distribution
        weights = weights / np.sum(weights)
        # Sample from analytical w
        idx = onp.random.choice(indices,
                                N_samples,
                                p=weights.flatten())
        X_train = X[idx]
        # fit GMM
        clf = mixture.GaussianMixture(n_components=num_comp,
                                      covariance_type='full')
        clf.fit(X_train)
        return clf.weights_, clf.means_, clf.covariances_

    @partial(jit, static_argnums=(0,))
    def compute_w_gmm(self, x, *args):
        bounds, gmm_vars = args
        lb = bounds['lb']
        ub = bounds['ub']
        x = (x - lb) / (ub - lb)
        weights, means, covs = gmm_vars
        gmm_mode = lambda w, mu, cov:  w*multivariate_normal.pdf(x, mu, cov)
        w = np.sum(vmap(gmm_mode)(weights, means, covs), axis = 0)
        return w

    @partial(jit, static_argnums=(0,))
    def LW_LCB(self, x, *args, kappa = 2.0):
        params, batch, norm_const, bounds, gmm_vars = args
        mean, std = self.predict(x,
                                 params,
                                 batch,
                                 norm_const)
        weights = self.compute_w_gmm(x, bounds, gmm_vars)
        lcb = mean - kappa*std*weights
        return lcb

    @partial(jit, static_argnums=(0,))
    def LW_LCB_sp(self, x, *args, kappa = 2.0):
        params, batch, norm_const, bounds, gmm_vars = args
        x = x[None,:]
        mean, std = self.predict(x,
                                 params,
                                 batch,
                                 norm_const)
        weights = self.compute_w_gmm(x, bounds, gmm_vars)
        lcb = mean - kappa*std*weights
        return lcb[0]

    @partial(jit, static_argnums=(0,))
    def acq_value_and_grad(self, x, *args):
        fun = lambda x: self.LW_LCB_sp(x, *args)
        primals, f_vjp = vjp(fun, x)
        grads = f_vjp(np.ones_like(primals))[0]
        return primals, grads

    def compute_next_point(self, *args, num_restarts=10):
        params, batch, norm_const, bounds, gmm_vars = args
        # Define objective that returns NumPy arrays
        def objective(x):
            value, grads = self.acq_value_and_grad(x, *args)
            out = (onp.array(value), onp.array(grads))
            return out
        # Optimize with random restarts
        loc = []
        acq = []
        lb = bounds['lb']
        ub = bounds['ub']
        dim = batch['X'].shape[1]
        x0 = lb + (ub-lb)*lhs(dim, num_restarts)
        dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
        for i in range(num_restarts):
            pos, val = self.minimize(objective, x0[i,:], bnds = dom_bounds)
            loc.append(pos)
            acq.append(val)
        loc = np.vstack(loc)
        acq = np.vstack(acq)
        idx_best = np.argmin(acq)
        x_new = loc[idx_best:idx_best+1,:]
        return x_new
