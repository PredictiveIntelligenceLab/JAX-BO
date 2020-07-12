import numpy as onp
import jax.numpy as np
from jax import jit, vmap
from jax.scipy.stats import multivariate_normal
from KDEpy import FFTKDE
from scipy.interpolate import interp1d

@jit
def normalize(X, y):
    mu_X, sigma_X = X.mean(0), X.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    X = (X - mu_X)/sigma_X
    y = (y - mu_y)/sigma_y
    batch = {'X': X, 'y': y}
    norm_const = {'mu_X': mu_X, 'sigma_X': sigma_X,
                  'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def compute_w_gmm(x, bounds, gmm_vars):
    lb = bounds['lb']
    ub = bounds['ub']
    x = (x - lb) / (ub - lb)
    weights, means, covs = gmm_vars
    gmm_mode = lambda w, mu, cov:  w*multivariate_normal.pdf(x, mu, cov)
    w = np.sum(vmap(gmm_mode)(weights, means, covs), axis = 0)
    return w

def fit_kernel_density(X, xi):
    kde_pdf_x, kde_pdf_y  = FFTKDE().fit(onp.array(X)).evaluate()
    # Define the interpolation function
    interp1d_fun = interp1d(kde_pdf_x,
                            kde_pdf_y,
                            kind = 'linear',
                            fill_value = 'extrapolate')
    # Evaluate the weights on the input data
    pdf = interp1d_fun(xi)
    return np.clip(pdf, a_min=0.0) + 1e-8
