import numpy as onp
import jax.numpy as np
from jax import jit, vmap
from jax.experimental import stax
from jax.experimental.stax import Dense, Tanh
from jax.nn.initializers import glorot_normal, normal
from jax.scipy.stats import multivariate_normal
from KDEpy import FFTKDE
from scipy.interpolate import interp1d

from scipy.stats import gaussian_kde

#import matplotlib.pyplot as plt



@jit
def normalize(X, y, bounds):
    mu_y, sigma_y = y.mean(0), y.std(0)
    X = (X - bounds['lb'])/(bounds['ub']-bounds['lb'])
    y = (y - mu_y)/sigma_y
    batch = {'X': X, 'y': y}
    norm_const = {'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const


@jit
def normalize_MultifidelityGP(XL, yL, XH, yH, bounds):
    y = np.concatenate([yL, yH], axis = 0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - bounds['lb'])/(bounds['ub']-bounds['lb'])
    XH = (XH - bounds['lb'])/(bounds['ub']-bounds['lb'])
    yL = (yL - mu_y)/sigma_y
    yH = (yH - mu_y)/sigma_y
    y = (y - mu_y)/sigma_y
    batch = {'XL': XL, 'XH': XH, 'y': y, 'yL': yL, 'yH': yH}
    norm_const = {'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def normalize_GradientGP(XF, yF, XG, yG):
    y = np.concatenate([yF, yG], axis = 0)
    batch = {'XF': XF, 'XG': XG, 'yF': yF, 'yG': yG, 'y': y}
    norm_const = {'mu_X': 0.0, 'sigma_X': 1.0,
                  'mu_y': 0.0, 'sigma_y': 1.0}
    return batch, norm_const


@jit
def normalize_HeterogeneousMultifidelityGP(XL, yL, XH, yH, bounds):
    y = np.concatenate([yL, yH], axis = 0)
    mu_X, sigma_X = XL.mean(0), XL.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - mu_X)/sigma_X
    XH = (XH - bounds['lb'])/(bounds['ub']-bounds['lb'])
    yL = (yL - mu_y)/sigma_y
    yH = (yH - mu_y)/sigma_y
    y = (y - mu_y)/sigma_y
    batch = {'XL': XL, 'XH': XH, 'y': y, 'yL': yL, 'yH': yH}
    norm_const = {'mu_X': mu_X, 'sigma_X': sigma_X,
                  'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def standardize(X, y):
    mu_X, sigma_X = X.mean(0), X.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    X = (X - mu_X)/sigma_X
    y = (y - mu_y)/sigma_y
    batch = {'X': X, 'y': y}
    norm_const = {'mu_X': mu_X, 'sigma_X': sigma_X,
                  'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const


@jit
def standardize_MultifidelityGP(XL, yL, XH, yH):
    X = np.concatenate([XL, XH], axis = 0)
    y = np.concatenate([yL, yH], axis = 0)
    mu_X, sigma_X = X.mean(0), X.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - mu_X)/sigma_X
    XH = (XH - mu_X)/sigma_X
    yL = (yL - mu_y)/sigma_y
    yH = (yH - mu_y)/sigma_y
    y = (y - mu_y)/sigma_y
    batch = {'XL': XL, 'XH': XH, 'y': y, 'yL': yL, 'yH': yH}
    norm_const = {'mu_X': mu_X, 'sigma_X': sigma_X,
                  'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def standardize_HeterogeneousMultifidelityGP(XL, yL, XH, yH):
    y = np.concatenate([yL, yH], axis = 0)
    mu_XL, sigma_XL = XL.mean(0), XL.std(0)
    min_XH, max_XH = XH.min(0), XH.max(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    XL = (XL - mu_XL)/sigma_XL
    XH = (XH - min_XH)/(max_XH-min_XH)
    yL = (yL - mu_y)/sigma_y
    yH = (yH - mu_y)/sigma_y
    y = (y - mu_y)/sigma_y
    batch = {'XL': XL, 'XH': XH, 'y': y, 'yL': yL, 'yH': yH}
    norm_const = {'mu_XL': mu_XL, 'sigma_XL': sigma_XL,
                  'min_XH': min_XH, 'max_XH': max_XH,
                  'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const

@jit
def compute_w_gmm(x, **kwargs):
    bounds = kwargs['bounds']
    lb = bounds['lb']
    ub = bounds['ub']
    x = (x - lb) / (ub - lb)
    weights, means, covs = kwargs['gmm_vars']
    gmm_mode = lambda w, mu, cov:  w*multivariate_normal.pdf(x, mu, cov)
    w = np.sum(vmap(gmm_mode)(weights, means, covs), axis = 0)
    return w

def fit_kernel_density(X, xi, weights = None, bw=None):
#    kde_pdf_x, kde_pdf_y  = FFTKDE().fit(onp.array(X)).evaluate()

    X, weights = onp.array(X), onp.array(weights)
    X = X.flatten()
    if bw is None:
        try:
            sc = gaussian_kde(X, weights=weights)
            bw = onp.sqrt(sc.covariance).flatten()
        except:
            bw = 1.0
        if bw < 1e-8:
            bw = 1.0

    #print("bw", bw)

    kde_pdf_x, kde_pdf_y = FFTKDE(bw=bw).fit(X, weights).evaluate()


    #plt.hist(X, bins = 50, density = True)
    #plt.plot(kde_pdf_x, kde_pdf_y, 'r-')
    #plt.show()

    # Define the interpolation function
    interp1d_fun = interp1d(kde_pdf_x,
                            kde_pdf_y,
                            kind = 'linear',
                            fill_value = 'extrapolate')

    # Evaluate the weights on the input data
    pdf = interp1d_fun(xi)
    return np.clip(pdf, a_min=0.0) + 1e-8

def init_NN(Q):
    layers = []
    num_layers = len(Q)
    for i in range(0, num_layers-2):
        layers.append(Dense(Q[i+1],
                            W_init=glorot_normal(dtype=np.float64),
                            b_init=normal(dtype=np.float64)))
        layers.append(Tanh)
    layers.append(Dense(Q[-1],
                  W_init=glorot_normal(dtype=np.float64),
                  b_init=normal(dtype=np.float64)))
    net_init, net_apply = stax.serial(*layers)
    return net_init, net_apply
