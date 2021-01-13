import jax.numpy as np
from jax import jit
from jax.scipy.stats import norm

# Caution: all functions are designed for single point evaluation
# (use vmap to vectorize)

@jit
def EI(mean, std, best):
    # from https://people.orie.cornell.edu/pfrazier/Presentations/2011.11.INFORMS.Tutorial.pdf
    delta = -(mean - best)
    deltap = -(mean - best)
    deltap = np.clip(deltap, a_min=0.)
    Z = delta/std
    EI = deltap - np.abs(deltap)*norm.cdf(-Z) + std*norm.pdf(Z)
    return -EI[0]

@jit
def EIC(mean, std, norm_const, best):
    # Constrained expected improvement
    delta = -(mean[0,:] - best)
    deltap = -(mean[0,:] - best)
    deltap = np.clip(deltap, a_min=0.)
    Z = delta/std[0,:]
    EI = deltap - np.abs(deltap)*norm.cdf(-Z) + std*norm.pdf(Z)
    constraints = np.prod(norm.cdf(mean[1:,:]/std[1:,:]), axis = 0)
    return -EI[0]*constraints[0]
    
    # Normalized constrainted EI
#    mu_y = norm_const['mu_y']
#    sigma_y = norm_const['sigma_y']
#    delta = -(mean[0,:] - best) / sigma_y
#    deltap = -(mean[0,:] - best) / sigma_y
#    deltap = np.clip(deltap, a_min=0.)
#    Z = delta/(std[0,:]/sigma_y)
#    EI = deltap - np.abs(deltap)*norm.cdf(-Z) + std*norm.pdf(Z)
#    constraints = np.prod(norm.cdf(mean[1:,:]/std[1:,:]), axis = 0)
#    return -EI[0]*constraints[0]
    
#    xi = 0.01
#    mu_y = norm_const['mu_y']
#    sigma_y = norm_const['sigma_y']
#    imp = -(mean[0,:] - best + xi) / sigma_y
#    Z = imp / (std[0,:]/sigma_y)
#    EI = (imp * norm.cdf(Z) + (std[0,:]/sigma_y) * norm.pdf(Z)) * np.clip(std[0,:], a_min=0.)
#    constraints = np.prod(norm.cdf(mean[1:,:]/std[1:,:]), axis = 0)
#    return -EI[0]*constraints[0]


@jit
def LCBC(mean, std, norm_const, kappa = 2.0):
    threshold = 3.0
    lcb = (mean[0,:] - norm_const['mu_y']) / norm_const['sigma_y'] - threshold - kappa*std[0,:] / norm_const['sigma_y']
    constraints = np.prod(norm.cdf(mean[1:,:]/std[1:,:]), axis = 0)
    return lcb[0]*constraints[0]

@jit
def LCB(mean, std, kappa = 2.0):
    lcb = mean - kappa*std
    return lcb[0]

@jit
def US(std):
    return -std[0]

@jit
def LW_LCB(mean, std, weights, kappa = 2.0):
    lw_lcb = mean - kappa*std*weights
    return lw_lcb[0]

@jit
def LW_US(std, weights):
    lw_us = std*weights
    return -lw_us[0]

@jit
def CLSF(mean, std, kappa = 1.0):
    acq = np.log(np.abs(mean)+1e-8) - kappa*np.log(std+1e-8)
    return acq[0]

@jit
def LW_CLSF(mean, std, weights, kappa = 1.0):
    acq = np.log(np.abs(mean)+1e-8) - kappa*(np.log(std+1e-8) + np.log(weights+1e-8))
    return acq[0]
