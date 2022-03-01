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
def EIC(mean, std, best):
    # Constrained expected improvement
    delta = -(mean[0,:] - best)
    deltap = -(mean[0,:] - best)
    deltap = np.clip(deltap, a_min=0.)
    Z = delta/std[0,:]
    EI = deltap - np.abs(deltap)*norm.cdf(-Z) + std*norm.pdf(Z)
    constraints = np.prod(norm.cdf(mean[1:,:]/std[1:,:]), axis = 0)
    return -EI[0]*constraints[0]

@jit
def LCBC(mean, std, kappa = 2.0, threshold = 3.0):
    lcb = mean[0,:] - threshold - kappa*std[0,:]
    constraints = np.prod(norm.cdf(mean[1:,:]/std[1:,:]), axis = 0)
    return lcb[0]*constraints[0]

@jit
def LW_LCBC(mean, std, weights, kappa = 2.0, threshold = 3.0):
    lcb = mean[0,:] - threshold - kappa*std[0,:]*weights
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

