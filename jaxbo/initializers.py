import jax.numpy as np
from jax import random

def random_init_GP(rng_key, dim):
    logsigma_f = np.log(4.0*random.uniform(rng_key, (1,)))
    loglength  = np.log(4.0*random.uniform(rng_key, (dim,)))
    logsigma_n = np.array([-4.0]) + random.normal(rng_key, (1,))
    hyp = np.concatenate([logsigma_f, loglength, logsigma_n])
    return hyp

def random_init_MultifidelityGP(rng_key, dim):
    key1, key2 = random.split(rng_key)
    logsigma_fL = np.log(4.0*random.uniform(key1, (1,)))
    loglength_L = np.log(4.0*random.uniform(key1, (dim,)))
    logsigma_fH = np.log(4.0*random.uniform(key2, (1,)))
    loglength_H = np.log(4.0*random.uniform(key2, (dim,)))
    rho = 2.0*random.normal(rng_key, (1,))
    logsigma_nL = np.array([-4.0]) + random.normal(key1, (1,))
    logsigma_nH = np.array([-4.0]) + random.normal(key2, (1,))
    hyp = np.concatenate([logsigma_fL, loglength_L,
                          logsigma_fH, loglength_H,
                          rho, logsigma_nL, logsigma_nH])
    return hyp
