import jax.numpy as np
from jax import random

def random_init(rng_key, dim):
    logsigma_f = np.log(2.0*random.uniform(rng_key, (1,)))
    log_ell = np.log(random.uniform(rng_key, (dim,)))
    logsigma_n = np.array([-4.0]) + random.normal(rng_key, (1,))
    hyp = np.concatenate([logsigma_f, log_ell, logsigma_n])
    return hyp
