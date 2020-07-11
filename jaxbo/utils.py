import jax.numpy as np

def normalize(X, y):
    mu_X, sigma_X = X.mean(0), X.std(0)
    mu_y, sigma_y = y.mean(0), y.std(0)
    X = (X - mu_X)/sigma_X
    y = (y - mu_y)/sigma_y
    batch = {'X': X, 'y': y}
    norm_const = {'mu_X': mu_X, 'sigma_X': sigma_X,
                  'mu_y': mu_y, 'sigma_y': sigma_y}
    return batch, norm_const
