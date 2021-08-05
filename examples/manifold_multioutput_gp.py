#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 21:57:09 2021

@author: mohamedazizbhouri
"""

import numpy as onp
import jax.numpy as np
from jax import random, vmap
from jax.config import config
config.update("jax_enable_x64", True)

from pyDOE import lhs
import matplotlib.pyplot as plt
plt.close('all')
import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import griddata

from jaxbo.models import ManifoldGP_MultiOutputs
from jaxbo.utils import normalize
from jaxbo.test_functions import *

from jaxbo.input_priors import uniform_prior


onp.random.seed(1234)

def f(x):
    x1, x2 = x[0], x[1]
    a = 1.0
    b = 5.1 / (4*np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8*np.pi)
    f = a * (x2 - b*x1**2 + c*x1 -r)**2 + s * (1-t) * np.cos(x1) + s
    return f


def constraint1(x):
    x1, x2 = (x[0]-2.5)/7.5, (x[1] - 7.5)/7.5
    g1 = (4 - 2.1*x1**2 + 1./3*x1**4)*x1**2 + x1*x2 + (-4+4*x2**2)*x2**2 + 3*np.sin(6*(1-x1)) + 3*np.sin(6*(1-x2))
    return g1 - 6.

# Dimension of the problem
dim = 2

# Boundary of the domain
lb = np.array([-5.0, 0.0])
ub = np.array([10.0, 15.0])

bounds = {'lb': lb, 'ub': ub}

# Visualization of the function and constraints in 2D grid
nn = 100
xx = np.linspace(lb[0], ub[0], nn)
yy = np.linspace(lb[1], ub[1], nn)
XX, YY = np.meshgrid(xx, yy)
X_star = np.concatenate([XX.flatten()[:,None], 
                         YY.flatten()[:,None]], axis = 1)
y_f_star = vmap(f)(X_star)

y1_c_star = vmap(constraint1)(X_star)

Y_f_star = griddata(onp.array(X_star), onp.array(y_f_star), (onp.array(XX), onp.array(YY)), method='cubic')
Y1_c_star = griddata(onp.array(X_star), onp.array(y1_c_star), (onp.array(XX), onp.array(YY)), method='cubic')

# Problem settings
# Number of initial data for objective and constraints
N_f = 200
N_c = 200
noise_f = 0.00
noise_c = 0.00
nIter = 1 # 10

# Define prior distribution
p_x = uniform_prior(lb, ub)

# JAX-BO setting
options = {'kernel': 'RBF',
           'input_prior': p_x}
layers = [2, 5, 5, 1]
gp_model = ManifoldGP_MultiOutputs(options, layers)

# Domain bounds (already defined before where we visualized the data)
bounds = {'lb': lb, 'ub': ub}

# Initial training data for objective
X_f = lb + (ub-lb)*lhs(dim, N_f)
y_f = vmap(f)(X_f)
y_f = y_f + noise_f*y_f_star.std(0)*onp.random.normal(0, 1, size=y_f.shape)

# Initial training data for constraints
X_c = lb + (ub-lb)*lhs(dim, N_c)
y1_c = vmap(constraint1)(X_c)
y1_c = y1_c + noise_c*y1_c_star.std(0)*onp.random.normal(0, 1, size=y1_c.shape)

# Visualize the initial data for objective and constraints

plt.figure(figsize = (10,5))
plt.subplot(1, 2, 1)
fig = plt.contourf(XX, YY, Y_f_star)
plt.plot(X_f[:,0], X_f[:,1], 'ro', label = "Initial objective data")
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Output 1')
plt.colorbar(fig)

plt.subplot(1, 2, 2)
fig = plt.contourf(XX, YY, Y1_c_star)
plt.plot(X_c[:,0], X_c[:,1], 'bo', label = "Initial constraint data")
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Output 2')
plt.colorbar(fig)
plt.show()
#plt.savefig('manifold_multioutput_gp/initial_data.png', dpi = 100)  

# Main Bayesian optimization loop
rng_key = random.PRNGKey(0)

# Fetch normalized training data (for objective and all the constraints)
norm_batch_f, norm_const_f = normalize(X_f, y_f, bounds)
norm_batch_c1, norm_const_c1 = normalize(X_c, y1_c, bounds)

# Define a list using the normalized data and the normalizing constants
norm_batch_list = [norm_batch_f, norm_batch_c1]
norm_const_list = [norm_const_f, norm_const_c1]

# Train GP model with 100 random restart
#    print('Train GP...')
rng_key = random.PRNGKey(0)
        
opt_params_list = gp_model.train(norm_batch_list,
                                 rng_key,
                                 num_restarts = 5)#, verbose = False, maxfun=1500) #20) maxfun=15000

# Visualize the final outputs 
kwargs = {'params': opt_params_list,
          'batch': norm_batch_list,
          'norm_const': norm_const_list,
          'bounds': bounds,
          'rng_key': rng_key,
          'gmm_vars': None}

# Making prediction on the posterior objective and all constraints
mean, std = gp_model.predict_all(X_star, **kwargs)

mean = onp.array(mean)
std = onp.array(std)

mean[0:1,:] = mean[0:1,:] * norm_const_list[0]['sigma_y'] + norm_const_list[0]['mu_y']
std[0:1,:] = std[0:1,:] * norm_const_list[0]['sigma_y']

mean[1:2,:] = mean[1:2,:] * norm_const_list[1]['sigma_y'] + norm_const_list[1]['mu_y']
std[1:2,:] = std[1:2,:] * norm_const_list[1]['sigma_y']

# Check accuracy
error = np.linalg.norm(mean[0,:]-y_f_star,2)/np.linalg.norm(y_f_star,2)
print("Relative L2 error output 1: %e" % (error))

# Check accuracy
error = np.linalg.norm(mean[1,:]-y1_c_star,2)/np.linalg.norm(y1_c_star,2)
print("Relative L2 error output 2: %e" % (error))

y_f_pred = onp.array(mean[0,:])
y1_c_pred = onp.array(mean[1,:])

# Convert the numpy variable into grid data for visualization
Y_f_pred = griddata(onp.array(X_star), y_f_pred, (onp.array(XX), onp.array(YY)), method='cubic')
Y1_c_pred = griddata(onp.array(X_star), y1_c_pred, (onp.array(XX), onp.array(YY)), method='cubic')

# Visualization
plt.figure(figsize = (16,10))
plt.subplot(2, 3, 1)
fig = plt.contourf(XX, YY, Y_f_star)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Exact Output 1')
plt.colorbar(fig)

plt.subplot(2, 3, 2)
fig = plt.contourf(XX, YY, Y_f_pred)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Pred Output 1')
plt.colorbar(fig)

plt.subplot(2, 3, 3)
fig = plt.contourf(XX, YY, onp.abs(Y_f_pred-Y_f_star))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Absolute error Output 1')
plt.colorbar(fig)

plt.subplot(2, 3, 4)
fig = plt.contourf(XX, YY, Y1_c_star)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Exact Output 2')
plt.colorbar(fig)

plt.subplot(2, 3, 5)
fig = plt.contourf(XX, YY, Y1_c_pred)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Pred Output 2')
plt.colorbar(fig)

plt.subplot(2, 3, 6)
fig = plt.contourf(XX, YY, onp.abs(Y1_c_pred-Y1_c_star))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Absolute error Output 2')
plt.colorbar(fig)
plt.show()

#plt.savefig('manifold_multioutput_gp/prediction.png', dpi = 100)  
