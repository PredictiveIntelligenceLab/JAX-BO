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

from scipy.optimize import minimize
from pyDOE import lhs
import matplotlib.pyplot as plt
plt.close('all')
from matplotlib import rc
import matplotlib
matplotlib.use('Agg')
from scipy.interpolate import griddata

from jaxbo.models import MultipleIndependentOutputsGP, GP, DeepMultifidelityGP_MultiOutputs, ManifoldGP_MultiOutputs
from jaxbo.utils import normalize, compute_w_gmm#, normalize_constraint
from jaxbo.test_functions import *

from jax.scipy.stats import norm
import jaxbo.acquisitions as acquisitions

from jaxbo.input_priors import uniform_prior, gaussian_prior


onp.random.seed(1234)

# Example from 
# https://asmedigitalcollection.asme.org/mechanicaldesign/article/141/12/121001/975244?casa_token=45A-r7iV9IUAAAAA:ji-aHZ_T_HQ5Q1xgNxloqrG2LjOpFkXMItdWnuGH9d02MysONc3VTfrtM8GSB5oTdE2jcQ
# Section 4, and constraint in section 4.2 



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


plt.figure(figsize = (16, 5))
plt.subplot(1, 2, 1)
fig = plt.contourf(XX, YY, Y_f_star)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Exact objective')
plt.colorbar(fig)

plt.subplot(1, 2, 2)
fig = plt.contourf(XX, YY, Y1_c_star)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'constraint1')
plt.colorbar(fig)
plt.savefig('constrained_bo_ManifoldGP_MultiOutputs/objective_and_constraint.png', dpi = 100)  

# Visualize the feasible domain and the location of the best value of this problem

judge1 = (y1_c_star >= 0)

total_judge = judge1 

valid_index = np.where(total_judge)
#print(valid_index)

valid_x = X_star[valid_index]
valid_y = y_f_star[valid_index]

#print(valid_x.shape, valid_y.shape)
idx_best = np.argmin(valid_y)
x_best = valid_x[idx_best]
y_best = valid_y[idx_best]

plt.figure(figsize = (6,4))
fig = plt.contourf(XX, YY, Y_f_star)
plt.plot(valid_x[:,0], valid_x[:, 1], 'r.', markersize = 2.)
plt.plot(x_best[0], x_best[1], 'y.', markersize = 8.)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Exact objective')
plt.colorbar(fig)
plt.savefig('constrained_bo_ManifoldGP_MultiOutputs/exact_objective.png', dpi = 100)  

print("best y", y_best, "best x", x_best)

true_x = x_best
true_y = y_best

# Problem settings

# Number of initial data for objective and constraints
N_f = 50 # 20 # 200
N_c = 100 # 50 # 200
noise_f = 0.00
noise_c = 0.01 # 0.00
nIter = 10 # 10

# Define prior distribution
p_x = uniform_prior(lb, ub)

# JAX-BO setting

options = {'kernel': 'RBF',
           'input_prior': p_x,
           'net_arch': 'MLP',
           'constrained_criterion': 'LCBC',
           'criterion': 'LW_CLSF',
           'kappa': 2.0,
           'nIter': nIter}
layers = [2, 5, 5, 2]
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
plt.title(r'Exact objective')
plt.colorbar(fig)

plt.subplot(1, 2, 2)
fig = plt.contourf(XX, YY, Y1_c_star)
plt.plot(X_c[:,0], X_c[:,1], 'bo', label = "Initial constraint data")
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'constraint1')
plt.colorbar(fig)
plt.legend()
plt.savefig('constrained_bo_ManifoldGP_MultiOutputs/initial_data.png', dpi = 100)  

# Main Bayesian optimization loop
rng_key = random.PRNGKey(0)
for it in range(options['nIter']):
    print('--- Iteration %d/%d ---' % (it+1, options['nIter']))

    # Fetch normalized training data (for objective and all the constraints)
    norm_batch_f, norm_const_f = normalize(X_f, y_f, bounds)
    norm_batch_c1, norm_const_c1 = normalize(X_c, y1_c, bounds)
    
    # Define a list using the normalized data and the normalizing constants
    norm_batch_list = [norm_batch_f, norm_batch_c1]
    norm_const_list = [norm_const_f, norm_const_c1]

    # Train GP model with 100 random restart
    print('Train GP (objective)...')
    rng_key = random.PRNGKey(0)
        
    opt_params_list = gp_model.train(norm_batch_list,
                                     rng_key,
                                     num_restarts = 5)#, verbose = False, maxfun=1500) #20) maxfun=15000
    
    # Fit GMM
    if options['constrained_criterion'] == 'LW_LCBC' or options['constrained_criterion'] == 'LW_CLSF' or options['constrained_criterion'] == 'LW-US':
        print('Fit GMM...')
        rng_key = random.split(rng_key)[0]
        kwargs = {'params': opt_params_list,
                  'batch': norm_batch_list,
                  'norm_const': norm_const_list,
                  'bounds': bounds,
                  'rng_key': rng_key}
        gmm_vars = gp_model.fit_gmm(**kwargs, N_samples = 10000)
    else:
        gmm_vars = None
    
    # Find the next acquisition point with 50 random restart
    print('Computing next acquisition point (objective)...')
    rng_key = random.PRNGKey(0)
    kwargs = {'params': opt_params_list,
              'batch': norm_batch_list,
              'norm_const': norm_const_list,
              'bounds': bounds,
              'kappa': options['kappa'],
              'gmm_vars': gmm_vars,
              'rng_key': rng_key}
        
    # Acquire objective data
    new_X_f,_,_ = gp_model.constrained_compute_next_point_lbfgs(num_restarts=50, **kwargs)
    new_y_f = vmap(f)(new_X_f) # This is the output of the solver for generating the objective function
    new_y_f = new_y_f + noise_f*y_f_star.std(0)*onp.random.normal(new_y_f.shape)
        
    new_y1_c = vmap(constraint1)(new_X_f) # This is the output of the solver for generating the constraint1 functions
        
    # # Augment training data
    print('Updating data-set...')
    X_f = np.concatenate([X_f, new_X_f], axis = 0)
    X_c = np.concatenate([X_c, new_X_f], axis = 0)
    
    y_f = np.concatenate([y_f, new_y_f], axis = 0)
    y1_c = np.concatenate([y1_c, new_y1_c], axis = 0)

    # # Print current best
    print('True location: ({}), True value: {}'.format(true_x, true_y))
    print('New  location: ({}), New  value: {}'.format(new_X_f, new_y_f))
        
    mean, std = gp_model.predict_all(X_star, **kwargs)
    mean = onp.array(mean)
    std = onp.array(std)
    mean[0:1,:] = mean[0:1,:] * norm_const_list[0]['sigma_y'] + norm_const_list[0]['mu_y']
    std[0:1,:] = std[0:1,:] * norm_const_list[0]['sigma_y']
    mean[1:2,:] = mean[1:2,:] * norm_const_list[1]['sigma_y'] + norm_const_list[1]['mu_y']
    std[1:2,:] = std[1:2,:] * norm_const_list[1]['sigma_y']
    judge1_f = (mean[1,:]+0*std[1,:] >= 0)
    total_judge_f = judge1_f
    valid_index_f = np.where(total_judge_f)
    valid_x_f = X_star[valid_index_f]
    valid_y_f = mean[0,valid_index_f]
    idx_best = np.argmin(valid_y_f)
    best_x = valid_x_f[idx_best,:]
    best_y = valid_y_f.min()
    
    print('Best location: ({}), Best value: {}'.format(best_x, best_y))
    
# Visualize the final outputs 
# Compute the weight
if options['constrained_criterion'] == 'LW_LCBC':
    w_pred = compute_w_gmm(X_star, **kwargs)
    

# Compute the upper and lower bounds of the posterior distributions
lower = mean - 2.0*std
upper = mean + 2.0*std

print(mean.shape, std.shape, lower.shape, upper.shape)

# Evaluate the acquisition function
acq_fn1 = lambda x: gp_model.constrained_acquisition(x, **kwargs)
LW_LCBCacq = vmap(acq_fn1)(X_star)

# Compute the ratio and weights derived by the constraints and convert everything into numpy for plotting
ratio1 = mean[1,:] / std[1,:]

weight1 = norm.cdf(mean[1,:]/std[1,:])

LW_LCBCacq = onp.array(LW_LCBCacq)

mean = onp.array(mean)
std = onp.array(std)

ratio1 = onp.array(ratio1)

weight1 = onp.array(weight1)

y_f_pred = onp.array(mean[0,:])
y1_c_pred = onp.array(mean[1,:])

y_f_std = onp.array(std[0,:])

try:
    w_pred = onp.array(w_pred)
except:
    w_pred = onp.ones_like(y_f_std)
    

kappa = 2.

# Convert the numpy variable into grid data for visualization
Y_f_pred = griddata(onp.array(X_star), y_f_pred, (onp.array(XX), onp.array(YY)), method='cubic')
Y1_c_pred = griddata(onp.array(X_star), y1_c_pred, (onp.array(XX), onp.array(YY)), method='cubic')

Y_f_std = griddata(onp.array(X_star), y_f_std, (onp.array(XX), onp.array(YY)), method='cubic')

Ratio1 = griddata(onp.array(X_star), ratio1, (onp.array(XX), onp.array(YY)), method='cubic')

Weight1 = griddata(onp.array(X_star), weight1, (onp.array(XX), onp.array(YY)), method='cubic')

LW_LCBCacq = griddata(onp.array(X_star), LW_LCBCacq.flatten(), (onp.array(XX), onp.array(YY)), method='cubic')

W_pred = griddata(onp.array(X_star), w_pred.flatten(), (onp.array(XX), onp.array(YY)), method='cubic')



LCBacq = Y_f_pred - 3. - kappa*Y_f_std


# Visualization
plt.figure(figsize = (16,10))
plt.subplot(2, 4, 1)
fig = plt.contourf(XX, YY, Y1_c_star)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Exact constraint1')
plt.colorbar(fig)

plt.subplot(2, 4, 2)
fig = plt.contourf(XX, YY, Y1_c_pred)
plt.plot(X_c[:,0], X_c[:,1], 'r.')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Pred constraint1')
plt.colorbar(fig)

plt.subplot(2, 4, 3)
fig = plt.contourf(XX, YY, Ratio1)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Ratio1')
plt.colorbar(fig)


plt.subplot(2, 4, 4)
fig = plt.contourf(XX, YY, np.clip(Weight1, 0, np.inf))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Weight1')
plt.colorbar(fig)


plt.subplot(2, 4, 5)
fig = plt.contourf(XX, YY, Y_f_star)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Exact objective')
plt.colorbar(fig)

plt.subplot(2, 4, 6)
fig = plt.contourf(XX, YY, Y_f_pred)
plt.plot(X_f[:,0], X_f[:,1], 'r.')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Pred objective')
plt.colorbar(fig)

plt.subplot(2, 4, 7)
fig = plt.contourf(XX, YY, LCBacq)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'LCB')
plt.colorbar(fig)


plt.subplot(2, 4, 8)
fig = plt.contourf(XX, YY, LW_LCBCacq)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'LW_LCBC')
plt.colorbar(fig)

plt.savefig('constrained_bo_ManifoldGP_MultiOutputs/pred_post.png', dpi = 100)  

# Data we collected and the ground truth
plt.figure(figsize = (15, 5))
plt.subplot(1, 3, 1)
fig = plt.contourf(XX, YY, Y_f_star)
plt.plot(valid_x[:,0], valid_x[:, 1], 'r.', markersize = 2.)
plt.plot(true_x[0], true_x[1], 'k.', markersize = 10.)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Exact objective')
plt.colorbar(fig)


plt.subplot(1, 3, 2)
fig = plt.contourf(XX, YY, Y_f_pred)
plt.plot(X_f[:,0], X_f[:,1], 'r.')
plt.plot(true_x[0], true_x[1], 'k.', markersize = 10.)
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Pred objective')
plt.colorbar(fig)


plt.subplot(1, 3, 3)
fig = plt.contourf(XX, YY, W_pred)
plt.plot(X_f[:,0], X_f[:,1], 'r.')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Pred output weight')
plt.colorbar(fig)
plt.savefig('constrained_bo_ManifoldGP_MultiOutputs/pred_post_and_ground_truth.png', dpi = 100)  

