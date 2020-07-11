import numpy as onp
import jax.numpy as np
from jax import random, vmap
from jax.config import config
config.update("jax_enable_x64", True)

from scipy.optimize import minimize
from pyDOE import lhs
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import griddata

from jaxbo.models import GP
from jaxbo.utils import normalize
from jaxbo.test_functions import *

onp.random.seed(1234)

if __name__ == "__main__":
    N = 5
    num_BO_iter = 20
    noise = 0.0

    # Define test function
    f, p_x, dim, lb, ub = oakley()
    # f, p_x, dim, lb, ub = michalewicz(2)
    # f, p_x, dim, lb, ub = rosenbrock()
    # f, p_x, dim, lb, ub = ursem_waves()
    # f, p_x, dim, lb, ub = himmelblau()
    # f, p_x, dim, lb, ub = hartmann6()
    # f, p_x, dim, lb, ub = bukin()
    # f, p_x, dim, lb, ub = branin()
    # f, p_x, dim, lb, ub = modified_branin()
    # f, p_x, dim, lb, ub = ackley(2)
    # f, p_x, dim, lb, ub = bird()

    # Fetch domain bounds
    bounds = {'lb': lb, 'ub': ub}

    # Initial training data
    X = lb + (ub-lb)*lhs(dim, N)
    y = vmap(f)(X)
    y = y + noise*y.std(0)**onp.random.normal(y.shape)

    # Test data
    if dim == 1:
        create_plots = True
        nn = 1000
        X_star = np.linspace(lb[0], ub[0], nn)
        y_star = vmap(f)(X_star)
    elif dim == 2:
        create_plots = True
        nn = 80
        xx = np.linspace(lb[0], ub[0], nn)
        yy = np.linspace(lb[1], ub[1], nn)
        XX, YY = np.meshgrid(xx, yy)
        X_star = np.concatenate([XX.flatten()[:,None], YY.flatten()[:,None]], axis = 1)
        y_star = vmap(f)(X_star)
    else:
        create_plots = False
        nn = 20000
        X_star = lb + (ub-lb)*lhs(dim, nn)
        y_star = vmap(f)(X_star)

    # True location of global minimum
    idx_true = np.argmin(y_star)
    true_x = X_star[idx_true,:]
    true_y = y_star.min()
    dom_bounds = tuple(map(tuple, np.vstack((lb, ub)).T))
    result = minimize(f, true_x, jac=None, method='L-BFGS-B', bounds = dom_bounds)
    true_x, true_y = result.x, result.fun

    # Define model
    gp_model = GP()
    rng_key = random.PRNGKey(0)

    # Main Bayesian optimization loop
    for it in range(num_BO_iter):
        print('-------------------------------------------------------------------')
        print('------------------------- Iteration %d/%d -------------------------' % (it+1, num_BO_iter))
        print('-------------------------------------------------------------------')

        # Fetch normalized training data
        norm_batch, norm_const = normalize(X, y)

        # Train GP model
        print('Train GP...')
        rng_key = random.split(rng_key)[0]
        opt_params = gp_model.train(norm_batch,
                                    rng_key,
                                    num_restarts = 10)

        # Fit GMM
        print('Fit GMM...')
        rng_key = random.split(rng_key)[0]
        gmm_vars = gp_model.fit_gmm(opt_params,
                                    norm_batch,
                                    norm_const,
                                    bounds,
                                    p_x,
                                    rng_key,
                                    N_samples = 10000)

        # Compute next point via minimizing the LW-LCB acquisition
        print('Computing next acquisition point...')
        new_X = gp_model.compute_next_point(opt_params,
                                            norm_batch,
                                            norm_const,
                                            bounds,
                                            gmm_vars,
                                            num_restarts=10)

        # Acquire data
        new_y = vmap(f)(new_X)
        new_y = new_y + noise*new_y.std(0)*onp.random.normal(new_y.shape)

        # Augment training data
        print('Updating data-set...')
        X = np.concatenate([X, new_X], axis = 0)
        y = np.concatenate([y, new_y], axis = 0)

        # Print current best
        idx_best = np.argmin(y)
        best_x = X[idx_best,:]
        best_y = y.min()
        print('True location: (%f,%f), True value: %f' % (true_x[0], true_x[1], true_y))
        print('Best location: (%f,%f), Best value: %f' % (best_x[0], best_x[1], best_y))

    # Test accuracy
    mean, std = gp_model.predict(X_star,
                             opt_params,
                             norm_batch,
                             norm_const)
    lower = mean - 2.0*std
    upper = mean + 2.0*std
    # Check accuracy
    error = np.linalg.norm(mean-y_star,2)/np.linalg.norm(y_star,2)
    print("Relative L2 error u: %e" % (error))

    if create_plots:
        # Compute predictions
        w_pred = gp_model.compute_w_gmm(X_star,
                                        bounds,
                                        gmm_vars)
        lw_ucb = gp_model.LW_LCB(X_star,
                                 opt_params,
                                 norm_batch,
                                 norm_const,
                                 bounds,
                                 gmm_vars)
        x_new = gp_model.compute_next_point(opt_params,
                                            norm_batch,
                                            norm_const,
                                            bounds,
                                            gmm_vars,
                                            num_restarts=10)

        # Convert to NumPy
        X_star = onp.array(X_star)
        y_star = onp.array(y_star)
        mean = onp.array(mean)
        std = onp.array(std)
        w_pred = onp.array(w_pred)
        lw_ucb = onp.array(lw_ucb)
        XX = onp.array(XX)
        YY = onp.array(YY)
        Y_star = griddata(X_star, y_star, (XX, YY), method='cubic')
        Y_pred = griddata(X_star, mean, (XX, YY), method='cubic')
        Y_std  = griddata(X_star, std, (XX, YY), method='cubic')
        W_star = griddata(X_star, w_pred, (XX, YY), method='cubic')
        A_star = griddata(X_star, lw_ucb, (XX, YY), method='cubic')

        # Plot
        plt.rcParams.update({'font.size': 16})
        plt.rcParams['axes.linewidth']=3
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)

        plt.figure(figsize = (16,8))
        plt.subplot(1, 4, 1)
        fig = plt.contourf(XX, YY, Y_star)
        plt.plot(X[:,0], X[:,1], 'r.', ms = 6, alpha = 0.8)
        # plt.plot(true_x[0], true_x[1], 'md', ms = 8, alpha = 1.0)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(r'Exact u(x)')
        plt.axis('square')

        plt.subplot(1, 4, 2)
        fig = plt.contourf(XX, YY, Y_pred)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(r'Predicted mean')
        plt.axis('square')

        plt.subplot(1, 4, 3)
        fig = plt.contourf(XX, YY, 2.0*Y_std)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(r'Two stds')
        plt.axis('square')

        plt.subplot(1, 4, 4)
        fig = plt.contourf(XX, YY, np.abs(Y_star-Y_pred))
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(r'Absolute error')
        plt.axis('square')
        plt.savefig('function_prediction.png', dpi = 300)

        idx_max = np.argmin(lw_ucb)
        plt.figure(figsize = (12,5))
        plt.subplot(1, 2, 1)
        fig = plt.contourf(XX, YY, W_star)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(r'$w_{GMM}(x)$')
        plt.axis('square')
        plt.subplot(1, 2, 2)
        fig = plt.contourf(XX, YY, A_star)
        plt.colorbar(fig)
        # plt.plot(x0[:,0], x0[:,1], 'ms')
        # plt.plot(X_star[idx_max,0], X_star[idx_max,1], 'md')
        plt.plot(x_new[:,0], x_new[:,1], 'md', label = 'new X')
        plt.legend(frameon = False)
        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(r'LW-UCB(x)')
        plt.axis('square')
        plt.savefig('acquisition.png', dpi = 300)
