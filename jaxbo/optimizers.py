from scipy.optimize import minimize

def minimize_lbfgs(objective, x0, verbose = False, maxfun = 15000, bnds = None):
    if verbose:
        def callback_fn(params):
            print("Loss: {}".format(objective(params)[0]))
    else:
        callback_fn = None
        
    result = minimize(objective, x0, jac=True,
                      method='L-BFGS-B', bounds = bnds,
                      callback=callback_fn, options = {'maxfun':maxfun})
    return result.x, result.fun
