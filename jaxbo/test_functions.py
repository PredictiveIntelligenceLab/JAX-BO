import jax.numpy as np
from jaxbo.input_priors import uniform_prior, gaussian_prior

# Caution: all functions are designed for single point evaluation
# (use vmap to vectorize)

def oakley():
    dim = 2
    lb = -4.0*np.ones(dim)
    ub = 4.0*np.ones(dim)
    p_x = gaussian_prior(np.zeros(dim), np.eye(dim))
    def f(x):
        return 5.0 + x[0] + x[1] + 2.0*np.cos(x[0]) + 2.0*np.sin(x[1])
    return f, p_x, dim, lb, ub

def michalewicz(dim):
    lb = 0.0*np.ones(dim)
    ub = np.pi*np.ones(dim)
    p_x = gaussian_prior(np.zeros(dim) + 0.5*np.pi, 0.1*np.eye(dim))
    def f(x):
        m = 10.0
        y = 0.0
        for i in range(dim):
            y += np.sin(x[i]) * np.sin( (i+1) * x[i]**2 / np.pi )**(2*m)
        return -y
    return f, p_x, dim, lb, ub

def ackley(dim):
    lb = -32.768*np.ones(dim)
    ub = 32.768*np.ones(dim)
    p_x = uniform_prior(lb, ub)
    def f(x):
        a = 20
        b = 0.2
        c = 2*np.pi
        return - a * np.exp( -b * np.sqrt(np.sum(x**2)/dim) ) \
               - np.exp( np.sum(np.cos(c*x))/dim ) \
               + a + np.exp(1.0)
    return f, p_x, dim, lb, ub

def bird():
    dim = 2
    lb = -2.0*np.pi*np.ones(dim)
    ub = 2.0*np.pi*np.ones(dim)
    p_x = uniform_prior(lb, ub)
    def f(x):
        x1, x2 = x[0], x[1]
        y = np.sin(x1) * np.exp( (1-np.cos(x2))**2 ) \
          + np.cos(x2) * np.exp( (1-np.sin(x1))**2 ) + (x1-x2)**2
        return y
    return f, p_x, dim, lb, ub

def rosenbrock():
    dim = 2
    lb = np.array([-1.0, -1.0])
    ub = np.array([0.5, 1.0])
    p_x = uniform_prior(lb, ub)
    def f(x):
        y = 74.0 + 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
        y -= 400.0 * np.exp(-((x[0] + 1.0) ** 2 + (x[1] + 1.0) ** 2) / 0.1)
        return y
    return f, p_x, dim, lb, ub

def branin():
    dim = 2
    lb = np.array([-5.0, 0.0])
    ub = np.array([10.0, 15.0])
    p_x = uniform_prior(lb, ub)
    def f(x):
        a = 1.0
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8*np.pi)
        x1, x2 = x[0], x[1]
        y = a * (x2 - b*x1**2 + c*x1 -r)**2 + s * (1-t) * np.cos(x1) + s
        return y
    return f, p_x, dim, lb, ub

def modified_branin():
    dim = 2
    lb = np.array([-5.0, 0.0])
    ub = np.array([10.0, 15.0])
    p_x = uniform_prior(lb, ub)
    def f(x):
        a = 1.0
        b = 5.1 / (4*np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8*np.pi)
        x1, x2 = x[0], x[1]
        f1 = a * (x2 - b*x1**2 + c*x1 -r)**2
        f2 = s * (1-t) * np.cos(x1) * np.cos(x2)
        f3 = np.log(x1**2 + x2**2 + 1)
        y = -1/(f1+f2+f3+s)
        return y
    return f, p_x, dim, lb, ub

def ursem_waves():
    dim = 2
    lb = np.array([-0.9, -1.2])
    ub = np.array([1.2, 1.2])
    p_x = uniform_prior(lb, ub)
    def f(x):
        x1, x2 = x[0], x[1]
        u = -0.9*x1**2
        v = (x2**2 - 4.5*x2**2) * x1 * x2
        w = 4.7*np.cos(3*x1 - x2**2 * (2+x1)) * np.sin(2.5*np.pi*x1)
        return u + v + w
    return f, p_x, dim, lb, ub

def himmelblau():
    dim = 2
    lb = np.array([-6.0, -6.0])
    ub = np.array([6.0, 6.0])
    p_x = uniform_prior(lb, ub)
    def f(x):
        x1, x2 = x[0], x[1]
        y = (x1**2 + x2 - 11)**2 + (x1 + x2**2 - 7)**2
        return y
    return f, p_x, dim, lb, ub

def bukin():
    dim = 2
    lb = np.array([-15.0, -3.0])
    ub = np.array([-5.0, 3.0])
    p_x = uniform_prior(lb, ub)
    def f(x):
        x1, x2 = x[0], x[1]
        y = 100 * np.sqrt( np.abs(x2-0.01*x1**2) ) + 0.01 * np.abs(x1+10)
        return y
    return f, p_x, dim, lb, ub

def hartmann6():
    dim = 6
    lb = np.zeros(dim)
    ub = np.ones(dim)
    p_x = uniform_prior(lb, ub)
    def f(x):
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])
        arg = np.dot(A, (x-P).T**2)
        y = -np.dot(alpha, np.diag(np.exp(-arg)))
        return y
    return f, p_x, dim, lb, ub

def forrester():
    dim = 1
    lb = np.zeros(dim)
    ub = np.ones(dim)
    p_x = uniform_prior(lb, ub)
    def f_H(x):
        x = x.flatten()
        y = (6.0*x-2.0)**2 * np.sin(12.0*x-4.0)
        return y[0]
    def f_L(x):
        x = x.flatten()
        y = 0.5*f_H(x) + 10.0*(x-0.5) - 5.0
        return y[0]
    return (f_L, f_H), p_x, dim, lb, ub
