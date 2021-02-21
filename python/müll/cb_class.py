import math
import numpy as np
import pandas as pd
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

ys = [  4.3, 4.7, 4.7, 3.1, 5.2, 6.7, 4.5, 3.6, 7.2, 10.9, 6.6, 5.8,
        6.3, 4.7, 8.2, 6.2, 4.2, 4.1, 3.3, 4.6, 6.3,  4.0, 3.1, 3.5,
        7.8, 5.0, 5.7, 5.8, 6.4, 5.2, 8.0, 4.9, 6.1,  8.0, 7.7, 4.3,
       12.5, 7.9, 3.9, 4.0, 4.4, 6.7, 3.8, 6.4, 7.2,  4.8, 10.5]

def inv(A):
    return np.linalg.inv(A) 

def cholesky(A):
    return np.linalg.cholesky(A)

def gradient(f):
    return nd.Gradient(f) 

def hessian(f):
    return nd.Hessian(f) 

def quantile(arr, q):
    return np.percentile(arr, q)

#Maximum likelyhood estimation
def mle(pdf_i, xdata, ydata, p0=None, info=False):
    def _maximize():
        f_neg = lambda var: -_L(var, xdata, ydata)
        opt = minimize(f_neg, p0, method='Nelder-Mead')
        return opt.x

    def _Lik(theta, x, y):
        return np.prod( [pdf_i(x[i],y[i],theta) for i in range(len(x))] )

    def _L(theta, x, y):
        return np.sum( [np.log(pdf_i(x[i],y[i],theta)) for i in range(len(x))] )
    
    opt = _maximize()
    if info:
        I = -hessian(lambda var: _L(var, xdata, ydata))(opt)
        V = inv(I) 
        return opt, V, I
    return opt

# Confidence interval
def native_ci(x, q, f, opt, V):
    confInt_L, confInt_U = [None]*len(x), [None]*len(x)
    for i in range(len(x)):
        dg = gradient(lambda var: f(x[i], var))
        h = 1.7 * np.sqrt( dg(opt).T @ V @ dg(opt) ) # TODO Quantile
        confInt_L[i] = f(x[i], opt) - h
        confInt_U[i] = f(x[i], opt) + h
    return confInt_L, confInt_U

# Confidence band
def native_cb(x, q, f, opt, V, I):
    confBand_L, confBand_U = [None]*len(x), [None]*len(x)
    for i in range(len(x)):
        dg = gradient(lambda var: f(x[i], var))
        h = np.sqrt(1.96) * np.sqrt( dg(opt).T @ V @ dg(opt) ) # TODO Quantile
        confBand_L[i] = f(x[i], opt) - h
        confBand_U[i] = f(x[i], opt) + h
    return confBand_L, confBand_U

# Confidence band: random points on elipsoidal region + bootstraping chi^2
def bootstrap1_cb(x, q, f, pdf, opt, V, I, xdata, n=50, B=100):
    confBand_L, confBand_U = [None]*len(x), [None]*len(x)
    # Calculate bootstrap version of chi square
    C = []
    theta = []
    # Bootstrap theta samples
    for _ in range(B):
        # generate y sample from CDF
        xs_sample = np.random.choice(xdata, n)
        ys_sample = [generate_sample(lambda var: pdf(x_i,var,opt), 0.01, 15) for x_i in xs_sample]
        # MLE
        theta_sample = mle(pdf, xs_sample, ys_sample, p0=opt)
        # calculate statistic: Chi^2
        theta_diff = np.subtract(theta_sample, opt)
        C_sample = theta_diff.T @ I @ theta_diff 
        C.append( C_sample )
        theta.append( theta_sample )
    # Sort and get quantile
    for i in range(len(C)):
        for j in range(i+1,len(C)):
            if C[j] < C[i]: 
                tmpC = C[i] 
                C[i] = C[j]
                C[j] = tmpC
                tmpTheta = theta[i]
                theta[i] = theta[j]
                theta[j] = tmpTheta
    chi_square = quantile(C, q) # chi_square = 10.64
    theta = [theta[i] for i in range(len(theta)) if C[i] < chi_square]    
    # maximize and minimize the value of f
    confBand_L, confBand_U = [None]*len(x), [None]*len(x)
    for theta_sample in theta:
        for i in range(len(x)):
            fx = f(x[i], theta_sample)
            if confBand_L[i] is None or fx < confBand_L[i]: 
                confBand_L[i] = fx
            if confBand_U[i] is None or fx > confBand_U[i]: 
                confBand_U[i] = fx
    return confBand_L, confBand_U

def pdf(x, y, theta):
    a = (theta[0] / theta[1])**2
    b = (theta[1])**2 / theta[0]
    try: return y**(a-1) * np.exp(-y/b) / (math.gamma(a) * b**a)
    except: return 0
    # return y**(theta[0]-1) * np.exp(-y/theta[1]) / (math.gamma(theta[0]) * theta[1]**theta[0])

def generate_sample(pdf, x_min, x_max, step=0.01):
    sum = 0
    x = x_min
    r = np.random.uniform(0, 1)
    while sum < r and x <= x_max:
        sum += pdf(x) * step
        x += step
    return x

def w(x, theta):
    return (1+theta[0]) * theta[0] * theta[1]**2 * x / (2 * (1 - theta[0]*theta[1]*x))

x = np.linspace(0.0, 0.1, num=100)
xs = [i for i in range(len(ys))]
opt, V, I = mle(pdf, xs, ys, [6,2], info=True)

plt.plot(x, [w(x_i,opt) for x_i in x])
nat_ci_L, nat_ci_U = native_ci(x, 0.1, w, opt, V)
plt.plot(x, nat_ci_L, color='green', linestyle='dashed')
plt.plot(x, nat_ci_U, color='green', linestyle='dashed')
nat_cb_L, nat_cb_U = native_cb(x, 0.1, w, opt, V, I)
plt.plot(x, nat_cb_L, color='red', linestyle='dashed')
plt.plot(x, nat_cb_U, color='red', linestyle='dashed')
bs1_cb_L, bs1_cb_U = bootstrap1_cb(x, 90, w, pdf, opt, V, I, [None])
plt.plot(x, bs1_cb_L, color='blue', linestyle='dashed')
plt.plot(x, bs1_cb_U, color='blue', linestyle='dashed')

# plt.plot(x, ys, 'o', color='black', markersize='3')
# plt.plot(x, [w(x_i,9.20,0.63) for x_i in x])

# plt.plot(x, [pdf(x_i, x_i, [9.2,0.63]) for x_i in x])
# test = [generate_sample(lambda var: pdf(var,var,opt),0.01,15) for x_i in x]
# plt.hist(test, density=True)

plt.show()