from re import T
import numpy as np
import pandas as pd
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

xs = [2.0, 7.0, 12.0, 19.5, 29.5, 39.5, 54.5, 75.0]
data = [ [ 1.26,   2.78,  0.63,  0.34], 
         [ 3.53,    4.1,  1.31,  0.91],
         [11.98,  13.14,  9.86,  6.53], 
         [90.82,  97.12, 75.85, 59.46], 
         [83.45, 116.62,  104., 80.85], 
         [55.98,  67.28, 79.33, 82.66],
         [66.32,  78.53,  69.1, 67.27], 
         [37.42,  55.35, 60.76,  73.2] ]
x_i = []
y_i = []
for i in range(len(xs)):
    for j in range(len(data[i])):
        x_i.append(xs[i])
        y_i.append(data[i][j])

def inv(A):
    return np.linalg.inv(A) 

def Gradient(f):
    return nd.Gradient(f) 

def Hessian(f):
    return nd.Hessian(f) 

def quantile(arr, q):
    arr_sorted = np.sort(arr)
    return np.percentile(arr_sorted, q)

#Maximum likelyhood estimation
def mle(f, xs, ys, p0, info=False):
    def _maximize():
        f_neg = lambda var: -_L(var, xs, ys)
        opt = minimize(f_neg, p0, method='Nelder-Mead')
        return opt.x

    def _fi(x, y, theta):
        mu = f(x,theta)
        sigma = theta[0]
        return np.exp( -(y - mu)**2 / (2 * sigma**2) ) / np.sqrt(2 * np.pi * sigma**2)

    def _Lik(theta, x, y):
        return np.prod( [_fi(x[i],y[i],theta) for i in range(len(x))] )

    def _L(theta, x, y):
        return np.sum( [np.log(_fi(x[i],y[i],theta)) for i in range(len(x))] )
    
    opt = _maximize()
    if info:
        I = -Hessian(lambda var: _L(var, xs, ys))(opt)
        V = inv(I) 
        return opt, V, I
    return opt

# Confidence interval
def native_ci(x, q, f, opt, V):
    confInt = []
    for i in range(len(x)):
        dg = Gradient(lambda var: f(x[i], var))
        h = 1.96 * np.sqrt( dg(opt).T @ V @ dg(opt) ) # TODO Quantile
        confInt.append(h)
    return confInt

# Confidence band
def native_cb(x, q, f, opt, V, I):
    confBand = []
    for i in range(len(x)):
        dg = Gradient(lambda var: f(x[i], var))
        h = np.sqrt(10.64) * np.sqrt( dg(opt).T @ V @ dg(opt) ) # TODO Quantile
        confBand.append(h)
    return confBand

# Confidence band
def bootstrap_cb(x, q, f, opt, V, I, n=100, B=50):
    y_sample = lambda var: f(var, opt) + np.random.normal(0, opt[0])

    sampleC = []
    for _ in range(B):
        xs_sample = np.random.choice(x, n)
        ys_sample = [y_sample(x_i) for x_i in xs_sample]
        theta_sample = mle(f, xs_sample, ys_sample, opt)
        theta_diff = np.subtract(theta_sample, opt)
        sampleC.append( theta_diff.T @ I @ theta_diff )
    chi_square = quantile(sampleC, q)

    confBand = []
    for i in range(len(x)):
        dg = Gradient(lambda var: f(x[i], var))
        h = np.sqrt(chi_square) * np.sqrt( dg(opt).T @ V @ dg(opt) ) # TODO Quantile
        confBand.append(h)
    return confBand

# Confidence band
def full_bootstrap_cb(x, q, f, opt, V, I, n=100, B=50):
    y_sample = lambda var: f(var, opt) + np.random.normal(0, opt[0])

    sampleC = []
    for _ in range(B):
        xs_sample = np.random.choice(x, n)
        ys_sample = [y_sample(x_i) for x_i in xs_sample]
        theta_sample = mle(f, xs_sample, ys_sample, opt)
        theta_diff = np.subtract(theta_sample, opt)
        sampleC.append( theta_diff.T @ I @ theta_diff )
    chi_square = quantile(sampleC, q)

    confBand = []
    for i in range(len(x)):
        dg = Gradient(lambda var: f(x[i], var))
        h = np.sqrt(chi_square) * np.sqrt( dg(opt).T @ V @ dg(opt) ) # TODO Quantile
        confBand.append(h)
    return confBand

########## Regression ##########
eta = lambda x,theta: (theta[1] + theta[3]*x + theta[5] * x**2) * np.exp(theta[4]*(x-theta[2]))/( 1 + np.exp(theta[4]*(x-theta[2])) )
opt, V, I = mle(eta, x_i, y_i, [10.17, 153.79, 17.26, -2.58, 0.455, 0.01746], info=True) 
x = np.linspace(1, 80, num=100)
avg = [eta(x[i], opt) for i in range(len(x))]
########## Native Confidence ##########
# plt.plot(xs, data, 'o', color='black', markersize='3')
# plt.plot(x, avg)
# h_ci = native_ci(x, 90, eta, opt, V)
# plt.plot(x, [avg[i] - h_ci[i]   for i in range(len(x))], color='green',   linestyle='dashed')
# plt.plot(x, [avg[i] + h_ci[i]   for i in range(len(x))], color='green',   linestyle='dashed')
# h_cb = native_cb(x, 90, eta, opt, V, I)
# plt.plot(x, [avg[i] - h_cb[i]   for i in range(len(x))], color='blue',   linestyle='dashed')
# plt.plot(x, [avg[i] + h_cb[i]   for i in range(len(x))], color='blue',   linestyle='dashed')
# plt.show()
########## Bootstrap Confidence ##########
plt.plot(xs, data, 'o', color='black', markersize='3')
plt.plot(x, avg)
h_bscb = bootstrap_cb(x, 90, eta, opt, V, I)
plt.plot(x, [avg[i] - h_bscb[i] for i in range(len(x))], color='red', linestyle='dashed')
plt.plot(x, [avg[i] + h_bscb[i] for i in range(len(x))], color='red', linestyle='dashed')
plt.show()

###############################
#     return nd.Gradient(lambda var: Lik(var, y))(theta)
# oder
#     gradient=[]
#     for j in range(len(theta)):
#         sum = 0
#         for i in range(len(y)):
#             logf_i = lambda var: np.log( f(i,y[i],theta[:j]+[var]+theta[j+1:]) )
#             dlogf_i = nd.Derivative(logf_i)
#             try:
#                 sum += dlogf_i(theta[j])
#             except ZeroDivisionError:
#                 pass
#         gradient.append(sum)
#     return gradient