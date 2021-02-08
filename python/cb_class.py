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

def cholesky(A):
    return np.linalg.cholesky(A)

def gradient(f):
    return nd.Gradient(f) 

def hessian(f):
    return nd.Hessian(f) 

def quantile(arr, q):
    return np.percentile(arr, q)

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
        I = -hessian(lambda var: _L(var, xs, ys))(opt)
        V = inv(I) 
        return opt, V, I
    return opt

# Confidence interval
def native_ci(x, q, f, opt, V):
    confInt = []
    for i in range(len(x)):
        dg = gradient(lambda var: f(x[i], var))
        h = 1.96 * np.sqrt( dg(opt).T @ V @ dg(opt) ) # TODO Quantile
        confInt.append(h)
    return confInt

# Confidence band
def native_cb(x, q, f, opt, V, I):
    confBand = []
    for i in range(len(x)):
        dg = gradient(lambda var: f(x[i], var))
        h = np.sqrt(10.64) * np.sqrt( dg(opt).T @ V @ dg(opt) ) # TODO Quantile
        confBand.append(h)
    return confBand

# Confidence band
def bootstrap_cb(x, q, f, opt, V, I, n=20, B=50):
    y_sample = lambda var: f(var, opt) + np.random.normal(0, opt[0])
    C = []
    theta = []
    # Calculate C samples
    for _ in range(B):
        xs_sample = np.random.choice(x, n)
        ys_sample = [y_sample(x_i) for x_i in xs_sample]
        theta_sample = mle(f, xs_sample, ys_sample, opt)
        theta_diff = np.subtract(theta_sample, opt)
        C_sample = theta_diff.T @ I @ theta_diff 
        C.append( C_sample )
        theta.append( theta_sample )
    # Sort C and corresponding theta
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
    theta = [theta[i] for i in range(len(theta)) if C[i] < 10.64]    
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

# # Confidence band
# def bootstrap_cb(x, q, f, opt, V, I, n=50, B=100):
#     L = cholesky(V)
#     confBand_L, confBand_U = [None]*len(x), [None]*len(x)
#     for _ in range(B):
#         # obtain a point of theta on the edge of the confidence region
#         z = np.random.normal(0, 1, len(opt))
#         z_sum = np.sum([z[j]**2 for j in range(len(z))])
#         Z = [5.2 * z[i]**2 / z_sum for i in range(len(z))]
#         W = L @ Z
#         theta_sample = np.add(W, opt)
#         test = W.T @ I @ W 
#         for i in range(len(x)):
#             fx = f(x[i], theta_sample)
#             if confBand_L[i] is None or fx < confBand_L[i]: 
#                 confBand_L[i] = fx
#             if confBand_U[i] is None or fx > confBand_U[i]: 
#                 confBand_U[i] = fx
#     return confBand_L, confBand_U

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
        dg = gradient(lambda var: f(x[i], var))
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
h_ci = native_ci(x, 90, eta, opt, V)
plt.plot(x, [avg[i] - h_ci[i]   for i in range(len(x))], color='green',   linestyle='dashed')
plt.plot(x, [avg[i] + h_ci[i]   for i in range(len(x))], color='green',   linestyle='dashed')
h_cb = native_cb(x, 90, eta, opt, V, I)
plt.plot(x, [avg[i] - h_cb[i]   for i in range(len(x))], color='blue',   linestyle='dashed')
plt.plot(x, [avg[i] + h_cb[i]   for i in range(len(x))], color='blue',   linestyle='dashed')
y_L_bscb, y_U_bscb = bootstrap_cb(x, 90, eta, opt, V, I)
plt.plot(x, y_L_bscb, color='red', linestyle='dashed')
plt.plot(x, y_U_bscb, color='red', linestyle='dashed')
plt.show()

###############################
#     return nd.gradient(lambda var: Lik(var, y))(theta)
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