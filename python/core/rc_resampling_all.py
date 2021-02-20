import math
import numpy as np
import pandas as pd
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root
from scipy.integrate import quad

def mle(x, y, pdf, guess, info=False):
    def _L(theta,x,y):
        return np.sum( [np.log(pdf(x[i],y[i],theta)) for i in range(len(x))] )
    opt = minimize(lambda var: -_L(var,x,y), guess, method='Nelder-Mead').x.tolist()
    if info:
        I = ( -nd.Hessian(lambda var: _L(var,x,y))(opt) ).tolist()
        V = np.linalg.inv(I).tolist()
        return opt, V, I
    return opt

def generate_y_nonparam(x, y, B=1000, n=50):
    def _generate_y(y,r):
        sortedY = y.copy()
        n = len(sortedY)
        for i in range(n):
            for j in range(i+1,n):
                if sortedY[j] < sortedY[i]: 
                    tmpY = sortedY[i] 
                    sortedY[i] = sortedY[j]
                    sortedY[j] = tmpY
        sum = 0
        i = -1
        while sum < r:
            sum += 1 / n
            i += 1
        if i < 0: return 0
        return sortedY[i] 
    X = []
    Y = []
    for _ in range(B):
        x_sample = []
        y_sample = []
        for _ in range(n): 
            r = np.random.uniform(0, 1)
            i = np.random.choice(range(len(x)))
            x_sample.append(x[i])
            y_sample.append(_generate_y(y[i],r))
        X.append(x_sample)
        Y.append(y_sample)
    return X, Y

def generate_y_param(x, cdf, y0, B=1000, n=50):
    def _generate_y(x,r,y0):
        return root(lambda var: (cdf(x,var) - r), y0).x[0]
    X = []
    Y = []
    for _ in range(B):
        x_sample = []
        y_sample = []
        for _ in range(n): 
            r = np.random.uniform(0, 1)
            i = np.random.choice(range(len(x)))
            x_sample.append(x[i])
            y_sample.append(_generate_y(x[i],r,y0[i]))
        X.append(x_sample)
        Y.append(y_sample)
    return X, Y

# Given:
guess = [10.17, 153.79, 17.26, -2.58, 0.455, 0.01746]
xs = [2.0, 7.0, 12.0, 19.5, 29.5, 39.5, 54.5, 75.0]
ys = [ [ 1.26,   2.78,  0.63,  0.34], 
       [ 3.53,    4.1,  1.31,  0.91],
       [11.98,  13.14,  9.86,  6.53], 
       [90.82,  97.12, 75.85, 59.46], 
       [83.45, 116.62,  104., 80.85], 
       [55.98,  67.28, 79.33, 82.66],
       [66.32,  78.53,  69.1, 67.27], 
       [39.42,  55.35, 60.76,  73.2] ]
def w(x, theta):
    t2 = theta[1]
    t3 = theta[2]
    t4 = theta[3]
    t5 = theta[4]
    t6 = theta[5]
    return (t2 + t4*x + t6*x**2) * np.exp(t5*(x-t3))/( 1 + np.exp(t5*(x-t3)) )
def pdf(x, y, theta):
    mu = w(x, theta)
    sigma = theta[0]
    return np.exp( -(y - mu)**2 / (2 * sigma**2) ) / np.sqrt(2 * np.pi * sigma**2)
def cdf(x, y, theta):
    value, error = quad(lambda var: pdf(x,var,theta), -np.inf, y)
    return value

# # MLE
# opt, V, I = mle(
#     [xs[i] for i in range(len(ys)) for _   in ys[i]], 
#     [y_i   for i in range(len(ys)) for y_i in ys[i]], 
#     pdf, 
#     guess, 
#     info=True
# )

# # generate BS samples
# # Y_samples = generate_samples(lambda y: pdf(0,y,opt))
# X_samples, Y_samples = generate_y_param(
#     xs, 
#     lambda x,y: cdf(x,y,opt), 
#     [w(x_i, opt) for x_i in xs], 
#     1000, 
#     50
# )

# # Test: parametric vs non parametric sampling
# mle_estimate_test = [w(x_i, opt) for x_i in xs[2:3]]
# X_samples_test, Y_samples_test = generate_y_nonparam(xs[2:3], ys[2:3], 1, 1000)
# plt.subplot(411)
# plt.hist(Y_samples_test, density=True)
# X_samples_test, Y_samples_test = generate_y_param(xs[2:3], lambda x,y: cdf(x,y,opt), mle_estimate_test, 1, 1000)
# plt.subplot(412)
# plt.hist(Y_samples_test, density=True)
# plt.subplot(413)
# plt.scatter(ys[2], [0]*len(ys[2]))
# plt.subplot(414)
# y = np.linspace(5,15,num=100)
# plt.plot(y, [pdf(xs[2],y_i,opt) for y_i in y])
# plt.show()

# # jth MLE
# Theta_samples = []
# for i in range(len(X_samples)):
#     theta_sample = mle(X_samples[i], Y_samples[i], pdf, guess)
#     Theta_samples.append(theta_sample)

# # Test: generated MLEs scatter plot 
# thetaX = [Theta_samples[i][2] for i in range(len(Theta_samples))]
# thetaY = [Theta_samples[i][3] for i in range(len(Theta_samples))]
# plt.scatter(thetaX, thetaY)
# [optX, optY] = opt[2:4]
# plt.scatter(optX, optY)
# plt.show()

# dict = {
#     'opt': opt,
#     'I': I,
#     'V': V,
#     'X_samples': X_samples,
#     'Y_samples': Y_samples,
#     'Theta_samples': Theta_samples,
# }
# f = open("python/dict.txt","w")
# f.write( str(dict) )
# f.close()

###############################################################################

f = open("python/dict.txt")
dict = f.read()
dict = eval(dict)
opt = dict['opt']
I = dict['I']
V = dict['V'] 
X_samples = dict['X_samples']
Y_samples = dict['Y_samples']
Theta_samples = dict['Theta_samples']  
B = len(Theta_samples)
p = len(opt)
f.close()

# Test: generated MLEs scatter plot 
thetaX = [Theta_samples[i][2] for i in range(len(Theta_samples))]
thetaY = [Theta_samples[i][3] for i in range(len(Theta_samples))]
plt.scatter(thetaX, thetaY)
[optX, optY] = opt[2:4]
plt.scatter(optX, optY)
plt.show()

# jth GoF test statistic
C = []
for j in range(B):
    c_sample = np.subtract(opt, Theta_samples[j]).T @ I @ np.subtract(opt, Theta_samples[j])
    C.append(c_sample)
# order test values
for i in range(B):
    for j in range(i+1,B):
        if C[j] > C[i]: 
            tmpC = C[i] 
            C[i] = C[j]
            C[j] = tmpC
            tmpTheta = Theta_samples[i] 
            Theta_samples[i] = Theta_samples[j]
            Theta_samples[j] = tmpTheta
# rounded integer subscript
c_critval = np.percentile(C, 90) 

# Test: histogram of Chi^2 distribution
plt.hist(C, density=True)
plt.show()

# Test: MLE
x = np.linspace(1, 80, num=100)
mle_estimate = [w(x_i, opt) for x_i in x]
plt.plot(x, mle_estimate)

# Test: delta methode
confBand_L, confBand_U = [None]*len(x), [None]*len(x)
for i in range(len(x)):
    dg = nd.Gradient(lambda var: w(x[i], var))
    h = np.sqrt( c_critval * dg(opt).T @ V @ dg(opt) ) # TODO Quantile
    confBand_L[i] = mle_estimate[i] - h
    confBand_U[i] = mle_estimate[i] + h
plt.plot(x, confBand_L, color='red', linestyle='dashed')
plt.plot(x, confBand_U, color='red', linestyle='dashed')

# Test: R_alpha
R_alpha = [Theta_samples[i] for i in range(B) if C[i] < c_critval]
confBand_L, confBand_U = [None]*len(x), [None]*len(x)
for theta in R_alpha:
    for i in range(len(x)):
        fx = w(x[i], theta)
        if confBand_L[i] is None or fx < confBand_L[i]: 
            confBand_L[i] = fx
        if confBand_U[i] is None or fx > confBand_U[i]: 
            confBand_U[i] = fx
plt.plot(x, confBand_L, color='green', linestyle='dashed')
plt.plot(x, confBand_U, color='green', linestyle='dashed')

# Test: dR_alpha nelder-mead
L = np.linalg.cholesky(V)
def _restircted_search(z,x):
    z_sum = np.sum([z[j]**2 for j in range(p)])
    z_sum = np.sqrt(z_sum)
    Z = [np.sqrt(c_critval) * z[i] / z_sum for i in range(p)]
    W = L @ Z
    theta = np.add(W, opt)
    return w(x, theta)
confBand_L, confBand_U = [None]*len(x), [None]*len(x)
for i in range(len(x)):
    min = minimize(lambda var: _restircted_search(var,x[i]), [0]*len(opt), method='Nelder-Mead').x
    confBand_L[i] = _restircted_search(min,x[i])
    max = minimize(lambda var: -_restircted_search(var,x[i]), [0]*len(opt), method='Nelder-Mead').x
    confBand_U[i] = _restircted_search(max,x[i])
plt.plot(x, confBand_L, color='black', linestyle='dashed')
plt.plot(x, confBand_U, color='black', linestyle='dashed')

# Test: dR_alpha
dR_alpha = []
L = np.linalg.cholesky(V)
for j in range(1000):
    z = np.random.normal(0, 1, p)
    z_sum = np.sum([z[j]**2 for j in range(p)])
    z_sum = np.sqrt(z_sum)
    Z = [np.sqrt(c_critval) * z[i] / z_sum for i in range(p)]
    W = L @ Z
    dR_alpha.append(np.add(W, opt).tolist())
confBand_L, confBand_U = [None]*len(x), [None]*len(x)
for theta in dR_alpha:
    for i in range(len(x)):
        fx = w(x[i], theta)
        if confBand_L[i] is None or fx < confBand_L[i]: 
            confBand_L[i] = fx
        if confBand_U[i] is None or fx > confBand_U[i]: 
            confBand_U[i] = fx
plt.plot(x, confBand_L, color='blue', linestyle='dashed')
plt.plot(x, confBand_U, color='blue', linestyle='dashed')
plt.show()

# Test: Likelihood based Confidence regions
thetaX = [Theta_samples[i][2] for i in range(len(Theta_samples))]
thetaY = [Theta_samples[i][3] for i in range(len(Theta_samples))]
ax1 = plt.subplot(211)
ax1.set_xlim([15, 20])
ax1.scatter(thetaX, thetaY)
thetaX = [R_alpha[i][2] for i in range(len(R_alpha))]
thetaY = [R_alpha[i][3] for i in range(len(R_alpha))]
ax1.scatter(thetaX, thetaY, color='green')
thetaX = [dR_alpha[i][2] for i in range(len(dR_alpha))]
thetaY = [dR_alpha[i][3] for i in range(len(dR_alpha))]
ax2 = plt.subplot(212)
ax2.set_xlim([15, 20])
ax2.scatter(thetaX, thetaY, color='red')
[optX, optY] = opt[2:4]
ax1.scatter(optX, optY)
ax2.scatter(optX, optY)
plt.show()