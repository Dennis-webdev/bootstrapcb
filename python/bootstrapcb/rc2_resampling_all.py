import math
import numpy as np
import pandas as pd
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def mle(y, pdf, guess, info=False):
    def _L(theta,y):
        return np.sum( [np.log(pdf(y[i],theta)) for i in range(len(y))] )
    opt = minimize(lambda var: -_L(var,y), guess, method='Nelder-Mead').x
    if info:
        I = -nd.Hessian(lambda var: _L(var,ys))(opt)
        V = np.linalg.inv(I)
        return opt.tolist(), V.tolist(), I.tolist()
    return opt.tolist()

def generate_samples(var, B=1000, n=50):
    def _generate_y_nonparam(y, r):
        n = len(y)
        sortedY = y
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
        return y[i] 
    def _generate_y_param(pdf, r, step=0.01):
        sum = 0
        y_i = 0
        while sum < r:
            sum += pdf(y_i) * step
            y_i += step
        return y_i
    try: 
        _ = iter(var)
        _generate_y = lambda r: _generate_y_nonparam(var,r)
    except: 
        _generate_y = lambda r: _generate_y_param(var,r)
    Y = []
    for _ in range(B):
        y_sample = []
        for _ in range(n): 
            r = np.random.uniform(0, 1)
            y_sample.append(_generate_y(r))
        Y.append(y_sample)
    return Y

# Given:
guess = [5.9,2]
ys = [  4.3, 4.7, 4.7, 3.1, 5.2, 6.7, 4.5, 3.6, 7.2, 10.9, 6.6, 5.8,
        6.3, 4.7, 8.2, 6.2, 4.2, 4.1, 3.3, 4.6, 6.3,  4.0, 3.1, 3.5,
        7.8, 5.0, 5.7, 5.8, 6.4, 5.2, 8.0, 4.9, 6.1,  8.0, 7.7, 4.3,
       12.5, 7.9, 3.9, 4.0, 4.4, 6.7, 3.8, 6.4, 7.2,  4.8, 10.5]
def w(x, theta):
    mu = theta[0]
    sigma = theta[1]
    a = (mu / sigma)**2
    b = sigma**2 / mu
    return (1+a) * a * b**2 * x / (2 * (1 - a*b*x))
def pdf(x, y, theta):
    mu = theta[0]
    sigma = theta[1]
    a = (mu / sigma)**2
    b = sigma**2 / mu
    try:
        return y**(a-1) * np.exp(-y/b) / (math.gamma(a) * b**a)
    except:
        return 0
def cdf(x, y, theta, step=0.01):
    sum = 0
    y_i = 0
    while y_i < y:
        sum += pdf(0,y_i,theta) * step
        y_i += step
    return sum

# # MLE
# opt, V, I = mle(ys, lambda y,theta: pdf(0,y,theta), guess, info=True)

# # generate BS samples
# # Y_samples = generate_samples(lambda y: pdf(0,y,opt))
# Y_samples = generate_samples(ys, 1000, 50)

# # Test: parametric vs non parametric sampling
# Y_samples_test = generate_samples(ys, 1, 1000)
# plt.subplot(411)
# plt.hist(Y_samples_test[0], density=True)
# Y_samples_test = generate_samples(lambda y: pdf(0,y,opt), 1, 1000)
# plt.subplot(412)
# plt.hist(Y_samples_test[0], density=True)
# plt.subplot(413)
# x = np.linspace(0,15,num=len(ys))
# plt.plot(x, [pdf(0,x_i,opt) for x_i in x])
# plt.subplot(414)
# plt.scatter(ys, [0.2]*len(ys))
# plt.show()

# # jth MLE
# Theta_samples = []
# for y_sample in Y_samples:
#     theta_sample = mle(y_sample, lambda y,theta: pdf(0,y,theta), guess)
#     Theta_samples.append(theta_sample)

# # Test: generated MLEs scatter plot 
# thetaX = [x for [x,_] in Theta_samples]
# thetaY = [y for [_,y] in Theta_samples]
# plt.scatter(thetaX, thetaY)
# [optX, optY] = opt
# plt.scatter(optX, optY)
# plt.show()

# dict = {
#     'opt': opt,
#     'I': I,
#     'V': V,
#     'Y_samples': Y_samples,
#     'Theta_samples': Theta_samples,
# }
# f = open("python/dict.txt","w")
# f.write( str(dict) )
# f.close()

###############################################################################

f = open("python/dict_musigma_1000_param.txt")
dict = f.read()
dict = eval(dict)
opt = dict['opt']
I = dict['I']
V = dict['V'] 
Y_samples = dict['Y_samples']
Theta_samples = dict['Theta_samples']  
B = len(Theta_samples)
p = len(opt)
f.close()

# Test: generated MLEs scatter plot 
thetaX = [x for [x,_] in Theta_samples]
thetaY = [y for [_,y] in Theta_samples]
plt.scatter(thetaX, thetaY)
[optX, optY] = opt
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
        if C[j] < C[i]: 
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
x = np.linspace(0,0.1,num=100)
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
for j in range(B):
    z = np.random.normal(0, 1, p)
    z_sum = np.sum([z[j]**2 for j in range(p)])
    z_sum = np.sqrt(z_sum)
    Z = [np.sqrt(c_critval) * z[i] / z_sum for i in range(p)]
    W = L @ Z
    dR_alpha.append(np.add(W, opt))
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
thetaX = [x for [x,_] in Theta_samples]
thetaY = [y for [_,y] in Theta_samples]
plt.scatter(thetaX, thetaY, color='green')
thetaX = [x for [x,_] in R_alpha]
thetaY = [y for [_,y] in R_alpha]
plt.scatter(thetaX, thetaY)
thetaX = [x for [x,_] in dR_alpha]
thetaY = [y for [_,y] in dR_alpha]
plt.scatter(thetaX, thetaY, color='red')
[optX, optY] = opt
plt.scatter(optX, optY)
plt.show()