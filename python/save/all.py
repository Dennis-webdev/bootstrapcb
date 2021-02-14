import math
import numpy as np
import pandas as pd
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Given:
def w(x, theta):
    mu = theta[0]
    sigma = theta[1]
    a = (mu / sigma)**2
    b = sigma**2 / mu
    return (1+a) * a * b**2 * x / (2 * (1 - a*b*x))
def quantile(q, T):
    return np.percentile(T, q)

f = open("dict_musigma_1000_nonparam.txt")
dict = f.read()
dict = eval(dict)
opt = dict['opt']
I = dict['I']
V = dict['V'] 
Y = dict['Y']
Theta = dict['Theta']   
C = []
B = len(Theta)
p = len(opt)
f.close()

x = np.linspace(0.0,0.1,num=100)
mle_estimate = [w(x_i,opt) for x_i in x]

thetaX = [x for [x,_] in Theta]
thetaY = [y for [_,y] in Theta]
plt.scatter(thetaX, thetaY)
[optX, optY] = opt
plt.scatter(optX, optY)
plt.show()

###############################################################

# jth GoF test statistic
for j in range(B):
    c_sample = np.subtract(opt, Theta[j]).T @ I @ np.subtract(opt, Theta[j])
    C.append(c_sample)
# order test values
for i in range(B):
    for j in range(i+1,B):
        if C[j] < C[i]: 
            tmpC = C[i] 
            C[i] = C[j]
            C[j] = tmpC
            tmpTheta = Theta[i] 
            Theta[i] = Theta[j]
            Theta[j] = tmpTheta
# rounded integer subscript
c_critval = quantile(90, C) 

plt.hist(C, density=True)
plt.show()

###############################################################

thetaX = [Theta[i][0] for i in range(B) if C[i] < c_critval]
thetaY = [Theta[i][1] for i in range(B) if C[i] < c_critval]
plt.scatter(thetaX, thetaY)
thetaX_out = [Theta[i][0] for i in range(B) if C[i] > c_critval]
thetaY_out = [Theta[i][1] for i in range(B) if C[i] > c_critval]
plt.scatter(thetaX_out, thetaY_out, color='green')
[optX, optY] = opt
plt.scatter(optX, optY)
# plt.show()

# maximize and minimize the value of f
confBand_L, confBand_U = [None]*len(x), [None]*len(x)
Theta = [Theta[i] for i in range(B) if C[i] < c_critval]
for theta_sample in Theta:
    for i in range(len(x)):
        fx = w(x[i], theta_sample)
        if confBand_L[i] is None or fx < confBand_L[i]: 
            confBand_L[i] = fx
        if confBand_U[i] is None or fx > confBand_U[i]: 
            confBand_U[i] = fx

###############################################################

Theta = []
L = np.linalg.cholesky(V)
for j in range(B):
    # obtain a point of theta on the edge of the confidence region
    z = np.random.normal(0, 1, p)
    z_sum = np.sum([z[j]**2 for j in range(p)])
    z_sum = np.sqrt(z_sum)
    Z = [np.sqrt(c_critval) * z[i] / z_sum for i in range(p)]
    W = L @ Z
    Theta.append(np.add(W, opt))

thetaX = [x for [x,_] in Theta]
thetaY = [y for [_,y] in Theta]
plt.scatter(thetaX, thetaY, color="red")
plt.show()

# maximize and minimize the value of f
confBand2_L, confBand2_U = [None]*len(x), [None]*len(x)
# Theta = [Theta[i] for i in range(B) if C[i] < c_critval]
for theta_sample in Theta:
    for i in range(len(x)):
        fx = w(x[i], theta_sample)
        if confBand2_L[i] is None or fx < confBand2_L[i]: 
            confBand2_L[i] = fx
        if confBand2_U[i] is None or fx > confBand2_U[i]: 
            confBand2_U[i] = fx

###############################################################

# plot
plt.plot(x, mle_estimate)
plt.plot(x, confBand_L, color='green', linestyle='dashed')
plt.plot(x, confBand_U, color='green', linestyle='dashed')

# plot2
plt.plot(x, confBand2_L, color='red', linestyle='dashed')
plt.plot(x, confBand2_U, color='red', linestyle='dashed')
plt.show()