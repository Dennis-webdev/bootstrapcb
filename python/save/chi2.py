
import math
import numpy as np
import pandas as pd
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Given:
ys = [  4.3, 4.7, 4.7, 3.1, 5.2, 6.7, 4.5, 3.6, 7.2, 10.9, 6.6, 5.8,
        6.3, 4.7, 8.2, 6.2, 4.2, 4.1, 3.3, 4.6, 6.3,  4.0, 3.1, 3.5,
        7.8, 5.0, 5.7, 5.8, 6.4, 5.2, 8.0, 4.9, 6.1,  8.0, 7.7, 4.3,
       12.5, 7.9, 3.9, 4.0, 4.4, 6.7, 3.8, 6.4, 7.2,  4.8, 10.5]

def pdf(x, y, theta):
    return y**(theta[0]-1) * np.exp(-y/theta[1]) / (math.gamma(theta[0]) * theta[1]**theta[0])

def w(x, theta):
    return (1+theta[0]) * theta[0] * theta[1]**2 * x / (2 * (1 - theta[0]*theta[1]*x))

# MLE
def L(var,y):
    return np.sum( [np.log(pdf(0,y[i],var)) for i in range(len(y))] )

opt = minimize(lambda var: -L(var,ys), [9,1], method='Nelder-Mead').x
I = -nd.Hessian(lambda var: L(var,ys))(opt)
V = np.linalg.inv(I) 

# GoF test statistic
chi2 = 10.64 # np.subtract(opt, theta).T @ I @ np.subtract(opt, theta) 

def bootstrap(x):
    theta = []
    C = []
    for j in range(100):
        # jth BS sample
        def _generate_sample(n):
            y = []
            for i in range(n): 
                sum = 0
                y_i = 0
                r = np.random.uniform(0, 1)
                while sum < r and y_i <= 15:
                    sum += pdf(0,y_i,opt) * 0.01
                    y_i += 0.01
                y.append(y_i)
            return y
        y_sample = _generate_sample(50)
        # jth MLE
        theta_sample = minimize(lambda var: -L(var,y_sample), [9,1], method='Nelder-Mead').x
        theta.append(theta_sample)
        # jth GoF test statistic
        C_sample = np.subtract(opt, theta_sample).T @ I @ np.subtract(opt, theta_sample) 
        C.append(C_sample)
    # order test values
    for i in range(len(C)):
        for j in range(i+1,len(C)):
            if C[j] < C[i]: 
                tmpC = C[i] 
                C[i] = C[j]
                C[j] = tmpC
                tmpTheta = theta[i]
                theta[i] = theta[j]
                theta[j] = tmpTheta
    # rounded integer subscript
    chi2_sample = np.percentile(C, 90)
    theta = [theta[i] for i in range(len(theta)) if C[i] < chi2_sample]    
    # maximize and minimize the value of f
    confBand_L, confBand_U = [None]*len(x), [None]*len(x)
    for theta_sample in theta:
        for i in range(len(x)):
            fx = w(x[i], theta_sample)
            if confBand_L[i] is None or fx < confBand_L[i]: 
                confBand_L[i] = fx
            if confBand_U[i] is None or fx > confBand_U[i]: 
                confBand_U[i] = fx
    return confBand_L, confBand_U

x = np.linspace(0.0,0.1,num=100)
plt.plot(x, [w(x_i,opt) for x_i in x])
cb_L, cb_U = bootstrap(x)
plt.plot(x, cb_L, color='green', linestyle='dashed')
plt.plot(x, cb_U, color='green', linestyle='dashed')
plt.show()
    
# def test():
#     y = np.linspace(0,15,num=100)
#     plt.plot(y,[pdf(0,y_i,opt) for y_i in y])
#     plt.hist(C, density=True)
#     plt.show()
# test()
