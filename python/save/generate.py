import math
import numpy as np
import pandas as pd
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Given:
guess = [5.9,2]
ys = [  4.3, 4.7, 4.7, 3.1, 5.2, 6.7, 4.5, 3.6, 7.2, 10.9, 6.6, 5.8,
        6.3, 4.7, 8.2, 6.2, 4.2, 4.1, 3.3, 4.6, 6.3,  4.0, 3.1, 3.5,
        7.8, 5.0, 5.7, 5.8, 6.4, 5.2, 8.0, 4.9, 6.1,  8.0, 7.7, 4.3,
       12.5, 7.9, 3.9, 4.0, 4.4, 6.7, 3.8, 6.4, 7.2,  4.8, 10.5]
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

# MLE
def L(theta,y):
    return np.sum( [np.log(pdf(0,y[i],theta)) for i in range(len(y))] )
def mle(y):
    max = minimize(lambda var: -L(var,y), guess, method='Nelder-Mead').x
    return max.tolist()
opt = mle(ys)
try: 
    I = -nd.Hessian(lambda var: L(var,ys))(opt)
    I = I.tolist()
    V = np.linalg.inv(I)
    V = V.tolist()
except:
    I, V = [], []

B=1000
n=50
Theta = []
Y = []
for j in range(B):
    def _generate_sample_param(r, step=0.01):
        sum = 0
        y_i = 0
        while sum < r:
            sum += pdf(0,y_i,opt) * step
            y_i += step
        return y_i
    def _generate_sample_nonparam(r, step=0.01):
        n = len(ys)
        for i in range(n):
            for j in range(i+1,n):
                if ys[j] < ys[i]: 
                    tmpYs = ys[i] 
                    ys[i] = ys[j]
                    ys[j] = tmpYs
        sum = 0
        i = -1
        while sum < r:
            sum += 1 / n
            i += 1
        if i < 0: return 0
        return ys[i] 
    # jth BS sample
    y_sample = []
    for i in range(n): 
        r = np.random.uniform(0, 1)
        y_sample.append(_generate_sample_nonparam(r))
    Y.append(y_sample)
    # jth MLE
    theta_sample = mle(y_sample)
    Theta.append(theta_sample)

dict = {
    'opt': opt,
    'I': I,
    'V': V,
    'Y': Y,
    'Theta': Theta,
}
f = open("python/dict.txt","w")
f.write( str(dict) )
f.close()