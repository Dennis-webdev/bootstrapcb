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
    return y**(a-1) * np.exp(-y/b) / (math.gamma(a) * b**a)
def cdf(x, y, theta, step=0.01):
    sum = 0
    y_i = 0
    while y_i < y:
        sum += pdf(0,y_i,theta) * step
        y_i += step
    return sum
def w(x, theta):
    mu = theta[0]
    sigma = theta[1]
    a = (mu / sigma)**2
    b = sigma**2 / mu
    return (1+a) * a * b**2 * x / (2 * (1 - a*b*x))

# MLE
def L(theta,y):
    return np.sum( [np.log(pdf(0,y[i],theta)) for i in range(len(y))] )
def mle(y):
    return minimize(lambda var: -L(var,y), guess, method='Nelder-Mead').x

# GoF test statistic
def Statistic_Test(y, theta):
    sum = 0
    n = len(y)
    for i in range(n):
        for j in range(i+1,n):
            if y[j] < y[i]: 
                tmpY = y[i] 
                y[i] = y[j]
                y[j] = tmpY
    for i in range(1,n+1):
        Z_i = cdf(0,y[i - 1],theta)
        Z_n_1 = cdf(0,y[n+1-i - 1],theta)
        sum += (np.log(Z_i) + np.log(1 - Z_n_1)) * (2*i - 1) / n
    return -n - sum
def quantile(q, T):
    return np.percentile(T, q)

f = open("dict.txt")
dict = f.read()
dict = eval(dict)
opt = dict['opt']
Y = dict['Y']
Theta = dict['Theta']   
T = []
A2 = []
B = len(Theta)
f.close()

thetaX = [x for [x,_] in Theta]
thetaY = [y for [_,y] in Theta]
plt.scatter(thetaX, thetaY)
[optX, optY] = opt
plt.scatter(optX, optY)
plt.show()

# jth GoF test statistic
for j in range(B):
    a_sample = Statistic_Test(Y[j],Theta[j])
    A2.append(a_sample)
# order test values
for i in range(B):
    for j in range(i+1,len(A2)):
        if A2[j] < A2[i]: 
            tmpA2 = A2[i] 
            A2[i] = A2[j]
            A2[j] = tmpA2
# rounded integer subscript
a_sample = quantile(90, A2) 

plt.hist(A2, density=True)
plt.show()

for j in range(B):
    t_sample = Statistic_Test(ys,Theta[j])
    T.append(t_sample)

thetaX = [Theta[i][0] for i in range(B) if T[i] < a_sample]
thetaY = [Theta[i][1] for i in range(B) if T[i] < a_sample]
plt.scatter(thetaX, thetaY)
thetaX_out = [Theta[i][0] for i in range(B) if T[i] > a_sample]
thetaY_out = [Theta[i][1] for i in range(B) if T[i] > a_sample]
plt.scatter(thetaX_out, thetaY_out, color='green')
[optX, optY] = opt
plt.scatter(optX, optY)
plt.show()

# maximize and minimize the value of f
x = np.linspace(0.0,0.1,num=100)
mle_estimate = [w(x_i,opt) for x_i in x]
confBand_L, confBand_U = [None]*len(x), [None]*len(x)
Theta = [Theta[i] for i in range(B) if T[i] < t_sample]
for theta_sample in Theta:
    for i in range(len(x)):
        fx = w(x[i], theta_sample)
        if confBand_L[i] is None or fx < confBand_L[i]: 
            confBand_L[i] = fx
        if confBand_U[i] is None or fx > confBand_U[i]: 
            confBand_U[i] = fx

# plot
plt.plot(x, mle_estimate)
plt.plot(x, confBand_L, color='green', linestyle='dashed')
plt.plot(x, confBand_U, color='green', linestyle='dashed')
plt.show()