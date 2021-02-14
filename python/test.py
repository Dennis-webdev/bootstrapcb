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

f = open("python/dict_musigma_1000_nonparam.txt")
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
# plt.scatter(thetaX, thetaY)
[optX, optY] = opt
# plt.scatter(optX, optY)
# plt.show()

n = B
theta_hat = np.sum(thetaX) / n
S_2 = np.sum([(thetaX[i] - theta_hat)**2 for i in range(n)])
S = np.sqrt(S_2)
x_critval = 1.282 * S / np.sqrt(n)

theta_hat = np.sum(thetaY) / n
S_2 = np.sum([(thetaY[i] - theta_hat)**2 for i in range(n)])
S = np.sqrt(S_2)
y_critval = 1.282 * S / np.sqrt(n)

cover_x = [Theta[i][0] for i in range(B) if np.abs(thetaX[i] - optX) < x_critval and np.abs(thetaY[i] - optY) < y_critval]
cover_y = [Theta[i][1] for i in range(B) if np.abs(thetaX[i] - optX) < x_critval and np.abs(thetaY[i] - optY) < y_critval]
plt.scatter(cover_x, cover_y)
out_x = [Theta[i][0] for i in range(B) if np.abs(thetaX[i] - optX) > x_critval or np.abs(thetaY[i] - optY) > y_critval]
out_y = [Theta[i][1] for i in range(B) if np.abs(thetaX[i] - optX) > x_critval or np.abs(thetaY[i] - optY) > y_critval]
plt.scatter(out_x, out_y, color='green')
[optX, optY] = opt
plt.scatter(optX, optY)
plt.show()
