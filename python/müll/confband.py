import numpy as np
import pandas as pd
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

x = np.linspace(1, 80, num=100)
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

# Confidence Interval
def native_ci(x, alpha, eta, theta, V):
    errorInt = []
    for i in range(len(x)):
        g = lambda var: eta(x[i], var)
        dg = Derivative(g)
        h = 1.96 * np.sqrt( dg(theta).T @ V @ dg(theta) ) # TODO Quantile
        errorInt.append(h)
    return errorInt

# Confidence Band
def bootstrap_cb(x, alpha, eta, theta, V, I, n=50, B=30):
    sampleC = []
    for _ in range(B):
        sampleX = np.random.choice(x, n)
        sampleY = [y_sample(x_i, eta, theta) for x_i in sampleX]
        sampleTheta, sampleV = MLE(sampleX, sampleY, eta, theta)
        thetaDiff = np.subtract(sampleTheta, theta)
        sampleC.append( thetaDiff.T @ I @ thetaDiff )
    sampleC = np.sort(sampleC)
    chi_square = np.percentile(sampleC, 100*(1-alpha))
    errorInt = [] 
    for i in range(len(x)):
        g = lambda var: eta(x[i], var)
        dg = Derivative(g)
        h = np.sqrt(chi_square) * np.sqrt( dg(theta).T @ V @ dg(theta) ) # TODO Quantile
        errorInt.append(h)
    return errorInt

# PDF functions
def pdf(x, theta, type='uniform'):
    # Uniform
    pdf = lambda var: np.exp( -(var - theta[0])**2 / (2 * theta[1]**2) ) / np.sqrt(2 * np.pi * theta[1]**2)
    return eval(pdf, x)

# Optimization
def maximize(_f, x0):
    f = lambda var: -_f(var)
    opt = minimize(f, x0, method='Nelder-Mead')
    return opt.x

# Differentiation
def Hessian(f):
    return nd.Hessian(f)

def Derivative(f):
    return nd.Derivative(f)

def Inverse(A):
    return np.linalg.inv(A) # TODO 

#########################################################################################################
# Model + Likelyhood functions
def eta(x, theta): 
    eta = lambda var: (theta[1] + theta[3]*var + theta[5] * var**2) * np.exp(theta[4]*(var-theta[2]))/( 1 + np.exp(theta[4]*(var-theta[2])) )
    return eval(eta, x)

def Likelyhood(theta, eta, x, y):
    f_i = []
    for i in range(len(x)):
        mu = eta(x[i], theta)
        sigma = theta[0]
        f_i.append( pdf(y[i], [mu, sigma], type='uniform') )
    return np.prod( f_i )

def logLikelyhood(theta, eta, x, y):
    return np.log( Likelyhood(theta, eta, x, y) )

def logLikelyhood_sum(theta, eta, x, y):
    logf_i = []
    for i in range(len(x)):
        mu = eta(x[i], theta)
        sigma = theta[0]
        f_i = pdf(y[i], [mu, sigma], type='uniform')
        logf_i.append( np.log(f_i) )
    return np.sum( logf_i )

def y_sample(x, eta, theta): 
    y = lambda var: eta(var, theta) + np.random.normal(0, theta[0])
    return eval(y, x)

# MLE estimation
def MLE(x, y, eta, theta):
    L = lambda var: logLikelyhood_sum(var, eta, x, y)
    opt = maximize(L, x0=theta)
    ddL = Hessian(L)
    I = -ddL(opt)
    V = Inverse(I) 
    return opt, V, I

opt, V = MLE(x_i, y_i, eta=eta, theta=[10.17, 153.79, 17.26, -2.58, 0.455, 0.01746])
mean = eta(x, opt)
h_ci = native_ci(x, alpha=0.1, eta=eta, theta=opt, V=V)
# h_bsci = bootstrap_ci(x, a=0.1, TODO)
# h_cb = native_cb(x, a=0.1, TODO)
# h_bscb = bootstrap_cb(x, a=0.1, TODO)
h_bscb = bootstrap_cb(x, alpha=0.1, eta=eta, theta=opt, V=V, I=I)
plt.plot(xs, data, 'o', color='black', markersize='3')
plt.plot(x, mean)
plt.plot(x, [mean[i] - h_ci[i]   for i in range(len(x))], color='red',   linestyle='dashed')
plt.plot(x, [mean[i] + h_ci[i]   for i in range(len(x))], color='red',   linestyle='dashed')
plt.plot(x, [mean[i] - h_bscb[i] for i in range(len(x))], color='green', linestyle='dashed')
plt.plot(x, [mean[i] + h_bscb[i] for i in range(len(x))], color='green', linestyle='dashed')
plt.show()
