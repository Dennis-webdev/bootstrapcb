import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Utilities
def isiterable(obj):
    try:
        it = iter(obj)
    except TypeError: 
        return False
    return True

def eval(f, vars):
    if isiterable(vars):
        return [ f(var) for var in vars ]
    return f(vars)

def replace(v, indize, value):
    v_new = v.copy()
    if isiterable(indize):
        for i in range(len(indize)):
            v_new[ indize[i] ] = value[i]
    else:
        v_new[ indize ] = value  
    return v_new      

# Differentiation
def derivative(f, vars, n=1, h=0.001, method='central'): 
    if method == 'central':
        df = lambda var: (f(var + h) - f(var - h))/(2*h)
    elif method == 'forward':
        df = lambda var: (f(var + h) - f(var))/(h)
    elif method == 'backward':
        df = lambda var: (f(var) - f(var - h))/(h)
    if n>1: 
        deriv = lambda x: derivative(df, x, n-1, h, method) 
        return eval(deriv, vars)
    return eval(df, vars)

def Derivative(f, vec, h=0.001):
    n = len(vec)
    D = np.zeros(n)
    for i in range(n):
        d1 = f( replace(vec, i, vec[i]+h) )
        d2 = f( replace(vec, i, vec[i]-h) )
        d = d1/(2*h) - d2/(2*h) #( d1-d2 )/(2*h)
        D[i] = d
    return D 

def Hessian(f, vec, h=0.001):
    n = len(vec)
    H = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            vec_d1 = replace(vec, i, vec[i]+h)
            vec_d2 = replace(vec, i, vec[i]-h)
            vec_dd1 = replace(vec_d1, j, vec_d1[j]+h)
            vec_dd2 = replace(vec_d1, j, vec_d1[j]-h)
            vec_dd3 = replace(vec_d2, j, vec_d2[j]+h)
            vec_dd4 = replace(vec_d2, j, vec_d2[j]-h)
            dd1 = f( vec_dd1 )
            dd2 = f( vec_dd2 )
            dd3 = f( vec_dd3 )
            dd4 = f( vec_dd4 )
            d =  dd1/(4*h**2) - dd2/(4*h**2) - dd3/(4*h**2) + dd4/(4*h**2) #( (d1-d2)/(2*h) - (d3-d4)/(2*h) )/(2*h)
            H[i][j] = d
    return H

# test = lambda x: x[1]*x[0]**2 + x[1]**2 + x[2]**2
# print(Hessian(test, [3,2,10]))

# Model function
def eta(x, theta): 
    eta = lambda var: (theta[1] + theta[3]*var + theta[5] * var**2) * np.exp(theta[4]*(var-theta[2]))/( 1 + np.exp(theta[4]*(var-theta[2])) )
    return eval(eta, x)

# Confidence Interval
def h_ci(x, alpha, eta, theta, V):
    errorInt = []
    for i in range(len(x)):
        g = lambda theta: eta(x[i], theta)
        dg = Derivative(g, theta)
        h = 1.96 * np.sqrt( dg.T @ V @ dg ) # TODO Quantile
        errorInt.append(h)
    return errorInt

# Confidence Band
def h_cb(x, alpha, eta, theta, V):
    errorInt = []
    for i in range(len(x)):
        g = lambda theta: eta(x[i], theta)
        dg = Derivative(g, theta)
        h = 1.645 * np.sqrt( dg.T @ V @ dg ) # TODO Quantile
        errorInt.append(h)
    return errorInt

# PDF functions
def y_sample(eta, x, theta): 
    y = lambda var: eta(var, theta) + np.random.normal(0, theta[0])
    return eval(y, x)

def pdf(y, eta, x, theta):
    pdf = lambda var: np.exp( -(var - eta(x, theta))**2 / (2 * theta[0]**2) ) / (theta[0] * np.sqrt(2 * np.pi))
    return eval(pdf, y)

# def cdf(y, eta, x, theta): # TODO

# y = np.linspace(-40, 40, num=100)
# test = lambda y: pdf(y, eta, xs[0], [10.17, 153.79, 17.26, -2.58, 0.455, 0.01746])
# test_s = [ y_sample(eta, xs[0], [10.17, 153.79, 17.26, -2.58, 0.455, 0.01746]) for _ in range(1000) ]
# plt.plot(y, test(y))
# plt.hist(test_s, density=True)
# plt.show()

# Likelyhood functions
def Lik(theta, eta, x, y):
    return np.prod( [pdf(y[i], eta, x[i], theta) for i in range(len(x))] )
    # return np.prod( [pdf(y[i], x, theta) for i in range(len(y))] )

def L(theta, eta, x, y):
    return np.log( Lik(theta, eta, x, y) )

def L_sum(theta, eta, x, y):
    return np.sum( np.log( [pdf(y[i], eta, x[i], theta) for i in range(len(x))] ) )
    # return np.sum( [np.log(pdf(y[i], x, theta)) for i in range(len(y))] ) 

def maximum(f, x0=0, h=0.001, max=100):
    count = 0
    df = derivative(f, x0, 1, h)
    while count<max and np.abs(df)>h:
        df = derivative(f, x0, 1, h)
        ddf = derivative(f, x0, 2, h)
        x0 = x0 - df/ddf
    return x0

# MLE estimation
logLikelyhood = lambda theta: L(theta, eta, x_i, y_i)
# TODO theta=[153.79, 17.26, -2.58, 0.455, 0.01746], sigma=3.189, sigma^2=10.17 
theta_0=[10.17, 153.79, 17.26, -2.58, 0.455, 0.01746]
theta = theta_0# []
# for i in range(len(theta_0)):
#     llh_i = lambda var: logLikelyhood( replace(theta_0, i, var) ) 
#     theta.append(maximum(llh_i, theta_0[i]))
V = np.linalg.inv( -Hessian(logLikelyhood, theta) ) # TODO inverting algorithm

# y = np.linspace(9, 13, num=100)
# test = lambda i: logLikelyhood([i, 153.79, 17.26, -2.58, 0.455, 0.01746])# [10.17, 153.79, 17.26, -2.58, 0.455, 0.01746]
# plt.plot(y, [ test(y[i]) for i in range(len(y)) ])
# # plt.plot(y, [ derivative(test, y[i]) for i in range(len(y)) ])
# plt.show()
# print(Derivative(logLikelyhood, [10.17, 153.79, 17.26, -2.58, 0.455, 0.01746]))

h_ci = h_ci(x, 0.1, eta, theta, V)
h_cb = h_cb(x, 0.1, eta, theta, V)
plt.plot(xs, data, 'o', color='black', markersize='3')
plt.plot(x, eta(x, theta))
plt.plot(x, [eta(x[i], theta) - h_ci[i] for i in range(len(x))], linestyle='dashed')
plt.plot(x, [eta(x[i], theta) + h_ci[i] for i in range(len(x))], linestyle='dashed')
# plt.plot(x, [eta(x[i], theta) - h_cb[i] for i in range(len(x))], linestyle='dashed')
# plt.plot(x, [eta(x[i], theta) + h_cb[i] for i in range(len(x))], linestyle='dashed')
plt.show()