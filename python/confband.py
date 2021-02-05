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

# Model functions
def eta(x, theta=[153.79, 17.26, -2.58, 0.455, 0.01746]): 
    f = lambda var: (theta[0] + theta[2]*var + theta[4]*var**2) * np.exp(theta[3]*(var-theta[1]))/( 1 + np.exp(theta[3]*(var-theta[1])) )
    return eval(f, x)

def y(x, theta=[153.79, 17.26, -2.58, 0.455, 0.01746], sigma=10.17): 
    f = lambda var: eta(var, theta) + np.random.normal(0, sigma**2)
    return eval(f, x)

def pdf(y, x, theta=[153.79, 17.26, -2.58, 0.455, 0.01746], sigma=10.17):
    f = lambda var: np.exp( -(var - eta(x, theta))**2 / (2*sigma**2) ) / np.sqrt(2*np.pi*sigma**2)
    return eval(f, y)

# def g(theta, x):
#     return eta(x, theta)

def Lik(theta, x, y):
    return np.prod( pdf(y, x, theta) )
    # return np.prod( [pdf(y[i], x, theta) for i in range(len(y))] )

def L(theta, x, y):
    return np.log( Lik(theta, x, y) )

def L_sum(theta, x, y):
    return np.sum( np.log( pdf(y, x, theta) ) )
    # return np.sum( [np.log(pdf(y[i], x, theta)) for i in range(len(y))] ) 

def h(x, data, theta=[153.79, 17.26, -2.58, 0.455, 0.01746], sigma=10.17):
    errorInt = []
    for i in range(len(x)):
        logLikelyhood = lambda theta: L(theta, x[i], data[i])
        # TODO max logLikelyhood
        V = np.linalg.inv( -Hessian(logLikelyhood, theta) )
        g = lambda theta: eta(x[i], theta)
        dg = Derivative(g, theta)
        h = (2*sigma) * np.sqrt( dg.T @ V @ dg )
        errorInt.append(h)
    return errorInt
         
h = h(xs, data)
print(h)
plt.plot(xs, data, 'o', color='black', markersize='3')
plt.plot(x, eta(x))
plt.plot(xs, [eta(xs[i]) - h[i] for i in range(len(xs))], linestyle='dashed')
plt.plot(xs, [eta(xs[i]) + h[i] for i in range(len(xs))], linestyle='dashed')
plt.show()