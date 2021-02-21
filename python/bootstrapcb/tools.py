import math
import sys
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

def generate_y_nonparam(x, y, pdf, opt, B=100, n=50):
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
    Theta = []
    for i in range(B):
        x_sample = []
        y_sample = []
        for _ in range(n): 
            r = np.random.uniform(0, 1)
            index = np.random.choice(range(len(x)))
            x_sample.append(x[index])
            y_sample.append(_generate_y(y[index],r))
        theta_sample = mle(x_sample, y_sample, pdf, opt)
        X.append(x_sample)
        Y.append(y_sample)
        Theta.append(theta_sample)
        print("Generate bootstrap samples: {0}%".format( int(100*(i+1)/B) ), end="\r")
    return X, Y, Theta

def generate_y_param(x, y, pdf, opt, B=100, n=50):  
    def _cdf(x, y, theta):
        try:
            value, error = quad(lambda var: pdf(x,var,theta), -np.inf, y)
            return value
        except: 
            print("pdf integration failed")
            return 0
    def _generate_y(x,r,y0):
        value = root(lambda var: (_cdf(x,var,opt)-r), y0) 
        return value.x[0]
    X = []
    Y = []
    Theta = []
    for i in range(B):
        x_sample = []
        y_sample = []
        for _ in range(n): 
            r = np.random.uniform(0, 1)
            index = np.random.choice(range(len(x)))
            y0 = np.sum(y[index]) / len(y[index])
            x_sample.append(x[index])
            y_sample.append(_generate_y(x[index],r,y0))
        theta_sample = mle(x_sample, y_sample, pdf, opt)
        X.append(x_sample)
        Y.append(y_sample)
        Theta.append(theta_sample)
        print("Generate bootstrap samples: {0}%".format( int(100*(i+1)/B) ), end="\r")
    return X, Y, Theta

def conf_band_delta(x, q, f, opt, V):
    mean, confBand_L, confBand_U = [None]*len(x), [None]*len(x), [None]*len(x)
    for i in range(len(x)):
        dg = nd.Gradient(lambda var: f(x[i], var))
        h = np.sqrt( 8.4 * dg(opt).T @ V @ dg(opt) ) # TODO Quantile
        mean[i] = f(x[i], opt)
        confBand_L[i] = mean[i] - h
        confBand_U[i] = mean[i] + h
    return confBand_L, confBand_U

def conf_band_bs_ralpha(x, q, f, opt, I, theta_samples):
    B = len(theta_samples)
    # jth GoF test statistic
    C = []
    for theta in theta_samples:
        c_sample = np.subtract(opt, theta).T @ I @ np.subtract(opt, theta)
        C.append(c_sample)
    # order test values
    for i in range(B):
        for j in range(i+1,B):
            if C[j] > C[i]: 
                tmpC = C[i] 
                C[i] = C[j]
                C[j] = tmpC
                tmpTheta = theta_samples[i] 
                theta_samples[i] = theta_samples[j]
                theta_samples[j] = tmpTheta
    # rounded integer subscript
    c_critval = np.percentile(C, q) 
    R_alpha = [theta_samples[i] for i in range(B) if C[i] < c_critval]
    confBand_L, confBand_U = [None]*len(x), [None]*len(x)
    for theta in R_alpha:
        for i in range(len(x)):
            fx = f(x[i], theta)
            if confBand_L[i] is None or fx < confBand_L[i]: 
                confBand_L[i] = fx
            if confBand_U[i] is None or fx > confBand_U[i]: 
                confBand_U[i] = fx
    return confBand_L, confBand_U

def conf_band_bs_dralpha(x, q, f, opt, V, I, theta_samples):
    p = len(opt)
    B = len(theta_samples)
    L = np.linalg.cholesky(V)
    # jth GoF test statistic
    C = []
    for theta in theta_samples:
        c_sample = np.subtract(opt, theta).T @ I @ np.subtract(opt, theta)
        C.append(c_sample)
    # order test values
    for i in range(B):
        for j in range(i+1,B):
            if C[j] > C[i]: 
                tmpC = C[i] 
                C[i] = C[j]
                C[j] = tmpC
                tmpTheta = theta_samples[i] 
                theta_samples[i] = theta_samples[j]
                theta_samples[j] = tmpTheta
    # rounded integer subscript
    c_critval = np.percentile(C, q) 
    dR_alpha = []
    for j in range(1000):
        z = np.random.normal(0, 1, p)
        z_sum = np.sum([z[j]**2 for j in range(p)])
        z_sum = np.sqrt(z_sum)
        Z = [np.sqrt(c_critval) * z[i] / z_sum for i in range(p)]
        W = L @ Z
        dR_alpha.append(np.add(W, opt))
    confBand_L, confBand_U = [None]*len(x), [None]*len(x)
    for theta in dR_alpha:
        for i in range(len(x)):
            fx = f(x[i], theta)
            if confBand_L[i] is None or fx < confBand_L[i]: 
                confBand_L[i] = fx
            if confBand_U[i] is None or fx > confBand_U[i]: 
                confBand_U[i] = fx
    return confBand_L, confBand_U

def conf_band_approx_dralpha(x, q, f, opt, V, I, theta_samples):
    p = len(opt)
    B = len(theta_samples)
    L = np.linalg.cholesky(V)
    # jth GoF test statistic
    C = []
    for theta in theta_samples:
        c_sample = np.subtract(opt, theta).T @ I @ np.subtract(opt, theta)
        C.append(c_sample)
    # order test values
    for i in range(B):
        for j in range(i+1,B):
            if C[j] > C[i]: 
                tmpC = C[i] 
                C[i] = C[j]
                C[j] = tmpC
                tmpTheta = theta_samples[i] 
                theta_samples[i] = theta_samples[j]
                theta_samples[j] = tmpTheta
    # rounded integer subscript
    c_critval = np.percentile(C, q) 
    def _restircted_search(z,x):
        z_sum = np.sum([z[j]**2 for j in range(p)])
        z_sum = np.sqrt(z_sum)
        Z = [np.sqrt(c_critval) * z[i] / z_sum for i in range(p)]
        W = L @ Z
        theta = np.add(W, opt)
        return f(x, theta)
    confBand_L, confBand_U = [None]*len(x), [None]*len(x)
    for i in range(len(x)):
        min = minimize(lambda var: _restircted_search(var,x[i]), [0]*len(opt), method='Nelder-Mead').x
        confBand_L[i] = _restircted_search(min,x[i])
        max = minimize(lambda var: -_restircted_search(var,x[i]), [0]*len(opt), method='Nelder-Mead').x
        confBand_U[i] = _restircted_search(max,x[i])
    return confBand_L, confBand_U