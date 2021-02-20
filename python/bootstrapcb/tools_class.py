import math
import sys
import os
import numpy as np
import pandas as pd
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import root
from scipy.integrate import quad

class cb_class:
    # Data
    _xdata = None
    _ydata = None
    xs = None
    ys = None
    # Model
    _f = None
    _p0 = None
    _pdf = None
    _cdf = None
    # MLE
    _opt = None
    _I = None
    _V = None
    # BS
    _parametric = True
    _x_samples = None
    _y_samples = None
    _theta_samples = None

    def __init__(self, xdata, ydata, f, p0=None, pdf=None, cdf=None):
        # get data
        self._xdata = xdata
        self._ydata = ydata
        self.xs, self.ys = [], []
        for i in range(len(xdata)):
            for j in range(len(ydata[i])):
                self.xs.append(xdata[i])
                self.ys.append(ydata[i][j])
        # get parameter guess
        if not p0:
            p0 = [0]*self.listarg_len(self._f)
        # get normal distribution if neither is given
        if not pdf and not cdf:
            def pdf(x,y,theta): 
                return np.exp(-(y-self._f(x,theta))**2 / (2*theta[0]**2)) / np.sqrt(2*np.pi*theta[0]**2)
            def cdf(x,y,theta): 
                return (1 + math.erf((y-self._f(x,theta)) / np.sqrt(2*theta[0]**2))) / 2
        # get pdf if cdf is given
        elif cdf and not pdf:
            def pdf(x,y,theta): 
                return nd.Gradient(lambda var: self._cdf(x,var,theta))(y)
        # get cdf if pdf is given
        elif not cdf and pdf:
            def cdf(x,y,theta): 
                return quad(lambda var: self._pdf(x,var,theta), -np.inf, y)
        self._f = f
        self._p0 = p0
        self._pdf = pdf
        self._cdf = cdf

    def _generate_y_nonparam(self,i,r):
        sortedY = self._ydata[i].copy()
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

    def _generate_y_param(self,i,r):
        try:
            y0 = self._f(self._xdata[i],self._opt)
            value = root(lambda var: (self._cdf(self._xdata[i],var,self._opt)-r), y0) 
        except:
            print("Couldnt find root")
            exit(1)
        return value.x[0]

    # Likelihood
    def _L(self,theta,x,y):
        return np.sum( [np.log(self._pdf(x[i],y[i],theta)) for i in range(len(x))] )

    def listarg_len(function):
        count = 0
        while True:
            try:
                function([0]*count)
            except:
                continue
            else:
                return count

    def get_bs_samples(self, B=100, n=50):
        try: 
            print("Importing samples")
            file_path = os.getcwd()+"/dict.tmp"
            file = open(file_path)
            dict = file.read()
            dict = eval(dict) 
            # Check
            assert dict['parametric'] == self._parametric 
            assert len(dict['y_samples']) == B
            assert len(dict['y_samples'][0]) == n
            assert len(dict['theta_samples'][0]) == len(self._p0)
            # Import
            self._x_samples = dict['x_samples']
            self._y_samples = dict['y_samples']
            self._theta_samples = dict['theta_samples']  
            file.close()
            print("Successfully imported samples")
        except:
            print("Couldn't import samples, now generating...")
            # Generate
            X, Y, Theta = [], [], []
            for i in range(B):
                x_sample = []
                y_sample = []
                for _ in range(n): 
                    r = np.random.uniform(0, 1)
                    index = np.random.choice(range(len(self._xdata)))
                    x_sample.append(self._xdata[index])
                    if self._parametric:
                        y_sample.append(self._generate_y_param(index,r))
                    else :
                        y_sample.append(self._generate_y_nonparam(index,r))
                theta_sample = minimize(lambda var: -self._L(var,x_sample,y_sample), self._p0, method='Nelder-Mead').x.tolist()
                X.append(x_sample)
                Y.append(y_sample)
                Theta.append(theta_sample)
                print("Generate bootstrap samples: {0}%".format( int(100*(i+1)/B) ), end="\r")
            self._x_samples, self._y_samples, self._theta_samples = X, Y, Theta
            dict = {
                'parametric': self._parametric,
                'x_samples': self._x_samples,
                'y_samples': self._y_samples,
                'theta_samples': self._theta_samples
            }
            file_path = os.getcwd()+"/dict.tmp"
            file = open(file_path,"w")
            file.write( str(dict) ) 
            file.close() 
            print("\n")
            print("Successfully imported samples")
        return self._x_samples, self._y_samples, self._theta_samples

    def mle(self, x):
        try:
            self._opt = minimize(lambda var: -self._L(var,self.xs,self.ys), self._p0, method='Nelder-Mead').x.tolist()
            self._I = ( -nd.Hessian(lambda var: self._L(var,self.xs,self.ys))(self._opt) ).tolist()
            self._V = np.linalg.inv(self._I).tolist()
        except:
            print("MLE failed")
            exit(1)
        return [self._f(x_i, self._opt) for x_i in x], self._V

    def conf_band_delta(self, x, q):
        # get MLE
        if not self._opt:
            self.mle()
        # get BS samples
        if not self._theta_samples:
            self.get_bs_samples()
        confBand_L, confBand_U = [None]*len(x), [None]*len(x)
        for i in range(len(x)):
            dg = nd.Gradient(lambda var: self._f(x[i], var))
            h = np.sqrt( 8.4 * dg(self._opt).T @ self._V @ dg(self._opt) ) # TODO Quantile
            fx = self._f(x[i], self._opt)
            confBand_L[i] = fx - h
            confBand_U[i] = fx + h
        return confBand_L, confBand_U

    def conf_band_bs_ralpha(self, x, q):
        # get MLE
        if not self._opt:
            self.mle()
        p = len(self._opt)
        # get BS samples
        if not self._theta_samples:
            self.get_bs_samples()
        B = len(self._theta_samples)
        # jth GoF test statistic
        C = []
        theta_samples = self._theta_samples.copy()
        for theta in theta_samples:
            c_sample = np.subtract(self._opt, theta).T @ self._I @ np.subtract(self._opt, theta)
            C.append(c_sample)
        # order test values
        for i in range(B):
            for j in range(i+1,B):
                if C[j] > C[i]: 
                    tmpC = C[i] 
                    C[i] = C[j]
                    C[j] = tmpC
                    tmpTheta = self._theta_samples[i] 
                    theta_samples[i] = theta_samples[j]
                    theta_samples[j] = tmpTheta
        # rounded integer subscript
        c_critval = np.percentile(C, q) 
        R_alpha = [theta_samples[i] for i in range(B) if C[i] < c_critval]
        confBand_L, confBand_U = [None]*len(x), [None]*len(x)
        for theta in R_alpha:
            for i in range(len(x)):
                fx = self._f(x[i], theta)
                if confBand_L[i] is None or fx < confBand_L[i]: 
                    confBand_L[i] = fx
                if confBand_U[i] is None or fx > confBand_U[i]: 
                    confBand_U[i] = fx
        return confBand_L, confBand_U

    def conf_band_bs_dralpha(self, x, q):
        # get MLE
        if not self._opt:
            self.mle()
        p = len(self._opt)
        # get BS samples
        if not self._theta_samples:
            self.get_bs_samples()
        B = len(self._theta_samples)
        # jth GoF test statistic
        L = np.linalg.cholesky(self._V)
        C = []
        theta_samples = self._theta_samples.copy()
        for theta in theta_samples:
            c_sample = np.subtract(self._opt, theta).T @ self._I @ np.subtract(self._opt, theta)
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
            dR_alpha.append(np.add(W, self._opt))
        confBand_L, confBand_U = [None]*len(x), [None]*len(x)
        for theta in dR_alpha:
            for i in range(len(x)):
                fx = self._f(x[i], theta)
                if confBand_L[i] is None or fx < confBand_L[i]: 
                    confBand_L[i] = fx
                if confBand_U[i] is None or fx > confBand_U[i]: 
                    confBand_U[i] = fx
        return confBand_L, confBand_U

    def conf_band_approx_dralpha(self, x, q):
        # get MLE
        if not self._opt:
            self.mle()
        p = len(self._opt)
        # get BS samples
        if not self._theta_samples:
            self.get_bs_samples()
        B = len(self._theta_samples)
        # jth GoF test statistic
        L = np.linalg.cholesky(self._V)
        C = []
        theta_samples = self._theta_samples.copy()
        for theta in theta_samples:
            c_sample = np.subtract(self._opt, theta).T @ self._I @ np.subtract(self._opt, theta)
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
            theta = np.add(W, self._opt)
            return self._f(x, theta)
        confBand_L, confBand_U = [None]*len(x), [None]*len(x)
        for i in range(len(x)):
            min = minimize(lambda var: _restircted_search(var,x[i]), [0]*p, method='Nelder-Mead').x.tolist()
            confBand_L[i] = _restircted_search(min,x[i])
            max = minimize(lambda var: -_restircted_search(var,x[i]), [0]*p, method='Nelder-Mead').x.tolist()
            confBand_U[i] = _restircted_search(max,x[i])
        return confBand_L, confBand_U