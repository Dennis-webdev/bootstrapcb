import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/dennis/git/bootstrapcb/python')
import bootstrapcb as cb

confidence_level = 90
xdata = [0]*47 #df["lambda"].values
ydata = [  4.3, 4.7, 4.7, 3.1, 5.2, 6.7, 4.5, 3.6, 7.2, 10.9,  6.6, 5.8,
           6.3, 4.7, 8.2, 6.2, 4.2, 4.1, 3.3, 4.6, 6.3,  4.0,  3.1, 3.5,
           7.8, 5.0, 5.7, 5.8, 6.4, 5.2, 8.0, 4.9, 6.1,  8.0,  7.7, 4.3,
          12.5, 7.9, 3.9, 4.0, 4.4, 6.7, 3.8, 6.4, 7.2,  4.8, 10.5       ]
guess = [5.9,2]
def f(x, theta):
    mu = theta[0]
    sigma = theta[1]
    a = (mu / sigma)**2
    b = sigma**2 / mu
    return (1+a) * a * b**2 * x / (2 * (1 - a*b*x))
def pdf(x, y, theta): 
    mu = theta[0]   
    sigma = theta[1]
    a = (mu / sigma)**2 
    b = sigma**2 / mu 
    try:
        return y**(a-1) * np.exp(-y/b) / (math.gamma(a) * b**a)
    except: return 0
x = np.linspace(0,0.1,num=100)
opt, V, I = cb.mle(xdata, ydata, pdf, guess, info=True)
mean = [f(x_i, opt) for x_i in x] 
df = pd.DataFrame(data={ 
    "x": x, 
    "mean": mean  
}) 
df = df.set_index(["x"])  
x_samples, y_samples, theta_samples = cb.generate_y_param(xdata, ydata, pdf, opt, 100, 50)
conf_l, conf_u = cb.conf_band_delta(x, confidence_level, f, opt, V)
df["cb_l"] = conf_l 
df["cb_u"] = conf_u    
conf_l, conf_u = cb.conf_band_approx_dralpha(x, confidence_level, f, opt, V, I, theta_samples)
df["bscb_l"] = conf_l
df["bscb_u"] = conf_u
conf_l, conf_u = cb.conf_band_bs_dralpha(x, confidence_level, f, opt, V, I, theta_samples)
df["bscb*_l"] = conf_l
df["bscb*_u"] = conf_u
