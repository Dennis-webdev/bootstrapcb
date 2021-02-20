import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/dennis/git/bootstrapcb/python')
import bootstrapcb as cb

confidence_level = 90
xdata = [2.0, 7.0, 12.0, 19.5, 29.5, 39.5, 54.5, 75.0]
ydata = [ [ 1.26,   2.78,  0.63,  0.34], 
          [ 3.53,    4.1,  1.31,  0.91],
          [11.98,  13.14,  9.86,  6.53], 
          [90.82,  97.12, 75.85, 59.46], 
          [83.45, 116.62,  104., 80.85], 
          [55.98,  67.28, 79.33, 82.66],
          [66.32,  78.53,  69.1, 67.27], 
          [39.42,  55.35, 60.76,  73.2] ]
xs, ys = [], []
for i in range(len(xdata)):
    for j in range(len(ydata[i])):
        xs.append(xdata[i])
        ys.append(ydata[i][j])
x = np.linspace(0,80,num=100)

def f(x, theta):
    t2 = theta[1]
    t3 = theta[2]
    t4 = theta[3]
    t5 = theta[4]
    t6 = theta[5]
    return (t2 + t4*x + t6*x**2) * np.exp(t5*(x-t3))/( 1 + np.exp(t5*(x-t3)) )
p0 = [10.17, 153.79, 17.26, -2.58, 0.455, 0.01746]
pdf = lambda x,y,theta: np.exp(-(y-f(x,theta))**2 / (2*theta[0]**2)) / np.sqrt(2*np.pi*theta[0]**2)
    
opt, V, I = cb.mle(xs, ys, pdf, p0, info=True) 
mean = [f(x_i, opt) for x_i in x] 
df = pd.DataFrame(data={ 
    "x": x, 
    "mean": mean  
}) 
df = df.set_index(["x"])

x_samples, y_samples, theta_samples = cb.generate_y_param(xdata, ydata, pdf, opt, B=10, n=50)
for theta in theta_samples:
    plt.plot(x, [f(x_i, theta) for x_i in x])
plt.show()

conf_l, conf_u = cb.conf_band_bs_ralpha(x, confidence_level, f, opt, I, theta_samples)
df["cb_l"] = conf_l  
df["cb_u"] = conf_u  
