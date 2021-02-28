import numpy as np
import matplotlib.pyplot as plt

def logistic(r, x):
    return r * x * (1 - x)

n = 10000
r = np.linspace(2.5, 4, n)
x = 1e-5 * np.ones(n)

iterations = 10000
last = 1000

fig, ax1 = plt.subplots(1, 1)

for i in range(iterations):
    x = logistic(r, x)
    # We display the bifurcation diagram.
    if i >= (iterations - last):
        ax1.plot(r, x, ',k', alpha=.25)
        
ax1.set_xlim(2.5, 4)
ax1.set_title("Bifurcation diagram")
plt.show()

fig.savefig("foo.png")