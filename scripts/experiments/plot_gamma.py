from __future__ import division, print_function
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

def plot_gamma(alpha, beta):
    # Numpy calls k the "shape"
    k = alpha 

    # Numpy calls theta the "scale"
    theta = 1 / beta
    
    x = np.linspace(0, 0.2, 100000)
    y = stats.gamma.pdf(x, k, scale=theta)

    label = ('$k={:.1f}$, $\\theta={:.1f}$, $\\alpha={:.1f}$, $\\beta={:.1f}$'
             .format(k, theta, alpha, beta))

    plt.plot(x, y, label=label)

# plot_gamma(1,0.5)
# plot_gamma(2,0.5)
# plot_gamma(3,0.5)
# plot_gamma(5,1)
plot_gamma(1,1)
plot_gamma(0.5,1)
plot_gamma(0.1,1)
plot_gamma(0.1,0.5)
plot_gamma(0.1,0.1)

plt.legend()
plt.show()
