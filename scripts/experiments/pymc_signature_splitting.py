from __future__ import division, print_function
import pymc as mc
import numpy as np

"""
* CDP's reply to a SO question: http://stats.stackexchange.com/a/46628
* http://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions

TODO: tinker with Gammas http://en.wikipedia.org/wiki/Gamma_distribution
"""

data = np.concatenate([np.random.normal(5,1,100), np.random.normal(100,5,50)])
N = len(data)

# Priors for the means ($\mu$)
mu_1 = mc.Normal("mu1", 100, 1/100)
mu_2 = mc.Normal("mu2", 100, 1/100)

# Priors for the precision ($\tau$)
tau_1 = mc.Gamma("tau1", 0.5, 0.5)
tau_2 = mc.Gamma("tau2", 0.5, 0.5)

# Prior for the switch point
sp = mc.DiscreteUniform("switchpoint", lower=0, upper=N)

@mc.deterministic
def mu(sp=sp, mu_1=mu_1, mu_2=mu_2):
    out = np.empty(N)
    out[:sp] = mu_1
    out[sp:] = mu_2
    return out

@mc.deterministic
def tau(sp=sp, tau_1=tau_1, tau_2=tau_2):
    out = np.empty(N)
    out[:sp] = tau_1
    out[sp:] = tau_2
    return out

observation = mc.Normal("obs", mu, tau, value=data, observed=True)
#observation = mc.Normal("obs", mu, 1/100, value=data, observed=True)

model = mc.Model([observation, mu_1, mu_2, tau_1, tau_2, sp])
#model = mc.Model([observation, mu_1, mu_2, sp])

mcmc = mc.MCMC(model)
mcmc.sample(40000, 10000, 1)
