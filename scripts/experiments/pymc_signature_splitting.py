from __future__ import division, print_function
import pymc as mc
import numpy as np
from pda.channel import Channel
from scipy import stats
from os import path

"""
* CDP's reply to a SO question: http://stats.stackexchange.com/a/46628
* http://en.wikipedia.org/wiki/Conjugate_prior#Continuous_distributions

TODO: tinker with Gammas http://en.wikipedia.org/wiki/Gamma_distribution
"""

# np.random.normals are specified using a mean and standard deviation (scale)
# data = np.concatenate([np.random.normal(loc=  5, scale=1, size=100),
#                        np.random.normal(loc=100, scale=5, size= 50),
#                        np.random.normal(loc=  5, scale=1, size=100),
#                        np.random.normal(loc=100, scale=5, size= 50)])

DATA_DIR = '/data/mine/domesticPowerData/BellendenRd/wattsUp'
#SIG_DATA_FILENAME = 'breadmaker1.csv'
SIG_DATA_FILENAME = 'washingmachine1.csv'

chan = Channel()
chan.load_wattsup(path.join(DATA_DIR, SIG_DATA_FILENAME))
data = chan.series.values[138:1647]# [:1353][:153]

switchpoints = set()

config = {'min len': 60, 'mcmc iterations': 20000, 'mcmc burn in': 5000,
          'mu norm': 100, 'tau norm': 1/(100**2), 
          'alpha gamma': 0.5, 'beta gamma': 1.0}

def run_mcmc(data):

    N = len(data)

    # Priors for the means ($\mu$)
    # mc.Normals are specified using a mean and a precision tau where tau = 1/std**2
    mu_1 = mc.Normal("mu1", mu=config['mu norm'], tau=config['tau norm'])
    mu_2 = mc.Normal("mu2", mu=config['mu norm'], tau=config['tau norm'])

    # Priors for the precision ($\tau$)
    tau_1 = mc.Gamma("tau1", alpha=config['alpha gamma'], beta=config['beta gamma'])
    tau_2 = mc.Gamma("tau2", alpha=config['alpha gamma'], beta=config['beta gamma'])

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
    model = mc.Model([observation, mu_1, mu_2, tau_1, tau_2, sp])
    mcmc = mc.MCMC(model)
    mcmc.sample(config['mcmc iterations'], config['mcmc burn in'], 1)
    return mcmc

def split(data, offset=0):
    if len(data) < config['min len']:
        return
    mcmc = run_mcmc(data)
    mode = stats.mode(mcmc.trace('switchpoint')[:])[0]
    if len(mode) > 1:
        raise Exception('TODO handle case where there is more than one mode')
    else:
        mode = mode[0]

    print(" mode =", mode)
    switchpoints.add(mode+offset)
    if mode < config['min len'] or mode > len(data-config['min len']):
        return # this chunk can't be split any more
    split(data[:mode], offset=offset)
    split(data[mode:], offset=offset+mode)

def print_vars():
    print('')
    vars = ['mu1', 'mu2', 'tau1', 'tau2', 'switchpoint']
    n_vars = len(vars)
    fig = plt.figure()

    # plot data
    ax = fig.add_subplot(n_vars+1, 1, 1)
    ax.plot(data)
    ax.set_title('data')

    # plot vars
    for i, var in enumerate(vars):
        trace = mcmc.trace(var)[:]
        print('{:>5s} ={:7.2f}'.format(var, stats.mode(trace)[0][0]))
        print(i)
        ax = fig.add_subplot(n_vars+1, 1, i+2)
        ax.hist(trace, normed=True)
        ax.set_title(var)
        ax.set_yticklabels([])

    plt.subplots_adjust(hspace=0.8)
    plt.draw()

print("")
split(data)
print("switchpoints:", switchpoints)
plot(data)

for point in switchpoints:
    plot([point, point], [0, 3000], color='k')

scatter(list(switchpoints), [0]*len(switchpoints))
title(str(config) + SIG_DATA_FILENAME)
