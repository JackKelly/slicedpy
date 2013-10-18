from __future__ import print_function, division
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
import numpy as np
from os import path
from pda.channel import Channel

"""
Documentation for changepoint R package:
http://cran.r-project.org/web/packages/changepoint/changepoint.pdf

"""
cpt = importr('changepoint')

# data = FloatVector(np.concatenate([np.random.normal( 0,1,100),
#                                    np.random.normal( 5,1,100),
#                                    np.random.normal( 0,1,100),
#                                    np.random.normal(10,1,100)]))

DATA_DIR = '/data/mine/domesticPowerData/BellendenRd/wattsUp'
SIG_DATA_FILENAME = 'breadmaker1.csv'
#SIG_DATA_FILENAME = 'washingmachine1.csv'
#SIG_DATA_FILENAME = 'kettle1.csv'

chan = Channel()
chan.load_wattsup(path.join(DATA_DIR, SIG_DATA_FILENAME))
data = chan.series.values# [142:1647]# [:1353][:153]
data = FloatVector(data)

changepoints = cpt.PELT_mean_norm(data, pen=100000*log(len(data)))

plot(data)

for point in changepoints:
    plot([point, point], [0, 3000], color='k')

scatter(list(changepoints), [0]*len(changepoints))

print(changepoints)
