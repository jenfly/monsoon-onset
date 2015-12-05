import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra

# ----------------------------------------------------------------------
ensodir = atm.homedir() + 'dynamics/calc/ENSO/'

enso = {}
enso['MEI'] = pd.read_csv(ensodir + 'enso_mei.csv', index_col=0)
enso['ONI'] = pd.read_csv(ensodir + 'enso_oni.csv', index_col=0)

#season = {'MEI' : 'DECJAN', 'ONI' : 'DJF'}
season = {'MEI' : 'JULAUG', 'ONI' : 'JJA'}
#season = {'MEI' : 'NOVDEC', 'ONI' : 'NDJ'}
threshold = {'MEI' : 1, 'ONI' : 0.5}

xmin, xmax = 1979, 2015
xticks = range(1978, xmax, 2)

plt.figure(figsize=(12, 10))
nrows = len(enso.keys())
for i, key in enumerate(enso):
    data = enso[key][season[key]]
    plt.subplot(nrows, 1, i + 1)
    plt.bar(data.index, data.values, color='k', alpha=0.3)
    iwarm = data.values > threshold[key]
    icold = data.values < -threshold[key]
    plt.bar(data[iwarm].index, data[iwarm].values, color='r')
    plt.bar(data[icold].index, data[icold].values, color='b')
    plt.xticks(xticks)
    plt.xlim(xmin, xmax)
    plt.grid()
    plt.title('ENSO ' + key + ' ' + season[key])


data = enso['ONI']['JJA'].copy().loc[1979:2014]
data.sort(ascending=False)
nyrs = 5
print('El Nino Top %d' % nyrs)
print(data[:nyrs])
print('La Nina Top %d' % nyrs)
print(data[-1:-nyrs-1:-1])
