import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
import indices
import utils

# ----------------------------------------------------------------------
datadir = atm.homedir() + 'datastore/merra/daily/'
filename1 = datadir + 'merra_uv200_40E-120E_60S-60N_1979.nc'
filename2 = datadir + 'merra_DUDP200_40E-120E_90S-90N_1979.nc'
data = xray.Dataset()
with xray.open_dataset(filename1) as ds:
    data['U'] = atm.squeeze(ds['U'])
    data['V'] = atm.squeeze(ds['V'])
with xray.open_dataset(filename2) as ds:
    data['DUDP'] = atm.squeeze(ds['DUDP'])
data['OMEGA'] = np.nan * data['U']
data['DOMEGADP'] = np.nan * data['U']
data = data.rename({'Day' : 'day'})


nt = 5      # Rolling pentad
lon1, lon2 = 60, 100
taxis = 0

for nm in data.data_vars:
    print('Eddy decomposition for ' + nm)
    comp = utils.eddy_decomp(data[nm], nt, lon1, lon2, taxis)
    for compnm in comp:
        data[compnm] = comp[compnm]
    
    

