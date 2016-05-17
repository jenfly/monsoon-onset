import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import atmos as atm
import indices

# ----------------------------------------------------------------------
# Changepoint onset at individual points

years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/analysis/'
onset_nm, pts_nm = 'CHP_MFC', 'CHP_PCP'

yearstr = '%d-%d.nc' % (min(years), max(years))
filestr = datadir + 'merra_index_%s_' + yearstr
indfile = filestr % onset_nm
datafile = filestr % ('pts_' + pts_nm)

with xray.open_dataset(indfile) as index:
    index.load()
with xray.open_dataset(datafile) as data:
    data.load()

# Smooth data
nroll_x, nroll_y = 3, 3
for nm in data.data_vars:
    data[nm] = atm.rolling_mean(data[nm], nroll_x, axis=-1, center=True)
    data[nm] = atm.rolling_mean(data[nm], nroll_y, axis=-2, center=True)  

# Climatology mean and standard deviation
databar = data.mean(dim='year')
datastd = data.std(dim='year')

# Regression of gridpoint indices onto large-scale index
reg, pts_mask = {}, {}
for nm in data.data_vars:
    reg[nm] = atm.regress_field(data[nm], index[nm], axis=0)
    pts_mask[nm] = (reg[nm]['p'] >= 0.05)
    

# Plot climatology
def plot_clim(varbar, varstd, clev_bar=10, clev_std=5):
    atm.contourf_latlon(varstd, clev=clev_std, cmap='hot_r', symmetric=False,
                        extend='max')
    atm.contour_latlon(varbar, clev=clev_bar, colors='k', linewidths=2)
    
