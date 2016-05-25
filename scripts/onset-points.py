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

version = 'merra2'
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
datafile = datadir + version + '_index_pts_CHP_CMAP_1980-2014.nc'
indfile = datadir + version + '_index_CHP_MFC_1980-2015.nc'
titlestr = 'CMAP 1980-2014'

# Smoothing parameters
#nroll_x, nroll_y = 3, 3
nroll_x, nroll_y = None, None

with xray.open_dataset(indfile) as index:
    index.load()
with xray.open_dataset(datafile) as data:
    data.load()

# Overlapping years
years = atm.get_coord(data, 'year')
index = index.sel(year=years)

# Smooth data
for nm in data.data_vars:
    if nroll_x is not None:
        data[nm] = atm.rolling_mean(data[nm], nroll_x, axis=-1, center=True)
    if nroll_y is not None:
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
    atm.contourf_latlon(varstd, clev=clev_std, cmap='Blues', symmetric=False,
                        extend='max')
    atm.contour_latlon(varbar, clev=clev_bar, colors='k', linewidths=2)

# Plot regression
def plot_reg(var, mask, clev=0.2, xsample=1, ysample=1):    
    xname = atm.get_coord(mask, 'lon', 'name')
    yname = atm.get_coord(mask, 'lat', 'name')
    atm.contourf_latlon(var, clev=clev)
    atm.stipple_pts(mask, xname=xname, yname=yname, xsample=xsample, 
                    ysample=ysample)

nm = 'onset'
plt.figure(figsize=(8, 11))
plt.subplot(2, 1, 1)
plot_clim(databar[nm], datastd[nm])                    
plt.subplot(2, 1, 2)
plot_reg(reg[nm]['m'], pts_mask[nm])
  
