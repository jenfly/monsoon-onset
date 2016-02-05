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

years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/daily/'
files = {}
files['MFC'] = [datadir + 'merra_MFC_ps-300mb_%d.nc' % yr for yr in years]
files['PCP'] = [datadir + 'merra_precip_%d.nc' % yr for yr in years]
evapstr = 'merra_EVAP_40E-120E_90S-90N_%d.nc'
files['EVAP'] = [datadir + evapstr % yr for yr in years]

# Lat-lon box for MFC budget
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
latlonstr = '%d-%dE, %d-%dN' % (lon1, lon2, lat1, lat2)

# ----------------------------------------------------------------------
# Read data

# MFC and precip over SASM region
nroll = 7
ts = utils.get_mfc_box(files['MFC'], files['PCP'], files['EVAP'], years,
                       nroll, lat1, lat2, lon1, lon2)

# Residual MFC - (P - E)
ts['RESID_UNSM'] = ts['MFC_UNSM'] - (ts['PCP_UNSM'] - ts['EVAP_UNSM'])
ts['RESID'] = atm.rolling_mean(ts['RESID_UNSM'], nroll, axis=-1)

# ----------------------------------------------------------------------
# Plot daily tseries of MFC budget components

def plot_tseries(ts, year=None, smooth=True, ax=None):
    if smooth:
        keys = ['MFC', 'PCP', 'EVAP', 'RESID']                
    else:
        keys = ['MFC_UNSM', 'PCP_UNSM', 'EVAP_UNSM', 'RESID_UNSM']
    tseries = ts[keys]
    if year is None:
        tseries = tseries.mean(dim='year')
        title = 'Climatological Mean'
    else:
        tseries = tseries.sel(year=year).drop('year')
        title = year
    tseries = tseries.to_dataframe()
    tseries.plot(ax=ax, grid=True, legend=False)
    ax.set_title(title, loc='left', fontsize=11)
    
plt.figure(figsize=(12, 9))
suptitle = 'MFC Budget (%s) - Daily Tseries' % latlonstr
plt.suptitle(suptitle)
nrow, ncol = 2, 2
plotyears = [None, years[0], years[1], years[2]]
for y, year in enumerate(plotyears):
    ax = plt.subplot(nrow, ncol, y + 1)
    plot_tseries(ts, year, ax=ax)
    if y == 0:
        ax.legend(loc='upper left', fontsize=9)
        atm.text('RESID = MFC - (P-E)', (0.03, 0.67), fontsize=9)
        
        
# Average values over JJAS and LRS - interannual variability

