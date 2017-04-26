import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xarray as xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
from indices import onset_SJKE, summarize_indices, plot_index_years

# ----------------------------------------------------------------------
# Compute SJ indices (Boos and Emmanuel 2009)
datadir = atm.homedir() + 'datastore/merra/daily/'
years = np.arange(1979, 2015)
filestr = 'merra_uv850_40E-120E_60S-60N_'
datafiles = [datadir + filestr + '%d.nc' % y for y in years]

# Read daily data from each year
ds = atm.combine_daily_years(['U', 'V'], datafiles, years)

# Remove extra dimension from data
u = atm.squeeze(ds['U'])
v = atm.squeeze(ds['V'])

# Calculate OCI index
sjke = onset_SJKE(u, v)

# Summary plot and timeseries in individual years
summarize_indices(years, sjke['onset'])
plot_index_years(sjke, suptitle='SJ', yearnm='Year', daynm='Day')
