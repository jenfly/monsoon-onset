import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
from indices import onset_OCI, summarize_indices, plot_index_years

# ----------------------------------------------------------------------
# Compute SJ indices (Boos and Emmanuel 2009)
datadir = atm.homedir() + 'datastore/merra/daily/'
years = np.arange(1979, 2015)
filestr = 'merra_uv850_40E-120E_60S-60N_'
datafile = datadir + filestr + '%d-%d.nc'% (years.min(), years.max())

# Read years from individual files and save to datafile
combine_years = False
if combine_years:
    datafiles = [datadir + filestr + '%d.nc' % y for y in years]

    # Read daily data from each year
    ds = atm.combine_daily_years(['U', 'V'], datafiles, years)

    # Remove extra dimension from data
    u = atm.squeeze(ds['U'])
    v = atm.squeeze(ds['V'])

    # Save to file
    print('Saving to ' + datafile)
    atm.save_nc(datafile, u, v)

# Read combined years of daily data from file
print('Loading ' + datafile)
with xray.open_dataset(datafile) as ds:
    u = ds['U'].load()
    v = ds['V'].load()

# Calculate OCI index
sj = onset_SJ(u, v)

# Summary plot and timeseries in individual years
summarize_indices(years, sj['onset'])
plot_index_years(sj, suptitle='SJ', yearnm='Year', daynm='Day')
