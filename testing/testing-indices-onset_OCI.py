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
# Compute OCI indices (Wang et al 2009)
datadir = atm.homedir() + 'datastore/merra/daily/'
years = np.arange(1979, 2015)
filestr = 'merra_u850_40E-120E_60S-60N_'
datafile = datadir + filestr + 'apr-sep_%d-%d.nc'% (years.min(), years.max())

# Read years from individual files and save to datafile
combine_years = False
if combine_years:
    datafiles = [datadir + filestr + '%d.nc' % y for y in years]

    # Day range to extract (inclusive)
    # Apr 1 - Oct 1 non-leap year | Mar 31 - Sep 30 leap year
    daymin = 91
    daymax = 274

    # Read daily data from each year
    u = atm.combine_daily_years('U', datafiles, years)
    u = atm.squeeze(atm.subset(u, {'Day' : (daymin, daymax)}))

    # Save to file
    print('Saving to ' + datafile)
    atm.save_nc(datafile, u)

# Read combined years of daily data from file
print('Loading ' + datafile)
with xray.open_dataset(datafile) as ds:
    u = ds['U'].load()

# Calculate OCI index
oci = onset_OCI(u)

# Summary plot and timeseries in individual years
summarize_indices(years, oci['onset'])
plot_index_years(oci, suptitle='OCI', yearnm='Year', daynm='Day')
