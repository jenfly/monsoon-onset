import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
from indices import onset_TT, summarize_indices, plot_index_years

# ----------------------------------------------------------------------
# Compute TT indices (Goswami et al 2006)
datadir = atm.homedir() + 'datastore/merra/daily/'
years = np.arange(1979, 2015)
months = [4, 5, 6, 7, 8, 9]
# Common set of days for leap and non-leap years
dmin, dmax = 91, 274
days = np.arange(dmin, dmax + 1)
filestr = 'merra_T200-600_'
datafile = datadir + filestr + 'apr-sep_%d-%d.nc'% (years.min(), years.max())

combine_years = False

# Read months, years from individual files and save to datafile
if combine_years:
    for y, year in enumerate(years):
        for m, mon in enumerate(months):
            filn = datadir + filestr + '%d%02d.nc' % (year, mon)
            print('Loading ' + filn)
            with xray.open_dataset(filn) as ds:
                ds.load()
                if m == 0:
                    dsyr = ds
                else:
                    dsyr = xray.concat((dsyr, ds), dim='day')
        dsyr.coords['year'] = year
        # Align leap and non-leap years
        dsyr = dsyr.reindex(day=days)
        if y == 0:
            ds_all = dsyr
        else:
            ds_all = xray.concat((ds_all, dsyr), dim='year')
    print('Saving to ' + datafile)
    ds_all.to_netcdf(datafile)

# Read combined years of daily data from file
print('Loading ' + datafile)
with xray.open_dataset(datafile) as ds:
    T = ds['Tbar'].load()

lat = atm.get_coord(T, 'lat')
lon = atm.get_coord(T, 'lon')
axlims = (lat.min(), lat.max(), lon.min(), lon.max())

plt.figure()
atm.pcolor_latlon(T[0,0], axlims=axlims)

# Calculate OCI index
tt = onset_TT(T, north=(5, 30, 40, 100), south=(-15, 5, 40, 100))

# Summary plot and timeseries in individual years
summarize_indices(years, tt['onset'])
plot_index_years(tt, suptitle='TT', yearnm='year', daynm='day')
