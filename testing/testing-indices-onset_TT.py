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
from indices import (onset_TT, summarize_indices, plot_index_years,
                     plot_tseries_together)

# ======================================================================#
# ************************* NOTES *************************************
# Need to troubleshoot several things before using this script for
# anything final
# - Weirdness with values of vertical mean T200-600 in lat-lon maps
#   and compared with the figures in Goswami et al 2006 (same shape
#   but magnitude off by several degrees).  Look at maps of T400,
#   T200-600 on individual days and means, try integrating over
#   different pressure ranges
#  - Weirdness with 1991 data - one day with values > 1e35
#  - Compare TT index from 200-600mb vertical mean vs. calculated
#    on individual pressure levels.
# ======================================================================

# ----------------------------------------------------------------------
# Compute TT indices (Goswami et al 2006)
datadir = atm.homedir() + 'datastore/merra/daily/'
years = np.arange(1979, 2015)
months = [4, 5, 6, 7, 8, 9]
filestr = 'merra_T200-600_'
datafiles = [datadir + filestr + 'apr-sep_%d.nc'% year for year in years]

combine_months = False
save = True

# Select vertical pressure level to use, or None to use 200-600mb
# vertical mean
# plev = 400
plev = None

# Read months from individual files and save to datafiles
if combine_months:
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
        savefile = datafiles[y]
        print('Saving to ' + savefile)
        dsyr.to_netcdf(savefile)


# Read daily data from each year
plist = [200, 400, 600]
if plev is not None:
    plist = np.union1d(plist, [plev])
T_p = {}
for p in plist:
    T1 = atm.combine_daily_years('T', datafiles, years, yearname='year',
                              subset1=('plev', p, p))
    T_p[p] = atm.squeeze(T1)
Tbar = atm.combine_daily_years('Tbar', datafiles, years, yearname='year')

if plev is None:
    T = Tbar
    varname = 'TT200-600'
else:
    T = T_p[plev]
    varname = 'TT%d' % plev

# Calculate TT index
# The north region should go up to 35 N but there is some weirdness
# with the topography so I'm setting it to 30 N for now
north=(5, 30, 40, 100)
south=(-15, 5, 40, 100)
suptitle = varname + ' N=%s S=%s' % (str(north), str(south))
tt = onset_TT(T, north=north, south=south)

# Some weirdness going on in 1991, for now just set to NaN
# Troubleshoot later
for nm in ['ttn', 'tts', 'tseries']:
    vals = tt[nm].values
    vals = np.ma.masked_array(vals, abs(vals) > 1e30).filled(np.nan)
    tt[nm].values = vals

# Plot TTN and TTS
plot_tseries_together(tt[['ttn', 'tts']], tt['onset'].values,
                      suptitle=suptitle, standardize=False)

# Summary plot and timeseries in individual years
summarize_indices(years, tt['onset'])
plt.suptitle(suptitle)
plot_index_years(tt, suptitle=suptitle, yearnm='year', daynm='day')

if save:
    atm.savefigs(varname + '_', 'png')

# Plot contour map of pressure-level data
p_plot = 400
T_plot = T_p[p_plot]
y, d = 0, 80
lat = atm.get_coord(T_plot, 'lat')
lon = atm.get_coord(T_plot, 'lon')
axlims = (lat.min(), lat.max(), lon.min(), lon.max())
plt.figure()
atm.pcolor_latlon(T_plot[y, d], axlims=axlims)
