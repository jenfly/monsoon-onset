import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
from indices import onset_HOWI, summarize_indices, plot_index_years

# ----------------------------------------------------------------------
# Compute HOWI indices (Webster and Fasullo 2003)
datadir = atm.homedir() + 'datastore/merra/daily/'
datafile = datadir + 'merra_vimt_ps-300mb_may-sep_1979-2014.nc'
lat1, lat2 = -20, 30
lon1, lon2 = 40, 100

with xray.open_dataset(datafile) as ds:
    uq_int = ds['uq_int'].load()
    vq_int = ds['vq_int'].load()

#npts = 100
npts = 50
pre_days = 'May 18-24'
post_days = 'June 8-14'
namestr = 'HOWI_%dpts_' % npts
exts = ['png', 'eps']
isave = False

howi, ds = onset_HOWI(uq_int, vq_int, npts)

# ----------------------------------------------------------------------
# MAPS
# ----------------------------------------------------------------------
# Plot climatological VIMT composites
lat = atm.get_coord(ds, 'lat')
lon = atm.get_coord(ds, 'lon')
x, y = np.meshgrid(lon, lat)
axlims = (lat1, lat2, lon1, lon2)
plt.figure(figsize=(12,10))
plt.subplot(221)
m = atm.init_latlon(lat1, lat2, lon1, lon2)
m.quiver(x, y, ds['uq_bar_pre'], ds['vq_bar_pre'])
plt.title(pre_days + ' VIMT Climatology')
plt.subplot(223)
m = atm.init_latlon(lat1, lat2, lon1, lon2)
m.quiver(x, y, ds['uq_bar_post'], ds['vq_bar_post'])
plt.title(post_days + ' VIMT Climatology')

# Plot difference between pre- and post- composites
plt.subplot(222)
m = atm.init_latlon(lat1, lat2, lon1, lon2)
#m, _ = atm.pcolor_latlon(ds['vimt_bar_diff'], axlims=axlims, cmap='hot_r')
m.quiver(x, y, ds['uq_bar_diff'], ds['vq_bar_diff'])
plt.title(post_days + ' minus ' + pre_days + ' VIMT Climatology')

# Top N difference vectors
plt.subplot(224)
m, _ = atm.pcolor_latlon(ds['vimt_bar_diff_masked'],axlims=axlims, cmap='hot_r')
plt.title('Magnitude of Top %d Difference Fluxes' % npts)

# Plot vector VIMT fluxes for a few individual years
ylist = [0, 1, 2, 3]
plt.figure(figsize=(12, 10))
for yr in ylist:
    plt.subplot(2, 2, yr + 1)
    m = atm.init_latlon(lat1, lat2, lon1, lon2)
    m.quiver(x, y, ds['uq'][yr].mean(dim='day'), ds['vq'][yr].mean(dim='day'))
    m.contour(x, y, ds['mask'].astype(float), [0.99], colors='red')
    plt.title('%d May-Sep VIMT Fluxes' % ds.year[yr])

# ----------------------------------------------------------------------
# TIMESERIES
# ----------------------------------------------------------------------

onset = howi.onset
retreat = howi.retreat
length = retreat - onset
days = howi.day
years = howi.year.values
yearstr = '%d-%d' % (years[0], years[-1])
nroll = howi.attrs['nroll']

# Timeseries with and without rolling mean
def index_tseries(days, ind, ind_roll, titlestr):
    plt.plot(days, ind, label='daily')
    plt.plot(days, ind_roll, label='%d-day rolling' % nroll)
    plt.grid()
    plt.legend(loc='lower right')
    plt.title(titlestr)

plt.figure(figsize=(12, 10))
plt.subplot(221)
index_tseries(days, ds.howi_clim_norm, ds.howi_clim_norm_roll,
              'HOWI ' + yearstr + ' Climatology')
for yr in [0, 1, 2]:
    plt.subplot(2, 2, yr + 2)
    index_tseries(days, ds.howi_norm[yr], ds.howi_norm_roll[yr],
                  'HOWI %d' % years[yr])

# HOWI index with onset and retreat in individual years
def onset_tseries(days, ind, d_onset, d_retreat):
    ylim1, ylim2 = -1, 2
    plt.plot(days, ind)
    plt.plot(d_onset, ind.sel(day=d_onset), 'ro', label='onset')
    plt.plot(d_onset-1, ind.sel(day=d_onset-1), 'k.', label='onset-1')
    plt.plot(d_retreat, ind.sel(day=d_retreat), 'bo', label='retreat')
    plt.plot(d_retreat-1, ind.sel(day=d_retreat-1), 'k.', label='retreat')
    plt.grid()
    plt.ylim(ylim1, ylim2)

ylist = range(4)
titlestr = ['', '', '', '']
ylist = ylist + [int(onset.argmin()), int(onset.argmax()),
                 int(retreat.argmin()), int(retreat.argmax()),
                 int(length.argmin()), int(length.argmax())]
titlestr = titlestr + ['Earliest Onset', 'Latest Onset', 'Earliest Retreat',
                       'Latest Retreat', 'Shortest Monsoon', 'Longest Monsoon']
for i, yr in enumerate(ylist):
    if i % 4 == 0:
        plt.figure(figsize=(12, 10))
        yplot = 1
    else:
        yplot += 1
    plt.subplot(2, 2, yplot)
    onset_tseries(days, howi.tseries[yr], onset[yr], retreat[yr])
    plt.title('%d %s' % (years[yr], titlestr[i]))

# ----------------------------------------------------------------------
# Onset and retreat indices

summarize_indices(years, onset, retreat, 'HOWI')

# ----------------------------------------------------------------------
# Save figures
if isave:
    for ext in exts:
        atm.savefigs(namestr, ext)

# ----------------------------------------------------------------------
# Plot timeseries of each year
namestr = namestr + 'tseries_'
plt.close('all')
plot_index_years(howi)

if isave:
    for ext in exts:
        atm.savefigs(namestr, ext)
