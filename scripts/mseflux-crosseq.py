import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
import collections
import pandas as pd

import atmos as atm
import precipdat
import merra
import indices
import utils

mpl.rcParams['font.size'] = 10

# ----------------------------------------------------------------------

onset_nm = 'CHP_MFC'
years, years2 = np.arange(1979, 2015), None
yearstr, savestr = '%d-%d Climatology' % (years.min(), years.max()), 'clim'

datadir = atm.homedir() + 'datastore/merra/analysis/'
savedir = 'figs/'
nroll = 7   # Number of days rolling mean for smoothing
lon1, lon2 = 60, 100
lat1, lat2 = -20, 20
# eqlat1, eqlat2 = -5, 5
eqlat1, eqlat2 = -2, 2

filestr = datadir + 'merra_%s_dailyrel_%s_%d.nc'
files = {}
for nm in ['VFLXQV', 'VFLXCPT', 'VFLXPHI', 'VFLXMSE']:
    files[nm] = [filestr % (nm, onset_nm, yr) for yr in years]

# Read dailyrel MSE flux components
data = xray.Dataset()
for nm in files:
    data[nm], onset, retreat = utils.load_dailyrel(files[nm])

# Latent heat
Lv = atm.constants.Lv
attrs = data['VFLXQV'].attrs
data['VFLXLQV'] = data['VFLXQV'] * Lv.values
attrs['units'] = attrs['units'] + ' * ' + Lv.attrs['units']
attrs['long_name'] = 'Northward flux of latent heat'
attrs['standard_name'] = 'northward_flux_of_latent_heat'
data['VFLXLQV'].attrs = attrs
data = data.drop('VFLXQV')

# Rename variables
nms = data.data_vars.keys()
for nm in nms:
    data = data.rename({nm : nm.replace('FLX', '')})

# Take subset and smooth with rolling mean
data = atm.subset(data, {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)})
daydim = atm.get_coord(data['VMSE'], 'dayrel', 'dim')
for nm in data.data_vars:
    data[nm] = atm.rolling_mean(data[nm], nroll, axis=daydim, center=True)

# Average over equatorial region
data_eq = xray.Dataset()
for nm in data.data_vars:
    data_eq[nm] = atm.mean_over_geobox(data[nm], eqlat1, eqlat2, lon1, lon2)


# Climatology
databar = atm.dim_mean(data, 'year')
databar_eq = atm.dim_mean(data_eq, 'year')

# ----------------------------------------------------------------------
# Plot latitude-day contours for tropics
days = atm.get_coord(databar, 'dayrel')
lat = atm.get_coord(databar, 'lat')
fig_kw = {'figsize' : (14, 10), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.05, 'right' : 0.98, 'bottom' : 0.05,
               'top' : 0.92, 'wspace' : 0.01, 'hspace' : 0.1}
suptitle = '%d-%dE Vert. Int. Meridional MSE Fluxes (J/m/s) - ' % (lon1, lon2)
suptitle = suptitle + yearstr
nrow, ncol = (2, 2)
cmap = 'RdBu_r'
grp = atm.FigGroup(nrow, ncol, advance_by='col', fig_kw=fig_kw,
                   gridspec_kw=gridspec_kw, suptitle=suptitle)
for nm in ['VCPT', 'VPHI', 'VLQV', 'VMSE']:
    grp.next()
    plotdata = atm.dim_mean(databar[nm], 'lon')
    utils.contourf_lat_time(lat, days, plotdata, nm, cmap, onset_nm)
    #plt.contour(days, lat, plotdata.T, [0], colors='k')
    if grp.row < nrow - 1:
        plt.xlabel('')
    if grp.col > 0:
        plt.ylabel('')

# Daily timeseries plot of equatorial MSE flux
df = databar_eq.to_dataframe()
df.plot(figsize=(12, 8))
plt.grid()
plt.xticks(range(-120, 201, 30))
latstr = atm.latlon_str(eqlat1, eqlat2, 'lat')
title = 'Cross-Eq <VMSE> (%d-%dE, %s) %s' % (lon1, lon2, latstr, yearstr)
plt.title(title)
plt.ylabel('Vertically Integrated MSE (J/m/s)')

# Daily timeseries of equatorial MSE flux in each year
fig_kw = {'figsize' : (14, 10), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.05, 'right' : 0.98, 'bottom' : 0.05,
               'top' : 0.92, 'wspace' : 0.01, 'hspace' : 0.1}
suptitle = 'Cross-Eq <VMSE> (%d-%dE, %s)' % (lon1, lon2, latstr)
ylims = (-8e9, 5e9)
nrow, ncol = (3, 4)
grp = atm.FigGroup(nrow, ncol, advance_by='col', fig_kw=fig_kw,
                   gridspec_kw=gridspec_kw, suptitle=suptitle)
for y, year in enumerate(years):
    grp.next()
    plotdata = data_eq['VMSE'][y]
    plt.plot(days, plotdata, 'k')
    plt.title(year)
    plt.grid(True)
    plt.xlim(days.min(), days.max())
    plt.ylim(ylims)
