import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pandas as pd

import atmos as atm
import utils

mpl.rcParams['font.size'] = 11

# ----------------------------------------------------------------------
onset_nm = 'CHP_MFC'
years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/analysis/'
varnms = ['VFLXMSE', 'VFLXCPT', 'VFLXPHI', 'VFLXLQV']
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
eqlat1, eqlat2 = -2, 2
nroll = 7
scale, units, sector_units = 1e-9, '$10^9$W/m', 'PW'

relfiles = {}
yearstr = '%d-%d' % (min(years), max(years))
filestr = datadir + 'merra_%s_dailyrel_%s_%s.nc'
for nm in varnms:
    if nm == 'VFLXLQV':
        nm0 = 'VFLXQV'
    else:
        nm0 = nm
    relfiles[nm] = filestr % (nm0, onset_nm, yearstr)

# ----------------------------------------------------------------------
# Read data
data = xray.Dataset()
for nm in varnms:
    print('Loading ' + relfiles[nm])
    with xray.open_dataset(relfiles[nm]) as ds:
        if nm == 'VFLXLQV':
            var = ds['VFLXQV'].load()
            data[nm] = var * atm.constants.Lv.values
        else:
            data[nm] = ds[nm].load()

# Scale units and rename variables
data = data * scale
nms = data.data_vars.keys()
for nm in nms:
    data = data.rename({nm : nm.replace('FLX', '')})


# Take subset and smooth with rolling mean
daydim = atm.get_coord(data['VMSE'], 'dayrel', 'dim')
for nm in data.data_vars:
    data[nm] = atm.rolling_mean(data[nm], nroll, axis=daydim, center=True)

# Average over equatorial region
data_eq = atm.dim_mean(data, 'lat', eqlat1, eqlat2)

# Cross-equatorial flues integrated over sectors
a = atm.constants.radius_earth.values
eq_int = xray.Dataset()
eq_int.attrs['units'] = sector_units
lonranges = [(40, 60), (40, 100), (lon1, lon2)]
eq_int.attrs['lonranges'] = ['%dE-%dE' % lonrange for lonrange in lonranges]
for lonrange in lonranges:
    lon1, lon2 = lonrange
    dist = a * np.radians(lon2 - lon1)
    for nm in data_eq.data_vars:
        key = nm + '_%dE-%dE' % (lon1, lon2)
        eq_int[key] = atm.dim_mean(data_eq[nm], 'lon', lon1, lon2) * dist
# Convert to PW
eq_int = eq_int * 1e-15 / scale

# ----------------------------------------------------------------------
# Longitude-day contour plot

def contour_londay(var, clev=None, grp=None,n_pref=40,
                   yticks=np.arange(-120, 201, 30)):
    lon = atm.get_coord(var, 'lon')
    days = atm.get_coord(var, 'dayrel')
    if clev is None:
        cint = atm.cinterval(var, n_pref=n_pref, symmetric=True)
        clev = atm.clevels(var, cint, symmetric=True)
    plt.contourf(lon, days, var, clev, cmap='RdBu_r', extend='both')
    plt.grid()
    plt.colorbar()
    #plt.gca().invert_yaxis()
    plt.yticks(yticks)
    plt.axhline(0, color='k')
    if grp is not None and grp.row == grp.nrow - 1:
        plt.xlabel('Longitude')
    if grp is not None and grp.col == 0:
        plt.ylabel('Rel Day')


nrow, ncol = 2, 2
fig_kw = {'figsize' : (11, 7), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.07, 'right' : 0.99, 'bottom' : 0.07, 'top' : 0.9,
               'wspace' : 0.05}
suptitle = 'Cross-Eq <V*MSE> (%s)' % units
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw,
                   suptitle=suptitle)
for lonrange in [(40, 120), (lon1, lon2)]:
    for nm in data_eq.data_vars:
        grp.next()
        var = atm.subset(data_eq[nm], {'lon' : lonrange})
        contour_londay(var, grp=grp)
        plt.title(nm, fontsize=11)
    plt.gca().invert_yaxis()


# ----------------------------------------------------------------------
# Line plots of sector means

days = atm.get_coord(eq_int, 'dayrel')
nms = data_eq.data_vars
styles = {'VMSE' : {'color' : 'k', 'linewidth' : 2}, 'VCPT' : {'color' : 'k'},
          'VPHI' : {'color' : 'k', 'linestyle' : 'dashed'},
          'VLQV' : {'color' : 'k', 'alpha' : 0.4, 'linewidth' : 1.5}}
locs = {'40E-60E' : 'upper left', '40E-100E' : 'upper left',
        '60E-100E' : 'lower left'}
nrow, ncol = 3, 1
fig_kw = {'figsize' : (6, 9), 'sharex' : True}
gridspec_kw = {'left' : 0.15, 'right' : 0.92, 'bottom' : 0.07, 'top' : 0.9,
               'wspace' : 0.05}
suptitle = 'Sector Cross-Eq <V*MSE> (%s)' % eq_int.attrs['units']
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw,
                   suptitle=suptitle)
for lonrange in eq_int.attrs['lonranges']:
    grp.next()
    plt.title(lonrange, fontsize=11)
    for nm in nms:
        key = nm + '_' + lonrange
        plt.plot(days, eq_int[key], label=nm, **styles[nm])
    plt.legend(fontsize=9, loc=locs[lonrange], handlelength=3)
    plt.grid()
    plt.xticks(np.arange(-120, 201, 30))
    if grp.row == grp.nrow - 1:
        plt.xlabel('Rel Day')

# ----------------------------------------------------------------------
# OLD
# ----------------------------------------------------------------------
#
# onset_nm = 'CHP_MFC'
# years, years2 = np.arange(1979, 2015), None
# yearstr, savestr = '%d-%d Climatology' % (years.min(), years.max()), 'clim'
#
# datadir = atm.homedir() + 'datastore/merra/analysis/'
# savedir = 'figs/'
# nroll = 7   # Number of days rolling mean for smoothing
# lon1, lon2 = 60, 100
# lat1, lat2 = -20, 20
# # eqlat1, eqlat2 = -5, 5
# eqlat1, eqlat2 = -2, 2
#
# filestr = datadir + 'merra_%s_dailyrel_%s_%d.nc'
# files = {}
# for nm in ['VFLXQV', 'VFLXCPT', 'VFLXPHI', 'VFLXMSE']:
#     files[nm] = [filestr % (nm, onset_nm, yr) for yr in years]
#
# # Read dailyrel MSE flux components
# data = xray.Dataset()
# for nm in files:
#     data[nm], onset, retreat = utils.load_dailyrel(files[nm])
#
# # Latent heat
# Lv = atm.constants.Lv
# attrs = data['VFLXQV'].attrs
# data['VFLXLQV'] = data['VFLXQV'] * Lv.values
# attrs['units'] = attrs['units'] + ' * ' + Lv.attrs['units']
# attrs['long_name'] = 'Northward flux of latent heat'
# attrs['standard_name'] = 'northward_flux_of_latent_heat'
# data['VFLXLQV'].attrs = attrs
# data = data.drop('VFLXQV')
#
# # Rename variables
# nms = data.data_vars.keys()
# for nm in nms:
#     data = data.rename({nm : nm.replace('FLX', '')})
#
# # Take subset and smooth with rolling mean
# data = atm.subset(data, {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)})
# daydim = atm.get_coord(data['VMSE'], 'dayrel', 'dim')
# for nm in data.data_vars:
#     data[nm] = atm.rolling_mean(data[nm], nroll, axis=daydim, center=True)
#
# # Average over equatorial region
# data_eq = xray.Dataset()
# for nm in data.data_vars:
#     data_eq[nm] = atm.mean_over_geobox(data[nm], eqlat1, eqlat2, lon1, lon2)
#
#
# # Climatology
# databar = atm.dim_mean(data, 'year')
# databar_eq = atm.dim_mean(data_eq, 'year')
#
# # ----------------------------------------------------------------------
# # Plot latitude-day contours for tropics
# days = atm.get_coord(databar, 'dayrel')
# lat = atm.get_coord(databar, 'lat')
# fig_kw = {'figsize' : (14, 10), 'sharex' : True, 'sharey' : True}
# gridspec_kw = {'left' : 0.05, 'right' : 0.98, 'bottom' : 0.05,
#                'top' : 0.92, 'wspace' : 0.01, 'hspace' : 0.1}
# suptitle = '%d-%dE Vert. Int. Meridional MSE Fluxes (J/m/s) - ' % (lon1, lon2)
# suptitle = suptitle + yearstr
# nrow, ncol = (2, 2)
# cmap = 'RdBu_r'
# grp = atm.FigGroup(nrow, ncol, advance_by='col', fig_kw=fig_kw,
#                    gridspec_kw=gridspec_kw, suptitle=suptitle)
# for nm in ['VCPT', 'VPHI', 'VLQV', 'VMSE']:
#     grp.next()
#     plotdata = atm.dim_mean(databar[nm], 'lon')
#     utils.contourf_lat_time(lat, days, plotdata, nm, cmap, onset_nm)
#     #plt.contour(days, lat, plotdata.T, [0], colors='k')
#     if grp.row < nrow - 1:
#         plt.xlabel('')
#     if grp.col > 0:
#         plt.ylabel('')
#
# # Daily timeseries plot of equatorial MSE flux
# df = databar_eq.to_dataframe()
# df.plot(figsize=(12, 8))
# plt.grid()
# plt.xticks(range(-120, 201, 30))
# latstr = atm.latlon_str(eqlat1, eqlat2, 'lat')
# title = 'Cross-Eq <VMSE> (%d-%dE, %s) %s' % (lon1, lon2, latstr, yearstr)
# plt.title(title)
# plt.ylabel('Vertically Integrated MSE (J/m/s)')
#
# # Daily timeseries of equatorial MSE flux in each year
# fig_kw = {'figsize' : (14, 10), 'sharex' : True, 'sharey' : True}
# gridspec_kw = {'left' : 0.05, 'right' : 0.98, 'bottom' : 0.05,
#                'top' : 0.92, 'wspace' : 0.01, 'hspace' : 0.1}
# suptitle = 'Cross-Eq <VMSE> (%d-%dE, %s)' % (lon1, lon2, latstr)
# ylims = (-8e9, 5e9)
# nrow, ncol = (3, 4)
# grp = atm.FigGroup(nrow, ncol, advance_by='col', fig_kw=fig_kw,
#                    gridspec_kw=gridspec_kw, suptitle=suptitle)
# for y, year in enumerate(years):
#     grp.next()
#     plotdata = data_eq['VMSE'][y]
#     plt.plot(days, plotdata, 'k')
#     plt.title(year)
#     plt.grid(True)
#     plt.xlim(days.min(), days.max())
#     plt.ylim(ylims)
