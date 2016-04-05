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

mpl.rcParams['font.size'] = 10

# ----------------------------------------------------------------------


onset_nm = 'CHP_MFC'
years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/analysis/'

varnms = ['U200', 'V200', 'T200', 'H200', 'U850', 'V850', 'T850', 'H850',
          'THETA950', 'THETA_E950', 'QV950', 'T950', 'HFLUX', 'EFLUX',
          'EVAP', 'precip']
regdays = [-60, -30, 0]
seasons = ['JJAS']
lon1, lon2 = 60, 100
nroll = 5

# ----------------------------------------------------------------------
# Read data
def get_filenames(datadir, varnms, onset_nm, years, lon1, lon2, nroll):
    lonstr = atm.latlon_str(lon1, lon2, 'lon')
    yearstr = '_%d-%d.nc' % (min(years), max(years))
    filestr = datadir + 'merra_%s' 
    if nroll is not None:
        filestr = filestr + '_nroll%d' % nroll
    filestr = filestr + '_reg_%s_onset_' + onset_nm + yearstr
    filenames = {}
    for varnm in varnms:
        filenames[varnm] = {'latlon' : filestr % (varnm, 'latlon'),
                            'sector' : filestr % (varnm, lonstr)}        
    return filenames

datafiles = get_filenames(datadir, varnms, onset_nm, years, lon1, lon2, nroll)
reg = {}

for varnm in varnms:
    for key in ['latlon', 'sector']:
        filenm = datafiles[varnm][key]
        print('Loading ' + filenm)
        with xray.open_dataset(filenm) as ds:
            reg[varnm + '_' + key] = ds.load()

# ----------------------------------------------------------------------
def stipple_mask(p):
    return ((p >= 0.05) | np.isnan(p))

def sector_plot(varnm, reg, coeff='m', stipple_kw={}, grp=None, ylim=None,
                yticks=None):
    xname, yname = 'dayrel', 'lat'
    var = reg[varnm + '_sector'][coeff]
    p = reg[varnm + '_sector']['p']
    pts_mask = stipple_mask(p)
    lat = atm.get_coord(var, 'lat')
    days = atm.get_coord(var, 'dayrel')
    xsample = 6
    if max(np.diff(lat)) > 1:
        ysample = 2
    else:
        ysample = 4
    utils.contourf_lat_time(lat, days, var)
    atm.stipple_pts(pts_mask, xname, yname, xsample, ysample, **stipple_kw)
    plt.title(varnm)
    if grp is not None:
        if grp.col > 0:
            plt.ylabel('')
        if grp.row < grp.nrow - 1:
            plt.xlabel('')
    if ylim is not None:
        plt.ylim(ylim)
    if yticks is not None:
        plt.yticks(yticks)
            

def latlon_plot(varnm, reg, day_or_season, coeff='m', stipple_kw={},
                axlims=(-60, 60, 40, 120)):
    regdata = reg[varnm + '_latlon']
    keys = [key for key in regdata if key.endswith('_' + coeff)]
    cint = atm.cinterval(regdata[keys].to_array())
    clev = atm.clevels(regdata[keys].to_array(), cint, symmetric=True)
    xname, yname = 'lon', 'lat'
    xsample, ysample = 2, 2
    if isinstance(day_or_season, int):
        key = varnm + '_DAILY_'
        var = regdata[key + coeff].sel(dayrel=day_or_season)
        p = regdata[key + 'p'].sel(dayrel=day_or_season)
        titlestr = varnm + ' Day %d' % day_or_season
    else:
        key = varnm + '_' + day_or_season + '_'
        var = regdata[key + coeff]
        p = regdata[key + 'p']
        titlestr = varnm + ' ' + day_or_season
    pts_mask = stipple_mask(p)
    atm.contourf_latlon(var, clev=clev, axlims=axlims)
    atm.stipple_pts(pts_mask, xname, yname, xsample, ysample, **stipple_kw)
    plt.title(titlestr)
    

# Latitude-day contour plots of sector mean
nrow, ncol = 2, 2
fig_kw = {'figsize' : (14, 10), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.05, 'right' : 0.95, 'bottom' : 0.05, 'top' : 0.95,
               'wspace' : 0.05, 'hspace' : 0.1}
stipple_kw = {'markersize' : 2, 'markeredgewidth' : 1.5}
coeff = 'm'
suptitle = '%d-%dE Reg onto %s Onset - Reg Coefficients' % (lon1, lon2, onset_nm)
ylim, yticks = (-60, 60), np.arange(-60, 61, 20)

grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw,
                   suptitle=suptitle)
for varnm in varnms:
    grp.next()
    sector_plot(varnm, reg, coeff, stipple_kw, grp, ylim, yticks)
  

# Lat-lon maps
plt_list = regdays + seasons
nrow = 2
ncol = len(plt_list)
fig_kw = {'figsize' : (14, 10), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.05, 'right' : 0.95, 'bottom' : 0.05, 'top' : 0.95,
               'wspace' : 0.15, 'hspace' : 0.1}
stipple_kw = {'markersize' : 2, 'markeredgewidth' : 1.5}
coeff = 'm'
suptitle = 'Reg onto %s Onset - Reg Coefficients' % onset_nm
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw,
                   suptitle=suptitle)
for varnm in varnms:
    for day_or_season in plt_list:
        grp.next()
        latlon_plot(varnm, reg, day_or_season, coeff, stipple_kw)
    

