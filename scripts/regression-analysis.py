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
savedir = 'figs/'

varnms = ['U200', 'V200', 'T200', 'H200', 'U850', 'V850', 'T850', 'H850',
          'THETA950', 'THETA_E950', 'QV950', 'T950', 'HFLUX', 'EFLUX',
          'EVAP', 'precip']
regdays = [-60, -30, 0]
seasons = ['JJAS']
lon1, lon2 = 60, 100
nstd = 2.0 # Number of stdev for strong/weak composites
nroll = 5
savestr = savedir + 'reg_'
if nroll is not None:
    savestr = savestr + 'nroll%d_' % nroll

# ----------------------------------------------------------------------
# Read data
def get_filenames(datadir, varnms, onset_nm, years, lon1, lon2, nroll):
    lonstr = atm.latlon_str(lon1, lon2, 'lon')
    yearstr = '_%d-%d.nc' % (min(years), max(years))
    filestr = datadir + 'merra_%s'
    if nroll is not None:
        filestr = filestr + '_nroll%d' % nroll
    filestr = filestr + '_reg_%s_onset_' + onset_nm + yearstr
    filestr2 = datadir + 'merra_%s_dailyrel_' + onset_nm + yearstr
    filenames = {}
    for varnm in varnms:
        filenames[varnm] = {'latlon_reg' : filestr % (varnm, 'latlon'),
                            'sector_reg' : filestr % (varnm, lonstr),
                            'latlon_clim' : filestr2 % varnm,
                            'sector_clim' : filestr2 % varnm}
    return filenames

datafiles = get_filenames(datadir, varnms, onset_nm, years, lon1, lon2, nroll)
data = {'reg' : {}, 'clim' : {}, 'early' : {}, 'late' : {}}

# Load regression coefficients and climatology
for varnm in varnms:
    for key in ['latlon', 'sector']:
        for key2 in ['reg', 'clim']:
            filenm = datafiles[varnm][key + '_' + key2]
            print('Loading ' + filenm)
            with xray.open_dataset(filenm) as ds:
                data[key2][varnm + '_' + key] = ds.load()

# ----------------------------------------------------------------------
# Calculate strong/ weak composites for sector data
for varnm in varnms:
    key = varnm + '_sector'
    varbar = atm.dim_mean(data['clim'][key][varnm], 'lon', lon1, lon2)
    m = data['reg'][key]['m']
    data['late'][key] = varbar + m * nstd
    data['early'][key] = varbar - m * nstd

# ----------------------------------------------------------------------
def stipple_mask(p):
    return ((p >= 0.05) | np.isnan(p))

def sector_plot(var, p, stipple_kw={}, grp=None, ylim=None,
                yticks=None, clim=None):
    xname, yname = 'dayrel', 'lat'
    pts_mask = stipple_mask(p)
    lat = atm.get_coord(var, 'lat')
    days = atm.get_coord(var, 'dayrel')
    xsample = 3
    if max(np.diff(lat)) > 1:
        ysample = 1
    else:
        ysample = 2
    #utils.contourf_lat_time(lat, days, var)
    vals = np.ma.masked_array(var.T.values, mask=np.isnan(var.T))
    plt.pcolormesh(days, lat, vals, cmap='RdBu_r')
    cb = plt.colorbar()
    atm.stipple_pts(pts_mask, xname, yname, xsample, ysample, **stipple_kw)
    plt.title(varnm)
    plt.xlabel('Relative Day')
    plt.ylabel('Latitude')
    plt.grid(True)
    xticks = np.arange(-120, 201, 30)
    xlims = (-120, 200)
    plt.xlim(xlims)
    plt.xticks(xticks)
    if grp is not None:
        if grp.col > 0:
            plt.ylabel('')
        if grp.row < grp.nrow - 1:
            plt.xlabel('')
    if ylim is not None:
        plt.ylim(ylim)
    if yticks is not None:
        plt.yticks(yticks)
    if clim is not None:
        plt.clim(clim)
    else:
        clim = atm.climits(var, symmetric=True, percentile=99.9)
        plt.clim(clim)


def latlon_plot(varnm, reg, day_or_season, coeff='m', stipple_kw={},
                axlims=(-60, 60, 40, 120)):
    regdata = reg[varnm + '_latlon']
    keys = [key for key in regdata if key.endswith('_' + coeff)]
    clim = atm.climits(regdata[keys].to_array(), symmetric=True,
                        percentile=99.9)
    xname, yname = 'lon', 'lat'
    lat = atm.get_coord(regdata, 'lat')
    if max(np.diff(lat)) > 1:
        xsample, ysample = 1, 1
    else:
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
    atm.pcolor_latlon(var, axlims=axlims, fancy=False)
    plt.clim(clim)
    atm.stipple_pts(pts_mask, xname, yname, xsample, ysample, **stipple_kw)
    plt.title(titlestr)


# Latitude-day contour plots of sector mean
nrow, ncol = 2, 2
fig_kw = {'figsize' : (14, 10), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.05, 'right' : 0.98, 'bottom' : 0.05, 'top' : 0.95,
               'wspace' : 0.04, 'hspace' : 0.1}
stipple_kw = {'markersize' : 2, 'markeredgewidth' : 1.5}
coeff = 'm'
suptitle = '%d-%dE Reg onto %s Onset - Reg Coefficients' % (lon1, lon2, onset_nm)
ylim, yticks = (-60, 60), np.arange(-60, 61, 20)

grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw,
                   suptitle=suptitle)
for varnm in varnms:
    grp.next()
    var = data['reg'][varnm + '_sector'][coeff]
    p = data['reg'][varnm + '_sector']['p']
    sector_plot(var, p, stipple_kw, grp, ylim, yticks)
atm.savefigs(savestr + 'sector_onset_' + onset_nm, 'pdf', merge=True)
plt.close('all')

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
        latlon_plot(varnm, data['reg'], day_or_season, coeff, stipple_kw)
atm.savefigs(savestr + 'latlon_onset_' + onset_nm, 'pdf', merge=True)
plt.close('all')
