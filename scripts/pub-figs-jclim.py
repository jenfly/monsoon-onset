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
import indices
import utils

style = atm.homedir() + 'dynamics/python/mpl-styles/grl_article.mplstyle'
plt.style.use(style)
figwidth = 7.48
fontsize = mpl.rcParams['font.size']
labelsize = fontsize + 3
dashes = [6, 2]

# *** Consolidate code from momentum-budget.py and analyze-composites.py

# ----------------------------------------------------------------------
version = 'merra2'
years = np.arange(1980, 2016)
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
onset_nm = 'CHP_MFC'
pcp_nm = 'GPCP'
varnms = ['U200', 'V200', 'T200', 'U850', 'V850', 'TLML', 'QLML',
          'THETA_E_LML']
lat_extract = {'U200' : 0, 'V200' : 15, 'T200' : 30,
               'U850' : 15, 'V850' : 15}
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
nroll = 5 # n-day rolling averages for smoothing daily timeseries
ind_nm, npre, npost = 'onset', 120, 200
#ind_nm, npre, npost = 'retreat', 270, 89

yearstr = '%d-%d.nc' % (min(years), max(years))
indfile = datadir + version + '_index_%s_%s' % (onset_nm, yearstr)

filestr = datadir + version + '_%s_dailyrel_' + onset_nm + '_' + yearstr
datafiles = {nm : filestr % nm for nm in varnms}
datafiles['GPCP'] = datadir + 'gpcp_dailyrel_' + onset_nm + '_1997-2015.nc'

if ind_nm == 'retreat':
    for nm in datafiles:
        datafiles[nm] = datafiles[nm].replace('dailyrel', 'dailyrel_retreat')

# ----------------------------------------------------------------------
# Read data

# Large-scale onset/retreat indices
with xray.open_dataset(indfile) as index:
    index.load()
index['length'] = index['retreat'] - index['onset']

# Dailyrel climatology
keys_dict = {'PRECTOT' : 'PRECTOT', 'CMAP' : 'precip', 'GPCP' : 'PREC',
             'U200' : 'U', 'U850' : 'U', 'V200' : 'V', 'V850' : 'V',
             'THETA_E_LML' : 'THETA_E', 'TLML' : 'T', 'QLML' : 'Q',
             'T200' : 'T'}
data = {}
for nm in datafiles:
    print('Loading ' + datafiles[nm])
    with xray.open_dataset(datafiles[nm]) as ds:
        if 'year' in ds.dims:
            ds = ds.mean(dim='year')
        data[nm] = ds[keys_dict[nm]].load()


# ----------------------------------------------------------------------
# Daily timeseries

ts = xray.Dataset()
ts[pcp_nm] = atm.mean_over_geobox(data[pcp_nm], lat1, lat2, lon1, lon2)
ts['MFC'] = utils.daily_rel2onset(index['daily_ts'], index[ind_nm], npre,npost)
ts['CMFC'] = utils.daily_rel2onset(index['tseries'], index[ind_nm], npre,npost)
ts = ts.mean(dim='year')

# Extract variables at specified latitudes
for nm, lat0 in lat_extract.iteritems():
    var = atm.dim_mean(data[nm], 'lon', lon1, lon2)
    lat = atm.get_coord(var, 'lat')
    lat0_str = atm.latlon_labels(lat0, 'lat', deg_symbol=False)
    # key = nm + '_' + lat0_str
    key = nm
    lat_closest, _ = atm.find_closest(lat, lat0)
    print '%s %.2f %.2f' % (nm, lat0, lat_closest)
    ts[key] = atm.subset(var, {'lat' : (lat_closest, None)}, squeeze=True)

# Smooth with rolling mean
if nroll is not None:
    for nm in ts.data_vars:
        ts[nm] = atm.rolling_mean(ts[nm], nroll, center=True)
tseries = atm.subset(ts, {'dayrel' : (-npre, npost)})


# Smooth latitude-dayrel data with rolling mean
for nm in data:
    daydim = atm.get_coord(data[nm], 'dayrel', 'dim')
    data[nm] = atm.rolling_mean(data[nm], nroll, axis=daydim, center=True)

# ----------------------------------------------------------------------
# Plotting functions

def fix_axes(axlims):
    plt.gca().set_ylim(axlims[:2])
    plt.gca().set_xlim(axlims[2:])
    plt.draw()

def add_labels(grp, labels, pos, fontsize, fontweight='bold'):
    # Expand pos to list for each subplot, if needed
    try:
        n = len(pos[0])
    except TypeError:
        pos = [pos] * (grp.nrow * grp.ncol)
    i = 0
    for row in range(grp.nrow):
        for col in range(grp.ncol):
            grp.subplot(row, col)
            atm.text(labels[i], pos[i], fontsize=fontsize,
                     fontweight=fontweight)
            i += 1

def skip_ticklabel(xticks):
    xtick_labels = []
    for i, n in enumerate(xticks):
        if i % 2 == 0:
            xtick_labels = xtick_labels +  ['']
        else:
            xtick_labels = xtick_labels + [n]
    return xtick_labels


def daily_tseries(tseries, index, pcp_nm, npre, npost, grp, keys1=None,
                  keys2=None, units1=None, units2=None, ylims=None,
                  legend_loc=None, ind_nm='onset', grid=False, dashes=[6, 2],
                  dlist=[15], labelpad=1.5, legend=True):
    """Plot dailyrel timeseries climatology"""
    xlims = (-npre, npost)
    xticks = range(-npre, npost + 10, 30)
    xlabel = 'Days Since ' + ind_nm.capitalize()
    if ind_nm == 'onset':
        x0 = [0, index['length'].mean(dim='year')]
        xtick_labels = xticks
    else:
        x0 = [-index['length'].mean(dim='year'), 0]
        xtick_labels = skip_ticklabel(xticks)

    y2_opts={'color' : 'r', 'alpha' : 0.6}
    dashed = {'color' : 'k', 'linestyle' : '--', 'dashes' : dashes}
    styles = ['k', dashed, 'g', 'm']
    legend_kw = {}
    legend_kw['loc'] = legend_loc
    y1_label = units1
    y2_label = units2
    data1 = tseries[keys1]
    if keys2 is not None:
        data2 = tseries[keys2]
    else:
        data2 = None
    data1_styles = {nm : style for (nm, style) in zip(keys1, styles)}
    axs = utils.plotyy(data1, data2, xname='dayrel', data1_styles=data1_styles,
                       y2_opts=y2_opts, xlims=xlims, xticks=xticks, ylims=ylims,
                       xlabel=xlabel, y1_label=y1_label, y2_label=y2_label,
                       legend=legend, legend_kw=legend_kw, x0_axvlines=x0,
                       grid=grid)
    for ax, label in zip(axs, [y1_label, y2_label]):
        ax.set_ylabel(label, labelpad=labelpad)
    plt.gca().set_xticklabels(xtick_labels)
    if dlist is not None:
        for d0 in dlist:
            plt.axvline(d0, color='k', linestyle='--', dashes=dashes)


def contourf_latday(var, clev=None, title='', nc_pref=40, grp=None,
                    xlims=(-120, 200), xticks=np.arange(-120, 201, 30),
                    ylims=(-60, 60), yticks=np.arange(-60, 61, 20),
                    dlist=None, grid=False, ind_nm='onset'):
    vals = var.values.T
    lat = atm.get_coord(var, 'lat')
    days = atm.get_coord(var, 'dayrel')
    if var.min() >= 0:
        cmap, extend, symmetric = 'PuBuGn', 'max', False
    else:
        cmap, extend, symmetric = 'RdBu_r', 'both', True
    if clev == None:
        cint = atm.cinterval(vals, n_pref=nc_pref, symmetric=symmetric)
        clev = atm.clevels(vals, cint, symmetric=symmetric)
    elif len(atm.makelist(clev)) == 1:
        if var.name == 'PREC':
            clev = np.arange(0, 10 + clev/2.0, clev)
        else:
            clev = atm.clevels(vals, clev, symmetric=symmetric)
    cticks_dict = {'PRECTOT' : np.arange(0, 13, 2),
                   'PREC' : np.arange(0, 11, 2),
                   'T200' : np.arange(-208, 227, 2),
                   'U200' : np.arange(-60, 61, 10),
                   'PSI500' : np.arange(-800, 801, 200)}
    cticks = cticks_dict.get(var.name)
    plt.contourf(days, lat, vals, clev, cmap=cmap, extend=extend)
    plt.colorbar(ticks=cticks)
    atm.ax_lims_ticks(xlims, xticks, ylims, yticks)
    plt.grid(grid)
    plt.title(title)
    if dlist is not None:
        for d0 in dlist:
            plt.axvline(d0, color='k')
    if grp is not None and grp.row == grp.ncol - 1:
        plt.xlabel('Days Since ' + ind_nm.capitalize())
    if grp is not None and grp.col == 0:
        plt.ylabel('Latitude')

# ----------------------------------------------------------------------
# FIGURES

# Timeseries plots - setup figure for subplots
nrow, ncol = 2, 2
fig_kw = {'figsize' : (figwidth, 0.7 * figwidth)}
gridspec_kw = {'left' : 0.07, 'right' : 0.9, 'bottom' : 0.07, 'top' : 0.9,
               'wspace' : 0.5, 'hspace' : 0.35}
legend = True
legend_kw = {'loc' : 'upper left', 'framealpha' : 0.0}
labelpos = (-0.2, 1.05)
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)

# Plot daily tseries
legend = True
if ind_nm == 'onset':
    dlist = [15]
else:
    dlist = None
opts = []
opts.append({'keys1' : ['MFC', pcp_nm], 'keys2' : ['CMFC'],
             'units1' : 'mm day$^{-1}$', 'units2' : 'mm',
             'ylims' : (-3.5, 9), 'legend_loc' : 'upper left' })
opts.append({'keys1' : ['U850'], 'keys2' : ['V850'],
             'units1' : '   m s$^{-1}$', 'units2' :  '   m s$^{-1}$',
             'ylims' : (-7, 15), 'legend_loc' : 'upper left' })
for opt in opts:
    grp.next()
    daily_tseries(tseries, index, pcp_nm, npre, npost, grp, legend=legend,
                  ind_nm=ind_nm, dlist=dlist, **opt)

# Add a-d labels
labels = ['a', 'b', 'c', 'd']
add_labels(grp, labels, labelpos, labelsize)

# Lat-day contour plots
xticks = range(-npre, npost + 10, 30)
if ind_nm == 'onset':
    dlist = [0, index['length'].mean(dim='year')]
    d0 = 15
    xtick_labels = xticks
else:
    dlist = [-index['length'].mean(dim='year'), 0]
    d0 = None
    xtick_labels = skip_ticklabel(xticks)

keys = [pcp_nm, 'V200', 'U200', 'U850']
clevs = {pcp_nm : 1, 'U200' : 5, 'V200' : 1, 'U850' : 2}
nrow, ncol = 2, 2
fig_kw = {'figsize' : (figwidth, 0.64 * figwidth), 'sharex' : True,
          'sharey' : True}
gridspec_kw = {'left' : 0.07, 'right' : 0.99, 'bottom' : 0.07, 'top' : 0.94,
               'wspace' : 0.05}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for key in keys:
    grp.next()
    var = atm.dim_mean(data[key], 'lon', lon1, lon2)
    contourf_latday(var, clev=clevs[key], title=key.upper(), grp=grp,
                    dlist=dlist, ind_nm=ind_nm)
    if d0 is not None:
        plt.axvline(d0, color='k', linestyle='--', dashes=dashes)
plt.xticks(xticks, xtick_labels)
plt.xlim(-npre, npost)
labels = ['a', 'b', 'c', 'd']
x1, x2, y0 = -0.15, -0.05, 1.05
pos = [(x1, y0), (x2, y0), (x1, y0), (x2, y0)]
add_labels(grp, labels, pos, labelsize)
