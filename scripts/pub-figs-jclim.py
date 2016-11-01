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
yearstr = '1980-2015'
datadir = atm.homedir() + 'datastore/%s/figure_data/' % version
pcp_nm = 'GPCP'
ind_nm = 'onset'
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
npre, npost =  120, 200

datafiles = {}
datafiles['ubudget'] = datadir + 'merra2_ubudget_1980-2014_excl.nc'
filestr = datadir + version + '_%s_' + yearstr + '.nc'
for nm in ['latp', 'hov', 'latlon', 'tseries', 'psi_comp']:
    datafiles[nm] = filestr % nm
datafiles['gpcp'] = datadir + 'gpcp_dailyrel_1997-2015.nc'
datafiles['index'] = filestr % 'index_CHP_MFC'

# ----------------------------------------------------------------------
# Read data

data = {}
for nm in datafiles:
    print('Loading ' + datafiles[nm])
    with xray.open_dataset(datafiles[nm]) as ds:
        data[nm] = ds.load()

tseries = data['tseries']
index = data['index']
index['length'] = index['retreat'] - index['onset']

data_hov = {nm : data['hov'][nm] for nm in data['hov'].data_vars}
data_hov['GPCP'] = data['gpcp']['PCP_SECTOR']

# ----------------------------------------------------------------------
# Plotting functions and other utilities

def get_varnm(nm):
    varnms = {'U200' : 'U', 'V200' : 'V', 'T200' : 'T', 'TLML' : 'T',
              'QLML' : 'Q', 'THETA_E_LML' : 'THETA_E'}
    return varnms.get(nm)

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
    if var.min < 0:
        extend, symmetric = 'both', True
    else:
        extend, symmetric = 'max', False
    if var.name.startswith('PCP'):
        cmap = 'PuBuGn'
    else:
        cmap = 'RdBu_r'
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

# Overview map and timeseries plots - setup figure for subplots
nrow, ncol = 2, 2
fig_kw = {'figsize' : (figwidth, 0.7 * figwidth)}
gridspec_kw = {'left' : 0.07, 'right' : 0.9, 'bottom' : 0.07, 'top' : 0.9,
               'wspace' : 0.5, 'hspace' : 0.35}
legend = True
legend_kw = {'loc' : 'upper left', 'framealpha' : 0.0}
labelpos = (-0.2, 1.05)
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)

# Overview map
grp.next()
m = atm.init_latlon(0, 35, 58, 102, resolution='l', coastlines=False,
                    fillcontinents=True)
m.drawcoastlines(linewidth=0.5, color='0.5')
#plot_kerala(linewidth=1)
x = [lon1, lon1, lon2, lon2, lon1]
y = [lat1, lat2, lat2, lat1, lat1]
plt.plot(x, y, color='b', linewidth=2)


# Plot daily tseries
legend = True
dlist = [15]
opts = []
opts.append({'keys1' : ['MFC', pcp_nm], 'keys2' : ['CMFC'],
             'units1' : 'mm day$^{-1}$', 'units2' : 'mm',
             'ylims' : (-3.5, 9), 'legend_loc' : 'upper left' })
opts.append({'keys1' : ['U200_0N'], 'keys2' : ['V200_15N'],
             'units1' : '   m s$^{-1}$', 'units2' :  '   m s$^{-1}$',
             'ylims' : (-20, 0), 'legend_loc' : 'lower left' })
opts.append({'keys1' : ['T200_30N'], 'keys2' : ['T200_30S'],
             'units1' : '   K', 'units2' :  '   K',
             'ylims' : (218, 227), 'legend_loc' : 'upper left' })
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

keys = [pcp_nm, 'U200', 'PSI500', 'T200', 'THETA_E_LML', 'TLML', 'QLML']
clevs = {pcp_nm : 1, 'U200' : 5, 'V200' : 1, 'PSI500' : 5, 'T200' : 1,
         'THETA_E_LML' : 1, 'TLML' : 1, 'QLML' : 5e-4}
nrow, ncol = 2, 2
fig_kw = {'figsize' : (figwidth, 0.64 * figwidth), 'sharex' : True,
          'sharey' : True}
gridspec_kw = {'left' : 0.07, 'right' : 0.99, 'bottom' : 0.07, 'top' : 0.94,
               'wspace' : 0.05}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for key in keys:
    grp.next()
    var = data_hov[key]
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
