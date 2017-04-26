import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xarray as xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation
import collections
import pandas as pd

import atmos as atm
import precipdat
import merra
import indices
import utils

# ----------------------------------------------------------------------
#onset_nm = 'HOWI'
onset_nm, lrs_mean = 'CHP_MFC', 137
#onset_nm = 'CHP_PCP'

# CHP_MFC Early/Late Years
comp_yrs = collections.OrderedDict()
comp_yrs['early'] = [2004, 1999, 1990, 2000, 2001]
comp_yrs['late'] = [1983, 1992, 1997, 2014, 2012]
savestr = 'early_late'

clim_yearstr = '1979-2014'
datadir = atm.homedir() + 'datastore/merra/analysis/'
savedir = 'figs/'

varnms = ['precip', 'U200', 'V200', 'rel_vort200', 'Ro200',
           'abs_vort200', 'H200', 'T200',
           'U850', 'V850', 'H850', 'T850', 'QV850',
           'THETA950', 'THETA_E950', 'V*THETA_E950',
           'HFLUX', 'EFLUX', 'EVAP']

# Day ranges for composites
comp_keys = ['pre4', 'pre3', 'pre2', 'pre1']
#comp_keys = ['post1', 'post2', 'post3', 'post4']

compdays_all = {'pre4' : np.arange(-60, -45),
                'pre3' : np.arange(-45, -30),
                'pre2' : np.arange(-30, -15),
                'pre1' : np.arange(-15, 0),
                'post1' : np.arange(0, 15),
                'post2' : np.arange(15, 30),
                'post3' : np.arange(45, 90),
                'post4' : np.arange(0, lrs_mean)}

compdays = collections.OrderedDict()
for key in comp_keys:
    compdays[key] = compdays_all[key]

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# List of data files

def data_filenames(years, varnms, onset_nm, datadir):
    filestr = datadir + 'merra_%s_dailyrel_%s_%d.nc'
    files = {}
    for nm in varnms:
        files[nm] = [filestr % (nm, onset_nm, yr) for yr in years]
    return files

def clim_filenames(yearstr, varnms, onset_nm, datadir):
    filestr = datadir + 'merra_%s_dailyrel_%s_%s.nc'
    files = {nm : [filestr % (nm, onset_nm, yearstr)] for nm in varnms}
    return files

datafiles = {key: data_filenames(comp_yrs[key], varnms, onset_nm, datadir)
             for key in comp_yrs}
datafiles['clim'] = clim_filenames(clim_yearstr, varnms, onset_nm, datadir)


# ----------------------------------------------------------------------
# Get daily data relative to onset day

data = {}
for key in datafiles:
    data[key] = {}
    for nm in datafiles[key]:
        var, _, _ = utils.load_dailyrel(datafiles[key][nm])
        if 'year' in var.dims:
            var = var.mean(dim='year')
        data[key][nm] = var


# ----------------------------------------------------------------------
# Housekeeping

# Fill Ro200 with NaNs near equator
varnm = 'Ro200'
for key1 in data:
    if varnm in data[key1]:
        latbuf = 5
        lat = atm.get_coord(data[key1][varnm], 'lat')
        latbig = atm.biggify(lat, data[key1][varnm], tile=True)
        vals = data[key1][varnm].values
        vals = np.where(abs(latbig)>latbuf, vals, np.nan)
        data[key1][varnm].values = vals

# ----------------------------------------------------------------------
# Sector mean data

lonname, latname = 'XDim', 'YDim'
sectordata = {}
for key1 in data:
    sectordata[key1] = collections.OrderedDict()
    for varnm in data[key1]:
        var = atm.subset(data[key1][varnm], {lonname : (lon1, lon2)})
        sectordata[key1][varnm] = var.mean(dim=lonname)

# ----------------------------------------------------------------------
# Plotting params and utilities

def plusminus(num):
    if num == 0:
        numstr = '+0'
    else:
        numstr = atm.format_num(num, ndecimals=0, plus_sym=True)
    return numstr

# ----------------------------------------------------------------------
# Composite averages
print('Computing composites relative to onset day')
comp = {}
for nm in varnms:
    comp[nm] = {}
    for key in data:
        compdat = utils.composite(sectordata[key][nm], compdays_all,
                                  daynm='dayrel', return_avg=True)
        for dkey in compdat:
            key2 = key + '_' + dkey
            comp[nm][key2] = compdat[dkey]

# Get max/min values from all composites for setting consistent
# ylimits on plots
ylimits = {}
for nm in varnms:
    for i, key in enumerate(comp[nm]):
        val1 = comp[nm][key].min().values
        val2 = comp[nm][key].max().values
        if i == 0:
            ylim1, ylim2 = val1, val2
        else:
            ylim1, ylim2 = min([ylim1, val1]), max([ylim2, val2])
    # Add a bit of buffer space
    delta = (ylim2 - ylim1) * 0.05
    ylim1, ylim2 = ylim1 - delta, ylim2 + delta
    if nm == 'precip':
        ylim1 = 0
    ylimits[nm] = (ylim1, ylim2)

# ----------------------------------------------------------------------
# Line plots of sector data
compnms = {}
for key in compdays:
    d1 = plusminus(compdays[key].min())
    d2 = plusminus(compdays[key].max())
    compnms[key] = 'D0%s:D0%s' % (d1, d2)

suptitle = '%d-%dE Composites Relative to %s Onset Day' % (lon1, lon2, onset_nm)
ncol = len(compdays.keys())
nrow = 4
figsize = (12, 9)
xlimits = (-60, 60)
fmt = {'early' : 'k--', 'late' : 'k', 'clim' : 'k'}
lwidth = {'early' : 1, 'late' : 1, 'clim' : 2}
alph = {'early' : 1, 'late' : 1, 'clim' : 0.35}
if 'pre1' in compdays:
    legend_loc = 'upper right'
else:
    legend_loc = 'upper left'

for i, nm in enumerate(varnms):
    if i % nrow == 0:
        row = 1
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True,
                                 sharey='row')
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        plt.suptitle(suptitle)
    else:
        row += 1
    for j, dkey in enumerate(compdays):
        ax = axes[row - 1, j]
        for yrkey in ['clim'] + comp_yrs.keys():
            var = comp[nm][yrkey + '_' + dkey]
            lat = atm.get_coord(var, 'lat')
            ax.plot(lat, var, fmt[yrkey], linewidth=lwidth[yrkey],
                    alpha=alph[yrkey], label=yrkey)
        ax.grid(True)
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits[nm])
        if row == 1:
            ax.set_title(compnms[dkey])
        if row == nrow:
            ax.set_xlabel('Latitude')
        if j == 0:
            ax.set_ylabel(nm)
        if i == 0 and j == 0:
            ax.legend(fontsize=10, loc=legend_loc)
