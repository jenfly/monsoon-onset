import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
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
onset_nm = 'CHP_MFC'

years = np.arange(1979, 2015)

datadir = atm.homedir() + 'datastore/merra/daily/'
savedir = atm.homedir() + 'datastore/merra/analysis/'

# Number of days before and after onset to include
npre, npost = 120, 200

varnms = ['precip', 'U200', 'V200', 'rel_vort200', 'Ro200',
           'abs_vort200', 'H200', 'T200',
           'U850', 'V850', 'H850', 'T850', 'QV850',
           'T950', 'H950', 'QV950', 'V950', 'THETA950', 'THETA_E950',
           'V*THETA_E950']

keys_remove = ['T950', 'H950', 'QV950', 'V950',  'DSE950',
                'MSE950', 'V*DSE950', 'V*MSE950']

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# List of data files

def yrlyfile(var, plev, year, subset=''):
    return 'merra_%s%d_40E-120E_60S-60N_%s%d.nc' % (var, plev, subset, year)

def get_filenames(years, datadir):
    datafiles = {}
    datafiles['HOWI'] = ['merra_vimt_ps-300mb_%d.nc' % yr for yr in years]
    datafiles['CHP_MFC'] = ['merra_MFC_ps-300mb_%d.nc' % yr for yr in years]
    datafiles['CHP_PCP'] = ['merra_precip_%d.nc' % yr for yr in years]
    datafiles['precip'] = datafiles['CHP_PCP']

    for plev in [200, 850]:
        files = [yrlyfile('uv', plev, yr) for yr in years]
        for key in ['U', 'V', 'Ro', 'rel_vort', 'abs_vort']:
            datafiles['%s%d' % (key, plev)] = files
        for key in ['T', 'H', 'QV']:
            key2 = '%s%d' % (key, plev)
            datafiles[key2] = [yrlyfile(key, plev, yr) for yr in years]

    for plev in [950, 975]:
        for key in ['T', 'H','QV', 'V']:
            files = [yrlyfile(key, plev, yr) for yr in years]
            datafiles['%s%d' % (key, plev)] = files

    for varnm in datafiles:
        files = datafiles[varnm]
        datafiles[varnm] = [datadir + filenm for filenm in files]

    return datafiles

datafiles = get_filenames(years, datadir)


# ----------------------------------------------------------------------
# Onset index for each year

index = utils.get_onset_indices(onset_nm, datafiles[onset_nm], years)
onset = index['onset']

# ----------------------------------------------------------------------
# Get daily data

def get_savefile(savedir, varnm, onset_nm, year):
    return savedir + 'merra_%s_dailyrel_%s_%d.nc' % (varnm, onset_nm, year)

year = 1979
varnm = 'U200'
savestr = savedir + 'merra_%s_dailyrel_%s_%d.nc'
yearnm, daynm = 'year', 'day'

for y, year in enumerate(years):
    files = {key : [datafiles[key][y]] for key in datafiles}
    d_onset = onset[y]
    data = {}
    for varnm in varnms:
        savefile = get_savefile(savedir, varnm, onset_nm, year)
        print('Reading daily data for ' + varnm)
        data[varnm] = utils.get_data_rel(varnm, [year], files.get(varnm), data,
                                         d_onset, npre, npost)


def all_data(onset_nm, varnms, years, datafiles, npre, npost):

    # Read daily data fields and align relative to onset day
    yearnm, daynm = 'year', 'day'
    data = collections.OrderedDict()
    for varnm in varnms:
        print('Reading daily data for ' + varnm)
        var = utils.get_data_rel(varnm, years, datafiles, data, onset, npre,
                                 npost, yearnm, daynm)
        if utils.var_type(varnm) == 'basic':
            print('Aligning data relative to onset day')
            data[varnm] = utils.daily_rel2onset(var, onset, npre, npost,
                                                yearnm=yearnm, daynm=daynm)
        else:
            data[varnm] = var
    return index, data

data = collections.OrderedDict()
index = collections.OrderedDict()
for key in comp_yrs:
    index[key], data[key] = all_data(onset_nm, varnms, comp_yrs[key],
                                     datafiles[key], npre, npost)

    # Mean over years for each composite
    for nm in data[key]:
        data[key][nm] = data[key][nm].mean(dim='year')

# ----------------------------------------------------------------------
# Housekeeping

# Remove data that I don't want to include in plots
for key1 in comp_yrs:
    keys = data[key1].keys()
    for key in keys_remove:
        if key in keys:
            data[key1] = atm.odict_delete(data[key1], key)


# Fill Ro200 with NaNs near equator
varnm = 'Ro200'
for key1 in comp_yrs:
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
sectordata = collections.OrderedDict()
for key1 in comp_yrs:
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
nms =  data[data.keys()[0]].keys()

comp = collections.OrderedDict()
sectorcomp = collections.OrderedDict()
for nm in nms:
    comp[nm] = collections.OrderedDict()
    sectorcomp[nm] = collections.OrderedDict()
    for key in data:
        compdat = utils.composite(data[key][nm], compdays_all, daynm='dayrel',
                                  return_avg=True)
        compsec = utils.composite(sectordata[key][nm], compdays_all,
                                  daynm='dayrel', return_avg=True)
        for dkey in compdat:
            key2 = key + '_' + dkey
            comp[nm][key2] = compdat[dkey]
            sectorcomp[nm][key2] = compsec[dkey]

# Get max/min values from all composites for setting consistent
# ylimits on plots
ylimits = {}
for nm in nms:
    for i, key in enumerate(sectorcomp[nm]):
        val1 = sectorcomp[nm][key].min().values
        val2 = sectorcomp[nm][key].max().values
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

ncol = len(compdays.keys())
nrow = 4
figsize = (12, 9)
suptitle = '%d-%dE Composites Relative to %s Onset Day' % (lon1, lon2, onset_nm)

fmt_str = {'early' : 'k--', 'late' : 'k'}
for i, nm in enumerate(sectorcomp):
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
        ### Temporary - put real climatology in here
        lat = atm.get_coord(sectorcomp[nm]['early_' + dkey], 'lat')
        varbar = 0.5 * (sectorcomp[nm]['early_' + dkey] +
                        sectorcomp[nm]['late_' + dkey])
        ax.plot(lat, varbar, color='k', linewidth=2, alpha=0.3, label='clim')
        ###
        for yrkey in comp_yrs:
            var = sectorcomp[nm][yrkey + '_' + dkey]
            lat = atm.get_coord(var, 'lat')
            ax.plot(lat, var, fmt_str[yrkey], label=yrkey)
        ax.grid(True)
        ax.set_ylim(ylimits[nm])
        if row == 1:
            ax.set_title(compnms[dkey])
        if row == nrow:
            ax.set_xlabel('Latitude')
        if j == 0:
            ax.set_ylabel(nm)
        if i == 0 and j == 0:
            ax.legend(fontsize=8, loc='upper left')
