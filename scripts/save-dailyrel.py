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

def housekeeping(data, keys_remove):
    # Remove data that I don't want to include in plots
    keys = data.keys()
    for key in keys_remove:
        if key in keys:
            data = atm.odict_delete(data, key)

    # Fill Ro200 with NaNs near equator
    varnm = 'Ro200'
    if varnm in data:
        latbuf = 5
        lat = atm.get_coord(data[varnm], 'lat')
        latbig = atm.biggify(lat, data[varnm], tile=True)
        vals = data[varnm].values
        vals = np.where(abs(latbig)>latbuf, vals, np.nan)
        data[varnm].values = vals
    return data


savestr = savedir + 'merra_%s_dailyrel_%s_%d.nc'
yearnm, daynm = 'year', 'day'

for y, year in enumerate(years):
    files = {key : [datafiles[key][y]] for key in datafiles}
    d_onset = int(onset[y].values)
    data = {}
    for varnm in varnms:        
        print('Reading daily data for ' + varnm)
        data[varnm] = utils.get_data_rel(varnm, year, files.get(varnm), data,
                                         d_onset, npre, npost)
    data = housekeeping(data, keys_remove)
    for varnm in data:
        savefile = get_savefile(savedir, varnm, onset_nm, year)
        print('Saving to ' + savefile)
        atm.save_nc(savefile, data[varnm])



