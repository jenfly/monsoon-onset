import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import collections

import atmos as atm
import merra
import indices
import utils

# ----------------------------------------------------------------------
onset_nm = 'CHP_MFC'
years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/analysis/'
filestr = datadir + 'merra_ubudget200_ndays5_60E-100E_%d.nc'
savestr = datadir + 'merra_ubudget200_dailyrel_%s_ndays5_60E-100E_%d.nc'
datafiles = [filestr % yr for yr in years]
savefiles = [savestr % (onset_nm, yr) for yr in years]
subset1 = '40E-120E_90S-90N'
indfiles = ['%sdatastore/merra/daily/merra_MFC_%s_%d.nc' %
            (atm.homedir(), subset1, yr) for yr in years]

# Number of days before and after onset to include
npre, npost = 120, 200

# ----------------------------------------------------------------------
# Onset index for each year

index = utils.get_onset_indices(onset_nm, indfiles, years)
onset = index['onset'].values
retreat = index['retreat'].values

# ----------------------------------------------------------------------
# Get daily data
yearnm, daynm = 'year', 'day'

for y, year in enumerate(years):
    datafile = datafiles[y]
    d_onset, d_retreat = onset[y], retreat[y]
    ds_rel = xray.Dataset()
    print('Loading ' + datafile)
    with xray.open_dataset(datafiles[y]) as ds:
        ds_rel.attrs = ds.attrs
        for nm in ds.data_vars:
            var = atm.expand_dims(ds[nm], yearnm, year)
            ds_rel[nm] = utils.daily_rel2onset(var, d_onset, npre, npost,
                                               daynm, yearnm)
    ds_rel.attrs['d_onset'] = d_onset
    ds_rel.attrs['d_retreat'] = d_retreat
    savefile = savefiles[y]
    print('Saving to ' + savefile)
    ds_rel.to_netcdf(savefile)
