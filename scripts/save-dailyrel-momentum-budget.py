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
plevs = [1000,925,850,775,700,600,500,400,300,250,150,100,70,50,30,20]
datadir = atm.homedir() + 'datastore/merra/analysis/'
filestr = datadir + 'merra_ubudget%d_ndays5_60E-100E_%d.nc'
savestr = datadir + 'merra_ubudget%d_dailyrel_' + onset_nm + '_ndays5_60E-100E'
datafiles, savefiles = {}, {}
for plev in plevs:
    datafiles[plev] = [filestr % (plev, yr) for yr in years]
    savefiles[plev] = [savestr % plev + '_%d.nc' % yr for yr in years]
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

for plev in plevs:
    for y, year in enumerate(years):
        datafile = datafiles[plev][y]
        d_onset, d_retreat = onset[y], retreat[y]
        ds_rel = xray.Dataset()
        print('Loading ' + datafile)
        with xray.open_dataset(datafile) as ds:
            ds_rel.attrs = ds.attrs
            for nm in ds.data_vars:
                var = atm.expand_dims(ds[nm], yearnm, year)
                ds_rel[nm] = utils.daily_rel2onset(var, d_onset, npre, npost)
        ds_rel.attrs['d_onset'] = d_onset
        ds_rel.attrs['d_retreat'] = d_retreat
        savefile = savefiles[plev][y]
        print('Saving to ' + savefile)
        ds_rel.to_netcdf(savefile)


# ----------------------------------------------------------------------
# Compute climatologies and save

yearstr = '%d-%d' % (years.min(), years.max())
for plev in plevs:
    relfiles = savefiles[plev]
    savefile = savestr % plev + '_' + yearstr + '.nc'
    ds = atm.combine_daily_years(None, relfiles, years, yearname='year')
    ds = ds.mean(dim='year')
    ds.attrs['years'] = years
    print('Saving to ' + savefile)
    ds.to_netcdf(savefile)
