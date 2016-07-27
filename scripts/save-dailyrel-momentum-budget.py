import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')
sys.path.append('/home/jwalker/dynamics/python/monsoon-onset')

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
version = 'merra2'
years = np.arange(1980, 2016)
onset_nm = 'CHP_MFC'
plevs = [1000,925,850,775,700,600,500,400,300,250,200,150,100,70,50,30,20]
ind_nm, npre, npost = 'onset', 140, 230
#ind_nm, npre, npost = 'retreat', 270, 100

datadir = atm.homedir() + 'datastore/%s/analysis/' % version
savedir = atm.homedir() + 'eady/datastore/%s/analysis/' % version
filestr = datadir + version + '_ubudget%d_ndays5_60E-100E_%d.nc'
savestr = (savedir + version + '_ubudget%d_dailyrel_' + onset_nm +
           '_ndays5_60E-100E')
datafiles, savefiles = {}, {}
for plev in plevs:
    datafiles[plev] = [filestr % (plev, yr) for yr in years]
    savefiles[plev] = [savestr % plev + '_%d.nc' % yr for yr in years]
yearstr = '%d-%d' % (min(years), max(years))
indfile = savedir + version + '_index_%s_%s.nc' % (onset_nm, yearstr)

# ----------------------------------------------------------------------
# Onset index for each year

with xray.open_dataset(indfile) as index:
    index.load()
onset = index['onset'].values
retreat = index['retreat'].values

# ----------------------------------------------------------------------
# Get daily data

def get_data(datafile, year, d0, npre, npost):
    daymin, daymax = d0 - npre, d0 + npost
    if daymin < 1:
        file_pre = datafile.replace(str(year), str(year - 1))
    else:
        file_pre = None
    if daymax > len(atm.season_days('ANN', year)):
        file_post = datafile.replace(str(year), str(year + 1))
    else:
        file_post = None
    

for plev in plevs:
    for y, year in enumerate(years):
        datafile = datafiles[plev][y]
        d_onset, d_retreat = onset[y], retreat[y]
        d0 = index[ind_nm][y].values

        ds_rel = xray.Dataset()
        print('Loading ' + datafile)
        with xray.open_dataset(datafile) as ds:
            ds_rel.attrs = ds.attrs
            for nm in ds.data_vars:
                var = atm.expand_dims(ds[nm], 'year', year)
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

# ----------------------------------------------------------------------
# Concatenate plevels in climatology and save

files = [savestr % plev + '_' + yearstr + '.nc' for plev in plevs]
ubudget = xray.Dataset()
pname, pdim = 'Height', 1
subset_dict = {'lat' : (-60, 60), 'lon' : (40, 120)}
for i, plev in enumerate(plevs):
    filenm = files[i]
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        ds = atm.subset(ds, subset_dict)
        ds.load()
    for nm in ds.data_vars:
        ds[nm] = atm.expand_dims(ds[nm], pname, plev, axis=pdim)
    if i == 0:
        ubudget = ds
    else:
        ubudget = xray.concat([ubudget, ds], dim=pname)
ubudget.coords[pname].attrs['units'] = 'hPa'
savefile = files[0]
savefile = savefile.replace('%d' % plevs[0], '')
print('Saving to ' + savefile)
ubudget.to_netcdf(savefile)
