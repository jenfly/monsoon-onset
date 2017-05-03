import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')
sys.path.append('/home/jwalker/dynamics/python/monsoon-onset')

import numpy as np
import xarray as xray
import pandas as pd
import matplotlib.pyplot as plt
import collections

import atmos as atm
import merra
import indices
import utils

# ----------------------------------------------------------------------
version = 'merra2'
onset_nm = 'CHP_MFC'
datadir1 = atm.homedir() + 'datastore/%s/analysis/' % version
datadir2 = atm.homedir() + 'datastore/%s/analysis/' % version
yearstr = '1980-2015'
plevs = [1000,925,850,775,700,600,500,400,300,250,200,150,100,70,50,30,20]
lon1, lon2 = 60, 100
ndays = 5     # n-day rolling mean for smoothing
scale = 1e-4  # Scaling factor for all terms in momentum budget
ind_nm = 'onset'
# ind_nm = 'retreat


filestr = (datadir1 + version + '_ubudget%d_dailyrel_' + onset_nm +
           '_ndays%d_%dE-%dE_%s.nc' % (ndays, lon1, lon2, yearstr))
if ind_nm == 'retreat':
    filestr = filestr.replace('dailyrel', 'dailyrel_retreat')
datafiles = [filestr % plev for plev in plevs]
savefile_ubudget = datafiles[0].replace('%d' % plevs[0], '_sector')

# ----------------------------------------------------------------------
# Read ubudget data and save

def biggify(small, big):
    vals = atm.biggify(small, big, tile=True)
    var = xray.DataArray(vals, coords=big.coords, dims=big.dims)
    return var


def consolidate(ds):
    # Consolidate terms in ubudget
    groups = collections.OrderedDict()
    groups['ADV_AVG'] = ['ADV_AVG_AVG_X', 'ADV_AVG_AVG_Y', 'ADV_AVG_AVG_P']
    groups['ADV_AVST'] = ['ADV_AVG_ST_X', 'ADV_AVG_ST_Y', 'ADV_AVG_ST_P']
    groups['ADV_STAV'] = ['ADV_ST_AVG_X', 'ADV_ST_AVG_Y', 'ADV_ST_AVG_P']
    groups['ADV_CRS'] = ['ADV_AVST', 'ADV_STAV']
    groups['EMFC_TR'] = ['EMFC_TR_X', 'EMFC_TR_Y', 'EMFC_TR_P']
    groups['EMFC_ST'] = ['EMFC_ST_X', 'EMFC_ST_Y', 'EMFC_ST_P']

    groups['EMFC'] = ['EMFC_TR', 'EMFC_ST']
    groups['COR'] = ['COR_AVG', 'COR_ST']
    groups['ADV+COR'] = ['ADV_AVG', 'COR_AVG']
    groups['DMDY'] = ['ADV_AVG_AVG_Y', 'COR_AVG']
    groups['SUM'] = ['ADV_AVG', 'ADV_CRS', 'EMFC', 'COR', 'PGF_ST', 'ANA']

    print('Consolidating ubudget terms')
    big = ds['PGF_ST']
    
    for key in groups:
        print(key)
        nms = groups[key]
        ds[key] = biggify(ds[nms[0]], big)
        
        for nm in nms[1:]:
            #ds[key] = ds[key] + ds[nm]
            
            # Need to use xray.concat() and .sum() to handle NaNs properly
            var0 = biggify(ds[nm], big)
            var = xray.concat((ds[key], var0), dim='concat_dim')
            ds[key] = var.sum(dim='concat_dim')
            
        ds[key].attrs['group'] = nms

    return ds


def process_file(filenm, plev, lon1, lon2, pname='Height'):
    # Process data for a single file (one pressure level)
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        ds = consolidate(ds)
        print('Computing sector mean')
        ds = atm.dim_mean(ds, 'lon', lon1, lon2)
        ds.load()
    for nm in ds.data_vars:
        ds[nm] = atm.expand_dims(ds[nm], pname, plev, axis=1)
    return ds

# Read ubudget at each plevel and concatenate
for i, filenm in enumerate(datafiles):
    plev = plevs[i]
    ds = process_file(filenm, plev, lon1, lon2)
    if i == 0:
        ubudget = ds
    else:
        ubudget = xray.concat([ubudget, ds], dim='Height')

ubudget['Height'].attrs['units'] = 'hPa'

# Apply scaling
ubudget = ubudget / scale
ubudget.attrs['units'] = '%.0e m/s2' % scale
for nm in ubudget.data_vars:
    ubudget[nm].attrs['units'] = '%.0e m/s2' % scale

# Additional metadata
ubudget.attrs['scale'] = scale
ubudget.attrs['ndays'] = ndays
ubudget.attrs['lon1'] = lon1
ubudget.attrs['lon2'] = lon2

# Save to file
ubudget.to_netcdf(savefile_ubudget)
