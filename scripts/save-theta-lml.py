import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import collections

import atmos as atm

# ----------------------------------------------------------------------
datadir = atm.homedir() + 'datastore/merra/analysis/'
years = np.arange(1979, 2015)
onset_nm = 'CHP_MFC'
latlonstr = '40E-120E_90S-90N'
varnms = ['TLML', 'QLML', 'PS']
varnms_out = ['THETA_LML', 'THETA_E_LML']
files = {}
filestr = datadir + 'merra_%s_dailyrel_' + onset_nm + '_%d.nc'
for nm in varnms + varnms_out:
    files[nm] = [filestr % (nm, yr) for yr in years]

# Calculate potential temperatures in each year and save
for y, year in enumerate(years):
    data = xray.Dataset()
    for nm in varnms:
        filenm = files[nm][y]
        print('Loading ' + filenm)
        with xray.open_dataset(filenm) as ds:
            data[nm] = ds[nm].load()
    print('Computing theta and theta_e')
    data['THETA_LML'] = atm.potential_temp(data['TLML'], data['PS'])
    data['THETA_E_LML'] = atm.equiv_potential_temp(data['TLML'], data['PS'],
                                                   data['QLML'])
    for nm in varnms_out:
        filenm = filestr % (nm, year)
        print('Saving to ' + filenm)
        atm.save_nc(filenm, data[nm])

# Save climatology
yearstr = '%d-%d' % (years.min(), years.max())
filestr2 = filestr.replace('%d.nc', '%s.nc')
for varnm in varnms_out:
    var, onset, retreat = load_dailyrel(files[varnm])
    ds = xray.Dataset()
    ds[varnm], ds['D_ONSET'], ds['D_RETREAT'] = var, onset, retreat
    print('Computing climatological mean')
    ds = ds.mean(dim='year')
    ds[varnm].attrs = var.attrs
    filenm = filestr2 % (nm, yearstr)
    print('Saving to ' + filenm)
    ds.to_netcdf(filenm)
