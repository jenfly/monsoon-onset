import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import collections

import atmos as atm
import utils

# ----------------------------------------------------------------------
version = 'merra2'
years = np.arange(1980, 2015)
onset_nm = 'CHP_MFC'
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
latlonstr = '40E-120E_90S-90N'
varnms = ['TLML', 'QLML', 'PS']
varnms_out = ['THETA_LML', 'THETA_E_LML']
files = {}
filestr = datadir + version + '_%s_dailyrel_' + onset_nm + '_%d.nc'
for nm in varnms + varnms_out:
    files[nm] = [filestr % (nm, yr) for yr in years]

# Calculate potential temperatures in each year and save
for y, year in enumerate(years):
    data = xray.Dataset()
    for nm in varnms:
        filenm = files[nm][y]
        print('Loading ' + filenm)
        with xray.open_dataset(filenm) as ds:
            ds.load()
        for key in [nm, 'D_ONSET', 'D_RETREAT']:
            data[key] = ds[key]
    print('Computing theta and theta_e')
    data['THETA_LML'] = atm.potential_temp(data['TLML'], data['PS'])
    data['THETA_E_LML'] = atm.equiv_potential_temp(data['TLML'], data['PS'],
                                                   data['QLML'])
    for nm in varnms_out:
        filenm = filestr % (nm, year)
        print('Saving to ' + filenm)
        atm.save_nc(filenm, data[nm], data['D_ONSET'], data['D_RETREAT'])

# Save climatology
yearstr = '%d-%d' % (years.min(), years.max())
filestr2 = filestr.replace('%d.nc', yearstr + '.nc')
for varnm in varnms_out:
    var, _, _ = utils.load_dailyrel(files[varnm])
    print('Computing climatological mean')
    var = var.mean(dim='year')
    filenm = filestr2 % nm
    print('Saving to ' + filenm)
    atm.save_nc(filenm, var)
