import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xarray as xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
import indices
import utils

# ----------------------------------------------------------------------
# version, years = 'merra', np.arange(1979, 2015)
version, years = 'merra2', np.arange(1980, 2016)
daymin, daymax = 1, 400

datadir = atm.homedir() + 'datastore/%s/daily/' % version
savedir = atm.homedir() + 'datastore/%s/analysis/' % version
yearstr = '%d-%d.nc' % (min(years), max(years))
savefile = savedir + version + '_mfc_budget_' + yearstr
files = {}
filestr = '%d/' + version + '_%s_40E-120E_90S-90N_%d.nc'
files['MFC'] = [datadir + filestr % (yr, 'MFC', yr) for yr in years]
if version == 'merra':
    files['PRECTOT'] = [datadir + 'merra_precip_%d.nc' % yr for yr in years]
else:
    files['PRECTOT'] = [datadir + filestr % (yr, 'PRECTOT', yr) for yr in years]
files['EVAP'] = [datadir + filestr % (yr, 'EVAP', yr) for yr in years]
files['DQVDT_ANA'] = [datadir + filestr % (yr, 'DQVDT_ANA', yr) for yr in years]
files['TQV'] = [datadir + filestr % (yr, 'TQV', yr) for yr in years]

# Lat-lon box for MFC budget
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}

# ----------------------------------------------------------------------
# Read data
ts = xray.Dataset()
for nm in files:
    var = atm.combine_daily_years(nm, files[nm], years, yearname='year',
                                  subset_dict=subset_dict)
    var = atm.mean_over_geobox(var, lat1, lat2, lon1, lon2)
    units = var.attrs.get('units')
    if units in ['kg/m2/s', 'kg m-2 s-1']:
        var = atm.precip_convert(var, units, 'mm/day')
    elif units in ['kg/m2', 'kg m-2']:
        var.attrs['units'] = 'mm'
    long_name = var.attrs.get('long_name')
    var.attrs = {'units' : units, 'long_name' : long_name}
    ts[nm] = utils.wrapyear_all(var, daymin, daymax)

# dW/dt
dwdt = atm.gradient(ts['TQV'], ts['day'].values, axis=-1)
dwdt.attrs['units'] = 'mm/day'
dwdt.attrs['long_name'] = 'd/dt of ' + ts['TQV'].attrs.get('long_name')
ts['DWDT'] = dwdt
ts = ts.rename({'DQVDT_ANA' : 'DWDT_ANA'}).drop('TQV')
for nm, val in zip(['lat1', 'lat2', 'lon1', 'lon2'], [lat1, lat2, lon1, lon2]):
    ts.attrs[nm] = val

print('Saving to ' + savefile)
ts.to_netcdf(savefile)
