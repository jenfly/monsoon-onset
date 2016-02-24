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
year = 1979
datadir = atm.homedir() + 'datastore/merra/daily/'
files = collections.OrderedDict()
files['U'] = datadir + 'merra_uv200_40E-120E_60S-60N_%d.nc' % year
files['V'] = datadir + 'merra_uv200_40E-120E_60S-60N_%d.nc' % year
files['DUDP'] = datadir + 'merra_DUDP200_40E-120E_90S-90N_%d.nc' % year
files['H'] = datadir + 'merra_H200_40E-120E_60S-60N_%d.nc' % year
files['OMEGA'] = datadir + 'merra_OMEGA200_40E-120E_90S-90N_%d.nc' % year
files['DOMEGADP'] = datadir + 'merra_DOMEGADP200_40E-120E_90S-90N_%d.nc' % year

ndays = 5      # Rolling pentad
lon1, lon2 = 60, 100

# ----------------------------------------------------------------------
# Read data and calculate momentum budget

ubudget_all, data = utils.calc_ubudget(files, ndays, lon1, lon2)

# Consolidate terms together
groups = collections.OrderedDict()
groups['ADV_AVG'] = ['ADV_AVG_AVG_X', 'ADV_AVG_AVG_Y', 'ADV_AVG_AVG_P']
groups['ADV_CROSS'] = ['ADV_AVG_ST_X', 'ADV_AVG_ST_Y', 'ADV_AVG_ST_P',
                       'ADV_ST_AVG_X', 'ADV_ST_AVG_Y', 'ADV_ST_AVG_P']
groups['EMFC_TR'] = ['EMFC_TR_X', 'EMFC_TR_Y', 'EMFC_TR_P']
groups['EMFC_ST'] = ['EMFC_ST_X', 'EMFC_ST_Y', 'EMFC_ST_P']
groups['COR'] = ['COR_AVG', 'COR_ST']
groups['PGF'] = ['PGF_ST']
groups['ACCEL'] = ['ACCEL']

ubudget = xray.Dataset()
for key in groups:
    nms = groups[key]
    ubudget[key] = ubudget_all[nms[0]]
    for nm in nms[1:]:
        ubudget[key] = ubudget[key] + ubudget_all[nm]

# Tile the zonal mean value
adv_avg = atm.biggify(ubudget['ADV_AVG'], ubudget['ACCEL'], tile=True)
ubudget['ADV_AVG'] = xray.DataArray(adv_avg, coords=ubudget['COR'].coords)

# ----------------------------------------------------------------------
# Sector mean budget
lonname = atm.get_coord(ubudget, 'lon', 'name')
ubudget_sector = atm.subset(ubudget, {lonname : (lon1, lon2)})
ubudget_sector = ubudget_sector.mean(dim=lonname)

# ----------------------------------------------------------------------
axlims = (-60, 60, 40, 120)

day = 180
for nm in ubudget.data_vars:
    plt.figure()
    atm.pcolor_latlon(ubudget[nm].sel(day=day), axlims=axlims)
    plt.title(nm)

days = ubudget['day']
lat = atm.get_coord(ubudget, 'lat')
for nm in ubudget_sector.data_vars:
    plt.figure()
    utils.contourf_lat_time(lat, days, ubudget_sector[nm], nm)

df = ubudget_sector.sel(day=day).drop('day').to_dataframe()
df.plot(legend=False)
plt.legend(fontsize=10)
plt.grid()
ylims = plt.ylim()
plt.title('Day %d' % day)

keys = groups.keys()
keys.remove('ACCEL')
df_check = pd.DataFrame()
df_check['SUM'] = df[keys].sum(axis=1)
df_check['ACCEL'] = df['ACCEL']
df_check.plot()
plt.ylim(ylims)
plt.grid()
plt.title('Day %d' % day)
