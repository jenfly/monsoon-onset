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

ubudget, data = utils.calc_ubudget(files, ndays, lon1, lon2)
#
# # ----------------------------------------------------------------------
# # Read data
# data = xray.Dataset()
# for nm in files:
#     with xray.open_dataset(files[nm]) as ds:
#         data[nm] = atm.squeeze(ds[nm])
# data['PHI'] = atm.constants.g.values * data['H']
# data = data.rename({'Day' : 'day'})
#
# # Eddy decomposition
# for nm in data.data_vars:
#     print('Eddy decomposition for ' + nm)
#     comp = utils.eddy_decomp(data[nm], ndays, lon1, lon2, taxis)
#     for compnm in comp:
#         data[compnm] = comp[compnm]
#
# # ----------------------------------------------------------------------
# # Momentum budget calcs
#
# # du/dt = sum of terms in ubudget
# ubudget = xray.Dataset()
# readme = 'Momentum budget: ACCEL = sum of all other data variables'
# ubudget.attrs['readme'] = readme
# ubudget.attrs['ndays'] = ndays
# ubudget.attrs['lon1'] = lon1
# ubudget.attrs['lon2'] = lon2
#
# # Advective terms
# keypairs = [ ('AVG', 'AVG'), ('AVG', 'ST'), ('ST', 'AVG')]
# print('Computing advective terms')
# for pair in keypairs:
#     print(pair)
#     ukey, flowkey = pair
#     u = data['U_' + ukey]
#     dudp = data['DUDP_' + ukey]
#     uflow = data['U_' + flowkey]
#     vflow = data['V_' + flowkey]
#     omegaflow = data['OMEGA_' + flowkey]
#     adv = utils.advection(uflow, vflow, omegaflow, u, dudp)
#     for nm in adv.data_vars:
#         key = 'ADV_%s_%s_%s' % (ukey, flowkey, nm)
#         ubudget[key] = - adv[nm]
#
# # EMFD terms
# keys = ['TR', 'ST']
# print('Computing EMFD terms')
# for key in keys:
#     print(key)
#     u = data['U_' + key]
#     v = data['V_' + key]
#     omega = data['OMEGA_' + key]
#     dudp = data['DUDP_' + key]
#     domegadp = data['DOMEGADP_' + key]
#     emfd = utils.fluxdiv(u, v, omega, dudp, domegadp)
#     for nm in emfd.data_vars:
#         ubudget['EMFC_%s_%s' % (key, nm)] = - emfd[nm]
#
# # Coriolis terms
# latlon = utils.latlon_data(data['V_ST'])
# lat = latlon['LAT']
# f = atm.coriolis(lat)
# ubudget['COR_AVG'] = data['V_AVG'] * f
# ubudget['COR_ST'] = data['V_ST'] * f
#
# # Pressure gradient terms
# a = atm.constants.radius_earth.values
# coslat = latlon['COSLAT']
# lonrad = latlon['LONRAD']
# londim = atm.get_coord(data['PHI_ST'], 'lon', 'dim')
# ubudget['PGF_ST'] = - atm.gradient(data['PHI_ST'], lonrad, londim) / (a*coslat)
#
# # Time mean
# print('Computing rolling time mean')
# for nm in ubudget.data_vars:
#     ubudget[nm] = atm.rolling_mean(ubudget[nm], ndays, axis=taxis, center=True)
#
# # Acceleration
# nseconds = 60 * 60 * 24 * ndays
# delta_u = np.nan * data['U']
# u = data['U'].values
# delta_u.values[ndays//2:-ndays//2] = (u[ndays:] - u[:-ndays]) / nseconds
# ubudget['ACCEL'] = delta_u

# ----------------------------------------------------------------------
# Sector mean budget
lonname = atm.get_coord(ubudget, 'lon', 'name')
ubudget_sector = atm.subset(ubudget, {lonname : (lon1, lon2)})
ubudget_sector = ubudget_sector.mean(dim=lonname)

# ----------------------------------------------------------------------
# Group terms together for summary
groups = collections.OrderedDict()
groups['ADV'] = ['ADV_AVG_AVG_X', 'ADV_AVG_AVG_Y', 'ADV_AVG_AVG_P']
groups['ADV_CROSS'] = ['ADV_AVG_ST_X', 'ADV_AVG_ST_Y', 'ADV_AVG_ST_P',
                       'ADV_ST_AVG_X', 'ADV_ST_AVG_Y', 'ADV_ST_AVG_P']
groups['EMFC_TR'] = ['EMFC_TR_X', 'EMFC_TR_Y', 'EMFC_TR_P']
groups['EMFC_ST'] = ['EMFC_ST_X', 'EMFC_ST_Y', 'EMFC_ST_P']
groups['COR'] = ['COR_AVG', 'COR_ST']
groups['PGF'] = ['PGF_ST']
groups['ACCEL'] = ['ACCEL']

ubudget_summary = xray.Dataset()
for key in groups:
    nms = groups[key]
    ubudget_summary[key] = ubudget[nms[0]]
    for nm in nms[1:]:
        ubudget_summary[key] = ubudget_summary[key] + ubudget[nm]

# ----------------------------------------------------------------------
axlims = (-60, 60, 40, 120)

day = 180
for nm in ubudget_summary.data_vars:
    plt.figure()
    atm.pcolor_latlon(ubudget_summary[nm].sel(day=day), axlims=axlims)
    plt.title(nm)

season = 'JUL'
ubudget_ssn = ubudget.sel(day=atm.season_days(season)).mean(dim='day')
atm.pcolor_latlon(ubudget_ssn['PGF_ST'], axlims=axlims)

# ----------------------------------------------------------------------


day = 150
#key = 'EMFC_TR_Y'
key = 'ADV_AVG_ST_Y'
plt.figure()
atm.pcolor_latlon(ubudget[key].sel(day=day), axlims=axlims)

df = ubudget_sector.sel(day=day).drop('day').to_dataframe()
#df.plot(legend=False)
#plt.legend(fontsize=10)

df_summary = pd.DataFrame()
for key in groups:
    df_summary[key] = df[groups[key]].sum(axis=1)
df_summary.plot(legend=False)
plt.legend(fontsize=10)
plt.grid()
plt.title('Day %d' % day)

keys = groups.keys()
keys.remove('ACCEL')
df_check = pd.DataFrame()
df_check['SUM'] = df_summary[keys].sum(axis=1)
df_check['ACCEL'] = df_summary['ACCEL']
df_check.plot()

# keys = ['ADV_AVG_AVG_Y', 'ADV_ST_AVG_X', 'EMFC_TR_X', 'EMFC_TR_Y', 'EMFC_ST_X',
#         'EMFC_ST_Y']
