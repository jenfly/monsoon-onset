import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
import indices
import utils

# ----------------------------------------------------------------------
datadir = atm.homedir() + 'datastore/merra/daily/'
filename1 = datadir + 'merra_uv200_40E-120E_60S-60N_1979.nc'
filename2 = datadir + 'merra_DUDP200_40E-120E_90S-90N_1979.nc'
filename3 = datadir + 'merra_H200_40E-120E_60S-60N_1979.nc'

nt = 5      # Rolling pentad
lon1, lon2 = 60, 100
taxis = 0

# ----------------------------------------------------------------------
# Read data
data = xray.Dataset()
with xray.open_dataset(filename1) as ds:
    data['U'] = atm.squeeze(ds['U'])
    data['V'] = atm.squeeze(ds['V'])
with xray.open_dataset(filename2) as ds:
    data['DUDP'] = atm.squeeze(ds['DUDP'])
with xray.open_dataset(filename3) as ds:
    data['PHI'] = atm.constants.g.values * atm.squeeze(ds['H'])
data['OMEGA'] = np.nan * data['U']
data['DOMEGADP'] = np.nan * data['U']
data = data.rename({'Day' : 'day'})

# Eddy decomposition
for nm in data.data_vars:
    print('Eddy decomposition for ' + nm)
    comp = utils.eddy_decomp(data[nm], nt, lon1, lon2, taxis)
    for compnm in comp:
        data[compnm] = comp[compnm]

# ----------------------------------------------------------------------
# Momentum budget calcs

# du/dt = sum of terms in ubudget
ubudget = xray.Dataset()

# Advective terms
keypairs = [ ('AVG', 'AVG'), ('AVG', 'ST'), ('ST', 'AVG')]
print('Computing advective terms')
for pair in keypairs:
    print(pair)
    ukey, flowkey = pair
    u = data['U_' + ukey]
    dudp = data['DUDP_' + ukey]
    uflow = data['U_' + flowkey]
    vflow = data['V_' + flowkey]
    omegaflow = data['OMEGA_' + flowkey]
    adv = utils.advection(uflow, vflow, omegaflow, u, dudp)
    for nm in adv.data_vars:
        key = 'ADV_%s_%s_%s' % (ukey, flowkey, nm)
        ubudget[key] = - adv[nm]

# EMFD terms
keys = ['TR', 'ST']
print('Computing EMFD terms')
for key in keys:
    print(key)
    u = data['U_' + key]
    v = data['V_' + key]
    omega = data['OMEGA_' + key]
    dudp = data['DUDP_' + key]
    domegadp = data['DOMEGADP_' + key]
    emfd = utils.fluxdiv(u, v, omega, dudp, domegadp)
    for nm in emfd.data_vars:
        ubudget['EMFC_%s_%s' % (key, nm)] = - emfd[nm]

# Coriolis terms
latname = atm.get_coord(data, 'lat', 'name')
lat = data[latname]
f = atm.coriolis(lat)
ubudget['COR_AVG'] = data['V_AVG'] * f
ubudget['COR_ST'] = data['V_ST'] * f

# Pressure gradient terms
a = atm.constants.radius_earth.values
coslat = np.cos(lat)
lonrad = np.radians(atm.get_coord(data, 'lon'))
londim = atm.get_coord(data['PHI_ST'], 'lon', 'dim')
ubudget['PGF_ST'] = - atm.gradient(data['PHI_ST'], lonrad, londim) / (a*coslat)

# Acceleration


# Time mean
print('Computing rolling time mean')
for nm in ubudget.data_vars:
    ubudget[nm] = atm.rolling_mean(ubudget[nm], nt, axis=taxis, center=True)

# ----------------------------------------------------------------------
# Sector mean budget
lonname = atm.get_coord(ubudget, 'lon', 'name')
ubudget_sector = atm.subset(ubudget, {lonname : (lon1, lon2)})
ubudget_sector = ubudget_sector.mean(dim=lonname)

# ----------------------------------------------------------------------
axlims = (-60, 60, 40, 120)

day = 150
#key = 'EMFC_TR_Y'
key = 'ADV_AVG_ST_Y'
plt.figure()
atm.pcolor_latlon(ubudget[key].sel(day=day), axlims=axlims)

df = ubudget_sector.sel(day=day).drop('day').to_dataframe()
df.plot(legend=False)
plt.legend(fontsize=10)

keys = ['ADV_AVG_AVG_Y', 'ADV_ST_AVG_X', 'EMFD_TR_X', 'EMFD_TR_Y', 'EMFD_ST_X',
        'EMFD_ST_Y']
df[keys].plot(legend=False)
plt.legend(fontsize=10)
