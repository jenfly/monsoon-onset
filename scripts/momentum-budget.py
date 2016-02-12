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
    adv = advection(uflow, vflow, omegaflow, u, dudp)
    for nm in adv.data_vars:
        key = 'ADV_%s_%s_%s' % (ukey, flowkey, nm)
        ubudget[key] = - adv[nm]

# EMFD terms



# ----------------------------------------------------------------------
# Sector mean budget
lonname = atm.get_coord(ubudget, 'lon', 'name')
ubudget_sector = atm.subset(ubudget, {lonname : (lon1, lon2)})
ubudget_sector = ubudget_sector.mean(dim=lonname)

day = 150
df = ubudget_sector.sel(day=day).drop('day').to_dataframe()
df.plot(legend=False)
plt.legend(fontsize=10)
