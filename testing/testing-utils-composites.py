import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import atmos as atm
import precipdat
import merra
import indices
from utils import days_premidpost, composite_premidpost


# ----------------------------------------------------------------------
# Define monsoon onset index

datadir = atm.homedir() + 'datastore/cmap/'
cmap_file = datadir + 'cmap.precip.pentad.mean.nc'

# Read CMAP data and average over box
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
titlestr = 'CMAP %d-%dE, %d-%dN ' % (lon1, lon2, lat1, lat2)
precip = precipdat.read_cmap(cmap_file)
precipbar = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)
nyears, npentad = precipbar.shape
years = precipbar.year.values

# Smooth with truncated Fourier series and calculate onset index
# with Wang & LinHo method
kmax = 12
threshold = 5.0
pcp_sm, Rsq = atm.fourier_smooth(precipbar, kmax)
p_onset = np.zeros(nyears)
for y, year in enumerate(years):
    p_onset[y], _, _ = indices.onset_WLH_1D(pcp_sm[y])

# Convert onset index from pentad to day of year
d_onset = np.zeros(nyears)
for y in range(nyears):
    d_onset[y] = atm.pentad_to_jday(p_onset[y], pmin=0)

# ----------------------------------------------------------------------
# Read daily mean MERRA data

datadir = atm.homedir() + 'datastore/merra/daily/'
filestr = datadir + 'merra_uv200_40E-120E_60S-60N_%d.nc'
files = []
for yr in years:
    files.append(filestr % yr)

ds = atm.combine_daily_years(['U', 'V', 'Ro'], files, years)
u = ds['U']
v = ds['V']
Ro = ds['Ro']



# ndays = 10
# varlist = ['U', 'V', 'Ro']
# comp = {}
# for y, yr in enumerate(years):
#     filn = filestr % yr
#     print('Loading ' + filn)
#     with xray.open_dataset(filn) as ds:
#         for varnm in varlist:
#             print(varnm)
#             var = ds[varnm]
#             var.coords['Year'] = yr
#             comp_in = composite_premidpost(var, d_onset[y], ndays)
#             if y == 0:
#                 comp[varnm] = comp_in
#             else:
#                 for key in comp_in:
#                     comp[varnm] = xray.concat((comp[varnm][key], comp_in[key]),
#                                               dim='Year')
# print('Done!')
