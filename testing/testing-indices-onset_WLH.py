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
from indices import onset_WLH, onset_WLH_1D


datadir = '/home/jennifer/datastore/cmap/'
#datadir = '/home/jwalker/datastore/cmap/'
cmap_file = datadir + 'cmap.precip.pentad.mean.nc'

# Years to include
yearmin = 1979
yearmax = 1997

# Calculate each year individually or use climatological precip
climatology = True

# Smoothing parameters
kmax = 12
kann = 4

# Threshold for onset criteria
threshold = 5.0

cmap = precipdat.read_cmap(cmap_file, yearmin, yearmax)
lat = atm.get_coord(cmap, 'lat')
lon = atm.get_coord(cmap, 'lon')

if climatology:
    precip = cmap.mean(dim='year')
    axis = 0
else:
    precip = cmap
    axis = 1

# Single grid point
sz = 8
lat0, lon0 = 11.25, 91.25
latval, ilat0 = atm.find_closest(precip.lat, lat0)
lonval, ilon0 = atm.find_closest(precip.lon, lon0)
pcp = precip[:, ilat0, ilon0]
pcp_sm = atm.fourier_smooth(pcp, kmax)
pcp_ann = atm.fourier_smooth(pcp, kann)
i_onset, i_retreat, i_peak = onset_WLH_1D(pcp_sm, threshold)
i_onset = int(i_onset)
i_retreat = int(i_retreat)
i_peak = int(i_peak)
plt.figure()
plt.plot(pcp, color='grey', label='raw')
plt.plot(pcp_ann, color='red', label='annual')
plt.plot(pcp_sm, color='blue', label='smoothed')
plt.plot(i_onset, pcp_sm[i_onset], 'bd', markersize=sz, label='onset')
plt.plot(i_peak, pcp_sm[i_peak], 'r*', markersize=sz, label='peak')
plt.plot(i_retreat, pcp_sm[i_retreat], 'go', markersize=sz, label='retreat')
plt.grid()
plt.legend()



wlh = onset_WLH(precip, axis)
onset = wlh['onset']
retreat = wlh['retreat']
peak = wlh['peak']
precip_sm = wlh['precip_sm']


plt.figure()
clev = np.arange(20, 60)
atm.contourf_latlon(onset, lat, lon, clev=clev, cmap='jet',
                    axlims=(0,50,50,180), symmetric=False)



# ----------------------------------------------------------------------
# Test with multi-dim data

# cmap = ds['precip']
# npentad = 73 # pentads/year
# dt = 5.0/365
# nyears = 1
# precip = cmap[:nyears*npentad]
# y = precip.values
# ntrunc = 12
# nharm = 3
#
# spec1 = Fourier(y, dt, axis=0)
# print(spec1)
#
# # Extract one grid point
# spec1.tseries = spec1.tseries[:, ilat0, ilon0]
# spec1.C_k = spec1.C_k[:, ilat0, ilon0]
# spec1.ps_k = spec1.ps_k[:, ilat0, ilon0]
# print(spec1)
#
# spec1 = test_fourier(spec1, ntrunc)
# plot_fourier(spec1, nharm, ntrunc)
