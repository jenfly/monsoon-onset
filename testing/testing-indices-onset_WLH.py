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
from indices import onset_WLH


datadir = '/home/jennifer/datastore/cmap/'
#datadir = '/home/jwalker/datastore/cmap/'
cmap_file = datadir + 'cmap.precip.pentad.mean.nc'

# Years to include
yearmin = 1979
yearmax = 1997

# Calculate each year individually or use climatological precip
climatology = True

precip = precipdat.read_cmap(cmap_file, yearmin, yearmax)
lat = atm.get_coord(precip, 'lat')
lon = atm.get_coord(precip, 'lon')

if climatology:
    precip = precip.mean(dim='year')
    axis = 0
else:
    axis = 1

dt = 5.0/365
wlh = onset_WLH(precip, axis, dt)
onset = wlh['onset']
retreat = wlh['retreat']
peak = wlh['peak']
precip_sm = wlh['precip_sm']
precip_ann = wlh['precip_ann']
precip_rel = wlh['precip_rel']

# Check timeseries for single grid point
#lat0, lon0 = 12.5, 90.0
lat0, lon0 = 11.25, 91.25
latval, ilat0 = atm.find_closest(precip.lat, lat0)
lonval, ilon0 = atm.find_closest(precip.lon, lon0)

pcp = precip[:, ilat0, ilon0]
psm = precip_sm[:, ilat0, ilon0]
pan = precip_ann[:, ilat0, ilon0]
prel = precip_rel[:, ilat0, ilon0]
i_onset = int(onset[ilat0, ilon0] - 1)
i_retreat = int(retreat[ilat0, ilon0] - 1)
i_peak = int(peak[ilat0, ilon0] - 1)

plt.figure()
plt.plot(pcp, color='grey', label='raw')
plt.plot(psm, color='black', label='smoothed')
plt.plot(pan, color='red', label='annual')
plt.plot(i_onset, psm[i_onset], 'r*')
plt.plot(i_peak, psm[i_peak], 'bd')
plt.plot(i_retreat, psm[i_retreat], 'go')
plt.grid()
plt.legend()

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
