import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
from indices import onset_HOWI

# ----------------------------------------------------------------------
# Compute HOWI indices (Webster and Fasullo 2003)
datadir = atm.homedir() + 'datastore/merra/daily/'
datafile = datadir + 'merra_vimt_ps-300mb_may-sep_1979-2014.nc'
lat1, lat2 = -20, 30
lon1, lon2 = 40, 100

with xray.open_dataset(datafile) as ds:
    uq_int = ds['uq_int'].load()
    vq_int = ds['vq_int'].load()

npts = 100
howi, ds = onset_HOWI(uq_int, vq_int, npts)

onset = howi.onset
retreat = howi.retreat
y=0

plt.figure()
plt.plot(howi.day, howi.index[y])
plt.plot(onset[y], howi.index[y].sel(day=onset[y]), 'r*')
plt.plot(retreat[y], howi.index[y].sel(day=retreat[y]), 'r*')
plt.grid()


# Climatological moisture fluxes
dsbar = ds.mean(dim='year')

# Pre- and post- monsoon climatology composites
days_pre = range(138, 145)  # May 18-24
days_post = range(159, 166) # June 8-14
dspre = atm.subset(dsbar, 'day', days_pre).mean(dim='day')
dspost = atm.subset(dsbar, 'day', days_post).mean(dim='day')
dsdiff = dspost - dspre

# Magnitude of vector fluxes
vimt = np.sqrt(dsdiff['uq_int']**2 + dsdiff['vq_int']**2)

# Top N difference vectors
def top_n(data, n):
    """Return a mask with the highest n values in 2D array."""
    vals = data.copy()
    mask = np.ones(vals.shape, dtype=bool)
    for k in range(n):
        i, j = np.unravel_index(np.nanargmax(vals), vals.shape)
        print(i, j)
        mask[i, j] = False
        vals[i, j] = np.nan
    return mask

N = 50
mask = top_n(vimt, N)
vimt_top = np.ma.masked_array(vimt, mask)

# Plot climatological VIMT composites
lat = atm.get_coord(dsbar, 'lat')
lon = atm.get_coord(dsbar, 'lon')
x, y = np.meshgrid(lon, lat)
axlims = (lat1, lat2, lon1, lon2)
plt.figure(figsize=(7,10))
plt.subplot(211)
m = atm.init_latlon(lat1, lat2, lon1, lon2)
m.quiver(x, y, dspre['uq_int'], dspre['vq_int'])
plt.title('May 18-24 VIMT Climatology')
plt.subplot(212)
m = atm.init_latlon(lat1, lat2, lon1, lon2)
m.quiver(x, y, dspost['uq_int'], dspost['vq_int'])
plt.title('June 8-14 VIMT Climatology')

# Plot difference between pre- and post- composites
plt.figure(figsize=(7,10))
plt.subplot(211)
m = atm.init_latlon(lat1, lat2, lon1, lon2)
m.quiver(x, y, dsdiff['uq_int'], dsdiff['vq_int'])
plt.title('June 8-14 minus May 18-24 VIMT Climatology')
plt.subplot(212)
atm.pcolor_latlon(vimt, axlims=axlims, cmap='hot_r')
plt.title('Magnitude of vector difference')

# Top N difference vectors
plt.figure()
atm.pcolor_latlon(vimt_top, lat, lon, axlims=axlims, cmap='hot_r')
plt.title('Top %d Magnitude of vector difference' % npts)
