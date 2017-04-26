import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xarray as xray
import numpy as np
import matplotlib.pyplot as plt
import atmos as atm

# ----------------------------------------------------------------------
mldfile = '/home/jennifer/datastore/mld/ifremer_mld_DT02_c1m_reg2.0.nc'
suptitle = 'Ifremer Mixed Layer Depths'

ds = xray.open_dataset(mldfile)
mld = ds['mld']
missval = mld.attrs['mask_value']
vals = mld.values
vals = np.ma.masked_array(vals, vals==missval)
vals = np.ma.filled(vals, np.nan)
mld.values = vals

# Sector mean
lon1, lon2 = 60, 100
mldbar = atm.subset(mld, {'lon' : (lon1, lon2)}).mean(dim='lon')


# ----------------------------------------------------------------------
# Plots

cmap = 'hot_r'
axlims = (-30, 30, 40, 120)
clim1, clim2 = 0, 80
figsize = (12, 9)
months = [4, 5, 6]

plt.figure(figsize=figsize)
plt.suptitle(suptitle)

# Lat-lon maps
for i, month in enumerate(months):
    plt.subplot(2, 2, i + 1)
    atm.pcolor_latlon(mld[month-1], axlims=axlims, cmap=cmap)
    plt.clim(clim1, clim2)
    plt.title(atm.month_str(month))

# Sector mean line plots
latmin, latmax = -30, 30
mldplot = atm.subset(mldbar, {'lat' : (latmin, latmax)})
lat = atm.get_coord(mldplot, 'lat')
plt.subplot(2, 2, 4)
for month in months:
    plt.plot(lat, mldplot[month-1], label=atm.month_str(month))
plt.legend()
plt.grid()
plt.title('%d-%d E Mean' % (lon1, lon2))
plt.xlabel('Latitude')
plt.ylabel('MLD (m)')
