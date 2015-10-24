import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra

# ----------------------------------------------------------------------
# Compute HOWI indices (Webster and Fasullo 2003)
datadir = atm.homedir() + 'datastore/merra/daily/'
datafile = datadir + 'merra_vimt_ps-300mb_may-aug_1979-2003.nc'
lat1, lat2 = -20, 30
lon1, lon2 = 40, 100

with xray.open_dataset(datafile) as ds:
    ds.load()

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
vimt_top = np.ma.masked_array(vimt, mask).filled(np.nan)

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
plt.title('Top %d Magnitude of vector difference' % N)



# ----------------------------------------------------------------------

#datadir = '/home/jennifer/datastore/merra/daily/'
datadir = '/home/jwalker/eady/datastore/merra/daily/'
filestr = 'merra_uv200_40E-120E_60S-60N_'
pathstr = datadir + filestr
years = np.arange(1979, 2015)
lon1, lon2 = 60, 100
#lat1, lat2 = 10, 30
lat1, lat2 = 10.625, 10.625
#lat1, lat2 = 0.625, 0.625
#lat1, lat2 = -5.625, -5.625
#lat1, lat2 = -10.625, -10.625

# ----------------------------------------------------------------------
def timeseries_allyears(pathstr, years, lat1, lat2, lon1, lon2):
    """Return the mean_over_geobox of daily data from selected years."""

    def get_year(ds, year, lat1, lat2, lon1, lon2):
        """Get daily data for this year, calculate mean_over_geobox,
        and add NaN if applicable so that non-leap and leap years
        can be concatenated together."""
        dsbar = xray.Dataset()
        nan = np.nan*np.ones((1,1))
        days = np.arange(1,367)
        for nm in ds.data_vars:
            print(nm)
            var = atm.mean_over_geobox(ds[nm], lat1, lat2, lon1, lon2)
            vals = var.values
            if not atm.isleap(year):
                vals = np.concatenate([vals, nan])
            coords = {'Day' : days, 'Height': var.coords['Height'], 'Year': year}
            dsbar[nm] = xray.DataArray(vals, name=var.name, dims=var.dims,
                                       attrs=var.attrs, coords=coords)
        return dsbar

    for i, year in enumerate(years):
        filename = '%s%d.nc' % (pathstr, year)
        print('Loading ' + filename)
        with xray.open_dataset(filename) as ds:
            data = get_year(ds, year, lat1, lat2, lon1, lon2)
        if i == 0:
            dsbar = data
        else:
            dsbar = xray.concat([dsbar, data], dim='Year')
    return dsbar


def plot_timeseries_year(dsbar, year, nroll=None):
    iplot = {'U' : 1, 'V' : 2, 'rel_vort' : 3, 'Ro' : 4}
    plt.figure(figsize=(12, 9))
    plt.suptitle(year)
    for nm in dsbar.data_vars:
        var = dsbar[nm].sel(Year=year)
        plt.subplot(2, 2, iplot[nm])
        plt.plot(var.Day, var, color='gray')
        if nroll is not None:
            data = pd.rolling_mean(np.squeeze(var.values), nroll)
            plt.plot(var.Day, data, color='black')
        plt.title(nm)


# ----------------------------------------------------------------------

dsbar = timeseries_allyears(pathstr, years, lat1, lat2, lon1, lon2)

nroll = 10
for year in [1979, 1980, 1981, 1982]:
    plot_timeseries_year(dsbar, year, nroll)


# ----------------------------------------------------------------------
