import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra

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
