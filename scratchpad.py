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
# EMFD
datadir = atm.homedir() + 'datastore/merra/daily/'
ds = xray.open_dataset(datadir + 'merra_uv200_40E-120E_60S-60N_1979.nc')
u = atm.squeeze(ds['U'])
v = atm.squeeze(ds['V'])

nroll = 7
u_tr = u - atm.rolling_mean(u, nroll, axis=0)
v_tr = v - atm.rolling_mean(v, nroll, axis=0)

emfd_tr, emfd_tr_x, emfd_tr_y = atm.divergence_spherical_2d(u_tr * u_tr,
                                                            u_tr * v_tr)

# ----------------------------------------------------------------------
# Lat-pres streamfunction
v = merra.read_daily('V', 1979, 7, days=range(1,6),
                     subset_dict={'lon' : (60, 100)})
v = v.mean(dim='TIME')
psi = atm.streamfunction(v)
psibar = psi.mean(dim='XDim')
plt.figure()
atm.contourf_latpres(psibar)

# ----------------------------------------------------------------------
# 01/14/2016 Plots for Simona

# Load data from compare-indices.py

keys = ['HOWI_100', 'OCI', 'SJKE', 'TT',  'WLH_MERRA_PRECIP_nroll7']
shortkeys = ['HOWI', 'OCI', 'SJKE', 'TT',  'WLH']
#shortkeys = [short[key] for key in keys]

years = index[keys[0]].year.values
onset = np.reshape(index[keys[0]].onset.values, (len(years), 1))
for key in keys[1:]:
    ind = np.reshape(index[key].onset.values, (len(years), 1))
    onset = np.concatenate([onset, ind], axis=1)
onset = pd.DataFrame(onset, index=years, columns=shortkeys)

# Add monsoon strength index
ind_comp = onset.copy()
ind_comp['JJAS_MFC'] = strength['MERRA_DET']

# Box plots of onset days
plt.figure()
onset.boxplot()
plt.xlabel('Onset Index')
plt.ylabel('Day of Year')

# ----------------------------------------------------------------------
# Model level MERRA data


varnm = 'T'

xsub = '[330:2:450]'
ysub = '[60:2:301]'
tsub = '[0:1:3]'
lev = 71
zsub = '[%d:1:%d]' % (lev, lev)

def datafile(year, mon, day, varnm, xsub, ysub, zsub, tsub):
    url = ('http://goldsmr3.sci.gsfc.nasa.gov:80/opendap/MERRA/MAI6NVANA.5.2.0/'
           '%d/%02d/MERRA100.prod.assim.inst6_3d_ana_Nv.%d%02d%02d.hdf'
           '?%s%s%s%s%s,XDim%s,YDim%s,Height%s,TIME%s') % (year, mon, year, mon,
           day, varnm, tsub, zsub, ysub, xsub, xsub, ysub, zsub, tsub)
    return url

year = 1979
month = 4
#jdays = atm.season_days(atm.month_str(month), atm.isleap(year))
days = range(1, atm.days_this_month(year, month) + 1)
urls = [datafile(year, month, day, varnm, xsub, ysub, zsub, tsub) for day
        in days]
savedir = atm.homedir() + '/datastore/merra/daily/'
savefile = '%smerra_%s_ML%02d_40-120E_60S-60N_%d%02d.nc' % (savedir, varnm, lev,
                                                            year, month)
var = atm.load_concat(urls, varnm, 'TIME')
print('Saving to ' + savefile)
atm.save_nc(savefile, var)

# for d, day in enumerate(days):
#     url = datafile(year, month, day, varnm, xsub, ysub, zsub, tsub)
#     print('Reading %s' % url)
#     ds = xray.open_dataset(url)
#     var_in = atm.squeeze(ds[varnm])
#     # Daily mean:
#     var_in = var_in.mean(dim='TIME')
#     var_in.coords['Day'] = day
#     if d == 0:
#         var = var_in
#     else:
#         var = xray.concat([var, var_in], dim='Day')
#
#
# T = ds['T']
# T = T[0]


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
