import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')
sys.path.append('/home/jwalker/dynamics/python/monsoon-onset')

import numpy as np
import xarray as xray
import pandas as pd
import matplotlib.pyplot as plt
import collections

import atmos as atm
import merra
import indices
import utils

# ----------------------------------------------------------------------
version = 'merra2'
yearstr = '1980-2015'
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
savedir = atm.homedir() + 'datastore/%s/figure_data/' % version

ndays = 5      # Rolling pentad
lon1, lon2 = 60, 100
eqlat1, eqlat2 = -5, 5

varnms = ['UFLXCPT', 'UFLXQV', 'UFLXPHI', 'VFLXCPT', 'VFLXQV', 'VFLXPHI',
          'LWTUP', 'SWGNT', 'LWGNT', 'SWTNT', 'HFLUX', 'EFLUX']

filestr = datadir + version + '_%s_dailyrel_CHP_MFC_' + yearstr + '.nc'
datafiles = {nm : filestr % nm for nm in varnms}

savestr = savedir + version + '_%s_' + yearstr + '.nc'
savefiles = {}
for nm in ['energy_budget', 'energy_budget_sector', 'energy_budget_eq']:
    savefiles[nm] = savestr % nm

# ----------------------------------------------------------------------

data = xray.Dataset()
for nm in varnms:
    filenm = datafiles[nm]
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        var = atm.subset(ds[nm], {'lat' : (-60, 60)})
        daydim = atm.get_coord(var, 'dayrel', 'dim')
        data[nm] = atm.rolling_mean(var, ndays, axis=daydim)

data['NETRAD'] = data['SWTNT'] - data['LWTUP'] - data['SWGNT']- data['LWGNT']
data['FNET'] = data['NETRAD'] + data['EFLUX'] + data['HFLUX']

Lv = atm.constants.Lv.values
for nm in ['UFLXQV', 'VFLXQV']:
    key = nm.replace('QV', 'LQV')
    data[key] = data[nm] * Lv
    data[key].attrs['units'] = 'J m-1 s-1'

data['UH'] = data['UFLXCPT'] + data['UFLXPHI'] + data['UFLXLQV']
data['VH'] = data['VFLXCPT'] + data['VFLXPHI'] + data['VFLXLQV']
data.attrs['ndays'] = ndays

print('Saving to ' + savefiles['energy_budget'])
data.to_netcdf(savefiles['energy_budget'])


# Sector mean data
data_sector = atm.dim_mean(data, 'lon', lon1, lon2)
data_sector.attrs['lon1'] = lon1
data_sector.attrs['lon2'] = lon2

# Equator mean data
data_eq = atm.dim_mean(data, 'lat', eqlat1, eqlat2)
data_eq.attrs['eqlat1'] = eqlat1
data_eq.attrs['eqlat2'] = eqlat2


a = atm.constants.radius_earth.values
dx = a * np.radians(lon2-lon1)
var = data_eq['UH']
data_eq['UH_DX'] = (var.sel(lon=lon2) - var.sel(lon=lon1))/dx

data_eq_bar = atm.dim_mean(data_eq, 'lon', lon1, lon2)

var = atm.subset(data_sector['VH'], {'lat' : (-30, 30)})
daydim = atm.get_coord(var, 'dayrel', 'dim')
var = atm.rolling_mean(var, ndays, axis=daydim)
days = atm.get_coord(var, 'dayrel')
lat = atm.get_coord(var, 'lat')


zerolat = np.nan * np.ones(len(days))
latmin = -15
latmax = 15
for i, day in enumerate(days[ndays:-ndays]):
    print(day)
    zerolat[i] = utils.find_zeros_1d(lat, var.sel(dayrel=day), latmin, latmax,
                                     interp=0.1, return_type='min')

#cint = atm.cinterval(var, n_pref=50, symmetric=True)
#clevs = atm.clevels(var, cint, symmetric=True)
clevs = np.arange(-4e9, 4.1e9, 0.2e9)
plt.figure()
plt.contourf(days, lat, var.T, clevs, cmap='RdBu_r', extend='both')
plt.colorbar()
plt.grid()
plt.plot(days, zerolat, 'k', linewidth=2)
#plt.contour(days, lat, var.T, [0], colors='0.5', linewidths=2)

# ----------------------------------------------------------------------
# Radiation terms - monthly data
#
# def concat_years_rad(years, datadir, nms_rad, subset_dict={'lon' : (40, 120)}):
#
#     def monthly_rad(datafiles, year, nms_rad, concat_dim='time'):
#         ds = atm.load_concat(datafiles, var_ids=nms_rad, concat_dim=concat_dim)
#         ds = ds.rename({concat_dim : 'month'})
#         ds['month'] = range(1, 13)
#         for nm in ds.data_vars:
#             ds[nm] = atm.expand_dims(ds[nm], 'year', year, axis=0)
#         return ds
#
#     prod = {yr : 100 for yr in range(1980, 1992)}
#     for yr in range(1992, 2001):
#         prod[yr] = 200
#     for yr in range(2001, 2011):
#         prod[yr] = 300
#     for yr in range(2011, 2016):
#         prod[yr] = 400
#
#     filestr = datadir + 'MERRA2_%d.tavgM_2d_rad_Nx.%d%02d.nc4'
#     files = {}
#     months = range(1, 13)
#     for yr in years:
#         files[yr] = [filestr % (prod[yr], yr, mon) for mon in months]
#
#     for i, year in enumerate(files):
#         dsyr = monthly_rad(files[year], year, nms_rad)
#         dsyr = atm.subset(dsyr, subset_dict)
#         if i == 0:
#             ds = dsyr
#         else:
#             ds = xray.concat([ds, dsyr], dim='year')
#     return ds
#
#
# nms_rad = ['SWTNT', 'LWTUP', 'SWGNT', 'LWGNT']
# ds_rad = concat_years_rad(years, datadir2, nms_rad)
# ds_rad['NETRAD'] = (ds_rad['SWTNT'] - ds_rad['LWTUP'] - ds_rad['SWGNT']
#                     - ds_rad['LWGNT'])
# ds_rad['NETRAD'].attrs['long_name'] = 'Net radiation into atmospheric column'
# ds_rad['NETRAD'].attrs['units'] = ds_rad['SWTNT'].attrs['units']
#
# savefile = datadir2 + 'merra2_rad_1980-2015.nc4'
# ds_rad.to_netcdf(savefile)

# ----------------------------------------------------------------------
# Julian day climatologies of EFLUX, HFLUX, uh, vh
