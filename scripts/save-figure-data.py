import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')
sys.path.append('/home/jwalker/dynamics/python/monsoon-onset')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import collections

import atmos as atm
import merra
import indices
import utils

# ----------------------------------------------------------------------
version = 'merra2'
onset_nm = 'CHP_MFC'
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
savedir = atm.homedir() + 'datastore/%s/figure_data/' % version
yearstr = '1980-2015'
lat1, lat2 = 10, 30
lon1, lon2 = 60, 100
lonstr = '%dE-%dE' % (lon1, lon2)
ndays = 5     # n-day rolling mean for smoothing
eqbuf = 5.0   # Latitude buffer around equator for psi decomposition
compdays = [0, 15] # Days for lat-lon composites

datafiles = {}
datafiles['index'] = datadir + version + '_index_%s_%s.nc' % (onset_nm,  yearstr)
datafiles['ubudget'] = (savedir + 'merra2_ubudget_sector_dailyrel_' + onset_nm +
                        '_ndays%d_' % ndays + lonstr + '_1980-2014_excl.nc')
datafiles['ps'] = atm.homedir() + 'dynamics/python/atmos-tools/data/topo/ncep2_ps.nc'
datafiles['GPCP'] = datadir + 'gpcp_dailyrel_' + onset_nm + '_1997-2015.nc'

# Sector mean data
filestr = datadir + version + '_%s' + '_dailyrel_%s_%s.nc' % (onset_nm, yearstr)
nms_sector = ['U', 'V']
datafiles['sector'] = {nm : filestr % (nm + '_sector_' + lonstr) for nm in nms_sector}

# Lat-lon data
nms_latlon = ['U200', 'V200', 'T200', 'TLML', 'QLML', 'THETA_E_LML']
datafiles['latlon'] = {nm : filestr % nm for nm in nms_latlon}


savefiles = {}
savestr = (savedir + version + '_%s_' + 'ndays%d_%dE-%dE_%s.nc'
                                         % (ndays, lon1, lon2, yearstr))
for nm in ['sector_latp', 'hov', 'psi_comp', 'tseries']:
    savefiles[nm] = savestr % nm
savefiles['latlon'] = savedir + version + '_latlon_ndays%d_%s.nc' % (ndays, yearstr)
savefiles['gpcp'] = savedir + 'gpcp_dailyrel_1997-2015.nc'

# ----------------------------------------------------------------------
# Read sector mean data and smooth with nday rolling mean

data_sector = xray.Dataset()
for nm in datafiles['sector']:
    filenm = datafiles['sector'][nm]
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        var = ds[nm].load()
    daydim = atm.get_coord(var, coord_name='dayrel', return_type='dim')
    data_sector[nm] = atm.rolling_mean(var, ndays, axis=daydim, center=True)

# ----------------------------------------------------------------------
# Lat-pres sector mean data

nms_latp = ['U', 'V']
data_latp = data_sector[nms_latp]

# Compute streamfunction
print('Computing streamfunction')
if (lon2 - lon1) < 360:
    sector_scale = (lon2 - lon1) / 360.
else:
    sector_scale = None
data_latp['PSI'] = atm.streamfunction(data_latp['V'], sector_scale=sector_scale)

# Topography for lat-pres contour plots
print('Loading topography')
lat = atm.get_coord(data_latp, 'lat')
lon = np.arange(0, 357.5, 0.5)
ps = atm.get_ps_clim(lat, lon, datafiles['ps'])
data_latp['PS'] = atm.dim_mean(ps, 'lon', lon1, lon2)

print('Saving to ' + savefiles['sector_latp'])
data_latp.to_netcdf(savefiles['sector_latp'])

# ----------------------------------------------------------------------
# Hovmoller data

def get_varnm(nm):
    varnms = {'U200' : 'U', 'V200' : 'V', 'T200' : 'T', 'TLML' : 'T',
              'QLML' : 'Q', 'THETA_E_LML' : 'THETA_E'}
    return varnms.get(nm)

# Read data, compute sector mean, smooth with nday rolling average
data_hov = xray.Dataset()

for nm in datafiles['latlon']:
    filenm = datafiles['latlon'][nm]
    varnm = get_varnm(nm)
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        var = atm.dim_mean(ds[varnm], 'lon', lon1, lon2)
        var.load()
    daydim = atm.get_coord(var, coord_name='dayrel', return_type='dim')
    data_hov[nm] = atm.rolling_mean(var, ndays, axis=daydim, center=True)

# Extract PSI500 from data_latp and add to dataset
plev = 500
psi = atm.subset(data_latp['PSI'], {'plev' : (plev, plev)}, squeeze=True)
data_hov['PSI%d' % plev] = psi

print('Saving to ' + savefiles['hov'])
data_hov.to_netcdf(savefiles['hov'])

# ----------------------------------------------------------------------
# Lat-lon data on selected days

data_latlon = xray.Dataset()
for nm in datafiles['latlon']:
    filenm = datafiles['latlon'][nm]
    varnm = get_varnm(nm)
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        var = ds[varnm]
        daydim = atm.get_coord(var, coord_name='dayrel', return_type='dim')
        var = atm.rolling_mean(var, ndays, axis=daydim, center=True)
        var = var.sel(dayrel=compdays)
        var.load()
    data_latlon[nm] = var

print('Saving to ' + savefiles['latlon'])
data_latlon.to_netcdf(savefiles['latlon'])

# ----------------------------------------------------------------------
# GPCP lat-lon, Hovmoller data, and average over box tseries

print('Loading ' + datafiles['GPCP'])
with xray.open_dataset(datafiles['GPCP']) as ds:
    var = atm.dim_mean(ds['PREC'], 'year')
    var.load()

data_gpcp = xray.Dataset({'PCP' : var})
data_gpcp['PCP_SECTOR'] = atm.dim_mean(var, 'lon', lon1, lon2)
data_gpcp['PCP_BOX'] = atm.mean_over_geobox(var, lat1, lat2, lon1, lon2)
attrs = data_gpcp['PCP_BOX'].attrs
attrs = atm.odict_delete(attrs, 'area_weighted')
attrs = atm.odict_delete(attrs, 'land_only')
data_gpcp['PCP_BOX'].attrs = attrs

# Smooth with nday rolling mean
for nm in data_gpcp.data_vars:
    data_gpcp[nm] = atm.rolling_mean(data_gpcp[nm], ndays, axis=daydim,
                                     center=True)

print('Saving to ' + savefiles['gpcp'])
data_gpcp.to_netcdf(savefiles['gpcp'])

# ----------------------------------------------------------------------
# Dailyrel timeseries at various latitudes

lat_extract = [-30, -15, 0, 15, 30]
npre, npost = 120, 200

with xray.open_dataset(datafiles['index']) as index:
    index.load()

ts = xray.Dataset()
ts['MFC'] = utils.daily_rel2onset(index['daily_ts'],index['onset'], npre, npost)
ts['CMFC'] = utils.daily_rel2onset(index['tseries'],index['onset'], npre, npost)
ts = atm.dim_mean(ts, 'year')
ts['GPCP'] = atm.subset(data_gpcp['PCP_BOX'], {'dayrel' : (-npre, npost)})

# Extract timeseries at each latitude
for lat0 in lat_extract:
    lat0_str = atm.latlon_labels(lat0, 'lat', deg_symbol=False)
    for nm in data_hov.data_vars:
        var = data_hov[nm]
        lat = atm.get_coord(var, 'lat')
        key = nm + '_' + lat0_str
        lat_closest, _ = atm.find_closest(lat, lat0)
        print '%s %.2f %.2f' % (nm, lat0, lat_closest)
        ts[key] = atm.subset(var, {'lat' : (lat_closest, None)}, squeeze=True)

print('Saving to ' + savefiles['tseries'])
ts.to_netcdf(savefiles['tseries'])

# ----------------------------------------------------------------------
# Calculate streamfunction components from ubudget

print('Loading ' + datafiles['ubudget'])
with xray.open_dataset(datafiles['ubudget']) as ubudget:
    ubudget.load()
scale = ubudget.attrs['scale']

print('Computing streamfunction components')
sector_scale = (lon2 - lon1) / 360.0
v = utils.v_components(ubudget, scale=scale, eqbuf=eqbuf)
psi_comp = xray.Dataset()
for nm in v.data_vars:
    psi_comp[nm] = atm.streamfunction(v[nm], sector_scale=sector_scale)
psi_comp.attrs['eqbuf'] = eqbuf

print('Saving to ' + savefiles['psi_comp'])
psi_comp.to_netcdf(savefiles['psi_comp'])


# ----------------------------------------------------------------------
# Cross-equatorial MSE fluxes
