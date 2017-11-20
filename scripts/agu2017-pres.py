import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pandas as pd

import atmos as atm
import indices
import utils


version = 'merra2'
years = np.arange(1980, 2016)
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
onset_nm = 'CHP_MFC'
onset_nms = ['CHP_MFC', 'MOK', 'HOWI', 'OCI']
pcp_nm = 'GPCP'
varnms = ['PRECTOT', 'U200', 'V200', 'U850', 'V850']
lat_extract = {'U200' : 0, 'V200' : 15, 'U850' : 15, 'V850' : 15}
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
nroll = 5 # n-day rolling averages for smoothing daily timeseries
ind_nm, npre, npost = 'onset', 120, 200
#ind_nm, npre, npost = 'retreat', 270, 89
fracmin = 0.5 # Precip JJAS frac of total for gridpoint masking

yearstr = '%d-%d.nc' % (min(years), max(years))
filestr = datadir + version + '_index_%s_' + yearstr
indfiles = collections.OrderedDict()
for nm in ['CHP_MFC', 'HOWI', 'OCI']:
    indfiles[nm] = filestr % nm
indfiles['MOK'] = atm.homedir() + 'dynamics/python/monsoon-onset/data/MOK.dat'
filestr2 = datadir + version + '_%s_dailyrel_' + onset_nm + '_' + yearstr
datafiles = {nm : filestr2 % nm for nm in varnms}
datafiles['CMAP'] = datadir + 'cmap_dailyrel_' + onset_nm + '_1980-2014.nc'
datafiles['GPCP'] = datadir + 'gpcp_dailyrel_' + onset_nm + '_1997-2015.nc'
mfcbudget_file = datadir + version + '_mfc_budget_' + yearstr


# -----------------------------
# Plot map of precip and winds pre- and post- onset
yr0 = 2000
datadir0 = datadir.replace('analysis', 'daily')
filestr0 = datadir0 + 'merra2_%s850_40E-120E_90S-90N_' + '%d.nc' % yr0
data0 = {}
for nm in ['U', 'V']:
    with xr.open_dataset(filestr0 % nm) as ds:
        data0[nm] = ds[nm].load()
filenm0 = 'datastore/gpcp/gpcp_daily_%d.nc' % yr0
with xr.open_dataset(atm.homedir() + filenm0) as ds:
    data0['GPCP'] = ds['PREC'].load()

day0 = 146
u = data0['U'].sel(day=day0)
v = data0['V'].sel(day=day0)
pcp = data0['GPCP'].sel(day=day0)


def precip_winds(u, v, pcp, axlims=(-30, 45, 40, 120), climits=(0, 20),
                 cticks=np.arange(4, 21, 2), clev=np.arange(4, 20.5, 1),
                 cmap='hot_r', scale=250, dx=5, dy=5,
                 xticks=range(40, 121, 20), yticks=range(-20, 41, 10)):
    lat1, lat2, lon1, lon2 = axlims
    subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}

    u = u[::dy, ::dx]
    v = v[::dy, ::dx]
    lat = atm.get_coord(u, 'lat')
    lon = atm.get_coord(u, 'lon')

    plt.figure()
    m = atm.init_latlon(lat1, lat2, lon1, lon2, coastlines=False)
    m.drawcoastlines(color='k', linewidth=0.5)
    m.shadedrelief(scale=0.3)
    atm.contourf_latlon(pcp, clev=clev, axlims=axlims, m=m, cmap=cmap,
                        extend='both', cb_kwargs={'ticks' : cticks})
    #atm.pcolor_latlon(pcp, axlims=axlims, cmap=cmap, cb_kwargs={'extend' : 'max'})
    plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
    plt.yticks(yticks, atm.latlon_labels(yticks, 'lat'))
    plt.clim(climits)
    #plt.quiver(lon, lat, u, v, linewidths=spd.values.ravel())
    plt.quiver(lon, lat, u, v, scale=scale, pivot='middle')

# ----------------------------------------------------------------------
# Read data

# Large-scale onset/retreat indices
index_all = collections.OrderedDict()
for nm in indfiles:
    print('Loading ' + indfiles[nm])
    if nm == 'MOK':
        mok = indices.onset_MOK(indfiles['MOK'], yearsub=years)
        index_all['MOK'] =  xr.Dataset({'onset' : mok})
    else:
        with xr.open_dataset(indfiles[nm]) as ds:
            index_all[nm] = ds.load()
index = index_all[onset_nm]

onset_all = pd.DataFrame()
for nm in index_all:
    onset_all[nm] = index_all[nm]['onset'].to_series()
onset_all = onset_all.astype(int)


# MFC budget
with xr.open_dataset(mfcbudget_file) as mfc_budget:
    mfc_budget.load()
mfc_budget = mfc_budget.rename({'DWDT' : 'dw/dt'})
mfc_budget = mfc_budget.drop('DWDT_ANA')
if nroll is not None:
    for nm in mfc_budget.data_vars:
        mfc_budget[nm] = atm.rolling_mean(mfc_budget[nm], nroll, center=True)
mfc_budget['CMFC'] = index['tseries']
df = mfc_budget.sel(year=2000).drop('year').to_dataframe()
df =  df[(df.index > 2) & (df.index < 399)]

# Dailyrel climatology
keys_dict = {'PRECTOT' : 'PRECTOT', 'CMAP' : 'precip', 'GPCP' : 'PREC',
             'U200' : 'U', 'U850' : 'U', 'V200' : 'V', 'V850' : 'V'}
data = {}
for nm in datafiles:
    print('Loading ' + datafiles[nm])
    with xr.open_dataset(datafiles[nm]) as ds:
        if 'year' in ds.dims:
            ds = ds.mean(dim='year')
        data[nm] = ds[keys_dict[nm]].load()

# Daily timeseries
ts = xr.Dataset()
for nm in ['GPCP', 'PRECTOT']:
    ts[nm] = atm.mean_over_geobox(data[nm], lat1, lat2, lon1, lon2)
ts['MFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['daily_ts'],
                                  index[ind_nm], npre + 5, npost + 5)
ts['CMFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['tseries'],
                                   index[ind_nm], npre + 5, npost + 5)


# Extract variables at specified latitudes
for nm, lat0 in lat_extract.iteritems():
    var = atm.dim_mean(data[nm], 'lon', lon1, lon2)
    lat = atm.get_coord(var, 'lat')
    lat0_str = atm.latlon_labels(lat0, 'lat', deg_symbol=False)
    # key = nm + '_' + lat0_str
    key = nm
    lat_closest, _ = atm.find_closest(lat, lat0)
    print '%s %.2f %.2f' % (nm, lat0, lat_closest)
    ts[key] = atm.subset(var, {'lat' : (lat_closest, None)}, squeeze=True)

# Compute climatology and smooth with rolling mean
if 'year' in ts.dims:
    ts = ts.mean(dim='year')
if nroll is not None:
    for nm in ts.data_vars:
        ts[nm] = atm.rolling_mean(ts[nm], nroll, center=True)
tseries = atm.subset(ts, {'dayrel' : (-npre, npost)})

tseries_df = tseries[['MFC', 'GPCP', 'CMFC', 'U850', 'V850']].to_dataframe()
