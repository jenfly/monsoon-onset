import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xarray as xray
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
#pts_nm = 'CHP_PCP'
#pts_nm = 'CHP_GPCP'
#pcp_nm = 'PRECTOT'
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


# ptsfile = datadir + version + '_index_pts_%s_' % pts_nm
# ptsmaskfile = None
# if pts_nm == 'CHP_CMAP':
#     ptsfile = ptsfile + '1980-2014.nc'
#     pts_xroll, pts_yroll = None, None
# elif pts_nm == 'CHP_GPCP':
#     ptsfile = ptsfile + '1997-2015.nc'
#     ptsmaskfile = atm.homedir() + 'datastore/gpcp/gpcp_daily_1997-2014.nc'
#     pts_xroll, pts_yroll = None, None
# else:
#     ptsfile = ptsfile + yearstr
#     pts_xroll, pts_yroll = 3, 3


# if ind_nm == 'retreat':
#     for nm in datafiles:
#         datafiles[nm] = datafiles[nm].replace('dailyrel', 'dailyrel_retreat')

# enso_nm = 'NINO3'
# #enso_nm = 'NINO3.4'
# ensodir = atm.homedir() + 'dynamics/python/data/ENSO/'
# ensofile = ensodir + ('enso_sst_monthly_%s.csv' %
#                       enso_nm.lower().replace('.', '').replace('+', ''))
# enso_keys = ['MAM', 'JJA']

# ----------------------------------------------------------------------
# Read data

# Large-scale onset/retreat indices
index_all = collections.OrderedDict()
for nm in indfiles:
    print('Loading ' + indfiles[nm])
    if nm == 'MOK':
        mok = indices.onset_MOK(indfiles['MOK'], yearsub=years)
        index_all['MOK'] =  xray.Dataset({'onset' : mok})
    else:
        with xray.open_dataset(indfiles[nm]) as ds:
            index_all[nm] = ds.load()
index = index_all[onset_nm]

onset_all = pd.DataFrame()
for nm in index_all:
    onset_all[nm] = index_all[nm]['onset'].to_series()
onset_all = onset_all.astype(int)
onset_all.to_csv('data/data_fig1b.csv')


# MFC budget
with xray.open_dataset(mfcbudget_file) as mfc_budget:
    mfc_budget.load()
mfc_budget = mfc_budget.rename({'DWDT' : 'dw/dt'})
mfc_budget = mfc_budget.drop('DWDT_ANA')
if nroll is not None:
    for nm in mfc_budget.data_vars:
        mfc_budget[nm] = atm.rolling_mean(mfc_budget[nm], nroll, center=True)
mfc_budget['CMFC'] = index['tseries']
df = mfc_budget.sel(year=2000).drop('year').to_dataframe()
mfc_budget.to_netcdf('data/data_fig1a_allyears.nc')
df.to_csv('data/data_fig1a.csv')

# Dailyrel climatology
keys_dict = {'PRECTOT' : 'PRECTOT', 'CMAP' : 'precip', 'GPCP' : 'PREC',
             'U200' : 'U', 'U850' : 'U', 'V200' : 'V', 'V850' : 'V'}
data = {}
for nm in datafiles:
    print('Loading ' + datafiles[nm])
    with xray.open_dataset(datafiles[nm]) as ds:
        if 'year' in ds.dims:
            ds = ds.mean(dim='year')
        data[nm] = ds[keys_dict[nm]].load()

# Daily timeseries
ts = xray.Dataset()
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
tseries_df.to_csv('data/data_fig1c-d.csv')
