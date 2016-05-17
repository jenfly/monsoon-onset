import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pandas as pd

import atmos as atm
import indices
import utils

# ----------------------------------------------------------------------
onset_nm = 'CHP_MFC'
years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/daily/'
savedir = atm.homedir() + 'datastore/merra/analysis/'
datafiles = {}
filestr = datadir + 'merra_%s_%s_%d.nc'
datafiles['CHP_MFC'] = [filestr % ('MFC', '40E-120E_90S-90N', y) for y in years]
datafiles['HOWI'] = [filestr % ('vimt', 'ps-300mb', y) for y in years]
datafiles['U850'] = [filestr % ('U850', '40E-120E_90S-90N', y) for y in years]
datafiles['V850'] = [filestr % ('V850', '40E-120E_90S-90N', y) for y in years]
yearstr = '%d-%d.nc' % (min(years), max(years))
savefile = savedir + 'merra_index_%s_' + yearstr

# Large-scale indices to save (set to [] if only doing grid points)
onset_nms = ['CHP_MFC', 'HOWI', 'OCI', 'SJKE']
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# Grid point calcs
pts_nm = 'CHP_PCP'
pts_subset = {'lon' : (lon1, lon2), 'lat' : (5, 30)}
xsample, ysample = 1, 1


# ----------------------------------------------------------------------
# Calculate and save large-scale onset/retreat indices

def save_index(index, onset_nm, savefile):
    filenm = savefile % onset_nm
    print('Saving to ' + filenm)
    index.to_netcdf(filenm)

if 'CHP_MFC' in onset_nms:
    # Accumulated MFC changepoint (Cook & Buckley 2009)
    onset_nm = 'CHP_MFC'
    mfc = atm.combine_daily_years('MFC', datafiles[onset_nm], years,
                                  yearname='year')
    mfcbar = atm.mean_over_geobox(mfc, lat1, lat2, lon1, lon2)
    mfc_acc = np.cumsum(mfcbar, axis=1)
    index = indices.onset_changepoint(mfc_acc)
    mfcbar.attrs = collections.OrderedDict({'units' : mfc.attrs['units']})
    index['daily_ts'] = mfcbar
    save_index(index, onset_nm, savefile)

if 'HOWI' in onset_nms:
    # HOWI index (Webster and Fasullo 2003)
    onset_nm = 'HOWI'
    npts, maxbreak = 100, 10
    ds = atm.combine_daily_years(['uq_int', 'vq_int'], datafiles[onset_nm],
                                 years, yearname='year')
    index, data = indices.onset_HOWI(ds['uq_int'], ds['vq_int'], npts,
                                     maxbreak=maxbreak)
    save_index(index, onset_nm, savefile)
    #save_index(data, onset_nm + '_data', savefile)

if 'OCI' in onset_nms or 'SJKE' in onset_nms:
    # 850 mb U, V for OCI and SJKE
    u850 = atm.combine_daily_years('U', datafiles['U850'], years,
                                   yearname='year')
    v850 = atm.combine_daily_years('V', datafiles['V850'], years,
                                   yearname='year')

if 'OCI' in onset_nms:
    # OCI index (Wang et al 2009)
    index = indices.onset_OCI(u850, yearnm='year', daynm='day')
    index['tseries'].attrs = {}
    save_index(index, 'OCI', savefile)

if 'SJKE' in onset_nms:
    # SJKE index (Boos and Emmanuel 2009)
    thresh_std = 0.7
    nroll = 7
    u_sm = atm.rolling_mean(u850, nroll, axis=1, center=True)
    v_sm = atm.rolling_mean(v850, nroll, axis=1, center=True)
    index = indices.onset_SJKE(u_sm, v_sm, thresh_std=thresh_std,
                               yearnm='year', daynm='day')
    index.attrs['nroll'] = nroll
    index['tseries'].attrs = {}
    save_index(index, 'SJKE', savefile)

# ----------------------------------------------------------------------
# Calculate onset/retreat indices at individual gridpoints

def get_data(datadir, year, pts_nm, pts_subset, xsample, ysample):
    if pts_nm == 'CHP_PCP':
        filenm = datadir + 'merra_precip_%d.nc' % year
        print('Loading ' + filenm)
        with xray.open_dataset(filenm) as ds:
            pcp = atm.subset(ds['PRECTOT'], pts_subset)
            pcp = pcp[:, ::ysample, ::xsample]
            pcp.load()
    pcp = atm.precip_convert(pcp, pcp.attrs['units'], 'mm/day')
    pcp_acc = np.cumsum(pcp, axis=0)
    return pcp_acc

def calc_points(pcp_acc):
    onset = (np.nan * pcp_acc[0]).drop('day')
    retreat = (np.nan * pcp_acc[0]).drop('day')
    lat = atm.get_coord(pcp_acc, 'lat')
    lon = atm.get_coord(pcp_acc, 'lon')
    for i, lat0 in enumerate(lat):
        for j, lon0 in enumerate(lon):
            print('%.1f E, %.1f N' % (lon0, lat0))
            chp = indices.onset_changepoint(pcp_acc[:, i, j])
            onset.values[i, j] = chp.onset.values
            retreat.values[i, j] = chp.retreat.values
    data = xray.Dataset({'onset' : onset, 'retreat' : retreat})
    return data

def yrly_file(savefile, year, pts_nm):
    filenm = savefile % ('pts_' + pts_nm)
    filenm = filenm.replace(yearstr, '%d.nc' % year)
    return filenm

# Calculate onset/retreat in each year
for year in years:
    pcp_acc = get_data(datadir, year, pts_nm, pts_subset, xsample, ysample)
    index = calc_points(pcp_acc)
    filenm = yrly_file(savefile, year, pts_nm)
    print('Saving to ' + filenm)
    index.to_netcdf(filenm)

# Combine years
filenm = savefile % ('pts_' + pts_nm)
files = [yrly_file(savefile, yr, pts_nm) % yr for yr in years]
ds = atm.load_concat(files, concat_dim='year')
ds['year'] = years
print('Saving to ' + filenm)
ds.to_netcdf(filenm)