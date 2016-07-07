import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')
sys.path.append('/home/jwalker/dynamics/python/monsoon-onset')

import xray
import numpy as np
import collections
import pandas as pd
import os

import atmos as atm
import precipdat
import indices
import utils

# ----------------------------------------------------------------------
version = 'merra2'
years = np.arange(1980, 2016)
#years = np.arange(1997, 2016) # GPCP

datadir = atm.homedir() + 'eady/datastore/%s/daily/' % version
savedir = atm.homedir() + 'eady/datastore/%s/analysis/' % version
datafiles = {}
filestr = datadir + '%d/' + version + '_%s%s_%d.nc'
subset1 = '_40E-120E_90S-90N'
datafiles['CHP_MFC'] = [filestr % (y, 'MFC', subset1, y) for y in years]
datafiles['HOWI'] = {}
for nm in ['UFLXQV', 'VFLXQV']:
    datafiles['HOWI'][nm] = [filestr % (y, nm, subset1, y) for y in years]
datafiles['U850'] = [filestr % (y, 'U850', subset1, y) for y in years]
datafiles['V850'] = [filestr % (y, 'V850', subset1, y) for y in years]
precname = {'merra' : 'precip', 'merra2' : 'PRECTOT_40E-120E_90S-90N'}[version]
datafiles['CHP_PCP'] = [filestr % (y, precname, '', y) for y in years]
datafiles['CHP_CMAP'] = [atm.homedir() + 'eady/datastore/cmap/' +
                     'cmap.enhanced.precip.pentad.mean.nc' for y in years]
datafiles['CHP_GPCP'] = [atm.homedir() + 'eady/datastore/gpcp/' +
                         'gpcp_daily_%d.nc' % yr for yr in years]

yearstr = '%d-%d.nc' % (min(years), max(years))
savefile = savedir + version + '_index_%s_' + yearstr

# Large-scale indices to save (set to [] if only doing grid points)
onset_nms = ['CHP_MFC', 'CHP_PCP', 'HOWI', 'OCI', 'SJKE']
#onset_nms = []
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# Grid point calcs
pts_nm = 'CHP_GPCP'
#pts_nm = 'CHP_PCP' # Set to None to omit
pts_subset = {'lon' : (55, 105), 'lat' : (-3, 40)}
xsample, ysample = 1, 1
xroll, yroll = 4, 4

# Options for large-scale and gridpoint CHP calcs
chp_opts = {'onset_range' : (1, 250), 'retreat_range' : (200, 400)}

# ----------------------------------------------------------------------
# Calculate and save large-scale onset/retreat indices

def save_index(index, onset_nm, savefile):
    filenm = savefile % onset_nm
    print('Saving to ' + filenm)
    index.to_netcdf(filenm)

def calc_chp(varnm, files, years, lat1, lat2, lon1, lon2, chp_opts):
    var = atm.combine_daily_years(varnm, files, years, yearname='year')
    varbar = atm.mean_over_geobox(var, lat1, lat2, lon1, lon2)
    daymax = max(chp_opts['retreat_range'])
    varbar = utils.wrapyear_all(varbar, daymin=1, daymax=daymax)
    var_acc = np.cumsum(varbar, axis=1)
    index = indices.onset_changepoint(var_acc, **chp_opts)
    varbar.attrs = collections.OrderedDict({'units' : var.attrs['units']})
    index['daily_ts'] = varbar
    return index

if 'CHP_MFC' in onset_nms:
    # Accumulated MFC changepoint (Cook & Buckley 2009)
    onset_nm = 'CHP_MFC'
    index = calc_chp('MFC', datafiles[onset_nm], years, lat1, lat2, lon1, lon2,
                     chp_opts)
    save_index(index, onset_nm, savefile)

if 'CHP_PCP' in onset_nms:
    # Accumulated precip changepoint (Cook & Buckley 2009)
    onset_nm = 'CHP_PCP'
    index = calc_chp('PRECTOT', datafiles[onset_nm], years, lat1, lat2, lon1,
                     lon2, chp_opts)
    save_index(index, onset_nm, savefile)

if 'HOWI' in onset_nms:
    # HOWI index (Webster and Fasullo 2003)
    onset_nm = 'HOWI'
    npts, maxbreak = 200, 10
    subset_dict = {'lon' : (40, 90), 'lat' : (0, 25)}
    ds = xray.Dataset()
    for nm in ['UFLXQV', 'VFLXQV']:
        ds[nm] =  atm.combine_daily_years(nm, datafiles[onset_nm][nm], years,
                                          yearname='year',
                                          subset_dict=subset_dict)
    index, data = indices.onset_HOWI(ds['UFLXQV'], ds['VFLXQV'], npts,
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

def get_data(filenm, year, pts_nm, pts_subset, xsample, ysample, xroll=None,
             yroll=None, daymax=366):
    print('Loading ' + filenm)

    def smooth_data(pcp, dimname, nroll):
        if nroll is None:
            return pcp
        dim = atm.get_coord(pcp, dimname, 'dim')
        coord = atm.get_coord(pcp, dimname)
        delta = max(abs(np.diff(coord))) * np.ceil(nroll / 2.0)
        subset = {dimname : (coord.min() + delta, coord.max() - delta)}
        pcp = atm.rolling_mean(pcp, nroll, axis=dim, center=True)
        pcp = atm.subset(pcp, subset)
        return pcp

    def get_year(filenm, pts_subset, xsample, ysample, xroll, yroll):
        if not os.path.isfile(filenm):
            return None

        with xray.open_dataset(filenm) as ds:
            # Get name of precip variable
            for pcpname in ['PRECTOT', 'PREC', ds.data_vars.keys()[0]]:
                if pcpname in ds.data_vars:
                    break
            pcp = atm.subset(ds[pcpname], pts_subset)
            pcp = smooth_data(pcp, 'lon', xroll)
            pcp = smooth_data(pcp, 'lat', yroll)
            pcp = pcp[:, ::ysample, ::xsample]
            pcp = atm.precip_convert(pcp, pcp.attrs['units'], 'mm/day')
            pcp.load()

        return pcp

    if pts_nm == 'CHP_CMAP':
        if daymax > 366:
            yearmax = year + 1
        else:
            yearmax = year
        pcp = precipdat.read_cmap(filenm, yearmin=year, yearmax=yearmax)
        if pcp.shape[0] > 1:
            pcp = utils.wrapyear(pcp[0], None, pcp[1], 1, daymax, year=year)
        pcp = atm.subset(pcp, pts_subset, squeeze=True)
        pcp = pcp[:, ::ysample, ::xsample]
        # Multiply pentad average rate for cumulative sum
        pcp = pcp * 5.0
    else:
        pcp = get_year(filenm, pts_subset, xsample, ysample, xroll, yroll)
        if daymax > 366:
            filenm_next = filenm.replace('%d' % year, '%d' % (year + 1))
            pcp_next = get_year(filenm_next, pts_subset, xsample, ysample,
                                xroll, yroll)
            pcp = utils.wrapyear(pcp, None, pcp_next, daymin=1, daymax=daymax,
                                 year=year)

    pcp_acc = np.cumsum(pcp, axis=0)
    return pcp_acc

def calc_points(pcp_acc, chp_opts):
    onset = (np.nan * pcp_acc[0]).drop('day')
    retreat = (np.nan * pcp_acc[0]).drop('day')
    lat = atm.get_coord(pcp_acc, 'lat')
    lon = atm.get_coord(pcp_acc, 'lon')
    for i, lat0 in enumerate(lat):
        for j, lon0 in enumerate(lon):
            print('%.1f E, %.1f N' % (lon0, lat0))
            chp = indices.onset_changepoint(pcp_acc[:, i, j], **chp_opts)
            onset.values[i, j] = chp.onset.values
            retreat.values[i, j] = chp.retreat.values
    data = xray.Dataset({'onset' : onset, 'retreat' : retreat})
    return data

def yrly_file(savefile, year, pts_nm):
    filenm = savefile % ('pts_' + pts_nm)
    filenm = filenm.replace(yearstr, '%d.nc' % year)
    return filenm

def gpcp_correct(index, year):
    # Set retreat values to NaN in GPCP 2015 because insufficient data
    if year == 2015:
        index['retreat'].values = np.nan * index['retreat'].values
    return index

if pts_nm is not None:
    daymax = max(chp_opts['retreat_range'])
    if pts_nm == 'CHP_CMAP':
        years = years[years < 2015]
    # Calculate onset/retreat in each year
    for year, filenm in zip(years, datafiles[pts_nm]):
        pcp_acc = get_data(filenm, year, pts_nm, pts_subset, xsample, ysample,
                           xroll=xroll, yroll=yroll, daymax=daymax)
        index = calc_points(pcp_acc, chp_opts)
        atm.disptime()
        if pts_nm == 'CHP_GPCP':
            index = gpcp_correct(index, year)
        filenm = yrly_file(savefile, year, pts_nm)
        print('Saving to ' + filenm)
        index.to_netcdf(filenm)

    # Combine years
    filenm = savefile % ('pts_' + pts_nm)
    filenm = filenm.replace(yearstr, '%d-%d.nc' % (min(years), max(years)))
    files = [yrly_file(savefile, yr, pts_nm) % yr for yr in years]
    ds = atm.load_concat(files, concat_dim='year')
    ds['year'] = years
    print('Saving to ' + filenm)
    ds.to_netcdf(filenm)
