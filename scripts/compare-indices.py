import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import collections
import pandas as pd
import atmos as atm
import precipdat
import merra
from indices import (onset_WLH, onset_WLH_1D, onset_HOWI, summarize_indices,
                     plot_index_years)

isave = True
exts = ['png', 'eps']
index = collections.OrderedDict()

# ----------------------------------------------------------------------
# Compute HOWI indices (Webster and Fasullo 2003)
datadir = atm.homedir() + 'datastore/merra/daily/'
datafile = datadir + 'merra_vimt_ps-300mb_may-sep_1979-2014.nc'
maxbreak = 10

ds = atm.ncload(datafile)
for npts in [50, 100]:
    howi, _ = onset_HOWI(ds['uq_int'], ds['vq_int'], npts, maxbreak=maxbreak)
    howi.attrs['title'] = 'HOWI (N=%d)' % npts
    index['HOWI_%d' % npts] = howi

# ----------------------------------------------------------------------
# Wang & LinHo method

def get_onset_WLH(years, days, pcp_sm, threshold, titlestr, pentad=True,
                  pcp_jan=None):
    nyears = len(years)
    i_onset = np.zeros(nyears)
    i_retreat = np.zeros(nyears)
    i_peak = np.zeros(nyears)
    for y, year in enumerate(years):
        i_onset[y], i_retreat[y], i_peak[y] = onset_WLH_1D(pcp_sm[y], threshold,
                                                           precip_jan=pcp_jan)

    # Convert from pentads to day of year
    if pentad:
        d_onset = [int(atm.pentad_to_jday(i, pmin=0)) for i in i_onset]
        d_retreat = [int(atm.pentad_to_jday(i, pmin=0)) for i in i_retreat]
    else:
        d_onset = days[i_onset.astype(int)]
        d_retreat = days[i_retreat.astype(int)]

    # Pack into Dataset
    index = xray.Dataset()
    days = xray.DataArray(days, {'day' : days})
    years = xray.DataArray(years, {'year' : years})
    index['tseries'] = xray.DataArray(pcp_sm, dims=['year', 'day'],
                                      coords={'year' : years, 'day': days})
    index['onset'] = xray.DataArray(d_onset, coords={'year' : years})
    index['retreat'] = xray.DataArray(d_retreat, coords={'year' : years})
    index.attrs['title'] = titlestr
    return index

# Lat-lon box
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# Threshold and smoothing parameters
threshold = 5.0
kmax = 12
nroll = {'CMAP' : 3, 'MERRA_MFC' : 7}

# Read CMAP pentad precip
datafile = atm.homedir() + 'datastore/cmap/cmap.precip.pentad.mean.nc'
precip = precipdat.read_cmap(datafile)
precipbar = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)
cmapdays = [atm.pentad_to_jday(p, pmin=1) for p in precip.pentad.values]

# MERRA moisture flux convergence
print('Calculating MFC')
mfc = atm.moisture_flux_conv(ds['uq_int'], ds['vq_int'], already_int=True)
mfcbar = atm.mean_over_geobox(mfc, lat1, lat2, lon1, lon2)

# Compute indices for each dataset
for name in ['CMAP', 'MERRA_MFC']:
    print('****' + name + '******')
    if name == 'CMAP':
        pcp = precipbar
        days = cmapdays
        pentad = True
        precip_jan = None # Calculate from pentad data
    elif name == 'MERRA_MFC':
        pcp = mfcbar
        days = mfcbar.day.values
        pentad = False
        precip_jan = 0.0 # Use zero for now
    years = pcp.year.values

    key = 'WLH_%s_kmax%d' % (name, kmax)
    print(key)
    pcp_sm, Rsq = atm.fourier_smooth(pcp, kmax)
    index[key] = get_onset_WLH(years, days, pcp_sm, threshold, key, pentad,
                               precip_jan)

    # Smooth with rolling mean
    key = 'WLH_%s_nroll%d' % (name, nroll[name])
    print(key)
    pcp_sm = np.zeros(pcp.shape)
    for y in range(pcp.shape[0]):
        pcp_sm[y] = pd.rolling_mean(pcp[y].values, nroll[name],
                                    center=True)
    index[key] = get_onset_WLH(years, days, pcp_sm, threshold, key, pentad,
                               precip_jan)

    # Unsmoothed pentad timeserires
    key = 'WLH_%s_unsmth' % name
    print(key)
    index[key] = get_onset_WLH(years, days, pcp, threshold, key, pentad,
                               precip_jan)

# ----------------------------------------------------------------------
# Summary plots
for key in index.keys():
    ind = index[key]
    summarize_indices(ind.year, ind.onset, ind.retreat, ind.title)
    if isave:
        for ext in exts:
            atm.savefigs('summary_' + key + '_', ext)
        plt.close('all')

# Plot daily timeseries for each year
tseries = index.keys()
for key in tseries:
    plot_index_years(index[key], suptitle=key)
    if isave:
        for ext in exts:
            atm.savefigs('tseries_' + key + '_', ext)
        plt.close('all')
