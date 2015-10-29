import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import atmos as atm
import precipdat
import merra
from indices import (onset_WLH, onset_WLH_1D, onset_HOWI, summarize_indices,
                     plot_index_years)

isave = True
exts = ['png', 'eps']
index = {}

# ----------------------------------------------------------------------
# Compute HOWI indices (Webster and Fasullo 2003)
datadir = atm.homedir() + 'datastore/merra/daily/'
datafile = datadir + 'merra_vimt_ps-300mb_may-sep_1979-2014.nc'
ds = atm.ncload(datafile)

for npts in [50, 100]:
    howi, _ = onset_HOWI(ds['uq_int'], ds['vq_int'], npts)
    howi.attrs['title'] = 'HOWI (N=%d)' % npts
    index['HOWI_%d' % npts] = howi

# ----------------------------------------------------------------------
# Wang & LinHo, CMAP precip
datafile = atm.homedir() + 'datastore/cmap/cmap.precip.pentad.mean.nc'

# Read data and average over box
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
titlestr = 'CMAP %d-%dE, %d-%dN ' % (lon1, lon2, lat1, lat2)
precip = precipdat.read_cmap(datafile)
precipbar = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)
years = precip.year.values
days = [atm.pentad_to_jday(p, pmin=1) for p in precip.pentad.values]

def get_onset_WLH(years, days, pcp_sm, threshold, titlestr):
    nyears = len(years)
    i_onset = np.zeros(nyears)
    i_retreat = np.zeros(nyears)
    i_peak = np.zeros(nyears)
    for y, year in enumerate(years):
        i_onset[y], i_retreat[y], i_peak[y] = onset_WLH_1D(pcp_sm[y], threshold)

    # Convert from pentads to day of year
    d_onset = [int(atm.pentad_to_jday(i, pmin=0)) for i in i_onset]
    d_retreat = [int(atm.pentad_to_jday(i, pmin=0)) for i in i_retreat]

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

# Threshold for onset criteria
threshold = 5.0

# Smooth with truncated Fourier series
kmax = 12
key = 'WLH_CMAP_kmax%d' % kmax
pcp_sm, Rsq = atm.fourier_smooth(precipbar, kmax)
index[key] = get_onset_WLH(years, days, pcp_sm, threshold, key)

# Smooth with rolling mean
nroll = 3
key = 'WLH_CMAP_nroll%d' % nroll
pcp_sm = np.zeros(precipbar.shape)
for y in range(precipbar.shape[0]):
    pcp_sm[y] = pd.rolling_mean(precipbar[y].values, nroll, center=True)
index[key] = get_onset_WLH(years, days, pcp_sm, threshold, key)

# Unsmoothed pentad timeserires
key = 'WLH_CMAP_unsmth'
index[key] = get_onset_WLH(years, days, precipbar, threshold, key)

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
    plot_index_years(index[key])
    if isave:
        for ext in exts:
            atm.savefigs('tseries_' + key + '_', ext)
        plt.close('all')
