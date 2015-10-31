import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import collections
import pandas as pd
import scipy.signal
import atmos as atm
import precipdat
import merra
import indices

# ----------------------------------------------------------------------
isave = False
exts = ['png', 'eps']
index = collections.OrderedDict()

datadir = atm.homedir() + 'datastore/merra/daily/'
vimtfile = datadir + 'merra_vimt_ps-300mb_apr-sep_1979-2014.nc'
precipfile = datadir + 'merra_precip_40E-120E_60S-60N_days91-274_1979-2014.nc'
cmapfile = atm.homedir() + 'datastore/cmap/cmap.precip.pentad.mean.nc'
eraIfile = atm.homedir() + ('datastore/era_interim/analysis/'
                            'era_interim_JJAS_60-100E_index.csv')
ocifile = datadir + 'merra_u850_40E-120E_60S-60N_1979-2014.nc'

# Lat-lon box for WLH method
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# HOWI index (Webster and Fasullo 2003)
maxbreak = 10
with xray.open_dataset(vimtfile) as ds:
    ds.load()
for npts in [50, 100]:
    howi, _ = indices.onset_HOWI(ds['uq_int'], ds['vq_int'], npts,
                                 maxbreak=maxbreak)
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
        vals = indices.onset_WLH_1D(pcp_sm[y], threshold, precip_jan=pcp_jan)
        i_onset[y], i_retreat[y], i_peak[y] = vals

    # Convert from pentads to day of year
    if pentad:
        d_onset = [int(atm.pentad_to_jday(i, pmin=0)) for i in i_onset]
        d_retreat = [int(atm.pentad_to_jday(i, pmin=0)) for i in i_retreat]
    else:
        d_onset = [np.nan if np.isnan(i) else days[int(i)] for i in i_onset]
        d_retreat = [np.nan if np.isnan(i) else days[int(i)] for i in i_retreat]

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


# Threshold and smoothing parameters
threshold = 5.0
kmax = 12
nroll = {'CMAP' : 3, 'MERRA_MFC' : 7, 'MERRA_PRECIP' : 7}

# Read CMAP pentad precip
cmap = precipdat.read_cmap(cmapfile)
cmapbar = atm.mean_over_geobox(cmap, lat1, lat2, lon1, lon2)
cmapdays = [atm.pentad_to_jday(p, pmin=1) for p in cmap.pentad.values]

# MERRA moisture flux convergence
print('Calculating MFC')
mfc = atm.moisture_flux_conv(ds['uq_int'], ds['vq_int'], already_int=True)
mfcbar = atm.mean_over_geobox(mfc, lat1, lat2, lon1, lon2)

# MERRA precip
print('Reading MERRA precip ' + precipfile)
with xray.open_dataset(precipfile) as dsprecip:
    precip = dsprecip['PRECTOT']
    precipbar = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)

# Compute indices for each dataset
for name in ['CMAP', 'MERRA_MFC', 'MERRA_PRECIP']:
    print('****' + name + '******')
    if name == 'CMAP':
        pcp = cmapbar
        days = cmapdays
        pentad = True
        precip_jan = None # Calculate from pentad data
    elif name == 'MERRA_MFC':
        pcp = mfcbar
        days = mfcbar.day.values
        pentad = False
        precip_jan = 0.0 # Use zero for now
    elif name == 'MERRA_PRECIP':
        pcp = precipbar
        days = precipbar.day.values
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
# OCI index (Wang et al 2009)
with xray.open_dataset(ocifile) as ds:
    u850 = ds['U'].load()

index['OCI'] = indices.onset_OCI(u850)

# ----------------------------------------------------------------------
# Summary plots
for key in index.keys():
    ind = index[key]
    indices.summarize_indices(ind.year, ind.onset, ind.retreat, ind.title)
    if isave:
        for ext in exts:
            atm.savefigs('summary_' + key + '_', ext)
        plt.close('all')

# Plot daily timeseries for each year
keys = index.keys()
for key in keys:
    indices.plot_index_years(index[key], suptitle=key)
    if isave:
        for ext in exts:
            atm.savefigs('tseries_' + key + '_', ext)
        plt.close('all')

# Compare onset indices to each other
short = { 'HOWI_50' : 'HOWI50',
          'HOWI_100' : 'HOWI100',
          'WLH_CMAP_kmax12' : 'W_C_k12',
          'WLH_CMAP_nroll3' : 'W_C_n3',
          'WLH_CMAP_unsmth' : 'W_C_u',
          'WLH_MERRA_MFC_kmax12' : 'W_MM_k12',
          'WLH_MERRA_MFC_nroll7' : 'W_MM_n7',
          'WLH_MERRA_MFC_unsmth' : 'W_MM_u',
          'WLH_MERRA_PRECIP_kmax12' : 'W_MP_k12',
          'WLH_MERRA_PRECIP_nroll7' : 'W_MP_n7',
          'WLH_MERRA_PRECIP_unsmth' : 'W_MP_u',
          'OCI' : 'OCI'}

# Subset of keys to include in correlation calcs
keys = ['HOWI_100', 'HOWI_50', 'OCI', 'WLH_CMAP_kmax12', 'WLH_CMAP_nroll3',
        'WLH_MERRA_PRECIP_nroll7']
shortkeys = [short[key] for key in keys]
years = index[keys[0]].year.values
onset = np.reshape(index[keys[0]].onset.values, (len(years), 1))
for key in keys[1:]:
    ind = np.reshape(index[key].onset.values, (len(years), 1))
    onset = np.concatenate([onset, ind], axis=1)
onset = pd.DataFrame(onset, index=years, columns=shortkeys)

# Plot matrix of scatter plots with correlation coeffs
titlestr = 'Onset Day - Scatter Plots and Correlation Coefficients'
atm.scatter_matrix(onset, corr_fmt='.2f', corr_pos=(0.1, 0.85), figsize=(16,10),
                   suptitle=titlestr)

if isave:
    for ext in exts:
        atm.savefigs('onset_scatter', ext)
        plt.close('all')

# ----------------------------------------------------------------------
# Monsoon strength indices

def detrend(vals, index):
    vals_det = scipy.signal.detrend(vals)
    vals_det = vals_det / np.std(vals_det)
    output = pd.Series(vals_det, index=index)
    return output

mfc_JJAS = atm.subset(mfcbar, 'day', atm.season_days('JJAS'))
mfc_JJAS = mfc_JJAS.mean(dim='day')
mfc_det = detrend(mfc_JJAS.values, mfc_JJAS.year)

era = pd.read_csv(eraIfile, index_col=0)
era_det = detrend(era.values.flatten(), era.index)

strength = mfc_JJAS.to_series().to_frame(name='MERRA')
strength['MERRA_DET'] = mfc_det
strength['ERAI'] = era
strength['ERAI_DET'] = era_det

strength.plot()
plt.title('JJAS Monsoon Strength')
plt.xlabel('Year')
plt.ylabel('Index (mm/day)')

atm.scatter_matrix(strength)
