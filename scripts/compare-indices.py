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

years = np.arange(1979, 2015)
ttfiles = [datadir + 'merra_T200-600_apr-sep_%d.nc' % yr for yr in years]

airfile = atm.homedir() + 'datastore/AIR/AIR_JJAS.csv'

# Lat-lon box for WLH method
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# HOWI index (Webster and Fasullo 2003)
maxbreak = 10
with xray.open_dataset(vimtfile) as ds:
    ds.load()
ds_howi = {}
for npts in [50, 100]:
    howi, ds_howi[npts] = indices.onset_HOWI(ds['uq_int'], ds['vq_int'], npts,
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
u850 = u850.rename({'Year' : 'year', 'Day' : 'day'})
index['OCI'] = indices.onset_OCI(u850, yearnm='year', daynm='day')
index['OCI'].attrs['title'] = 'OCI'

# ----------------------------------------------------------------------
# TT index (Goswami et al 2006)

# ***  NOTES ****
# Need to trouble shoot TT index before using in anything final.
# See testing/testing-indices-onset_TT.py for details.

# Select vertical pressure level to use, or None to use 200-600mb
# vertical mean
plev = None

# Read daily data from each year
if plev is None:
    T = atm.combine_daily_years('Tbar', ttfiles, years, yearname='year')
else:
    T = atm.combine_daily_years('T', ttfiles, years, yearname='year',
                                subset1=('plev', plev, plev))
    # Remove extra dimension (vertical)
    pdim = atm.get_coord(T, 'plev', 'dim')
    pname = atm.get_coord(T, 'plev', 'name')
    name, attrs, coords, dims = atm.meta(T)
    dims = list(dims)
    dims.pop(pdim)
    coords = atm.odict_delete(coords, pname)
    T = xray.DataArray(np.squeeze(T.values), dims=dims, coords=coords,
                       name=name, attrs=attrs)

# Calculate index
north=(5, 30, 40, 100)
south=(-15, 5, 40, 100)
index['TT'] = indices.onset_TT(T, north=north, south=south)

# Some weirdness going on in 1991, for now just set to NaN
for nm in ['ttn', 'tts', 'tseries']:
    vals = index['TT'][nm].values
    vals = np.ma.masked_array(vals, abs(vals) > 1e30).filled(np.nan)
    index['TT'][nm].values = vals
index['TT'].attrs['title'] = 'TT'

# ----------------------------------------------------------------------
# Monsoon strength indices

def detrend(vals, index):
    vals_det = scipy.signal.detrend(vals)
    vals_det = vals_det / np.std(vals_det)
    output = pd.Series(vals_det, index=index)
    return output

# MERRA MFC
mfc_JJAS = atm.subset(mfcbar, 'day', atm.season_days('JJAS'))
mfc_JJAS = mfc_JJAS.mean(dim='day')

# ERA-Interim MFC
era = pd.read_csv(eraIfile, index_col=0)

# All India Rainfall Index, convert to mm/day
air = pd.read_csv(airfile, skiprows=4, index_col=0).loc[years]
air /= len(atm.season_days('JJAS'))

strength = mfc_JJAS.to_series().to_frame(name='MERRA')
strength['ERAI'] = era
strength['AIR'] = air

# Detrended indices
for key in strength.columns:
    ind = strength[key].dropna()
    strength[key + '_DET'] = detrend(ind.values.flatten(), ind.index)

# ----------------------------------------------------------------------
# Short names
short = { 'HOWI_50' : 'HOWI_50',
          'HOWI_100' : 'HOWI_100',
          'WLH_CMAP_kmax12' : 'W_C_k12',
          'WLH_CMAP_nroll3' : 'W_C_n3',
          'WLH_CMAP_unsmth' : 'W_C_u',
          'WLH_MERRA_MFC_kmax12' : 'W_MM_k12',
          'WLH_MERRA_MFC_nroll7' : 'W_MM_n7',
          'WLH_MERRA_MFC_unsmth' : 'W_MM_u',
          'WLH_MERRA_PRECIP_kmax12' : 'W_MP_k12',
          'WLH_MERRA_PRECIP_nroll7' : 'W_MP_n7',
          'WLH_MERRA_PRECIP_unsmth' : 'W_MP_u',
          'OCI' : 'OCI',
          'TT' : 'TT'}


# ======================================================================
# PLOTS
# ======================================================================

# ----------------------------------------------------------------------
# Map showing averaging regions
axlims = (-45, 45, 0, 150)
mask = {}
for n in [50, 100]:
    mask1 = ds_howi[n]['mask']
    mask[n] = xray.DataArray(mask1.astype(float), dims=mask1.dims,
                             coords=mask1.coords)

plt.figure(figsize=(10, 7))
m, cs = atm.contour_latlon(mask[100], clev=[0.99], colors='red', axlims=axlims)

_, cs2 = atm.contour_latlon(mask[50], m=m, clev=[0.99], colors='red',
                            linestyles='dashed')

latlon = index['TT'].attrs['north']
atm.geobox(latlon[0],  latlon[1], latlon[2], latlon[3], m=m, color='green',
           label='TT (T200-600)')

latlon = index['TT'].attrs['south']
atm.geobox(latlon[0],  latlon[1], latlon[2], latlon[3], m=m, color='green')

latlon = index['OCI'].attrs['latlon']
atm.geobox(latlon[0],  latlon[1], latlon[2], latlon[3], m=m, color='blue',
           label='OCI (U850)')

atm.geobox(lat1,  lat2, lon1, lon2, m=m, color='magenta',
           label='WLH (MFC, PRECIP)')

plt.legend()
handles, labels = plt.gca().get_legend_handles_labels()
labels = ['HOWI_100 (VIMT)', 'HOWI_50 (VIMT)'] + labels
handles = [cs.collections[0], cs2.collections[0]] + handles
plt.legend(handles, labels, fontsize=12)

plt.title('Averaging Regions for Onset Indices')

if isave:
    for ext in exts:
        atm.savefigs('map_', ext)
plt.close('all')

# ----------------------------------------------------------------------
# Monsoon strength
styles = {'MERRA' : 'r--', 'MERRA_DET' : 'r', 'ERAI' : 'b--', 'ERAI_DET' : 'b',
          'AIR' : 'g--', 'AIR_DET' : 'g'}
strength.plot(figsize=(10,10), style=styles)
plt.title('JJAS Monsoon Strength')
plt.xlabel('Year')
plt.ylabel('Index (mm/day)')
plt.grid()
plt.legend(fontsize=12)
atm.scatter_matrix(strength)
plt.suptitle('JJAS Monsoon Strength')
if isave:
    for ext in exts:
        atm.savefigs('strength_', ext)
plt.close('all')

# ----------------------------------------------------------------------
# Histograms of each index
for key in index.keys():
    ind = index[key]
    if 'retreat' in ind.keys():
        retreat = ind.retreat
    else:
        retreat = None
    indices.summarize_indices(ind.year, ind.onset, retreat, ind.title)
    if isave:
        for ext in exts:
            atm.savefigs('hist_' + key + '_', ext)
plt.close('all')

# ----------------------------------------------------------------------
# Daily timeseries for each year
# keys = index.keys()
keys = ['OCI', 'TT']
for key in keys:
    indices.plot_index_years(index[key], suptitle=key)
    if isave:
        for ext in exts:
            atm.savefigs('tseries_' + key + '_', ext)
    plt.close('all')

# ----------------------------------------------------------------------
# Compare indices with each other

keys = ['HOWI_100', 'HOWI_50', 'OCI', 'TT', 'WLH_CMAP_kmax12',
        'WLH_CMAP_nroll3', 'WLH_MERRA_PRECIP_nroll7']
shortkeys = [short[key] for key in keys]

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

# Scatter plots with correlation coeffs
titlestr = 'Yearly Indices 1979-2014'
atm.scatter_matrix(ind_comp, corr_fmt='.2f', corr_pos=(0.1, 0.85),
                   figsize=(16,10), suptitle=titlestr)

if isave:
    for ext in exts:
        atm.savefigs('onset_', ext)
        plt.close('all')

# ----------------------------------------------------------------------
# Plot onset day vs. year along with histograms

# Subset of indices to focus on
keys_sub = ['HOWI_100', 'OCI', 'TT', 'WLH_CMAP_kmax12',
            'WLH_MERRA_PRECIP_nroll7']
shortkeys_sub = [short[key] for key in keys_sub]
n = len(keys_sub)

fig, axes = plt.subplots(n, 2, figsize=(10,10), sharex='col')
plt.subplots_adjust(left=0.08, right=0.95, wspace=0.2, hspace=0.3)
plt.suptitle('Onset Day Indices')
for i, key in enumerate(shortkeys_sub):
    ind = ind_comp[key]
    ax1, ax2 = axes[i, 0], axes[i, 1]
    ind.plot(ax=ax1)
    ax1.set_ylabel('Day')
    ax1.set_title(key, fontsize=12)
    if i == len(shortkeys_sub) - 1:
        ax1.set_xlabel('Year')
    indices.plot_hist(ind.values, binwidth=5, ax=ax2)
    if i < len(shortkeys_sub) - 1:
        ax2.set_xlabel('')
    ax2.set_ylabel('# Years')

# ----------------------------------------------------------------------
# Daily timeseries together

data = xray.Dataset()
for key in keys_sub:
    data[short[key]] = index[key]['tseries']

key_onset = 'HOWI_100'
d_onset = index[key_onset]['onset'].values
suptitle = key_onset + ' Onset'
indices.plot_tseries_together(data, onset=d_onset, suptitle=suptitle)

# Correlations between daily timeseries
def daily_corr(ind1, ind2, yearnm='year'):
    if ind1.name == ind2.name:
        raise ValueError('ind1 and ind2 have the same name ' + ind1.name)
    years = ind1[yearnm]
    corr = np.zeros(years.shape)
    for y, year in enumerate(years):
        df = atm.subset(ind1, yearnm, year).to_series().to_frame(name=ind1.name)
        df[ind2.name] = atm.subset(ind2, yearnm, year).to_series()
        corr[y] = df.corr().as_matrix()[0, 1]
    corr = pd.Series(corr, index=pd.Index(years, name=yearnm))
    return corr

def daily_corr_years(data, keys, yearnm='year'):
    years = data[yearnm].values
    corr = {}
    ones = pd.Series(np.ones(years.shape, dtype=float),
                     index=pd.Index(years, name=yearnm))
    for key1 in keys:
        ind1 = data[key1]
        corr[key1] = pd.DataFrame()
        for key2 in keys:
            if key2 == key1:
                corr[key1][key2] = ones
            else:
                corr[key1][key2] = daily_corr(ind1, data[key2])
    return corr

corr = daily_corr_years(data, shortkeys_sub)
n = len(shortkeys_sub)
ylim1, ylim2 = 0, 1
plt.figure(figsize=(8, 10))
for i, key in enumerate(shortkeys_sub):
    plt.subplot(n, 1, i + 1)
    corr[key].boxplot(column=shortkeys_sub)
    plt.ylim(ylim1, ylim2)
    plt.ylabel(key)
    if i < n - 1:
        plt.gca().set_xticklabels([])
plt.suptitle('Correlations between daily tseries')
