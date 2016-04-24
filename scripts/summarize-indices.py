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

mpl.rcParams['font.size'] = 11

# ----------------------------------------------------------------------
onset_nm = 'CHP_MFC'
years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/daily/'
datafiles = {}
filestr = datadir + 'merra_%s_%s_%d.nc'
datafiles['CHP_MFC'] = [filestr % ('MFC', '40E-120E_90S-90N', y) for y in years]
datafiles['HOWI'] = [filestr % ('vimt', 'ps-300mb', y) for y in years]
datafiles['OCI'] = [filestr % ('uv850', '40E-120E_60S-60N', y) for y in years]
datafiles['SJKE'] = datafiles['OCI']

# Lat-lon box
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# Read data and calculate onset/retreat indices
index = collections.OrderedDict()

# Accumulated MFC changepoint (Cook & Buckley 2009)
mfc = atm.combine_daily_years('MFC', datafiles['CHP_MFC'], years,
                              yearname='year')
mfcbar = atm.mean_over_geobox(mfc, lat1, lat2, lon1, lon2)
mfc_acc = np.cumsum(mfcbar, axis=1)
index['CHP_MFC'] = indices.onset_changepoint(mfc_acc)

# HOWI index (Webster and Fasullo 2003)
npts, maxbreak = 100, 10
ds = atm.combine_daily_years(['uq_int', 'vq_int'], datafiles['HOWI'], years,
                             yearname='year')
index['HOWI'], _ = indices.onset_HOWI(ds['uq_int'], ds['vq_int'], npts,
                                   maxbreak=maxbreak)

# 850 mb U, V for OCI and SJKE
ds = atm.combine_daily_years(['U', 'V'], datafiles['OCI'], years)
ds = ds.rename({'Year' : 'year', 'Day' : 'day'})
u850, v850 = atm.squeeze(ds['U']), atm.squeeze(ds['V'])

# OCI index (Wang et al 2009)
index['OCI'] = indices.onset_OCI(u850, yearnm='year', daynm='day')

# SJKE index (Boos and Emmanuel 2009)
thresh_std = 0.7
nroll = 7
u_sm = atm.rolling_mean(u850, nroll, axis=1, center=True)
v_sm = atm.rolling_mean(v850, nroll, axis=1, center=True)
index['SJKE'] = indices.onset_SJKE(u_sm, v_sm, thresh_std=thresh_std,
                                   yearnm='year', daynm='day')

# ----------------------------------------------------------------------
# Plot daily timeseries and fit in a single year
year = 2014
ts = index[onset_nm]['tseries'].sel(year=year)
ts_onset = index[onset_nm]['tseries_fit_onset'].sel(year=year)
ts_retreat = index[onset_nm]['tseries_fit_retreat'].sel(year=year)
days = index[onset_nm]['tseries']['day']

figsize = (5, 3.5)
gs_kw = {'left' : 0.15, 'bottom' : 0.15}
plt.subplots(1, 1, figsize=figsize, gridspec_kw=gs_kw)
plt.plot(days, ts, 'k')
plt.plot(days, ts_onset, 'r')
plt.plot(days, ts_retreat, 'b')
plt.grid()
plt.xlim(0, 366)
plt.xlabel('Day')
plt.ylabel('MFC_ACC (mm)')
atm.text(year, (0.05, 0.9))


# ----------------------------------------------------------------------
# Summarize onset, retreat, indices in dataframes

onset = pd.DataFrame()
for nm in index:
    onset[nm] = index[nm]['onset'].to_series()
corr = onset.corr()
retreat = index[onset_nm]['retreat'].to_series()
length = retreat - onset[onset_nm]

# Labels for onset indices correlations
labels = {}
for nm in onset.columns:
    labels[nm] = nm
    if nm != onset_nm:
        labels[nm] = labels[nm] + ' %.2f' % corr[onset_nm][nm]

xticks = np.arange(1980, 2016, 5)
xticklabels = [1980, '', 1990, '', 2000, '', 2010, '']

nrow, ncol = 2, 3
fig_kw = {'figsize' : (11, 7)}
gridspec_kw = {'left' : 0.08, 'right' : 0.94, 'wspace' : 0.2, 'hspace' : 0.3,
               'bottom' : 0.08, 'top' : 0.92}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)

# Onset index with other onset indices
grp.next()
for nm in onset.columns:
    plt.plot(years, onset[nm], label=labels[nm], **styles[nm])
plt.legend(fontsize=9, loc='upper left', ncol=2)
plt.grid()
plt.xlim(min(years) - 1, max(years) + 1)
plt.xticks(xticks, xticklabels)
plt.xlabel('Year')
plt.ylabel('Day of Year')
plt.title('Onset')

# Yearly timeseries of retreat and length
for ind, nm in zip([retreat, length], ['Retreat', 'Length']):
    grp.next()
    plt.plot(years, ind, **styles[onset_nm])
    plt.grid()
    plt.xticks(xticks, xticklabels)
    plt.xlim(min(years) - 1, max(years) + 1)
    plt.xlabel('Year')
    plt.title(nm)

# Histograms
pos = (0.05, 0.68)
for ind, flag in zip([onset[onset_nm], retreat, length], [True, True, False]):
    grp.next()
    indices.plot_hist(ind, pos=pos, incl_daystr=flag)
    if grp.col == 0:
        plt.ylabel('# Occurrences')
    else:
        plt.ylabel('')
