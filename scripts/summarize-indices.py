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

mpl.rcParams['font.size'] = 10

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
# Summarize onset, retreat, indices in dataframes

onset = pd.DataFrame()
for nm in index:
    onset[nm] = index[nm]['onset'].to_series()
corr = onset.corr()

styles = {'CHP_MFC' : {'color' : 'k', 'linewidth' : 2},
          'HOWI' : {'color' : 'b'}, 'OCI' : {'color' : 'r'},
          'SJKE' : {'color' : 'g'}}
labels = {}
for nm in onset.columns:
    labels[nm] = nm
    if nm != onset_nm:
        labels[nm] = labels[nm] + ' %.2f' % corr[onset_nm][nm]

plt.figure(figsize=(8, 6))
for nm in onset.columns.values:
    plt.plot(years, onset[nm], label=labels[nm], **styles[nm])
plt.legend(fontsize=9, loc='upper left')
plt.grid()
plt.xlim(min(years) - 1, max(years) + 1)
plt.xlabel('Year')
plt.ylabel('Onset (Day of Year)')
