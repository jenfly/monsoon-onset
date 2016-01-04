import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation
import collections
import pandas as pd
import atmos as atm
import precipdat
import merra
import indices
from utils import daily_rel2onset, comp_days_centered, composite

# ----------------------------------------------------------------------
years = range(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/daily/'
savedir = 'mp4/'
onsetfile = datadir + 'merra_vimt_ps-300mb_apr-sep_1979-2014.nc'
ensofile = atm.homedir() + 'dynamics/calc/ENSO/enso_oni.csv'
enso_ssn = 'JJA'
enso_nm = 'ONI JJA'

remove_tricky = False
years_tricky = [2002, 2004, 2007, 2009, 2010]

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# Monsoon onset day and index timeseries
onset_nm = 'HOWI'
maxbreak = 10
npts = 100
with xray.open_dataset(onsetfile) as ds:
    uq_int = ds['uq_int'].load()
    vq_int = ds['vq_int'].load()
    howi, _ = indices.onset_HOWI(uq_int, vq_int, npts, maxbreak=maxbreak)
    howi.attrs['title'] = 'HOWI (N=%d)' % npts

# Array of onset days
onset = howi['onset']

# Tile the climatology to each year
tseries_clim = howi['tseries_clim']
vals = atm.biggify(tseries_clim.values, howi['tseries'].values, tile=True)
_, _, coords, dims = atm.meta(howi['tseries'])
tseries_clim = xray.DataArray(vals, name=tseries_clim.name, coords=coords,
                              dims=dims)

# Daily timeseries for each year
tseries = xray.Dataset()
tseries[onset_nm] = howi['tseries']
tseries[onset_nm + '_clim'] = tseries_clim

# ----------------------------------------------------------------------
# MFC over SASM region
print('Calculating MFC')
mfc = atm.moisture_flux_conv(ds['uq_int'], ds['vq_int'], already_int=True)
mfcbar = atm.mean_over_geobox(mfc, lat1, lat2, lon1, lon2)

nroll = 7
mfcbar = atm.rolling_mean(mfcbar, nroll, axis=-1, center=True)
tseries['MFC'] = mfcbar


# ----------------------------------------------------------------------
# ENSO
enso = pd.read_csv(ensofile, index_col=0)
enso = enso[enso_ssn].loc[years]
enso = xray.DataArray(enso).rename({'Year' : 'year'})


# ======================================================================
# PLOTS
# ======================================================================

# ----------------------------------------------------------------------
# Daily timeseries plots together (day of year)
style = {onset_nm : 'k', onset_nm + '_clim' : 'k--', 'MFC' : 'b'}
onset_style = {onset_nm : 'k'}
d_onset = {onset_nm : onset.values}
indices.plot_tseries_together(tseries, onset=d_onset, data_style=style,
                              onset_style=onset_style, show_days=True)

# ----------------------------------------------------------------------
# Daily timeseries composites relative to onset day
keys = [onset_nm, 'MFC']
npre, npost = 30, 90
tseries_rel = xray.Dataset()
for key in keys:
    tseries_rel[key] = daily_rel2onset(tseries[key], onset, npre, npost,
                                       yearnm='year', daynm='day')
dayrel = tseries_rel['dayrel'].values

offset, factor = {}, {}
for key in keys:
    offset[key] = -np.nanmean(tseries[key].values.ravel())
    factor[key] = np.nanstd(tseries[key].values.ravel())


def plot_tseries(dayrel, ind, std, clr, key, xlabel, ylabel):
    plt.plot(dayrel, ind, clr, label=key)
    plt.fill_between(dayrel, ind-std, ind+std, color=clr, alpha=0.2)
    plt.legend(loc='lower right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.autoscale(tight=True)

clrs = {onset_nm : 'b', 'MFC' : 'g'}
plt.figure(figsize=(8, 10))
for i, key in enumerate(keys):
    ind = tseries_rel[key].mean(dim='year')
    std = tseries_rel[key].std(dim='year')

    # Individual timeseries
    plt.subplot(3, 1, i + 1)
    plot_tseries(dayrel, ind, std, clrs[key], key, '', 'Timeseries')
    if i == 0:
        plt.title('1979-2014 Climatological Composites')

    # Standardized timeseries together
    ind = (ind + offset[key]) / factor[key]
    std = std / factor[key]
    plt.subplot(3, 1, 3)
    xlabel = 'Day of year relative to ' + onset_nm + ' onset day'
    ylabel = 'Standardized Timeseries'
    plot_tseries(dayrel, ind, std, clrs[key], key, xlabel, ylabel)

# ----------------------------------------------------------------------
# Composite timeseries

nyrs = 5
comp_ind = {}
comp_ts = {}

# Earliest / latest onset years
onset_sorted = onset.to_series()
onset_sorted.sort()
comp_ind['Early'] = onset_sorted[:nyrs]
comp_ind['Late'] = onset_sorted[-1:-nyrs-1:-1]

# Nino / Nina years
enso_sorted = enso.to_series()
enso_sorted.sort()
comp_ind['Nina'] = enso_sorted[:nyrs]
comp_ind['Nino'] = enso_sorted[-1:-nyrs-1:-1]

comp_yrs = {key : comp_ind[key].index.values for key in comp_ind}

for key in comp_yrs:
    print(key)
    print(comp_ind[key])

# Composite timeseries
for key in comp_yrs:
    comp_ts[key] = atm.subset(tseries[onset_nm], 'year', comp_yrs[key])

# Plot composites and individual years
def ts_subplot(comp_ts, key, onset, enso, clim_ts):
    ts = comp_ts[key]
    days = ts['day'].values
    yrs = ts['year'].values
    for y, year in enumerate(yrs):
        onset_ind = onset.loc[year].values
        enso_ind = enso.loc[year].values
        label = '%d %d %.1f' % (year, onset_ind, enso_ind)
        plt.plot(days, ts[y], label=label)
    label = 'comp %.1f %.1f' % (onset.loc[yrs].mean().values,
                               enso.loc[yrs].mean().values)
    plt.plot(days, ts.mean(dim='year'), linewidth=2, color='k', label=label)
    label = 'clim %.1f %.1f' % (onset.mean().values, enso.mean().values)
    plt.plot(days, clim_ts, 'k--', linewidth=2, label=label)
    plt.xticks(range(75, 275, 25))
    plt.xlim(days.min(), days.max())
    plt.ylim(-1, 2)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=10)
    plt.title(key)

nrow, ncol = 2, 2
fig, axes = plt.subplots(nrow, ncol, figsize=(14, 10), sharex=True, sharey=True)
plt.subplots_adjust(left=0.08, right=0.97, wspace=0.1, hspace=0.2)
for i, key in enumerate(['Early', 'Nina', 'Late', 'Nino']):
    plt.subplot(nrow, ncol, i + 1)
    ts_subplot(comp_ts, key, onset, enso, tseries[onset_nm + '_clim'][0])
    row, col = atm.subplot_index(nrow, ncol, i + 1)
    if row == nrow:
        plt.xlabel('Day of Year')
    if col == 1:
        plt.ylabel(onset_nm)

# ----------------------------------------------------------------------
# Correlations between onset day and ENSO

df = pd.DataFrame()
df['onset'] = onset.to_series()
df['enso'] = enso.to_series()

corr = atm.corr_matrix(df)
r = corr['r'].as_matrix()[1, 0]
p = corr['p'].as_matrix()[1, 0]

plt.figure()
plt.plot(df['enso'], df['onset'], 'ko')
plt.xlabel('ENSO (%s)' % enso_nm)
plt.ylabel('Onset Day')
fmts = {'color' : 'black', 'fontweight' : 'bold', 'fontsize' : 14}
atm.text('r = %.2f' % r, (0.05, 0.9), **fmts)
atm.text('p = %.2f' % p, (0.05, 0.8), **fmts)
