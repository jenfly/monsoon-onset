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
#onset_nm = 'HOWI'
#onset_nm = 'CHP_MFC'
onset_nm = 'CHP_PRECIP'

years = range(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/daily/'
vimtfiles = [datadir + 'merra_vimt_ps-300mb_%d.nc' % yr for yr in years]
mfcfiles = [datadir + 'merra_MFC_ps-300mb_%d.nc' % yr for yr in years]
precipfiles = [datadir + 'merra_precip_%d.nc' % yr for yr in years]
ensofile = atm.homedir() + 'dynamics/calc/ENSO/enso_oni.csv'
enso_ssn = 'JJA'
enso_nm = 'ONI JJA'

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# MFC over SASM region
mfc = atm.combine_daily_years('MFC', mfcfiles, years, yearname='year')
mfcbar = atm.mean_over_geobox(mfc, lat1, lat2, lon1, lon2)
mfc_acc = np.cumsum(mfcbar, axis=1)

nroll = 7
mfcbar = atm.rolling_mean(mfcbar, nroll, axis=-1, center=True)
tseries = xray.Dataset()
tseries['MFC'] = mfcbar

# ----------------------------------------------------------------------
# Monsoon onset day and index timeseries

if onset_nm == 'HOWI':
    maxbreak = 10
    npts = 100
    ds = atm.combine_daily_years(['uq_int', 'vq_int'],vimtfiles, years,
                                 yearname='year')
    index, _ = indices.onset_HOWI(ds['uq_int'], ds['vq_int'], npts, maxbreak=maxbreak)
    index.attrs['title'] = 'HOWI (N=%d)' % npts
elif onset_nm == 'CHP_MFC':
    index = indices.onset_changepoint(mfc_acc)
elif onset_nm == 'CHP_PRECIP':
    precip = atm.combine_daily_years('PRECTOT', precipfiles, years, yearname='year',
                                     subset1=('lat', lat1, lat2),
                                     subset2=('lon', lon1, lon2))
    precip = atm.precip_convert(precip, precip.attrs['units'], 'mm/day')
    precipbar = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)
    precip_acc = np.cumsum(precipbar, axis=1)
    index = indices.onset_changepoint(precip_acc)

# Array of onset days
onset = index['onset']

# Tile the climatology to each year
if 'tseries_clim' in index:
    tseries_clim = index['tseries_clim']
else:
    tseries_clim = index['tseries'].mean(dim='year')
vals = atm.biggify(tseries_clim.values, index['tseries'].values, tile=True)
_, _, coords, dims = atm.meta(index['tseries'])
tseries_clim = xray.DataArray(vals, name=tseries_clim.name, coords=coords,
                              dims=dims)

# Daily timeseries for each year
tseries[onset_nm] = index['tseries']
tseries[onset_nm + '_clim'] = tseries_clim

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
comp_ts_rel = {}

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

# Timeseries relative to onset, shifted to 0 at onset day
npre, npost = 0, 200
ts_rel = daily_rel2onset(tseries[onset_nm], onset, npre, npost, yearnm='year',
                         daynm='day')
if onset_nm.startswith('CHP'):
    ts_rel = ts_rel - ts_rel[:, 0]

# Composite timeseries
for key in comp_yrs:
    comp_ts[key] = atm.subset(tseries[onset_nm], 'year', comp_yrs[key])
    comp_ts_rel[key] = atm.subset(ts_rel, 'year', comp_yrs[key])

# Plot composites and individual years
def ts_subplot(comp_ts, key, onset, enso, clim_ts, ymin=None, ymax=None,
                daynm='day'):
    ts = comp_ts[key]
    days = ts[daynm].values
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
    #plt.xticks(range(75, 275, 25))
    plt.xlim(days.min(), days.max())
    if ymin is not None:
        plt.ylim(ymin, ymax)
    plt.grid(True)
    plt.legend(loc='upper left', fontsize=10)
    plt.title(key)


def ts_plot_all(comp_ts, ts_clim, onset, enso, ymin, ymax, daynm):
    nrow, ncol = 2, 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 10), sharex=True, sharey=True)
    plt.subplots_adjust(left=0.08, right=0.97, wspace=0.1, hspace=0.2)
    if daynm == 'dayrel':
        xlabel = 'Day Since Onset'
        suptitle = onset_nm + ' Index, Relative to Onset Day'
    else:
        xlabel = 'Day of Year'
        suptitle = onset_nm + ' Index'
    for i, key in enumerate(['Early', 'Nina', 'Late', 'Nino']):
        plt.subplot(nrow, ncol, i + 1)
        ts_subplot(comp_ts, key, onset, enso, ts_clim, ymin, ymax, daynm)
        row, col = atm.subplot_index(nrow, ncol, i + 1)
        if row == nrow:
            plt.xlabel(xlabel)
        if col == 1:
            plt.ylabel(onset_nm)
    plt.suptitle(suptitle)

ylims = {'HOWI' : (-1, 2), 'CHP_MFC' : (-400, 400), 'CHP_PRECIP' : (0, 1700)}
ymin, ymax = ylims[onset_nm]
ts_plot_all(comp_ts, tseries[onset_nm + '_clim'][0], onset, enso,  ymin, ymax,
            'day')
ts_plot_all(comp_ts_rel, ts_rel.mean(dim='year'), onset, enso,  ymin, ymax,
            'dayrel')            
# ----------------------------------------------------------------------
# Correlations between onset day and ENSO

reg = atm.Linreg(enso, onset)
plt.figure()
reg.plot(annotation_pos=(0.05, 0.85))
plt.xlabel('ENSO (%s)' % enso_nm)
plt.ylabel('Onset Day')
plt.grid()
