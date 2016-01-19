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
onset_nm = 'CHP_PCP'

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
# MFC and precip over SASM region
nroll = 7
subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}

mfc = atm.combine_daily_years('MFC', mfcfiles, years, yearname='year',
                              subset_dict=subset_dict)
pcp = atm.combine_daily_years('PRECTOT', precipfiles, years, yearname='year',
                              subset_dict=subset_dict)

databox = {'MFC' : mfc, 'PCP' : pcp}
nms = databox.keys()
for nm in nms:
    var = databox[nm]
    var = atm.precip_convert(var, var.attrs['units'], 'mm/day')
    var = atm.mean_over_geobox(var, lat1, lat2, lon1, lon2)
    databox[nm + '_ACC'] = np.cumsum(var, axis=1)
    databox[nm] = atm.rolling_mean(var, nroll, axis=-1, center=True)

tseries = xray.Dataset(databox)

# ----------------------------------------------------------------------
# Monsoon onset day and index timeseries

if onset_nm == 'HOWI':
    maxbreak = 10
    npts = 100
    ds = atm.combine_daily_years(['uq_int', 'vq_int'],vimtfiles, years,
                                 yearname='year')
    index, _ = indices.onset_HOWI(ds['uq_int'], ds['vq_int'], npts,
                                  maxbreak=maxbreak)
    index.attrs['title'] = 'HOWI (N=%d)' % npts
elif onset_nm == 'CHP_MFC':
    index = indices.onset_changepoint(tseries['MFC_ACC'])
elif onset_nm == 'CHP_PCP':
    index = indices.onset_changepoint(tseries['PCP_ACC'])

# Array of onset days
onset = index['onset']
if 'retreat' in index:
    retreat = index['retreat']
else:
    retreat = np.nan * onset

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
keys = [onset_nm, onset_nm + '_clim', 'MFC']
indices.plot_tseries_together(tseries[keys], onset=d_onset, data_style=style,
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
ts_rel = xray.Dataset()
for key in tseries.data_vars:
    ts_rel[key] = daily_rel2onset(tseries[key], onset, npre, npost,
                                  yearnm='year', daynm='day')
    if key.startswith('CHP') or key.endswith('ACC'):
        ts_rel[key] = ts_rel[key] - ts_rel[key][:, 0]

# Composite timeseries
for key in comp_yrs:
    subset_dict = {'year' : (comp_yrs[key], None)}
    comp_ts[key] = atm.subset(tseries, subset_dict)
    comp_ts_rel[key] = atm.subset(ts_rel, subset_dict)

# Plot composites and individual years
def ts_subplot(ts, clim_ts, onset, retreat, enso,  ymin=None, ymax=None,
                daynm='day', title='', legend_loc='upper left',
                onset_lines=False, retreat_lines=False):
    clrs = ['b', 'g', 'r', 'c', 'm']
    days = ts[daynm].values
    yrs = ts['year'].values

    def get_label(s, d1, d2, x, dfmt='%d'):
        s = str(s)
        if np.isnan(d2):
            label = ('%s ' + dfmt + ' %.1f')  % (s, d1, x)
        else:
            label = ('%s ' + dfmt + ' ' + dfmt + ' %.1f')  % (s, d1, d2, x)
        return label

    # Individual years
    for y, year in enumerate(yrs):
        onset_ind = onset.loc[year].values
        retreat_ind = retreat.loc[year].values
        enso_ind = enso.loc[year].values
        label = get_label(year, onset_ind, retreat_ind, enso_ind)
        plt.plot(days, ts[y], clrs[y], label=label)
    # Composite average
    label = get_label('comp', onset.loc[yrs].mean().values,
                      retreat.loc[yrs].mean().values,
                      enso.loc[yrs].mean().values, '%.1f')
    plt.plot(days, ts.mean(dim='year'), linewidth=2, color='k', label=label)
    # Climatology
    label = get_label('clim', onset.mean().values, retreat.mean().values,
                      enso.mean().values, '%.1f')
    plt.plot(days, clim_ts, 'k--', linewidth=2, label=label)
    plt.xlim(days.min(), days.max())
    if ymin is not None:
        plt.ylim(ymin, ymax)
    for cond, ind in zip([onset_lines, retreat_lines], [onset, retreat]):
        if cond:
            for y, year in enumerate(yrs):
                d0 = ind.loc[year].values
                if np.isfinite(d0):
                    plt.plot([d0, d0], plt.gca().get_ylim(), clrs[y])
    plt.grid(True)
    if legend_loc is not None:
        plt.legend(loc=legend_loc, fontsize=10)
    plt.title(title)

def ts_plot_all(comp_ts, ts_clim, comp_keys, varnms, onset, retreat, enso,
                daynm, ylims, legend_var, figsize=(14, 14), onset_lines=False,
                retreat_lines=False,
                subplot_fmts = {'left' : 0.08, 'right' : 0.97, 'bottom' : 0.05,
                                'top' : 0.95, 'wspace' : 0.1, 'hspace' : 0.05}):
    nrow, ncol = len(varnms), len(comp_keys)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
    plt.subplots_adjust(**subplot_fmts)
    suptitle = onset_nm + ' Onset/ Retreat Index'
    if daynm == 'dayrel':
        xlabel = 'Day Since Onset'
        suptitle = suptitle + ', Daily Data Relative to Onset Day'
    else:
        xlabel = 'Day of Year'
    for i, key in enumerate(comp_keys):
        iplot = i + 1
        for nm in varnms:
            plt.subplot(nrow, ncol, iplot)
            row, col = atm.subplot_index(nrow, ncol, iplot)
            ymin, ymax = ylims[nm]
            if row == 1:
                title = key
            else:
                legend_loc = None
                title = ''
            if nm == legend_var:
                legend_loc = 'upper left'
            else:
                legend_loc = None
            ts_subplot(comp_ts[key][nm], ts_clim[nm], onset, retreat, enso,
                       ymin, ymax, daynm, title, legend_loc, onset_lines,
                       retreat_lines)
            if row == nrow:
                plt.xlabel(xlabel)
            else:
                plt.gca().set_xticklabels([])
            if col == 1:
                plt.ylabel(nm)
            iplot += ncol
    plt.suptitle(suptitle)

figsize = (14, 14)
subplot_fmts = {'left' : 0.08, 'right' : 0.97, 'bottom' : 0.05,
                'top' : 0.95, 'wspace' : 0.1, 'hspace' : 0.05}
onset_lines, retreat_lines = True, True
comp_keys = ['Early', 'Late']
varnms = [onset_nm, 'MFC', 'PCP', 'MFC_ACC', 'PCP_ACC']

# Daily data for full year
ylims = {'HOWI' : (-1, 2), 'MFC' : (-4, 10), 'PCP' : (0, 13),
        'MFC_ACC' : (-300, 400), 'PCP_ACC' : (0, 1500)}
ts_plot_all(comp_ts, tseries.mean(dim='year'), comp_keys, varnms, onset,
            retreat, enso, 'day', ylims, onset_nm, figsize, onset_lines,
            retreat_lines, subplot_fmts)

# Daily data relative to onset day
ylims = {'HOWI' : (-1, 2), 'MFC' : (-4, 10), 'PCP' : (0, 13),
        'MFC_ACC' : (0, 600), 'PCP_ACC' : (0, 1400)}
ts_plot_all(comp_ts_rel, ts_rel.mean(dim='year'), comp_keys, varnms, onset,
            retreat, enso, 'dayrel', ylims, 'PCP_ACC', figsize, onset_lines,
            retreat_lines, subplot_fmts)

# ----------------------------------------------------------------------
# Correlations between onset day and ENSO

reg = atm.Linreg(enso, onset)
plt.figure()
reg.plot(annotation_pos=(0.05, 0.85))
plt.xlabel('ENSO (%s)' % enso_nm)
plt.ylabel('Onset Day')
plt.grid()
