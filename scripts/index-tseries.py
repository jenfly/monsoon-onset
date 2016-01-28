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
onset_nm = 'CHP_MFC'
#onset_nm = 'CHP_PCP'

years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/daily/'
vimtfiles = [datadir + 'merra_vimt_ps-300mb_%d.nc' % yr for yr in years]
mfcfiles = [datadir + 'merra_MFC_ps-300mb_%d.nc' % yr for yr in years]
precipfiles = [datadir + 'merra_precip_%d.nc' % yr for yr in years]
ensofile = atm.homedir() + 'dynamics/calc/ENSO/enso_oni.csv'
enso_ssn = 'JJA'
enso_nm = 'ONI JJA'
plot_all_years = False

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# MFC and precip over SASM region
# **** Make this a function in utils.py ***************************

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
    databox[nm + '_UNSM'] = var
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
    length = retreat - onset
    index['length'] = length
else:
    retreat = np.nan * onset
    index['length'] = np.nan * onset

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

# ----------------------------------------------------------------------
# Climatology
index_clim = index.mean(dim='year')
tseries_clim = tseries.mean(dim='year')
enso_clim = enso.mean(dim='year').values

# ----------------------------------------------------------------------
# Timeseries relative to onset, shifted to 0 at onset day
npre, npost = 0, 200
tseries_rel = xray.Dataset()
for key in tseries.data_vars:
    tseries_rel[key] = daily_rel2onset(tseries[key], onset, npre, npost,
                                       yearnm='year', daynm='day')
    if key.startswith('CHP') or key.endswith('ACC'):
        tseries_rel[key] = tseries_rel[key] - tseries_rel[key][:, 0]

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

# Composite timeseries
# for key in comp_yrs:
#     subset_dict = {'year' : (comp_yrs[key], None)}
#     comp_ts[key] = atm.subset(tseries, subset_dict)
#     comp_ts_rel[key] = atm.subset(ts_rel, subset_dict)

# ======================================================================
# PLOTS
# ======================================================================

# ----------------------------------------------------------------------
if plot_all_years:
    # --- Daily timeseries plots together (day of year)
    style = {onset_nm : 'k', onset_nm + '_clim' : 'k--', 'MFC' : 'b'}
    onset_style = {onset_nm : 'k'}
    d_onset = {onset_nm : onset.values}
    keys = [onset_nm, onset_nm + '_clim', 'MFC']
    indices.plot_tseries_together(tseries[keys], onset=d_onset, data_style=style,
                                  onset_style=onset_style, show_days=True)

    # --- Daily timeseries composites relative to onset day - clim +/- 1 stdev
    keys = [onset_nm, 'MFC']
    npre, npost = 30, 90
    ts_rel = xray.Dataset()
    for key in keys:
        ts_rel[key] = daily_rel2onset(tseries[key], onset, npre, npost,
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
        ind = ts_rel[key].mean(dim='year')
        std = ts_rel[key].std(dim='year')

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

# # ----------------------------------------------------------------------
# # Plot composites and individual years
# def ts_subplot(ts, clim_ts, onset, retreat, enso,  ymin=None, ymax=None,
#                 daynm='day', title='', legend_loc='upper left',
#                 onset_lines=False, retreat_lines=False):
#     clrs = ['b', 'g', 'r', 'c', 'm']
#     days = ts[daynm].values
#     yrs = ts['year'].values
#
#     def get_label(s, d1, d2, x, dfmt='%d'):
#         s = str(s)
#         if np.isnan(d2):
#             label = ('%s ' + dfmt + ' %.1f')  % (s, d1, x)
#         else:
#             label = ('%s ' + dfmt + ' ' + dfmt + ' %.1f')  % (s, d1, d2, x)
#         return label
#
#     # Individual years
#     for y, year in enumerate(yrs):
#         onset_ind = onset.loc[year].values
#         retreat_ind = retreat.loc[year].values
#         enso_ind = enso.loc[year].values
#         label = get_label(year, onset_ind, retreat_ind, enso_ind)
#         plt.plot(days, ts[y], clrs[y], label=label)
#     # Composite average
#     label = get_label('comp', onset.loc[yrs].mean().values,
#                       retreat.loc[yrs].mean().values,
#                       enso.loc[yrs].mean().values, '%.1f')
#     plt.plot(days, ts.mean(dim='year'), linewidth=2, color='k', label=label)
#     # Climatology
#     label = get_label('clim', onset.mean().values, retreat.mean().values,
#                       enso.mean().values, '%.1f')
#     plt.plot(days, clim_ts, 'k--', linewidth=2, label=label)
#     plt.xlim(days.min(), days.max())
#     if ymin is not None:
#         plt.ylim(ymin, ymax)
#     for cond, ind in zip([onset_lines, retreat_lines], [onset, retreat]):
#         if cond:
#             for y, year in enumerate(yrs):
#                 d0 = ind.loc[year].values
#                 if np.isfinite(d0):
#                     plt.plot([d0, d0], plt.gca().get_ylim(), clrs[y])
#     plt.grid(True)
#     if legend_loc is not None:
#         plt.legend(loc=legend_loc, fontsize=10)
#     plt.title(title)
#
# def ts_plot_all(comp_ts, ts_clim, comp_keys, varnms, onset, retreat, enso,
#                 daynm, ylims, legend_var, figsize=(14, 14), onset_lines=False,
#                 retreat_lines=False,
#                 subplot_fmts = {'left' : 0.08, 'right' : 0.97, 'bottom' : 0.05,
#                                 'top' : 0.95, 'wspace' : 0.1, 'hspace' : 0.05}):
#     nrow, ncol = len(varnms), len(comp_keys)
#     fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
#     plt.subplots_adjust(**subplot_fmts)
#     suptitle = onset_nm + ' Onset/ Retreat Index'
#     if daynm == 'dayrel':
#         xlabel = 'Day Since Onset'
#         suptitle = suptitle + ', Daily Data Relative to Onset Day'
#     else:
#         xlabel = 'Day of Year'
#     for i, key in enumerate(comp_keys):
#         iplot = i + 1
#         for nm in varnms:
#             plt.subplot(nrow, ncol, iplot)
#             row, col = atm.subplot_index(nrow, ncol, iplot)
#             ymin, ymax = ylims[nm]
#             if row == 1:
#                 title = key
#             else:
#                 legend_loc = None
#                 title = ''
#             if nm == legend_var:
#                 legend_loc = 'upper left'
#             else:
#                 legend_loc = None
#             ts_subplot(comp_ts[key][nm], ts_clim[nm], onset, retreat, enso,
#                        ymin, ymax, daynm, title, legend_loc, onset_lines,
#                        retreat_lines)
#             if row == nrow:
#                 plt.xlabel(xlabel)
#             else:
#                 plt.gca().set_xticklabels([])
#             if col == 1:
#                 plt.ylabel(nm)
#             iplot += ncol
#     plt.suptitle(suptitle)
#
# figsize = (14, 14)
# subplot_fmts = {'left' : 0.08, 'right' : 0.97, 'bottom' : 0.05,
#                 'top' : 0.95, 'wspace' : 0.1, 'hspace' : 0.05}
# onset_lines, retreat_lines = True, True
# comp_keys = ['Early', 'Late']
# varnms = [onset_nm, 'MFC', 'PCP', 'MFC_ACC', 'PCP_ACC']
#
# # Daily data for full year
# ylims = {'HOWI' : (-1, 2), 'MFC' : (-4, 9), 'PCP' : (0, 13),
#         'MFC_ACC' : (-300, 400), 'PCP_ACC' : (0, 1500)}
# ylims['CHP_MFC'] = ylims['MFC_ACC']
# ts_plot_all(comp_ts, tseries.mean(dim='year'), comp_keys, varnms, onset,
#             retreat, enso, 'day', ylims, 'PCP_ACC', figsize, onset_lines,
#             retreat_lines, subplot_fmts)
#
# # Daily data relative to onset day
# ylims = {'HOWI' : (-1, 2), 'MFC' : (-4, 9), 'PCP' : (0, 13),
#         'MFC_ACC' : (0, 600), 'PCP_ACC' : (0, 1400)}
# ylims['CHP_MFC'] = ylims['MFC_ACC']
# ts_plot_all(comp_ts_rel, ts_rel.mean(dim='year'), comp_keys, varnms,
#             0.0*onset, index['length'], enso, 'dayrel', ylims, 'PCP_ACC',
#             figsize, onset_lines, retreat_lines, subplot_fmts)


# ----------------------------------------------------------------------
# Composites - individual years and averages

def ts_subplot(ts, ts_clim, d_onset, d_retreat, d_onset_clim, d_retreat_clim,
               enso_val, enso_val_clim, year, title='', ymin=None, ymax=None,
               daynm='day', legend_loc='upper left', onset_lines=True,
               retreat_lines=True,
               clim_leg=False):

    def get_label(s, d1, d2, x, dfmt='%d'):
        s = str(s)
        if np.isnan(d2):
            label = ('%s ' + dfmt + ' %.1f')  % (s, d1, x)
        else:
            label = ('%s ' + dfmt + ' ' + dfmt + ' %.1f')  % (s, d1, d2, x)
        return label

    days = ts[daynm].values

    # Single year
    if year == 'comp':
        lw, dfmt = 2, '%.1f'
    else:
        lw, dfmt = 1, '%.0f'
    label = get_label(year, d_onset, d_retreat, enso_val, dfmt)
    plt.plot(days, ts, 'k', label=label, linewidth=lw)

    # Climatology
    if clim_leg:
        label = get_label('clim', d_onset_clim, d_retreat_clim,
                          enso_val_clim, '%.1f')
    else:
        label = None
    plt.plot(days, ts_clim, 'k--', linewidth=2, label=label)

    # Onset/retreat lines and formatting
    plt.xlim(days.min(), days.max())
    if ymin is not None:
        plt.ylim(ymin, ymax)
    if onset_lines:
        plt.plot([d_onset, d_onset], plt.gca().get_ylim(), 'k')
        plt.plot([d_onset_clim, d_onset_clim], plt.gca().get_ylim(), 'k--')
    if retreat_lines:
        plt.plot([d_retreat, d_retreat], plt.gca().get_ylim(), 'k')
        plt.plot([d_retreat_clim, d_retreat_clim], plt.gca().get_ylim(), 'k--')
    plt.grid(True)
    if legend_loc is not None:
        plt.legend(loc=legend_loc, fontsize=10)
    plt.title(title)

def yearly_values(years, tsdata, index, enso):
    data = {}
    data['ts'] = tsdata.sel(year=years)
    data['onset'] = index['onset'].sel(year=years)
    data['retreat'] = index['retreat'].sel(year=years)
    data['enso'] = enso.sel(year=years)
    # Take average for multiple years
    if len(atm.makelist(years)) > 1:
        for key in data:
            data[key] = data[key].mean(dim='year')
    return data['ts'], data['onset'], data['retreat'], data['enso']


def ts_plot_all(tsdata, comp_yrs, keys, varnm, index, enso, ymin, ymax,
                suptitle='', figsize=(14, 14),
                subplot_fmts={'left' : 0.08, 'right' : 0.97, 'bottom' : 0.05,
                              'top' : 0.95, 'wspace' : 0.1, 'hspace' : 0.05}):

    daynm = tsdata.dims.keys()[0]
    ts_clim = tsdata[varnm].mean(dim='year')
    nrow, ncol = len(comp_yrs[keys[0]]) + 1, len(keys)
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
    plt.subplots_adjust(**subplot_fmts)
    plt.suptitle(suptitle, fontsize=14)

    for j, key in enumerate(keys):
        col = j + 1
        compyrs = comp_yrs[key]
        for i in range(nrow):
            plt.subplot(nrow, ncol, col + i*ncol)
            if i == 0:
                title = key
            else:
                title = ''
            if i < nrow - 1:
                # Individual year
                year, clim_leg = compyrs[i], False
                ts, d1, d2, enso_val = yearly_values(year, tsdata[varnm],
                                                     index, enso)
            else:
                # Composite average
                year, clim_leg = 'comp', True
                ts, d1, d2, enso_val = yearly_values(compyrs, tsdata[varnm],
                                                     index, enso)
            if daynm == 'dayrel':
                d1, d2 = 0, d2 -d1
                xlabel = 'Days Since Onset'
            else:
                xlabel = 'Day of Year'
            ts_subplot(ts, ts_clim, d1, d2, index_clim['onset'],
                       index_clim['retreat'], enso_val, enso_clim, year, title,
                       ymin, ymax, daynm=daynm, clim_leg=clim_leg)
            if i == nrow - 1:
                plt.xlabel(xlabel)
            else:
                plt.gca().set_xticklabels([])


ylims = {'HOWI' : (-1, 2), 'MFC' : (-4, 9), 'PCP' : (0, 13),
        'MFC_ACC' : (-300, 400), 'PCP_ACC' : (0, 1500)}

ylims_rel = {'HOWI' : (-1, 2), 'MFC' : (-4, 9), 'PCP' : (0, 13),
           'MFC_ACC' : (0, 600), 'PCP_ACC' : (0, 1400)}

figsize = (10, 10)
keys = ['Early', 'Late']
varnms = [onset_nm, 'MFC', 'PCP', 'MFC_ACC', 'PCP_ACC']
if onset_nm.startswith('CHP'):
    varnms.remove(onset_nm)

# Plot daily data for full year and daily data relative to onset day
for i, tsdata in enumerate([tseries, tseries_rel]):
    for varnm in varnms:
        if i == 0:
            ymin, ymax = ylims[varnm]
        else:
            ymin, ymax = ylims_rel[varnm]
        suptitle = '%s (%s Onset/Retreat Index)' % (varnm, onset_nm)
        ts_plot_all(tsdata, comp_yrs, keys, varnm, index, enso, ymin, ymax,
                    suptitle, figsize)

# ----------------------------------------------------------------------
# Variability in Accumulated precip / MFC over climatology
for ts, daynm in zip([tseries, tseries_rel], ['day', 'dayrel']):
    plt.figure(figsize=(8, 10))
    for i, key in enumerate(['MFC_ACC', 'PCP_ACC']):
        plt.subplot(2, 1, i + 1)
        tsbar = ts[key].mean(dim='year')
        tsplus = tsbar + ts[key].std(dim='year')
        tsminus = tsbar - ts[key].std(dim='year')
        tsmax = ts[key].max(dim='year')
        tsmin = ts[key].min(dim='year')
        days = tsbar[daynm]
        plt.plot(days, tsbar, 'b', label = 'Mean')
        plt.plot(days, tsplus, 'b--', label='Mean +/ 1 Std')
        plt.plot(days, tsminus, 'b--')
        plt.plot(days, tsmax, 'k-.', label='Max / Min')
        plt.plot(days, tsmin, 'k-.')
        plt.legend(loc='upper left', fontsize=12)
        plt.grid()
        plt.xlim(days.min(), days.max())
        plt.title(key)
        if daynm == 'day':
            plt.xlabel('Day of Year')
        else:
            plt.xlabel('Days Since Onset')
        plt.ylabel(key)

# ----------------------------------------------------------------------
# Cumulative and average rainfall over monsoon season

ssn = xray.Dataset()
ssn['onset'] = onset
ssn['retreat'] = retreat
ssn['length'] = retreat - onset

for key in ['MFC', 'PCP']:
    for key2 in ['_JJAS_AVG', '_JJAS_TOT', '_LRS_AVG', '_LRS_TOT']:
        ssn[key + key2] = xray.DataArray(np.nan * np.ones(len(years)),
                                         coords={'year' : years})

for key in ['MFC', 'PCP']:
    for y, year in enumerate(years):
        d1 = int(onset.values[y])
        d2 = int(retreat.values[y] - 1)
        days_jjas = atm.season_days('JJAS', atm.isleap(year))
        data = tseries[key + '_UNSM'].sel(year=year)
        data_jjas = atm.subset(data, {'day' : (days_jjas, None)})
        data_lrs = atm.subset(data, {'day' : (d1, d2)})
        ssn[key + '_JJAS_AVG'][y] = data_jjas.mean(dim='day').values
        ssn[key + '_LRS_AVG'][y] = data_lrs.mean(dim='day').values
        ssn[key + '_JJAS_TOT'][y] = ssn[key + '_JJAS_AVG'][y] * len(days_jjas)
        ssn[key + '_LRS_TOT'][y] = ssn[key + '_LRS_AVG'][y] * ssn['length'][y]
ssn = ssn.to_dataframe()

def line_plus_reg(years, ssn, key, clr):
    reg = atm.Linreg(years, ssn[key].values)
    plt.plot(years, ssn[key], clr, label=key)
    plt.plot(years, reg.predict(years), clr + '--')


plt.figure(figsize=(12, 10))
clrs = ['b', 'g', 'r', 'c']
for i, nm in enumerate(['TOT', 'AVG']):
    plt.subplot(2, 2, i + 1)
    for j, varnm in enumerate(['MFC_JJAS_', 'MFC_LRS_', 'PCP_JJAS_', 'PCP_LRS_']):
        key = varnm + nm
        line_plus_reg(years, ssn, key, clrs[j])
    if nm == 'TOT':
        plt.ylabel('Total (mm)')
    else:
        plt.ylabel('Avg (mm/day)')
plt.subplot(2, 2, 3)
line_plus_reg(years, ssn, 'onset', clrs[0])
line_plus_reg(years, ssn, 'length', clrs[1])
plt.subplot(2, 2, 4)
line_plus_reg(years, ssn, 'retreat', clrs[0])
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.xlabel('Year')
    plt.xlim(years.min(), years.max())
plt.suptitle('Monsoon Onset/Retreat Based on ' + onset_nm)

df1 = ssn[['onset', 'retreat', 'length']]
for key in ['_TOT', '_AVG']:
    keys = [nm + key for nm in ['MFC_JJAS', 'MFC_LRS', 'PCP_JJAS', 'PCP_LRS']]
    df2 = ssn[keys]
    atm.scatter_matrix_pairs(df1, df2)
    plt.suptitle('Monsoon Onset/Retreat Based on ' + onset_nm)
