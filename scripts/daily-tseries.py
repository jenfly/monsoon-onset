import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xarray as xray
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
import utils as utils

# ----------------------------------------------------------------------
#onset_nm = 'HOWI'
onset_nm = 'CHP_MFC'
#onset_nm = 'CHP_PCP'

enso_nm = 'ONI_JJA'
#enso_nm = 'ONI_MAM'

comp_keys, savestr = ['Early', 'Late'], 'early_late'
#comp_keys, savestr = ['Long', 'Short'], 'long_short'
#comp_keys, savestr = ['Nina (%s)' % enso_nm, 'Nino (%s)' % enso_nm], 'nina_nino'

years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/daily/'
vimtfiles = [datadir + 'merra_vimt_ps-300mb_%d.nc' % yr for yr in years]
mfcfiles = [datadir + 'merra_MFC_ps-300mb_%d.nc' % yr for yr in years]
precipfiles = [datadir + 'merra_precip_%d.nc' % yr for yr in years]
savedir = 'figs/'

plot_all_years = False
plot_acc_clim = False

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# Files for different onset indices
indfiles = { 'CHP_MFC' : mfcfiles, 'CHP_PCP' : precipfiles, 'HOWI' : vimtfiles}

# ----------------------------------------------------------------------
# Read data and calculate indices

# MFC and precip over SASM region
nroll = 7
tseries = utils.get_mfc_box(mfcfiles, precipfiles, None, years, nroll, lat1, lat2,
                            lon1, lon2)

# Monsoon onset day and index timeseries
if onset_nm.startswith('CHP'):
    # Use precip/MFC already loaded
    data = tseries[onset_nm.split('_')[1] + '_ACC']
else:
    data = None
index = utils.get_onset_indices(onset_nm, indfiles[onset_nm], years, data)
onset, retreat, length = index['onset'], index['retreat'], index['length']
tseries[onset_nm] = index['tseries']

# ENSO
enso = utils.get_enso_indices(years)
enso = xray.DataArray(enso[enso_nm]).rename({'Year' : 'year'})

# ----------------------------------------------------------------------
# Climatology

index_clim = index.mean(dim='year')
tseries_clim = tseries.mean(dim='year')
enso_clim = enso.mean(dim='year').values

# Tile the climatology to each year for plot_tseries_together
vals = atm.biggify(tseries_clim[onset_nm], index['tseries'].values, tile=True)
_, _, coords, dims = atm.meta(index['tseries'])
ts_clim = xray.DataArray(vals, name=tseries_clim[onset_nm].name, coords=coords,
                         dims=dims)
tseries[onset_nm + '_clim'] = ts_clim

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

def lowest_highest(ind, nyrs):
    ind_sorted = ind.to_series()
    ind_sorted.sort()
    lowest = ind_sorted[:nyrs].index.values
    highest = ind_sorted[-1:-nyrs-1:-1].index.values
    return lowest, highest

# Composite definitions
nyrs = 5
comp_yrs = {}
comp_yrs['Early'], comp_yrs['Late'] = lowest_highest(onset, nyrs)
comp_yrs['Short'], comp_yrs['Long'] = lowest_highest(length, nyrs)
key1, key2 = 'Nina (%s)' % enso_nm, 'Nino (%s)' % enso_nm
comp_yrs[key1], comp_yrs[key2] = lowest_highest(enso, nyrs)
atm.print_odict(comp_yrs)


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
    dayrel = ts_rel['dayrel'].values

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
                suptitle='', figsize=(14, 14), legend_loc='upper_left',
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
            d1_clim, d2_clim = index_clim['onset'], index_clim['retreat']
            if daynm == 'dayrel':
                d1, d2 = 0, d2 - d1
                d1_clim, d2_clim = 0, d2_clim - d1_clim
                xlabel = 'Days Since Onset'
            else:
                xlabel = 'Day of Year'
            ts_subplot(ts, ts_clim, d1, d2, d1_clim, d2_clim, enso_val,
                       enso_clim, year, title, ymin, ymax, daynm=daynm,
                       legend_loc=legend_loc, clim_leg=clim_leg)
            if i == nrow - 1:
                plt.xlabel(xlabel)
            else:
                plt.gca().set_xticklabels([])


ylims = {'HOWI' : (-1, 2), 'MFC' : (-4, 9), 'PCP' : (0, 13),
        'MFC_ACC' : (-300, 400), 'PCP_ACC' : (0, 1500)}

ylims_rel = {'HOWI' : (-1, 2), 'MFC' : (-4, 9), 'PCP' : (0, 13),
           'MFC_ACC' : (0, 600), 'PCP_ACC' : (0, 1400)}

figsize = (16, 12)
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
        if varnm in ['MFC', 'PCP', 'HOWI'] and i == 1:
            legend_loc = 'lower left'
        else:
            legend_loc = 'upper left'
        suptitle = '%s (%s Onset/Retreat)' % (varnm, onset_nm)
        ts_plot_all(tsdata, comp_yrs, comp_keys, varnm, index, enso, ymin, ymax,
                    suptitle, figsize, legend_loc=legend_loc)

filestr = 'daily_tseries-onset_%s-enso_%s-%s' % (onset_nm, enso_nm, savestr)
atm.savefigs(savedir + filestr, 'pdf', merge=True)
plt.close('all')

# ----------------------------------------------------------------------
# Variability in Accumulated precip / MFC over climatology
if plot_acc_clim:
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
