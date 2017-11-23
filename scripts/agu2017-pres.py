import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pandas as pd

import atmos as atm
import indices
import utils

style = atm.homedir() + 'dynamics/python/mpl-styles/presentation.mplstyle'
plt.style.use(style)
fontsize = mpl.rcParams['font.size']
labelsize = fontsize + 3
dashes = [6, 2]

version = 'merra2'
years = np.arange(1980, 2016)
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
onset_nm = 'CHP_MFC'
onset_nms = ['CHP_MFC', 'MOK', 'HOWI', 'OCI']
pcp_nm = 'GPCP'
varnms = ['PRECTOT', 'U200', 'V200', 'U850', 'V850']
lat_extract = {'U200' : 0, 'V200' : 15, 'U850' : 15, 'V850' : 15}
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
nroll = 5 # n-day rolling averages for smoothing daily timeseries
ind_nm, npre, npost = 'onset', 120, 200
#ind_nm, npre, npost = 'retreat', 270, 89
fracmin = 0.5 # Precip JJAS frac of total for gridpoint masking

yearstr = '%d-%d.nc' % (min(years), max(years))
filestr = datadir + version + '_index_%s_' + yearstr
indfiles = collections.OrderedDict()
for nm in ['CHP_MFC', 'HOWI', 'OCI']:
    indfiles[nm] = filestr % nm
indfiles['MOK'] = atm.homedir() + 'dynamics/python/monsoon-onset/data/MOK.dat'
filestr2 = datadir + version + '_%s_dailyrel_' + onset_nm + '_' + yearstr
datafiles = {nm : filestr2 % nm for nm in varnms}
datafiles['CMAP'] = datadir + 'cmap_dailyrel_' + onset_nm + '_1980-2014.nc'
datafiles['GPCP'] = datadir + 'gpcp_dailyrel_' + onset_nm + '_1997-2015.nc'
mfcbudget_file = datadir + version + '_mfc_budget_' + yearstr


# -----------------------------
# Plot map of precip and winds pre- and post- onset
datadir0 = datadir.replace('analysis', 'daily')
filestr0 = datadir0 + 'merra2_%s850_daily_clim_1980-2015.nc'
data0 = {}
for nm in ['U', 'V']:
    with xr.open_dataset(filestr0 % nm) as ds:
        data0[nm] = ds[nm].load()
filenm0 = 'datastore/gpcp/gpcp_daily_1997-2014.nc'
with xr.open_dataset(atm.homedir() + filenm0) as ds:
    data0['GPCP'] = ds['PREC'].load()


def precip_winds(data0, days, ax=None, axlims=(-30, 45, 40, 120), climits=(0, 20),
                 cticks=np.arange(4, 21, 2), clev=np.arange(4, 20.5, 1),
                 cmap='hot_r', scale=250, dx=5, dy=5,
                 xticks=range(40, 121, 20), yticks=range(-20, 41, 10),
                 box=(10, 30, 60, 100)):
    lat1, lat2, lon1, lon2 = axlims
    subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}

    u = data0['U'].sel(day=days).mean(dim='day')
    v = data0['V'].sel(day=days).mean(dim='day')
    pcp = data0['GPCP'].sel(day=days).mean(dim='day')
    u = u[::dy, ::dx]
    v = v[::dy, ::dx]
    lat = atm.get_coord(u, 'lat')
    lon = atm.get_coord(u, 'lon')

    if ax is not None:
        plt.sca(ax)
    m = atm.init_latlon(lat1, lat2, lon1, lon2, coastlines=False)
    m.drawcoastlines(color='k', linewidth=0.5)
    m.shadedrelief(scale=0.3)
    atm.contourf_latlon(pcp, clev=clev, axlims=axlims, m=m, cmap=cmap,
                        extend='max', colorbar=False)
    m.colorbar(ticks=cticks, location='bottom', size='4%', pad='10%')
    #atm.pcolor_latlon(pcp, axlims=axlims, cmap=cmap, cb_kwargs={'extend' : 'max'})
    plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
    plt.yticks(yticks, atm.latlon_labels(yticks, 'lat'))
    plt.clim(climits)
    #plt.quiver(lon, lat, u, v, linewidths=spd.values.ravel())
    plt.quiver(lon, lat, u, v, scale=scale, pivot='middle')
    if box is not None:
        atm.geobox(*box, m=m, color='k', linewidth=3)
    return m


#days1 = atm.season_days('MAM')
#days2 = atm.season_days('JJAS')
days1 = range(atm.mmdd_to_jday(4, 29), atm.mmdd_to_jday(5, 3))
days2 = range(atm.mmdd_to_jday(6, 13), atm.mmdd_to_jday(6, 18))

#box=(10, 30, 60, 100)
box = None
gridspec_kw = {'left' : 0.05, 'right' : 0.95, 'bottom' : 0.05, 'top' : 0.95,
               'wspace' : 0.05}
fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex='all', sharey='all',
                         gridspec_kw=gridspec_kw)

precip_winds(data0, days1, ax=axes[0], box=box)
precip_winds(data0, days2, ax=axes[1], box=box)
fig.savefig('figs/agu2017_01.png')

# Map of averaging region
axlims=(0, 35, 55, 105)
xticks, yticks = range(60, 101, 10), range(0, 31, 10)
plt.figure(figsize=(4, 3.5))
m = atm.init_latlon(*axlims, coastlines=False,  resolution='l')
m.drawcoastlines(color='k', linewidth=0.5, ax=plt.gca())
atm.mapticks(xticks, yticks)
m.shadedrelief(scale=0.3)
atm.geobox(10, 30, 60, 100, m=m, color='k', linewidth=3)
plt.savefig('figs/agu2017_02.png')
# ----------------------------------------------------------------------
# Read indices and dailyrel data

# Large-scale onset/retreat indices
index_all = collections.OrderedDict()
for nm in indfiles:
    print('Loading ' + indfiles[nm])
    if nm == 'MOK':
        mok = indices.onset_MOK(indfiles['MOK'], yearsub=years)
        index_all['MOK'] =  xr.Dataset({'onset' : mok})
    else:
        with xr.open_dataset(indfiles[nm]) as ds:
            index_all[nm] = ds.load()
index = index_all[onset_nm]
index['length'] = index['retreat'] - index['onset']

onset_all = pd.DataFrame()
for nm in index_all:
    onset_all[nm] = index_all[nm]['onset'].to_series()
onset_all = onset_all.astype(int)


# MFC budget
with xr.open_dataset(mfcbudget_file) as mfc_budget:
    mfc_budget.load()
mfc_budget = mfc_budget.rename({'DWDT' : 'dw/dt'})
mfc_budget = mfc_budget.drop('DWDT_ANA')
if nroll is not None:
    for nm in mfc_budget.data_vars:
        mfc_budget[nm] = atm.rolling_mean(mfc_budget[nm], nroll, center=True)
mfc_budget['CMFC'] = index['tseries']
df = mfc_budget.sel(year=2000).drop('year').to_dataframe()
df =  df[(df.index > 2) & (df.index < 399)]

# Dailyrel climatology
keys_dict = {'PRECTOT' : 'PRECTOT', 'CMAP' : 'precip', 'GPCP' : 'PREC',
             'U200' : 'U', 'U850' : 'U', 'V200' : 'V', 'V850' : 'V'}
data = {}
for nm in datafiles:
    print('Loading ' + datafiles[nm])
    with xr.open_dataset(datafiles[nm]) as ds:
        if 'year' in ds.dims:
            ds = ds.mean(dim='year')
        data[nm] = ds[keys_dict[nm]].load()

# Daily timeseries
ts = xr.Dataset()
for nm in ['GPCP', 'PRECTOT']:
    ts[nm] = atm.mean_over_geobox(data[nm], lat1, lat2, lon1, lon2)
ts['MFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['daily_ts'],
                                  index[ind_nm], npre + 5, npost + 5)
ts['CMFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['tseries'],
                                   index[ind_nm], npre + 5, npost + 5)


# Extract variables at specified latitudes
for nm, lat0 in lat_extract.iteritems():
    var = atm.dim_mean(data[nm], 'lon', lon1, lon2)
    lat = atm.get_coord(var, 'lat')
    lat0_str = atm.latlon_labels(lat0, 'lat', deg_symbol=False)
    # key = nm + '_' + lat0_str
    key = nm
    lat_closest, _ = atm.find_closest(lat, lat0)
    print '%s %.2f %.2f' % (nm, lat0, lat_closest)
    ts[key] = atm.subset(var, {'lat' : (lat_closest, None)}, squeeze=True)

# Compute climatology and smooth with rolling mean
if 'year' in ts.dims:
    ts = ts.mean(dim='year')
if nroll is not None:
    for nm in ts.data_vars:
        ts[nm] = atm.rolling_mean(ts[nm], nroll, center=True)
tseries = atm.subset(ts, {'dayrel' : (-npre, npost)})

tseries_df = tseries[['MFC', 'GPCP', 'CMFC', 'U850', 'V850']].to_dataframe()


# ----------------------------------------------------------------------------
# MFC Budget and changepoint linear fits

def plot_legend(axes, legend_kw={'fontsize' : 9, 'loc' : 'upper left',
                                 'handlelength' : 2.5, 'fontsize' : 14}):
    if isinstance(axes, list) and len(axes) > 1:
        atm.legend_2ax(axes[0], axes[1], **legend_kw)
    else:
        plt.legend(**legend_kw)

def plot_mfc(days, ts, nms=['Precipitation', 'Evaporation', 'MFC'],
             legend=True, dashes=(6, 2),  labelpad=1.5,
             axes_pos=None):
    styles = {'Precipitation' : {'color' : 'k', 'linestyle' : '--', 'dashes' : dashes},
              'Evaporation' : {'color' : '0.7', 'linewidth' : 2},
              'MFC' : {'color' : 'k', 'linewidth' : 1}}

    for nm in nms:
        plt.plot(days, ts[nm].values, label=nm, **styles[nm])
    plt.xlabel('Day of Year')
    plt.ylabel('mm day$^{-1}$', labelpad=labelpad)
    if axes_pos is not None:
        plt.gca().set_position(axes_pos)
    if legend:
        plot_legend(plt.gca())

def plot_cmfc(days, ind, twin=False, fit='onset', legend=True, labelpad=1.5,
              axes_pos=None):
    if axes_pos is not None:
        plt.gca().set_position(axes_pos)
    if twin:
        ax1 = plt.gca()
        ax2 = plt.twinx()
        plt.sca(ax2)
        axes = [ax1, ax2]
    else:
        axes = plt.gca()
    plt.plot(days, ind['tseries'], 'r', alpha=0.6, linewidth=2,
             label='Cumulative MFC')
    if fit is not None:
        ts_fit = ind['tseries_fit_' + fit]
        label = fit.capitalize() + ' Linear Fit'
        plt.plot(ts_fit['day'].values, ts_fit, 'b', label=label,  linewidth=2)
        plt.axvline(ind[fit], color='b')
    if twin:
        atm.fmt_axlabels('y', 'mm', color='r', alpha=0.6)
    plt.gca().set_ylabel('mm', labelpad=labelpad)
    if not twin:
        plt.xlabel('Day of Year')
    if legend:
        plot_legend(axes)

year = 2000
figsize = (10, 6)
#pos = mpl.transforms.Bbox(np.array([[0.125,0.1],[0.9, 0.9]]))
pos = None
nms_dict = {'PRECTOT' : 'Precipitation', 'EVAP' : 'Evaporation'}
ts = mfc_budget.sel(year=year).rename(nms_dict)
ind = index.sel(year=year)
days = ts['day'].values
nms = ['MFC']

figs = []
plt.figure(figsize=figsize)
plot_mfc(days, ts, nms=nms)
#figs.append(plt.gcf())
#plt.figure(figsize=figsize)
#plot_cmfc(days, ind, twin=False, fit=None)
plot_cmfc(days, ind, twin=True, fit=None)
figs.append(plt.gcf())
plt.figure(figsize=figsize)
plot_cmfc(days, ind, twin=False, fit='onset', axes_pos=pos)
figs.append(plt.gcf())
plt.figure(figsize=figsize)
plot_cmfc(days, ind, twin=False, fit='retreat', axes_pos=pos)
figs.append(plt.gcf())
plt.figure(figsize=figsize)
plot_mfc(days, ts, nms=nms)
plot_cmfc(days, ind, twin=True, fit=None, axes_pos=pos)
for x0 in ind['onset'], ind['retreat']:
    plt.axvline(x0, color='b')
figs.append(plt.gcf())
for i, fig in enumerate(figs):
    fig.savefig('figs/agu2017_mfc_%02d.png' % (i + 1))

# -------------------------------------------------------------------------
# Onset day vs. year

plt.figure(figsize=(6, 4))
plt.plot(index['year'].values, index['onset'].values, 'k')
plt.xlabel('Year')
plt.ylabel('Onset (day of year)')
plt.savefig('figs/agu2017_ind.png')

# -------------------------------------------------------------------------
def daily_tseries(tseries, index, npre, npost,dashes=[6, 2], labelpad=1.5):
    """Plot dailyrel timeseries climatology"""
    xlims = (-npre, npost)
    xticks = range(-npre, npost + 10, 30)
    days = tseries['dayrel'].values
    plt.plot(days, tseries['MFC'], 'k', label='MFC')
    plt.plot(days, tseries['GPCP'], 'k', linewidth=2, label='Precipitation')
    plt.xlabel('Days Since Onset')
    plt.ylabel('mm day$^{-1}$')
    plt.xticks(xticks)
    plt.xlim(xlims)
    x0_list = [0,  index['length'].mean(dim='year')]
    for x0 in x0_list:
        plt.axvline(x0, color='b')
    plt.axvline(15, color='b', linestyle='--', dashes=dashes)

    ax1 = plt.gca()
    ax2 = plt.twinx()
    plt.sca(ax2)
    #label = 'U$_{850}$ (15$^{\circ}$N)'
    label = 'U$_{850}$'
    plt.plot(days, tseries['U850'], 'm', linewidth=2, alpha=0.7,
             label=label)
    atm.fmt_axlabels('y', 'm s$^{-1}$', color='m', alpha=0.7)
    plt.gca().set_ylabel('m s$^{-1}$', labelpad=labelpad)
    legend_kw = {'fontsize' : 13, 'loc' : 'upper left',
                 'handlelength' : 2}
    atm.legend_2ax(ax1, ax2, **legend_kw)

plt.figure(figsize=(6, 4))
daily_tseries(tseries, index, npre, npost)
plt.savefig('figs/agu2017_ts.png')

    #
    #
    # keypairs = [(['MFC', pcp_nm], ['CMFC']), (['U850'], ['V850'])]
    # opts = [('upper left', 'mm day$^{-1}$', 'mm'),
    #         ('upper left', '   m s$^{-1}$', '   m s$^{-1}$')]
    # ylim_list = [(-3.5, 9), (-7, 15)]
    # y2_opts={'color' : 'r', 'alpha' : 0.6}
    # dashed = {'color' : 'k', 'linestyle' : '--', 'dashes' : dashes}
    # styles = ['k', dashed, 'g', 'm']
    # legend_kw = {}
    # for pair, opt, ylims in zip(keypairs, opts, ylim_list):
    #     grp.next()
    #     keys1, keys2 = pair
    #     legend_kw['loc'] = opt[0]
    #     y1_label = opt[1]
    #     y2_label = opt[2]
    #     data1 = tseries[keys1]
    #     if keys2 is not None:
    #         data2 = tseries[keys2]
    #     else:
    #         data2 = None
    #     data1_styles = {nm : style for (nm, style) in zip(keys1, styles)}
    #     axs = utils.plotyy(data1, data2, xname='dayrel', data1_styles=data1_styles,
    #                        y2_opts=y2_opts, xlims=xlims, xticks=xticks, ylims=ylims,
    #                        xlabel=xlabel, y1_label=y1_label, y2_label=y2_label,
    #                        legend=legend, legend_kw=legend_kw, x0_axvlines=x0,
    #                        grid=grid)
    #     for ax, label in zip(axs, [y1_label, y2_label]):
    #         ax.set_ylabel(label, labelpad=labelpad)
    #     plt.gca().set_xticklabels(xtick_labels)
    #     if dlist is not None:
    #         for d0 in dlist:
    #             plt.axvline(d0, color='k', linestyle='--', dashes=dashes)
