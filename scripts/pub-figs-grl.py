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
version = 'merra2'
years = np.arange(1980, 2016)
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
onset_nm = 'CHP_MFC'
onset_nms = ['CHP_MFC', 'MOK', 'HOWI', 'OCI']
pts_nm = 'CHP_PCP'
pcp_nm = 'PRECTOT'
varnms = ['PRECTOT', 'U200', 'V200', 'U850', 'V850']
lat_extract = {'U200' : 0, 'V200' : 15, 'U850' : 15, 'V850' : 15}
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
nroll = 5 # n-day rolling averages for smoothing daily timeseries
npre, npost = 120, 200

yearstr = '%d-%d.nc' % (min(years), max(years))
filestr = datadir + version + '_index_%s_' + yearstr
indfiles = {nm : filestr % nm for nm in ['CHP_MFC', 'HOWI', 'OCI']}
indfiles['MOK'] = atm.homedir() + 'dynamics/python/monsoon-onset/data/MOK.dat'
filestr2 = datadir + version + '_%s_dailyrel_' + onset_nm + '_' + yearstr
datafiles = {nm : filestr2 % nm for nm in varnms}
datafiles['CMAP'] = datadir + 'cmap_dailyrel_' + onset_nm + '_1980-2014.nc'
ptsfile = datadir + version + '_index_pts_%s_' % pts_nm
if pts_nm == 'CHP_CMAP':
    ptsfile = ptsfile + '1980-2014.nc'
    pts_xroll, pts_yroll = None, None
else:
    ptsfile = ptsfile + yearstr
    pts_xroll, pts_yroll = 3, 3
mfcbudget_file = datadir + version + '_mfc_budget_' + yearstr

enso_nm = 'NINO3'
#enso_nm = 'NINO3.4'
ensofile = atm.homedir() + ('dynamics/calc/ENSO/enso_%s.csv' %
                            enso_nm.lower().replace('.', '').replace('+', ''))
enso_keys = ['MAM', 'JJA']

# ----------------------------------------------------------------------
# Read data

# Large-scale onset/retreat indices
index_all = {}
for nm in indfiles:
    print('Loading ' + indfiles[nm])
    if nm == 'MOK':
        mok = indices.onset_MOK(indfiles['MOK'], yearsub=years)
        index_all['MOK'] =  xray.Dataset({'onset' : mok})
    else:
        with xray.open_dataset(indfiles[nm]) as ds:
            index_all[nm] = ds.load()

index = index_all[onset_nm]
index['length'] = index['retreat'] - index['onset']

onset_all = pd.DataFrame()
for nm in index_all:
    onset_all[nm] = index_all[nm]['onset'].to_series()

# Onset/retreat at grid points
print('Loading ' + ptsfile)
with xray.open_dataset(ptsfile) as index_pts:
    index_pts.load()
for nm in index_pts.data_vars:
    if pts_xroll is not None:
        index_pts[nm] = atm.rolling_mean(index_pts[nm], pts_xroll, axis=-1,
                                         center=True)
    if pts_yroll is not None:
        index_pts[nm] = atm.rolling_mean(index_pts[nm], pts_yroll, axis=-2,
                                         center=True)

# Regression of gridpoint indices onto large-scale index
print('Regression of gridpoint indices onto large-scale index')
pts_reg, pts_mask = {}, {}
for nm in index_pts.data_vars:
    ind = index[nm].sel(year=index_pts['year'])
    pts_reg[nm] = atm.regress_field(index_pts[nm], ind, axis=0)
    pts_mask[nm] = (pts_reg[nm]['p'] >= 0.05)

# MFC budget
with xray.open_dataset(mfcbudget_file) as mfc_budget:
    mfc_budget.load()
mfc_budget = mfc_budget.rename({'DWDT' : 'dw/dt'})
if nroll is not None:
    for nm in mfc_budget.data_vars:
        mfc_budget[nm] = atm.rolling_mean(mfc_budget[nm], nroll, center=True)

# Dailyrel climatology
keys_dict = {'PRECTOT' : 'PRECTOT', 'CMAP' : 'precip', 'U200' : 'U',
             'U850' : 'U', 'V200' : 'V', 'V850' : 'V'}
data = {}
for nm in datafiles:
    print('Loading ' + datafiles[nm])
    with xray.open_dataset(datafiles[nm]) as ds:
        if 'year' in ds.dims:
            ds = ds.mean(dim='year')
        data[nm] = ds[keys_dict[nm]].load()

# ENSO indices
enso = pd.read_csv(ensofile, index_col=0)
enso = enso.loc[years]
for key in enso_keys:
    if key not in enso.columns:
        months = atm.season_months(key)
        month_names = [(atm.month_str(m)).capitalize() for m in months]
        enso[key] = enso[month_names].mean(axis=1)
enso = enso[enso_keys]
col_names = [enso_nm + ' ' + nm for nm in enso.columns]
enso.columns = col_names

# ----------------------------------------------------------------------
# Daily timeseries

tseries = xray.Dataset()
tseries['MFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['daily_ts'],
                                       index['onset'], npre, npost)
tseries['CMFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['tseries'],
                                           index['onset'], npre, npost)
for nm in ['CMAP', 'PRECTOT']:
    tseries[nm] = atm.mean_over_geobox(data[nm], lat1, lat2, lon1, lon2)

# Extract variables at specified latitudes
for nm, lat0 in lat_extract.iteritems():
    var = atm.dim_mean(data[nm], 'lon', lon1, lon2)
    lat = atm.get_coord(var, 'lat')
    lat0_str = atm.latlon_labels(lat0, 'lat', deg_symbol=False)
    key = nm + '_' + lat0_str
    lat_closest, _ = atm.find_closest(lat, lat0)
    print '%s %.2f %.2f' % (nm, lat0, lat_closest)
    tseries[key] = atm.subset(var, {'lat' : (lat_closest, None)}, squeeze=True)

# Compute climatology and smooth with rolling mean
if 'year' in tseries.dims:
    tseries = tseries.mean(dim='year')
if nroll is not None:
    for nm in tseries.data_vars:
        tseries[nm] = atm.rolling_mean(tseries[nm], nroll, center=True)

# ----------------------------------------------------------------------
# Plotting functions

def fix_axes(axlims):
    plt.gca().set_ylim(axlims[:2])
    plt.gca().set_xlim(axlims[2:])
    plt.draw()

def plot_mfc_budget(mfc_budget, index, year, legend=True,
                    legend_kw={'fontsize' : 9, 'loc' : 'upper left',
                               'handlelength' : 2.5}):
    ts = mfc_budget.sel(year=year)
    ind = index.sel(year=year)
    days = ts['day'].values
    styles = {'PRECTOT' : {'color' : 'k'},
              'EVAP' : {'color' : 'k', 'linestyle' : 'dashed'},
              'MFC' : {'color' : 'k', 'linewidth' : 2},
              'dw/dt' : {'color' : '0.7', 'linewidth' : 2}}
    for nm in styles:
        plt.plot(days, ts[nm], label=nm, **styles[nm])
    plt.axvline(ind['onset'], color='k')
    plt.axvline(ind['retreat'], color='k')
    plt.xlabel('Day of Year')
    plt.ylabel('mm/day')
    ax1 = plt.gca()
    ax2 = plt.twinx()
    plt.sca(ax2)
    plt.plot(days, ind['tseries'], 'r', alpha=0.6, linewidth=2, label='CMFC')
    atm.fmt_axlabels('y', 'mm', color='r', alpha=0.6)
    if legend:
        atm.legend_2ax(ax1, ax2, **legend_kw)
    return ax1, ax2


def yrly_index(onset_all, legend=True):
    """Plot onset day vs. year for different onset definitions."""

    corr = onset_all.corr()[onset_nm]
    labels = {nm : nm for nm in onset_all.columns}
    labels['CHP_MFC'] = 'CHP'
    styles = {'CHP_MFC' : {'color' : 'k', 'linewidth' : 2},
              'OCI' : {'color' : 'r'}, 'HOWI' : {'color' : 'b'},
              'MOK' : {'color' : 'g'}}
    xticks = np.arange(1980, 2016, 5)
    xticklabels = [1980, '', 1990, '', 2000, '', 2010, '']

    for nm in onset_all.columns:
        plt.plot(years, onset_all[nm], label=labels[nm], **styles[nm])
    if legend:
        plt.legend(fontsize=9, loc='upper left', ncol=2, frameon=False,
                   framealpha=0.0)
    plt.grid()
    plt.xlim(min(years) - 1, max(years) + 1)
    plt.xticks(xticks, xticklabels)
    plt.xlabel('Year')
    plt.ylabel('Day of Year')
    #plt.title('Onset')

def daily_tseries(tseries, index, npre, npost, legend, grp):
    """Plot dailyrel timeseries climatology"""
    xlims = (-npre, npost)
    xticks = range(-npre, npost + 1, 30)
    x0 = [0, index['length'].mean(dim='year')]
    keypairs = [(['MFC', 'CMAP'], ['CMFC']),
                (['U850_15N'], ['V850_15N']),
                (['U200_0N'],['V200_15N'])]
    opts = [('upper left', 'mm/day', 'mm'),
            ('upper left', 'm/s', 'm/s'),
            ('lower left', 'm/s', 'm/s')]
    y2_opts={'color' : 'r', 'alpha' : 0.6}
    styles = ['k', 'k--', 'g', 'm']
    for pair, opt in zip(keypairs, opts):
        grp.next()
        keys1, keys2 = pair
        legend_kw['loc'] = opt[0]
        y1_label = opt[1]
        y2_label = opt[2]
        data1 = tseries[keys1]
        if keys2 is not None:
            data2 = tseries[keys2]
        else:
            data2 = None
        data1_styles = {nm : style for (nm, style) in zip(keys1, styles)}
        utils.plotyy(data1, data2, xname='dayrel', data1_styles=data1_styles,
                     y2_opts=y2_opts, xlims=xlims, xticks=xticks,
                     xlabel='Rel Day', y1_label=y1_label, y2_label=y2_label,
                     legend=legend, legend_kw=legend_kw, x0_axvlines=x0)


def contourf_latday(var, clev=None, title='', nc_pref=40, grp=None,
                    xlims=(-120, 200), xticks=np.arange(-120, 201, 30),
                    ylims=(-60, 60), yticks=np.arange(-60, 61, 20),
                    ssn_length=None):
    vals = var.values.T
    lat = atm.get_coord(var, 'lat')
    days = atm.get_coord(var, 'dayrel')
    if var.min() >= 0:
        cmap, extend, symmetric = 'PuBuGn', 'max', False
    else:
        cmap, extend, symmetric = 'RdBu_r', 'both', True
    if clev == None:
        cint = atm.cinterval(vals, n_pref=nc_pref, symmetric=symmetric)
        clev = atm.clevels(vals, cint, symmetric=symmetric)
    cticks_dict = {'precip' : np.arange(0, 13, 2),
                   'T200' : np.arange(-208, 227, 2),
                   'U200' : np.arange(-60, 61, 10),
                   'PSI500' : np.arange(-800, 801, 200)}
    cticks = cticks_dict.get(var.name)
    plt.contourf(days, lat, vals, clev, cmap=cmap, extend=extend)
    plt.colorbar(ticks=cticks)
    atm.ax_lims_ticks(xlims, xticks, ylims, yticks)
    plt.grid()
    plt.title(title)
    plt.axvline(0, color='k')
    if ssn_length is not None:
        plt.axvline(ssn_length, color='k')
    if grp is not None and grp.row == grp.ncol - 1:
        plt.xlabel('Rel Day')
    if grp is not None and grp.col == 0:
        plt.ylabel('Latitude')


def precip_maps(precip, days, grp, cmax=20, cint=1, axlims=(5, 35, 60, 100),
                cmap='PuBuGn', res='c'):
    """Lat-lon maps of precip on selected days."""
    clev = np.arange(0, cmax + cint/2.0, cint)
    cticks = np.arange(0, clev.max() + 1, 2)
    lat1, lat2, lon1, lon2 = axlims
    for day in days:
        grp.next()
        pcp = precip.sel(dayrel=day)
        m = atm.init_latlon(lat1, lat2, lon1, lon2, resolution=res)
        m = atm.contourf_latlon(pcp, m=m, clev=clev, axlims=axlims, cmap=cmap,
                                colorbar=False, extend='max')
        atm.text(day, (0.05, 0.9), fontsize=12, fontweight='bold')
    # plt.colorbar(ax=grp.axes.ravel().tolist(), orientation='vertical',
    #              shrink=0.8, ticks=cticks)
    atm.colorbar_multiplot(orientation='vertical', shrink=0.8, ticks=cticks)
    fix_axes(axlims)


def pts_clim(index_pts, nm, clev_bar=10, clev_std=np.arange(0, 21, 1),
             axlims=(5, 32, 60, 100), cmap='spectral', res='c'):
    """Plot climatological mean and standard deviation of grid point indices."""
    varbar = index_pts[nm].mean(dim='year')
    varstd = index_pts[nm].std(dim='year')
    lat1, lat2, lon1, lon2 = axlims
    m = atm.init_latlon(lat1, lat2, lon1, lon2, resolution=res)
    m = atm.contourf_latlon(varstd, m=m, clev=clev_std, axlims=axlims, cmap=cmap,
                            symmetric=False, colorbar=False, extend='max')
    m.colorbar(ticks=np.arange(0, 21, 2))
    _, cs = atm.contour_latlon(varbar, clev=clev_bar, axlims=axlims, colors='k',
                               linewidths=2)
    plt.clabel(cs, fmt='%.0f', fontsize=9)
    fix_axes(axlims)

# Plot regression
def plot_reg(pts_reg, pts_mask, nm, clev=0.2, xsample=1, ysample=1,
             axlims=(5, 32, 60, 100), cline=None, alpha=0.25,
             markersize=5):
    """Plot regression of grid point indices onto large-scale index."""
    var = pts_reg[nm]['m']
    mask = pts_mask[nm]
    xname = atm.get_coord(mask, 'lon', 'name')
    yname = atm.get_coord(mask, 'lat', 'name')
    atm.contourf_latlon(var, clev=clev, axlims=axlims, extend='both')
    atm.stipple_pts(mask, xname=xname, yname=yname, xsample=xsample,
                    ysample=ysample, alpha=alpha, markersize=markersize)
    if cline is not None:
        atm.contour_latlon(var, clev=[cline], axlims=axlims, colors='b',
                           linewidths=2)
    fix_axes(axlims)

# ----------------------------------------------------------------------
# FIGURES

# Daily MFC budget and CHP tseries fit in a single year
plotyear = 2000
nrow, ncol = 1, 2
fig_kw = {'figsize' : (10, 4)}
gridspec_kw = {'left' : 0.05, 'right' : 0.97, 'wspace' : 0.4}
legend = True
legend_kw = {'fontsize' : 9, 'loc' : 'upper left', 'handlelength' : 2.5,
             'frameon' : False, 'framealpha' : 0.0}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
grp.next()
plot_mfc_budget(mfc_budget, index, plotyear, legend=legend, legend_kw=legend_kw)

# Plot yearly tseries
grp.next()
yrly_index(onset_all, legend=True)

# Plot daily tseries
nrow, ncol = 2, 2
fig_kw = {'figsize' : (10, 6)}
gridspec_kw = {'left' : 0.1, 'right' : 0.9, 'bottom' : 0.06, 'top' : 0.95,
               'wspace' : 0.35, 'hspace' : 0.2}
legend_kw = {'fontsize' : 8, 'handlelength' : 2.5, 'frameon' : False,
             'framealpha' : 0.0}
legend = True
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
daily_tseries(tseries, index, npre, npost, legend, grp)
grp.axes[1,1].axis('off')


# Lat-day contour plots
keys = [pcp_nm, 'U200', 'V200', 'U850']
nrow, ncol = 2, 2
fig_kw = {'figsize' : (11, 7), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.07, 'right' : 0.99, 'bottom' : 0.07, 'top' : 0.95,
               'wspace' : 0.05}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for key in keys:
    grp.next()
    var = atm.dim_mean(data[key], 'lon', lon1, lon2)
    var = atm.rolling_mean(var, nroll, axis=0, center=True)
    contourf_latday(var, title=key.upper(), grp=grp,
                    ssn_length=index['length'].mean(dim='year'))

# Precip maps
#days = [-30, -15, 0, 15, 30, 45, 60, 75, 90]
days = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
nrow, ncol = 4, 3
cmax, cint = 20, 1
fig_kw = {'figsize' : (10, 8), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.07, 'right' : 0.99, 'wspace' : 0.15, 'hspace' : 0.05}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
precip_maps(data[pcp_nm], days, grp, cmax=cmax, cint=cint)

# Grid point indices
nm = 'onset'
cmap = 'spectral'
clev_bar = 10
clev_std = np.arange(0, 21, 1)
clev_reg = np.arange(-1.2, 1.25, 0.2)
xsample, ysample = 2, 2
nrow, ncol = 1, 2
fig_kw = {'figsize' : (9, 3.5)}
gridspec_kw = {'left' : 0.1, 'wspace' : 0.3}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
grp.next()
pts_clim(index_pts, nm, clev_bar=clev_bar, clev_std=clev_std, cmap=cmap)
grp.next()
plot_reg(pts_reg, pts_mask, nm, clev=clev_reg, xsample=xsample, ysample=ysample)
# NOTE!!!!!!
# Transparency gets lost if saved directly to eps
# Save this figure to pdf first, then run:
# pdftops -eps fig04.pdf fig04.eps



# ----------------------------------------------------------------------
# Table of summary stats on onset/retreat/length

nms = ['Mean', 'Std', 'Max', 'Min']
for nm in ['onset', 'retreat', 'length']:
    ind = index[nm].values
    series = pd.Series([ind.mean(), ind.std(), ind.max(), ind.min()], index=nms)
    if nm == 'onset':
        stats = series.to_frame(name=nm.capitalize())
    else:
        stats[nm.capitalize()] = series
stats = stats.T

def daystr(day):
    day = round(day)
    mm, dd = atm.jday_to_mmdd(day)
    mon = atm.month_str(mm)
    return '%.0f (%s-%.0f)' % (day, mon.capitalize(), dd)

for nm1 in stats.columns:
    for nm2 in stats.index:
        if nm1 != 'Std' and nm2 != 'Length':
            stats[nm1].loc[nm2] = daystr(stats[nm1].loc[nm2])
        else:
            stats[nm1].loc[nm2] = '%.0f' % stats[nm1].loc[nm2]

print(stats.to_latex())

# ----------------------------------------------------------------------
# Table of correlations between detrended indices

detrend = True
df_onset = onset_all
df_ind = index[['onset', 'retreat', 'length']].to_dataframe()
df_enso = enso
if detrend:
    df_onset = atm.detrend(df_onset)
    df_ind = atm.detrend(df_ind)
    df_enso = atm.detrend(df_enso)

corr_onset = df_onset.corr()
print(corr_onset.round(2))

df1 = pd.concat([df_ind, df_enso], axis=1)
df2 = df_ind

corr = {}
for key in ['r', 'm', 'p']:
    corr[key] = pd.DataFrame(np.ones((len(df1.columns), len(df2.columns))),
                             index=df1.columns, columns=df2.columns)

for key1 in df1.columns:
    for key2 in df2.columns:
        reg = atm.Linreg(df1[key1], df2[key2])
        corr['r'].loc[key1][key2] = reg.r
        corr['m'].loc[key1][key2] = reg.slope
        corr['p'].loc[key1][key2] = reg.p

# Minimum absolute value of r to be significant
rcrit = (abs(corr['r'][corr['p'] <= 0.05])).min().min()
def format_r(r):
    rstr = '%.2f' % r
    # if abs(r) >= rcrit:
    #     rstr = 'textbf ' + rstr
    return rstr

print('\n\n*** Correlation coefficients ***')
print(corr['r'].to_latex(float_format=format_r))
print('Lowest significant value of abs(r) %.2f' % rcrit)

print('\n\n*** Regression coefficients ***')
print(corr['m'].round(2))

# ----------------------------------------------------------------------
# Duration of transition

d0, peak1, peak2 = 0, 30, 90
d1_list = [7, 14, 21, 28, 5, 10, 15, 20, 25]

ts_peak = atm.dim_mean(tseries, 'dayrel', peak1, peak2)
ts0 = atm.subset(tseries, {'dayrel' : (d0, d0)}, squeeze=True)
df = pd.DataFrame()

for d1 in d1_list:
    ts1 = atm.subset(tseries, {'dayrel' : (d1, d1)}, squeeze=True)
    delta = (ts1 - ts0) / (ts_peak - ts0)
    delta = delta.round(2).to_array().to_series()
    df['D1=%d' % d1] = delta

print('Ratio of onset transition (day D1 minus day %d) to peak difference\n'
      '(peak days %d to %d minus day %d)' % (d0, peak1, peak2, d0))
print(df)


# ----------------------------------------------------------------------
# MFC budget timeseries in each year
nrow, ncol = 3, 4
fig_kw = {'figsize' : (11, 7)}
gridspec_kw = {'left' : 0.05, 'right' : 0.9, 'wspace' : 0.05, 'hspace' : 0.1}
legend_kw = {'fontsize' : 9, 'loc' : 'upper left', 'handlelength' : 2.5,
             'frameon' : False, 'framealpha' : 0.0}
xlims = (0, 400)
y1_lims = (-5, 15)
y2_lims = (-400, 400)
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for year in years:
    grp.next()
    if grp.row == 0 and grp.col == 0:
        legend = True
    else:
        legend = False
    ax1, ax2 = plot_mfc_budget(mfc_budget, index, year, legend=legend,
                               legend_kw=legend_kw)
    ax1.set_ylim(y1_lims)
    ax2.set_ylim(y2_lims)
    if grp.col > 0:
        ax1.set_ylabel('')
        ax1.set_yticklabels([])
    if grp.col < grp.ncol - 1:
        ax2.set_ylabel('')
        ax2.set_yticklabels([])
    if grp.row < grp.nrow - 1:
        for ax in [ax1, ax2]:
            ax.set_xlabel('')
            ax.set_xticklabels([])
    atm.text(year, (0.05, 0.9), fontsize=9)
# ----------------------------------------------------------------------
# nrow, ncol = 9, 4
# fig_kw = {'figsize' : (8.5, 11), 'sharex' : True, 'sharey' : True}
# gridspec_kw = {'left' : 0.06, 'right' : 0.97, 'bottom' : 0.06, 'top' : 0.98,
#                'wspace' : 0.05, 'hspace' : 0.05}
# grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
# days = atm.get_coord(index['tseries'], 'day')
# for y, year in enumerate(years):
#     grp.next()
#     plt.plot(days, index['tseries'][y], 'k')
#     for x0 in [index['onset'][y], index['retreat'][y]]:
#         plt.axvline(x0, color='k')
#     atm.text(year, (0.05, 0.85), fontsize=9)
# plt.xlim(0, 365)
# plt.ylim(-350, 350)
# xticks = np.arange(0, 365, 50)
# xtick_labels = [0, '', 100, '', 200, '', 300, '']
# plt.xticks(xticks, xtick_labels)


# data1_styles = {'tseries' : 'k'}
# y2_opts = {'color' : 'r', 'alpha' : 0.5, 'linewidth' : 1}
# for y, year in enumerate(years):
#     grp.next()
#     x0 = [index['onset'][y], index['retreat'][y]]
#     data2 = index.get('daily_ts')
#     if data2 is not None:
#         data2 = atm.rolling_mean(data2[y], nroll, center=True)
#     utils.plotyy(index['tseries'][y], data2=data2, xname='day',
#                  data1_styles=data1_styles, y2_opts=y2_opts,
#                  xlims=None, xticks=None, ylims=None, yticks=None, y2_lims=None,
#                  xlabel='', y1_label='', y2_label='', legend=False,
#                  x0_axvlines=x0)
#     atm.text(year, (0.05, 0.85), fontsize=9)