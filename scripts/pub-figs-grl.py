import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xarray as xray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pandas as pd

import atmos as atm
import indices
import utils

# Format for article publication or presentation slides
pres = True
if pres:
    figwidth = 12
    style = atm.homedir() + 'dynamics/python/mpl-styles/presentation.mplstyle'
else:
    figwidth = 7.48
    style = atm.homedir() + 'dynamics/python/mpl-styles/grl_article.mplstyle'

plt.style.use(style)
fontsize = mpl.rcParams['font.size']
labelsize = fontsize + 3
dashes = [6, 2]

# ----------------------------------------------------------------------
version = 'merra2'
years = np.arange(1980, 2016)
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
onset_nm = 'CHP_MFC'
onset_nms = ['CHP_MFC', 'MOK', 'HOWI', 'OCI']
#pts_nm = 'CHP_PCP'
pts_nm = 'CHP_GPCP'
#pcp_nm = 'PRECTOT'
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
ptsfile = datadir + version + '_index_pts_%s_' % pts_nm
ptsmaskfile = None
if pts_nm == 'CHP_CMAP':
    ptsfile = ptsfile + '1980-2014.nc'
    pts_xroll, pts_yroll = None, None
elif pts_nm == 'CHP_GPCP':
    ptsfile = ptsfile + '1997-2015.nc'
    ptsmaskfile = atm.homedir() + 'datastore/gpcp/gpcp_daily_1997-2014.nc'
    pts_xroll, pts_yroll = None, None
else:
    ptsfile = ptsfile + yearstr
    pts_xroll, pts_yroll = 3, 3
mfcbudget_file = datadir + version + '_mfc_budget_' + yearstr

if ind_nm == 'retreat':
    for nm in datafiles:
        datafiles[nm] = datafiles[nm].replace('dailyrel', 'dailyrel_retreat')

enso_nm = 'NINO3'
#enso_nm = 'NINO3.4'
ensodir = atm.homedir() + 'dynamics/python/data/ENSO/'
ensofile = ensodir + ('enso_sst_monthly_%s.csv' %
                      enso_nm.lower().replace('.', '').replace('+', ''))
enso_keys = ['MAM', 'JJA']

# ----------------------------------------------------------------------
# Read data

# Large-scale onset/retreat indices
index_all = collections.OrderedDict()
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
    pts_reg[nm]['pts_mask'] = (pts_reg[nm]['p'] >= 0.05)

# Mask out grid points where CHP index is ill-defined
def applymask(ds, mask_in):
    for nm in ds.data_vars:
        mask = atm.biggify(mask_in, ds[nm], tile=True)
        vals = np.ma.masked_array(ds[nm], mask=mask).filled(np.nan)
        ds[nm].values = vals
    return ds

if ptsmaskfile is not None:
    day1 = atm.mmdd_to_jday(6, 1)
    day2 = atm.mmdd_to_jday(9, 30)
    with xray.open_dataset(ptsmaskfile) as ds:
        pcp = ds['PREC'].sel(lat=index_pts.lat).sel(lon=index_pts.lon).load()
    pcp_ssn = atm.subset(pcp, {'day' : (day1, day2)})
    pcp_frac = pcp_ssn.sum(dim='day') / pcp.sum(dim='day')
    mask = pcp_frac < fracmin
    index_pts = applymask(index_pts, mask)
    for key in pts_reg:
        pts_reg[key] = applymask(pts_reg[key], mask)

# MFC budget
with xray.open_dataset(mfcbudget_file) as mfc_budget:
    mfc_budget.load()
mfc_budget = mfc_budget.rename({'DWDT' : 'dw/dt'})
mfc_budget['P-E'] = mfc_budget['PRECTOT'] - mfc_budget['EVAP']
if nroll is not None:
    for nm in mfc_budget.data_vars:
        mfc_budget[nm] = atm.rolling_mean(mfc_budget[nm], nroll, center=True)

# Dailyrel climatology
keys_dict = {'PRECTOT' : 'PRECTOT', 'CMAP' : 'precip', 'GPCP' : 'PREC',
             'U200' : 'U', 'U850' : 'U', 'V200' : 'V', 'V850' : 'V'}
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

ts = xray.Dataset()
for nm in ['GPCP', 'PRECTOT']:
    ts[nm] = atm.mean_over_geobox(data[nm], lat1, lat2, lon1, lon2)
ts['MFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['daily_ts'],
                                  index[ind_nm], npre, npost)
ts['CMFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['tseries'],
                                   index[ind_nm], npre, npost)


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


# Smooth latitude-dayrel data with rolling mean
for nm in data:
    daydim = atm.get_coord(data[nm], 'dayrel', 'dim')
    data[nm] = atm.rolling_mean(data[nm], nroll, axis=daydim, center=True)

# ----------------------------------------------------------------------
# Plotting functions

def fix_axes(axlims):
    plt.gca().set_ylim(axlims[:2])
    plt.gca().set_xlim(axlims[2:])
    plt.draw()

def add_labels(grp, labels, pos, fontsize, fontweight='bold'):
    # Expand pos to list for each subplot, if needed
    try:
        n = len(pos[0])
    except TypeError:
        pos = [pos] * (grp.nrow * grp.ncol)
    i = 0
    for row in range(grp.nrow):
        for col in range(grp.ncol):
            grp.subplot(row, col)
            atm.text(labels[i], pos[i], fontsize=fontsize,
                     fontweight=fontweight)
            i += 1

def skip_ticklabel(xticks):
    xtick_labels = []
    for i, n in enumerate(xticks):
        if i % 2 == 0:
            xtick_labels = xtick_labels +  ['']
        else:
            xtick_labels = xtick_labels + [n]
    return xtick_labels

def plot_mfc_budget(mfc_budget, index, year, legend=True,
                    legend_kw={'fontsize' : 9, 'loc' : 'upper left',
                               'handlelength' : 2.5},
                    dashes=[6, 2], netprecip=False, labelpad=1.5):
    ts = mfc_budget.sel(year=year)
    ind = index.sel(year=year)
    days = ts['day'].values
    styles = {'PRECTOT' : {'color' : 'k', 'linestyle' : '--', 'dashes' : dashes},
              'EVAP' : {'color' : 'k'},
              'MFC' : {'color' : 'k', 'linewidth' : 2},
              'dw/dt' : {'color' : '0.7', 'linewidth' : 2}}
    if netprecip:
        styles['P-E'] = {'color' : 'b', 'linewidth' : 2}
    for nm in styles:
        plt.plot(days, ts[nm], label=nm, **styles[nm])
    plt.axvline(ind['onset'], color='k')
    plt.axvline(ind['retreat'], color='k')
    plt.xlabel('Day of Year')
    plt.ylabel('mm day$^{-1}$', labelpad=labelpad)
    ax1 = plt.gca()
    ax2 = plt.twinx()
    plt.sca(ax2)
    plt.plot(days, ind['tseries'], 'r', alpha=0.6, linewidth=2, label='CMFC')
    atm.fmt_axlabels('y', 'mm', color='r', alpha=0.6)
    plt.gca().set_ylabel('mm', labelpad=labelpad)
    if legend:
        atm.legend_2ax(ax1, ax2, **legend_kw)
    return ax1, ax2


def yrly_index(onset_all, grid=False,legend=True,
               legend_kw={'loc' : 'upper left', 'ncol' : 2}):
    """Plot onset day vs. year for different onset definitions."""

    #corr = onset_all.corr()[onset_nm]
    labels = {nm : nm for nm in onset_all.columns}
    labels['CHP_MFC'] = 'CHP'
    styles = {'CHP_MFC' : {'color' : 'k', 'linewidth' : 2},
              'OCI' : {'color' : 'r'}, 'HOWI' : {'color' : 'g'},
              'MOK' : {'color' : 'b'}}
    styles['retreat'] = styles['CHP_MFC']
    styles['length'] = styles['CHP_MFC']
    xticks = np.arange(1980, 2016, 5)
    xticklabels = [1980, '', 1990, '', 2000, '', 2010, '']

    for nm in onset_all.columns:
        plt.plot(years, onset_all[nm], label=labels[nm], **styles[nm])
    if legend:
        plt.legend(**legend_kw)
    plt.grid(grid)
    plt.xlim(min(years) - 1, max(years) + 1)
    plt.xticks(xticks, xticklabels)
    plt.xlabel('Year')
    plt.ylabel('Day of Year')


def daily_tseries(tseries, index, pcp_nm, npre, npost, legend, grp,
                  ind_nm='onset', grid=False, dashes=[6, 2], dlist=[15],
                  labelpad=1.5):
    """Plot dailyrel timeseries climatology"""
    xlims = (-npre, npost)
    xticks = range(-npre, npost + 10, 30)
    xlabel = 'Days Since ' + ind_nm.capitalize()
    if ind_nm == 'onset':
        x0 = [0, index['length'].mean(dim='year')]
        xtick_labels = xticks
    else:
        x0 = [-index['length'].mean(dim='year'), 0]
        xtick_labels = skip_ticklabel(xticks)
    keypairs = [(['MFC', pcp_nm], ['CMFC']), (['U850'], ['V850'])]
    opts = [('upper left', 'mm day$^{-1}$', 'mm'),
            ('upper left', '   m s$^{-1}$', '   m s$^{-1}$')]
    ylim_list = [(-3.5, 9), (-7, 15)]
    y2_opts={'color' : 'r', 'alpha' : 0.6}
    dashed = {'color' : 'k', 'linestyle' : '--', 'dashes' : dashes}
    styles = ['k', dashed, 'g', 'm']
    legend_kw = {}
    for pair, opt, ylims in zip(keypairs, opts, ylim_list):
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
        axs = utils.plotyy(data1, data2, xname='dayrel', data1_styles=data1_styles,
                           y2_opts=y2_opts, xlims=xlims, xticks=xticks, ylims=ylims,
                           xlabel=xlabel, y1_label=y1_label, y2_label=y2_label,
                           legend=legend, legend_kw=legend_kw, x0_axvlines=x0,
                           grid=grid)
        for ax, label in zip(axs, [y1_label, y2_label]):
            ax.set_ylabel(label, labelpad=labelpad)
        plt.gca().set_xticklabels(xtick_labels)
        if dlist is not None:
            for d0 in dlist:
                plt.axvline(d0, color='k', linestyle='--', dashes=dashes)

def contourf_latday(var, clev=None, title='', nc_pref=40, grp=None,
                    xlims=(-120, 200), xticks=np.arange(-120, 201, 30),
                    ylims=(-60, 60), yticks=np.arange(-60, 61, 20),
                    dlist=None, grid=False, ind_nm='onset'):
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
    elif len(atm.makelist(clev)) == 1:
        if var.name == 'PREC':
            clev = np.arange(0, 10 + clev/2.0, clev)
        else:
            clev = atm.clevels(vals, clev, symmetric=symmetric)
    cticks_dict = {'PRECTOT' : np.arange(0, 13, 2),
                   'PREC' : np.arange(0, 11, 2),
                   'T200' : np.arange(-208, 227, 2),
                   'U200' : np.arange(-60, 61, 10),
                   'PSI500' : np.arange(-800, 801, 200)}
    cticks = cticks_dict.get(var.name)
    plt.contourf(days, lat, vals, clev, cmap=cmap, extend=extend)
    plt.colorbar(ticks=cticks)
    atm.ax_lims_ticks(xlims, xticks, ylims, yticks)
    plt.grid(grid)
    plt.title(title)
    if dlist is not None:
        for d0 in dlist:
            plt.axvline(d0, color='k')
    if grp is not None and grp.row == grp.ncol - 1:
        plt.xlabel('Days Since ' + ind_nm.capitalize())
    if grp is not None and grp.col == 0:
        plt.ylabel('Latitude')


def plot_maps(var, days, grp, cmin=0, cmax=20, cint=1, axlims=(5, 35, 60, 100),
                cmap='PuBuGn', res='c', extend='max', cticks=None,
                daypos=(0.05, 0.85)):
    """Lat-lon maps of precip on selected days."""
    clev = np.arange(cmin, cmax + cint/2.0, cint)
    if cticks is None:
        cticks = np.arange(cmin, clev.max() + 1, 2)
    lat1, lat2, lon1, lon2 = axlims
    for day in days:
        grp.next()
        pcp = var.sel(dayrel=day)
        m = atm.init_latlon(lat1, lat2, lon1, lon2, resolution=res)
        m = atm.contourf_latlon(pcp, m=m, clev=clev, axlims=axlims, cmap=cmap,
                                colorbar=False, extend=extend)
        atm.text(day, daypos, fontsize=12, fontweight='bold')
    # plt.colorbar(ax=grp.axes.ravel().tolist(), orientation='vertical',
    #              shrink=0.8, ticks=cticks)
    atm.colorbar_multiplot(orientation='vertical', shrink=0.8, ticks=cticks)
    fix_axes(axlims)
    xticks = [60, 70, 80, 90, 100]
    xtick_labels = atm.latlon_labels(xticks, 'lon')
    xtick_labels[1] = ''
    xtick_labels[3] = ''
    plt.xticks(xticks, xtick_labels)

def plot_kerala(color='b', linewidth=1):
    """Plot the boundaries of the Kerala region"""
    datadir = atm.homedir() + 'dynamics/python/monsoon-onset/data/'
    filenm = datadir + 'india_state.geojson'
    x, y = utils.kerala_boundaries(filenm)
    plt.plot(x, y, color, linewidth=linewidth)


def pts_clim(index_pts, nm, clev_bar=10, clev_std=np.arange(0, 21, 1),
             axlims=(5, 32, 60, 100), cmap='spectral', res='l',
             label_locs=None, inline_spacing=2):
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
    cs_opts = {'fmt' : '%.0f', 'fontsize' : 9,
               'inline_spacing' : inline_spacing}
    if label_locs is not None:
        cs_opts['manual'] = label_locs
    plt.clabel(cs, **cs_opts)
    plot_kerala()
    fix_axes(axlims)

# Plot regression
def plot_reg(pts_reg, nm, clev=0.2, xsample=1, ysample=1,
             axlims=(5, 32, 60, 100), cline=None, color='0.3', alpha=1.0,
             markersize=2, res='l'):
    """Plot regression of grid point indices onto large-scale index."""
    var = pts_reg[nm]['m']
    mask = pts_reg[nm]['pts_mask']
    xname = atm.get_coord(mask, 'lon', 'name')
    yname = atm.get_coord(mask, 'lat', 'name')
    lat1, lat2, lon1, lon2 = axlims
    m = atm.init_latlon(lat1, lat2, lon1, lon2, resolution=res)
    atm.contourf_latlon(var, m=m, clev=clev, axlims=axlims, extend='both')
    atm.stipple_pts(mask, xname=xname, yname=yname, xsample=xsample,
                    ysample=ysample, color=color, alpha=alpha,
                    markersize=markersize)
    if cline is not None:
        atm.contour_latlon(var, clev=[cline], axlims=axlims, colors='b',
                           linewidths=2)
    plot_kerala()
    fix_axes(axlims)

# ----------------------------------------------------------------------
# FIGURES

# Timeseries plots - setup figure for subplots
nrow, ncol = 2, 2
fig_kw = {'figsize' : (figwidth, 0.7 * figwidth)}
gridspec_kw = {'left' : 0.07, 'right' : 0.9, 'bottom' : 0.07, 'top' : 0.9,
               'wspace' : 0.5, 'hspace' : 0.35}
legend = True
legend_kw = {'loc' : 'upper left', 'framealpha' : 0.0}
labelpos = (-0.2, 1.05)
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)

# Daily MFC budget and CHP tseries fit in a single year
plotyear = 2000
if ind_nm == 'onset':
    grp.next()
    plot_mfc_budget(mfc_budget, index, plotyear, dashes=dashes, legend=legend,
                    legend_kw=legend_kw)

    # Plot yearly tseries
    grp.next()
    yrly_index(onset_all, legend=True)
else:
    for i in [0, 1]:
        grp.next()
        plt.axis('off')
    ax = plt.subplot(2, 1, 1)
    df = index[['length', 'retreat', 'onset']].to_dataframe()
    plt.boxplot(df.values, vert=False, labels=['Length', 'Retreat', 'Onset'],
                whis='range')
    plt.xlabel('Day of Year | Number of Days')
    plt.xlim(120, 320)
    plt.xticks(np.arange(120, 321, 20))
    pos = ax.get_position()
    pos2 = [pos.x0, pos.y0 + 0.05, pos.width, pos.height]
    ax.set_position(pos2)
    atm.text('a', (-0.16, 1.01), fontsize=labelsize, fontweight='bold')

# Plot daily tseries
legend = True
if ind_nm == 'onset':
    dlist = [15]
else:
    dlist = None
daily_tseries(tseries, index, pcp_nm, npre, npost, legend, grp, ind_nm=ind_nm,
              dlist=dlist)

# Add a-d labels
if ind_nm == 'onset':
    labels = ['a', 'b', 'c', 'd']
    add_labels(grp, labels, labelpos, labelsize)
else:
    labels = ['b', 'c']
    for i in [0, 1]:
        grp.subplot(1, i)
        atm.text(labels[i], labelpos, fontsize=labelsize, fontweight='bold')


# Lat-day contour plots
xticks = range(-npre, npost + 10, 30)
if ind_nm == 'onset':
    dlist = [0, index['length'].mean(dim='year')]
    d0 = 15
    xtick_labels = xticks
else:
    dlist = [-index['length'].mean(dim='year'), 0]
    d0 = None
    xtick_labels = skip_ticklabel(xticks)

keys = [pcp_nm, 'V200', 'U200', 'U850']
clevs = {pcp_nm : 1, 'U200' : 5, 'V200' : 1, 'U850' : 2}
nrow, ncol = 2, 2
fig_kw = {'figsize' : (figwidth, 0.64 * figwidth), 'sharex' : True,
          'sharey' : True}
gridspec_kw = {'left' : 0.07, 'right' : 0.99, 'bottom' : 0.07, 'top' : 0.94,
               'wspace' : 0.05}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for key in keys:
    grp.next()
    var = atm.dim_mean(data[key], 'lon', lon1, lon2)
    contourf_latday(var, clev=clevs[key], title=key.upper(), grp=grp,
                    dlist=dlist, ind_nm=ind_nm)
    if d0 is not None:
        plt.axvline(d0, color='k', linestyle='--', dashes=dashes)
plt.xticks(xticks, xtick_labels)
plt.xlim(-npre, npost)
labels = ['a', 'b', 'c', 'd']
x1, x2, y0 = -0.15, -0.05, 1.05
pos = [(x1, y0), (x2, y0), (x1, y0), (x2, y0)]
add_labels(grp, labels, pos, labelsize)

# Precip maps
axlims=(5, 35, 57, 103)
#days = [-30, -15, 0, 15, 30, 45, 60, 75, 90]
if ind_nm == 'onset':
    days = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
else:
    days = [-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10]
nrow, ncol = 4, 3
cmax, cint = 12, 1
fig_kw = {'figsize' : (figwidth, 0.8 * figwidth), 'sharex' : True,
          'sharey' : True}
gridspec_kw = {'left' : 0.07, 'right' : 0.99, 'wspace' : 0.15, 'hspace' : 0.05}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
plot_maps(data[pcp_nm], days, grp, cmax=cmax, cint=cint, axlims=axlims)
# -- Add MFC box to one subplot
if ind_nm == 'onset':
    x = [lon1, lon1, lon2, lon2, lon1]
    y = [lat1, lat2, lat2, lat1, lat1]
    grp.subplot(0, 0)
    plt.plot(x, y, color='m', linewidth=2)

# Grid point indices
cmap = 'spectral'
stipple_clr = '0.3'
if ind_nm == 'onset':
    label_locs = [(75, 10), (71, 10), (88, 15), (67, 17), (77, 21),
                  (75, 24), (95, 12)]
else:
    label_locs = [(95, 24), (85, 22), (75, 25), (76, 21), (88, 15)]
clev_bar = 10
clev_std = np.arange(0, 21, 1)
clev_reg = np.arange(-1.2, 1.25, 0.2)
if pts_nm == 'CHP_PCP':
    xsample, ysample = 2, 2
else:
    xsample, ysample = 1, 1
nrow, ncol = 1, 2
fig_kw = {'figsize' : (figwidth, 0.4 * figwidth)}
gridspec_kw = {'left' : 0.1, 'wspace' : 0.3}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
grp.next()
pts_clim(index_pts, ind_nm, clev_bar=clev_bar, clev_std=clev_std, cmap=cmap,
         label_locs=label_locs)
grp.next()
plot_reg(pts_reg, ind_nm, clev=clev_reg, xsample=xsample, ysample=ysample,
         color=stipple_clr)
add_labels(grp, ['a', 'b'], (-0.15, 1.05), labelsize)


# ----------------------------------------------------------------------
# Extra plots - maps of U850, V850 on various days
nms = ['U850', 'V850']
#nms = ['U850', 'V850', 'U200', 'V200']
suptitle_on = False
axlims=(-30, 30, 40, 120)
xticks = [40, 60, 80, 100, 120]
xtick_labels = atm.latlon_labels(xticks, 'lon')
yticks = [-30, -15, 0, 15, 30]
ytick_labels = atm.latlon_labels(yticks, 'lat')
daypos = 0.05, 1.02
opts = {'U850' : {'cmax' : 10, 'cint' : 1, 'ctick_int' : 2},
        'V850' : {'cmax' : 10, 'cint' : 1, 'ctick_int' : 2},
        'U200' : {'cmax' : 40, 'cint' : 5, 'ctick_int' : 10},
        'V200' : {'cmax' : 10, 'cint' : 1, 'ctick_int' : 2}}
plotdays = [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]
nrow, ncol = 4, 3
fig_kw = {'figsize' : (figwidth, 0.85 * figwidth), 'sharex' : True,
          'sharey' : True}
gridspec_kw = {'top' : 0.95, 'bottom' : 0.05, 'left' : 0.07, 'right' : 1.06,
               'wspace' : 0.3, 'hspace' : 0.25}
for nm in nms:
    var = data[nm]
    cmax = opts[nm]['cmax']
    cmin = -cmax
    cint = opts[nm]['cint']
    ctick_int = opts[nm]['ctick_int']
    cticks = np.arange(cmin, cmax + ctick_int/2.0, ctick_int)
    if suptitle_on:
        suptitle = nm
    else:
        suptitle = ''
    grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw,
                       suptitle=suptitle)
    plot_maps(var, plotdays, grp, cmin=cmin, cmax=cmax, cint=cint, axlims=axlims,
              cmap='RdBu_r', res='c', extend='both', cticks=cticks,
              daypos=daypos)
    plt.xticks(xticks, xtick_labels)
    plt.yticks(yticks, ytick_labels)


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

d0, peak1, peak2 = 0, 20, 100
#d1_list = [7, 14, 21, 28, 5, 10, 15, 20, 25]
d1_list = [5, 10, 15, 20, 25]

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
                               legend_kw=legend_kw, netprecip=True)
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
# Correlation between CHP_MFC and CHP_PCP
filestr = atm.homedir() + 'datastore/merra2/analysis/merra2_index_%s_' + yearstr
files = {nm : filestr % nm for nm in ['CHP_MFC', 'CHP_PCP']}
index1 = pd.DataFrame()
for nm in files:
    with xray.open_dataset(files[nm]) as ds:
        for nm2 in ['onset', 'retreat']:
            index1[nm2 + '_' + nm] = ds[nm2].load().to_series()
        index1['length_' + nm] = index1['retreat_' + nm] - index1['onset_' + nm]

years = index1.index
xticks = np.arange(1980, 2015, 10)
opts = {'CHP_MFC' : {}, 'CHP_PCP' : {'dashes' : [6, 2]}}
plt.figure()
for i, nm in enumerate(['onset', 'retreat', 'length']):
    plt.subplot(2, 2, i + 1)
    for nm2 in ['CHP_MFC', 'CHP_PCP']:
        plt.plot(years, index1[nm + '_' + nm2], 'k', label=nm2, **opts[nm2])
    plt.legend(fontsize=8, handlelength=3)
    plt.title(nm.capitalize())
    plt.xticks(xticks)

print(index1.corr())
keys = ['onset', 'retreat', 'length']
plt.figure()
plt.subplots_adjust(wspace=0.25, hspace=0.3, left=0.1, right=0.97, top=0.95)
for i, key in enumerate(keys):
    plt.subplot(2, 2, i + 1)
    key1 = key + '_CHP_MFC'
    key2 = key + '_CHP_PCP'
    reg = atm.Linreg(index1[key1], index1[key2])
    reg.plot(scatter_clr='k', scatter_sym='+', line_clr='k')
    plt.xlabel(key1)
    plt.ylabel(key2)


# ----------------------------------------------------------------------
# Extreme onset years

index1 = {}
index1['MOK'] = indices.onset_MOK(indfiles['MOK'])
index1['MOK_SUB'] = index1['MOK'].loc[years]
with xray.open_dataset(indfiles['CHP_MFC']) as ds:
    index1[onset_nm] = ds['onset'].load().to_series()

def extreme_years(ind, nstd=1):
    """Return years more than nstd away from the mean"""
    early = ind[ind - ind.mean() < -nstd * ind.std()]
    late = ind[ind - ind.mean() > nstd * ind.std()]
    return early, late

early, late = {}, {}
for nm in index1:
    early[nm], late[nm] = extreme_years(index1[nm])

# ----------------------------------------------------------------------
# Fourier harmonics

kmax_list = np.arange(2, 21, 1)
nms = [pcp_nm, 'U850', 'V850']
days = np.arange(-138, 227)
ts1 = ts.sel(dayrel=days)
ts_sm = {kmax : xray.Dataset() for kmax in kmax_list}
Rsq = {kmax : {} for kmax in kmax_list}
for kmax in kmax_list:
    for nm in nms:
        vals, Rsq[kmax][nm] = atm.fourier_smooth(ts1[nm], kmax)
        print kmax, nm, Rsq[kmax][nm]
        ts_sm[kmax][nm] = xray.DataArray(vals, coords=ts1[nm].coords)

# Find days where smoothed values are closest to actual timeseries
# values at days 0, 15
def closest_day(nm, ts1, ts_sm, d0, buf=20):
    val0 = ts1[nm].sel(dayrel=d0).values
    sm = atm.subset(ts_sm[nm], {'dayrel' : (d0 - buf, d0 + buf)})
    i0 = int(np.argmin(abs(sm - val0)))
    day0 = int(sm['dayrel'][i0])
    return day0

# Annual + semi-annual harmonics
xticks = np.arange(-120, 230, 30)
sz = 10
dlist = [0, 15]
for kmax in [2, 4, 6, 7, 8, 9, 10]:
    dclose = {nm : [] for nm in nms}
    for nm in nms:
        for d0 in dlist:
            day0 = closest_day(nm, ts1, ts_sm[kmax], d0)
            dclose[nm] = dclose[nm] + [day0]
    plt.figure()
    plt.suptitle('Fourier fit kmax = %d.  Delta = day %d to %d' %
                 (kmax, dlist[0], dlist[1]))
    for i, nm in enumerate(nms):
        dlist2 = dclose[nm]
        plt.subplot(2, 2, i + 1)
        plt.plot(days, ts1[nm], 'b')
        plt.plot(dlist, ts1[nm].sel(dayrel=dlist), 'b.', markersize=sz)
        plt.plot(days, ts_sm[kmax][nm], 'r')
        plt.plot(dlist2, ts_sm[kmax][nm].sel(dayrel=dlist2), 'r.', markersize=sz)
        plt.title(nm)
        plt.xticks(xticks)
        plt.grid()
        s = 'Rsq = %.2f\nNum days = %d' % (Rsq[kmax][nm], dlist2[1] - dlist2[0])
        atm.text(s, (0.05, 0.85))

# See which kmax is needed to minimize Rsq between tseries and Fourier
# fit over days 0-15
d1, d2 = 0, 15
ts_sub = atm.subset(ts1, {'dayrel' : (d1, d2)})
ts_sm_sub = {}
for kmax in kmax_list:
    ts_sm_sub[kmax] = atm.subset(ts_sm[kmax], {'dayrel' : (d1, d2)})

def get_rss_sub(kmax, nm, ts_sub, ts_sm_sub):
    var0 = ts_sub[nm].values
    var1 = ts_sm_sub[kmax][nm].values
    return np.sum(np.sqrt((var1 - var0)**2))

rss_sub = {}
for nm in nms:
    rss_sub[nm] = [get_rss_sub(kmax, nm, ts_sub, ts_sm_sub)
                   for kmax in kmax_list]

plt.figure()
plt.subplots_adjust(hspace=0.3)
plt.suptitle('RSS over days %d-%d for truncated Fourier fits' % (d1, d2))
for i, nm in enumerate(nms):
    plt.subplot(2, 2, i + 1)
    plt.plot(kmax_list, rss_sub[nm], 'k')
    plt.xlabel('kmax')
    plt.ylabel('RSS')
    plt.title(nm)
    plt.grid()

# ----------------------------------------------------------------------
# Calculate seasonal precip - totals and average daily rate

pcpfile = (atm.homedir() + 'datastore/merra2/analysis/' +
           'merra2_gpcp_mfc_box_daily.nc')
with xray.open_dataset(pcpfile) as pcpts:
    pcpts.load()

ssn = utils.get_strength_indices(years, pcpts, index['onset'],
                                 index['retreat'])

def detrend(df):
    df_detrend = df.copy()
    x = df.index.values
    for col in df.columns:
        y = df[col].values
        reg = atm.Linreg(x, y)
        df_detrend[col] = df[col] - reg.predict(x)
    return df_detrend

# Cumulative and average rainfall over monsoon season
i_detrend = True
df1 = ssn[['onset', 'retreat', 'length']]
for nm in df1.columns:
    df1 = df1.rename(columns={nm : nm.upper()})
if i_detrend:
    df1 = detrend(df1)
nms = ['MFC', 'PCP', 'GPCP', 'EVAP']
figsize = (7, 7)
fmts={'line_width': 1, 'annotation_pos': (0.05, 0.7), 'pmax_bold': 0.05,
     'scatter_size': 3, 'scatter_clr': 'k', 'scatter_sym': '+', 'line_clr': 'k'}
subplot_fmts={'right': 0.98, 'bottom': 0.05, 'top': 0.95, 'wspace': 0.1,
              'hspace': 0.15, 'left': 0.12}
for nm1 in ['_LRS']:
    for key in ['_TOT', '_AVG']:
        keys = [nm + nm1 + key for nm in nms]
        df2 = ssn[keys]
        newcols = {nm : nm.replace('_LRS', '') for nm in df2.columns}
        df2 = df2.rename(columns=newcols)
        if i_detrend:
            df2 = detrend(df2)
        atm.scatter_matrix_pairs(df1, df2, figsize=figsize, fmts=fmts,
                                 subplot_fmts=subplot_fmts)
        for i in range(9):
            plt.subplot(4, 3, i + 1)
            ax = plt.gca()
            ax.set_xticklabels([])


# # Daily timeseries of GPCP years
# yrs_gpcp = range(1997, 2015)
# days = pcpts['day']
# nroll = 5
# fig_kw = {'figsize' : (8, 11), 'sharex' : True, 'sharey' : True}
# gs_kw = {'left' : 0.05, 'right' : 0.95, 'hspace' : 0.05, 'wspace' : 0.05}
# nrow, ncol = 3, 3
# grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gs_kw)
# for yr in yrs_gpcp:
#     grp.next()
#     ts = atm.rolling_mean(pcpts['GPCP'].sel(year=yr), nroll)
#     plt.plot(days, ts, 'k')
#     plt.axvline(ssn['onset'].loc[yr])
#     plt.axvline(ssn['retreat'].loc[yr])
#     plt.title(yr)
#     plt.xlim(0, 366)

# ts = pcpts['GPCP'].sel(year=yrs_gpcp)
# d_onset = ssn['onset'].loc[yrs_gpcp].values
# ssn_length = ssn['length'].loc[yrs_gpcp].values
# tsrel = utils.daily_rel2onset(ts, d_onset, npre=120, npost=165)
# ts_acc = np.cumsum(tsrel.sel(dayrel=range(0,166)), axis=1)


# ----------------------------------------------------------------------
# Masking on lat-lon maps

plt.figure(figsize=(5, 4))
m = atm.init_latlon(0, 35, 58, 102, resolution='l', coastlines=False,
                    fillcontinents=True)
m.drawcoastlines(linewidth=0.5, color='0.5')
plot_kerala(linewidth=1)
x = [lon1, lon1, lon2, lon2, lon1]
y = [lat1, lat2, lat2, lat1, lat1]
plt.plot(x, y, color='m', linewidth=2)
_, cs = atm.contour_latlon(pcp_frac, m=m, clev=np.arange(0, 1, 0.1), linewidths=1.5,
                           axlims=(0, 35, 58, 102), colors='k')
label_locs = [(80, 5), (75, 6), (72, 8), (72, 10), (70, 15), (70, 18),
              (72, 25), (84, 5), (60, 5), (65, 3), (95, 18)]
cs_opts = {'fmt' : '%.1f', 'fontsize' : 9, 'manual' : label_locs,
           'inline_spacing' : 2}
plt.clabel(cs, **cs_opts)

# ------------------------------------------------------------------------
# Figure for AGU presentation
plt.figure(figsize=(8, 6))
m = atm.init_latlon(0, 35, 58, 102, resolution='l')
atm.contourf_latlon(pcp_frac, clev=np.arange(0, 1.1, 0.1), m=m,
                            axlims=(0, 35, 58, 102), cmap='PuBuGn')
_, cs = atm.contour_latlon(pcp_frac, m=m, clev=[0.5], linewidths=2, colors='k',
                           axlims=(0, 35, 58, 102))
label_locs = [(72, 8)]
cs_opts = {'fmt' : '%.1f', 'fontsize' : fontsize, 'manual' : label_locs,
           'inline_spacing' : 2}
plt.clabel(cs, **cs_opts)
plot_kerala(linewidth=1)
x = [lon1, lon1, lon2, lon2, lon1]
y = [lat1, lat2, lat2, lat1, lat1]
plt.plot(x, y, color='m', linewidth=2)
