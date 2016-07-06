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

style = atm.homedir() + 'dynamics/python/mpl-styles/grl_article.mplstyle'
plt.style.use(style)
figwidth = 7.48
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
    fracmin = 0.5
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

tseries = xray.Dataset()
tseries['MFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['daily_ts'],
                                       index[ind_nm], npre, npost)
tseries['CMFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['tseries'],
                                           index[ind_nm], npre, npost)
for nm in ['CMAP', 'GPCP', 'PRECTOT']:
    tseries[nm] = atm.mean_over_geobox(data[nm], lat1, lat2, lon1, lon2)

# Extract variables at specified latitudes
for nm, lat0 in lat_extract.iteritems():
    var = atm.dim_mean(data[nm], 'lon', lon1, lon2)
    lat = atm.get_coord(var, 'lat')
    lat0_str = atm.latlon_labels(lat0, 'lat', deg_symbol=False)
    # key = nm + '_' + lat0_str
    key = nm
    lat_closest, _ = atm.find_closest(lat, lat0)
    print '%s %.2f %.2f' % (nm, lat0, lat_closest)
    tseries[key] = atm.subset(var, {'lat' : (lat_closest, None)}, squeeze=True)

# Compute climatology and smooth with rolling mean
if 'year' in tseries.dims:
    tseries = tseries.mean(dim='year')
if nroll is not None:
    for nm in tseries.data_vars:
        tseries[nm] = atm.rolling_mean(tseries[nm], nroll, center=True)

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
                    dashes=[6, 2], netprecip=False):
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
    plt.ylabel('mm/day')
    ax1 = plt.gca()
    ax2 = plt.twinx()
    plt.sca(ax2)
    plt.plot(days, ind['tseries'], 'r', alpha=0.6, linewidth=2, label='CMFC')
    atm.fmt_axlabels('y', 'mm', color='r', alpha=0.6)
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
                  ind_nm='onset', grid=False, dashes=[6, 2], dlist=[15]):
    """Plot dailyrel timeseries climatology"""
    xlims = (-npre, npost)
    xticks = range(-npre, npost + 10, 30)
    if ind_nm == 'onset':
        x0 = [0, index['length'].mean(dim='year')]
        xtick_labels = xticks
    else:
        x0 = [-index['length'].mean(dim='year'), 0]
        xtick_labels = skip_ticklabel(xticks)
    keypairs = [(['MFC', pcp_nm], ['CMFC']), (['U850'], ['V850'])]
    opts = [('upper left', 'mm/day', 'mm'), ('upper left', 'm/s', 'm/s')]
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
        utils.plotyy(data1, data2, xname='dayrel', data1_styles=data1_styles,
                     y2_opts=y2_opts, xlims=xlims, xticks=xticks, ylims=ylims,
                     xlabel='Rel Day', y1_label=y1_label, y2_label=y2_label,
                     legend=legend, legend_kw=legend_kw, x0_axvlines=x0,
                     grid=grid)
        plt.gca().set_xticklabels(xtick_labels)
        if dlist is not None:
            for d0 in dlist:
                plt.axvline(d0, color='k', linestyle='--', dashes=dashes)

def contourf_latday(var, clev=None, title='', nc_pref=40, grp=None,
                    xlims=(-120, 200), xticks=np.arange(-120, 201, 30),
                    ylims=(-60, 60), yticks=np.arange(-60, 61, 20),
                    dlist=None, grid=False):
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
        plt.xlabel('Rel Day')
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


def pts_clim(index_pts, nm, clev_bar=10, clev_std=np.arange(0, 21, 1),
             axlims=(5, 32, 60, 100), cmap='spectral', res='c',
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
    fix_axes(axlims)

# Plot regression
def plot_reg(pts_reg, nm, clev=0.2, xsample=1, ysample=1,
             axlims=(5, 32, 60, 100), cline=None, color='0.3', alpha=1.0,
             markersize=2):
    """Plot regression of grid point indices onto large-scale index."""
    var = pts_reg[nm]['m']
    mask = pts_reg[nm]['pts_mask']
    xname = atm.get_coord(mask, 'lon', 'name')
    yname = atm.get_coord(mask, 'lat', 'name')
    atm.contourf_latlon(var, clev=clev, axlims=axlims, extend='both')
    atm.stipple_pts(mask, xname=xname, yname=yname, xsample=xsample,
                    ysample=ysample, color=color, alpha=alpha,
                    markersize=markersize)
    if cline is not None:
        atm.contour_latlon(var, clev=[cline], axlims=axlims, colors='b',
                           linewidths=2)
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
    plt.boxplot(df.values, vert=False, labels=['Length', 'Retreat', 'Onset'])
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

keys = [pcp_nm, 'U200', 'V200', 'U850']
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
                    dlist=dlist)
    if d0 is not None:
        plt.axvline(d0, color='k', linestyle='--', dashes=dashes)
plt.xticks(xticks, xtick_labels)
plt.xlim(-npre, npost)
labels = ['a', 'b', 'c', 'd']
x1, x2, y0 = -0.15, -0.05, 1.05
pos = [(x1, y0), (x2, y0), (x1, y0), (x2, y0)]
add_labels(grp, labels, pos, labelsize)

# Precip maps
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
plot_maps(data[pcp_nm], days, grp, cmax=cmax, cint=cint)

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
