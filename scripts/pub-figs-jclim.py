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
yearstr = '1980-2015'
datadir = atm.homedir() + 'datastore/%s/figure_data/' % version
pcp_nm = 'GPCP'
ind_nm = 'onset'
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
eqlat1, eqlat2 = -5, 5
plev_ubudget = 200
npre, npost =  120, 200

datafiles = {}
datafiles['ubudget'] = datadir + 'merra2_ubudget_1980-2014_excl.nc'
filestr = datadir + version + '_%s_' + yearstr + '.nc'
for nm in ['latp', 'hov', 'latlon', 'tseries', 'psi_comp', 'ebudget']:
    datafiles[nm] = filestr % nm
datafiles['gpcp'] = datadir + 'gpcp_dailyrel_1997-2015.nc'
datafiles['index'] = filestr % 'index_CHP_MFC'
datafiles['mld'] = atm.homedir() + 'datastore/mld/ifremer_mld_DT02_c1m_reg2.0.nc'

# ----------------------------------------------------------------------
# Read data

data = {}
for nm in datafiles:
    if nm == 'mld':
        decode_times = False
    else:
        decode_times = True
    print('Loading ' + datafiles[nm])
    with xray.open_dataset(datafiles[nm], decode_times=decode_times) as ds:
        data[nm] = ds.load()

tseries = data['tseries']
index = data['index']
index['length'] = index['retreat'] - index['onset']

data_hov = {nm : data['hov'][nm] for nm in data['hov'].data_vars}
data_hov['GPCP'] = data['gpcp']['PCP_SECTOR']

# Temporary fix for missing data in THETA_E_LML
var = data_hov['THETA_E_LML']
dmin, dmax = var['dayrel'].values.min(), var['dayrel'].values.max()
d1, d2 = 178, 183
var1 = var.sel(dayrel=range(dmin, d1))
var2 = var.sel(dayrel=range(d2, dmax + 1))
data_hov['THETA_E_LML'] = xray.concat((var1, var2), dim='dayrel')

# Surface moist static energy
Cp = atm.constants.Cp.values
Lv = atm.constants.Lv.values
data_hov['MSE_LML'] = (data_hov['TLML'] * Cp + data_hov['QLML'] * Lv) / 1e3
data_hov['MSE_LML'].name = 'MSE_LML'
data_hov['MSE_LML'].attrs['units'] = 'kJ kg^-1'

data_latp = data['latp']

data_latlon = {nm : data['latlon'][nm] for nm in data['latlon'].data_vars}
dlist = data['latlon']['dayrel'].values
data_latlon['GPCP'] = data['gpcp']['PCP'].sel(dayrel=dlist)
data_diff = {}
for nm in data_latlon:
    data_diff[nm] = data_latlon[nm][1] - data_latlon[nm][0]

subset_dict = {'plev' : (plev_ubudget, plev_ubudget)}
ubudget = atm.subset(data['ubudget'], subset_dict, squeeze=True)
for nm in ['U', 'V']:
    ubudget[nm] = atm.squeeze(data_latp[nm].sel(lev=plev_ubudget))
ubudget = ubudget.rename({'ADV_AVG' : 'ADV_MMC', 'COR_AVG' : 'COR_MMC',
                          'ADV_CRS' : 'CRS', 'PGF_ST' : 'PGF'})


ebudget = data['ebudget']
ebudget_eq = atm.dim_mean(ebudget, 'lat', eqlat1, eqlat2)
ebudget_sector = atm.dim_mean(ebudget, 'lon', lon1, lon2)
ebudget_eq_sector = atm.dim_mean(ebudget_eq, 'lon', lon1, lon2)

ps = data_latp['PS'] / 100

# ----------------------------------------------------------------------
# Plotting functions and other utilities

def get_varnm(nm):
    varnms = {'U200' : 'U', 'V200' : 'V', 'T200' : 'T', 'TLML' : 'T',
              'QLML' : 'Q', 'THETA_E_LML' : 'THETA_E'}
    return varnms.get(nm)

def get_colormap(nm):
    if nm.startswith('PCP') or nm == 'GPCP':
        cmap = 'PuBuGn'
    else:
        cmap = 'RdBu_r'
    return cmap

def fix_axes(axlims):
    plt.gca().set_ylim(axlims[:2])
    plt.gca().set_xlim(axlims[2:])
    plt.draw()

def add_labels(grp, labels, pos, fontsize, fontweight='bold'):
    # Expand pos to list for each subplot, if needed
    try:
        n = len(pos[0])
    except TypeError:
        pos = [pos] * len(labels)
    row, col = 0, 0
    for i in range(len(labels)):
        grp.subplot(row, col)
        atm.text(labels[i], pos[i], fontsize=fontsize,
                 fontweight=fontweight)
        col += 1
        if col == grp.ncol:
            col = 0
            row += 1


def skip_ticklabel(xticks):
    xtick_labels = []
    for i, n in enumerate(xticks):
        if i % 2 == 0:
            xtick_labels = xtick_labels +  ['']
        else:
            xtick_labels = xtick_labels + [n]
    return xtick_labels


def get_mld(ds):
    mld = ds['mld']
    missval = mld.attrs['mask_value']
    vals = mld.values
    vals = np.ma.masked_array(vals, vals==missval)
    vals = np.ma.filled(vals, np.nan)
    months = range(1, 13)
    lat = atm.get_coord(mld, 'lat')
    lon = atm.get_coord(mld, 'lon')
    coords = {'month' : months, 'lat' : lat, 'lon' : lon}
    mld = xray.DataArray(vals, dims=['month', 'lat', 'lon'], coords=coords)
    mld.values = vals
    return mld

def mld_map(mld, m=None, month=5, cmap='hot_r', climits=(0, 70),
            cticks=range(0, 71, 10), clevs=None):
    cb_kwargs = {'ticks' : cticks, 'extend' : 'max'}
    if min(climits) > 0 :
        cb_kwargs['extend'] = 'both'
    if clevs is None:
        # pcolormesh plot
        atm.pcolor_latlon(mld.sel(month=month), m=m, cmap=cmap,
                          cb_kwargs=cb_kwargs)
    else:
        # Contour plot
        atm.contourf_latlon(mld.sel(month=month), clev=clevs, m=m,
                            cmap=cmap, extend=cb_kwargs['extend'],
                            cb_kwargs=cb_kwargs)
    plt.clim(climits)


def precip_frac(pcp_frac, m=None):
    _, cs = atm.contour_latlon(pcp_frac, m=m, clev=np.arange(0, 1, 0.1),
                               linewidths=1.5, colors='k',
                               axlims=(0, 35, 58, 102))
    label_locs = [(80, 5), (75, 6), (72, 8), (72, 10), (70, 15), (70, 18),
                  (72, 25), (84, 5), (60, 5), (65, 3), (95, 18)]
    cs_opts = {'fmt' : '%.1f', 'fontsize' : 9, 'manual' : label_locs,
               'inline_spacing' : 2}
    plt.clabel(cs, **cs_opts)



def daily_tseries(tseries, index, pcp_nm, npre, npost, grp, keys1=None,
                  keys2=None, units1=None, units2=None, ylims=None,
                  legend_loc=None, ind_nm='onset', grid=False, dashes=[6, 2],
                  dlist=[15], labelpad=1.5, legend=True, xlabel=''):
    """Plot dailyrel timeseries climatology"""
    xlims = (-npre, npost)
    xticks = range(-npre, npost + 10, 30)
    if ind_nm == 'onset':
        x0 = [0, index['length'].mean(dim='year')]
        xtick_labels = xticks
    else:
        x0 = [-index['length'].mean(dim='year'), 0]
        xtick_labels = skip_ticklabel(xticks)

    y2_opts={'color' : 'r', 'alpha' : 0.6}
    dashed = {'color' : 'k', 'linestyle' : '--', 'dashes' : dashes}
    styles = ['k', dashed, 'g', 'm']
    legend_kw = {}
    legend_kw['loc'] = legend_loc
    y1_label = units1
    y2_label = units2
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


def latpres(data_latp, day, ps, xlims=(-60, 60), xticks=range(-60, 61, 15),
            title=None, clev_u=5, clev_psi=5, u_clr='#EE82EE', u_kw={},
            psi_kw={}):
    """Plot lat-pres contours of streamfunction and zonal wind.
    """
    xmin, xmax = xlims
    axlims = (xmin, xmax, 0, 1000)
    latp_data = atm.subset(data_latp, {'dayrel' : (day, day)}, squeeze=True)
    u = latp_data['U']
    psi = latp_data['PSI']

    atm.contour_latpres(u, clev=clev_u, topo=ps, colors=u_clr,
                        contour_kw=u_kw, axlims=axlims)
    atm.contour_latpres(psi, clev=clev_psi, omitzero=True, axlims=axlims,
                        contour_kw=psi_kw)

    plt.xticks(xticks, xticks)
    #plt.grid()
    if title is not None:
        plt.title(title)


def get_latmax(var):
    # Temporary - take subset to avoid wonky data at end of timeseries
    var = atm.subset(var.copy(), {'dayrel' : (-120, 170)})
    # ------------------------------------------
    lat = atm.get_coord(var, 'lat')
    coords={'dayrel': var['dayrel']}
    latdim = atm.get_coord(var, 'lat', 'dim')
    latmax = lat[np.nanargmax(var, axis=latdim)]
    latmax = xray.DataArray(latmax, dims=['dayrel'], coords=coords)
    return latmax

def annotate_latmax(var, ax=None, nroll=None, annotate=True):
    latmax = get_latmax(var)
    days = atm.get_coord(latmax, 'dayrel')
    if ax is None:
        ax = plt.gca()
    if nroll is not None:
        latmax = atm.rolling_mean(latmax, nroll, center=True)
    latmax_0 = latmax.sel(dayrel=0)
    ax.plot(days, latmax, 'k', linewidth=2, label='Latitude of Max')
    if annotate:
        ax.legend(loc='lower right', fontsize=10)
        s = atm.latlon_labels(latmax_0, latlon='lat', fmt='%.1f')
        ax.annotate(s, xy=(0, latmax_0), xycoords='data',
                    xytext=(-40, 20), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->"))
    return latmax

def contourf_latday(var, clev=None, title='', cticks=None, climits=None,
                    nc_pref=40, grp=None,
                    xlims=(-120, 200), xticks=np.arange(-120, 201, 30),
                    ylims=(-60, 60), yticks=np.arange(-60, 61, 20),
                    dlist=None, grid=False, ind_nm='onset'):
    var = atm.subset(var, {'lat' : ylims})
    vals = var.values.T
    lat = atm.get_coord(var, 'lat')
    days = atm.get_coord(var, 'dayrel')
    cmap = get_colormap(var.name)
    if var.min() < 0:
        symmetric = True
    else:
        symmetric = False
    if var.name.startswith('PCP'):
        extend = 'max'
    else:
        extend = 'both'
    if clev == None:
        cint = atm.cinterval(vals, n_pref=nc_pref, symmetric=symmetric)
        clev = atm.clevels(vals, cint, symmetric=symmetric)
    elif len(atm.makelist(clev)) == 1:
        if var.name == 'PREC':
            clev = np.arange(0, 10 + clev/2.0, clev)
        else:
            clev = atm.clevels(vals, clev, symmetric=symmetric)

    plt.contourf(days, lat, vals, clev, cmap=cmap, extend=extend)
    plt.colorbar(ticks=cticks)
    plt.clim(climits)
    atm.ax_lims_ticks(xlims, xticks, ylims, yticks)
    plt.grid(grid)
    plt.title(title)
    if dlist is not None:
        for d0 in dlist:
            plt.axvline(d0, color='k')
    if grp is not None and grp.row == grp.nrow - 1:
        plt.xlabel('Days Since ' + ind_nm.capitalize())
    if grp is not None and grp.col == 0:
        plt.ylabel('Latitude')


def latlon_and_sector(var, vardiff, lon1, lon2, grp, clim=None,
                      clim_diff=None, axlims=(-60, 60, 40, 120),
                      dashes=[6, 2], xticks=range(40, 121, 20)):
    subset_dict = {'lat' : (axlims[0], axlims[1]),
                   'lon' : (axlims[2], axlims[3])}
    xtick_labels = atm.latlon_labels(xticks, 'lon')
    for i in range(1, len(xtick_labels), 2):
        xtick_labels[i] = ''
    var = atm.subset(var, subset_dict)
    vardiff = atm.subset(vardiff, subset_dict)
    varbar = xray.Dataset()
    daynm = 'D%.0f'
    for day in var.dayrel:
        dnm = daynm % day
        varbar[dnm] = atm.dim_mean(var.sel(dayrel=day), 'lon', lon1, lon2)
    varbar['DIFF'] = atm.dim_mean(vardiff, 'lon', lon1, lon2)
    cmap = get_colormap(var.name)
    cmap_diff = 'RdBu_r'

    # Day 0
    grp.next()
    atm.pcolor_latlon(var[0], cmap=cmap, axlims=axlims)
    plt.clim(clim)
    if grp.row == 0:
        plt.title(daynm % var['dayrel'].values[0])
    ylimits = axlims[:2]
    plt.ylim(ylimits)
    plt.xticks(xticks, xtick_labels)
    plt.ylabel(var.name)

    # Day 0-15 difference
    grp.next()
    atm.pcolor_latlon(vardiff, cmap=cmap_diff, axlims=axlims)
    if clim_diff is None:
        vmax = np.nanmax(abs(vardiff))
        clim_diff = (-vmax, vmax)
    plt.clim(clim_diff)
    if grp.row == 0:
        plt.title('DIFF')
    plt.ylim(ylimits)
    plt.xticks(xticks, xtick_labels)
    plt.gca().set_yticklabels([])

    # Sector mean line plot
    grp.next()
    latnm = atm.get_coord(varbar, 'lat', 'name')
    xticks = np.arange(axlims[0], axlims[1] + 1, 20)
    xlims = axlims[:2]
    legend_kw = {'handlelength': 2, 'fontsize': 7, 'ncol' : 3}
    dashed = {'color' : 'k', 'linestyle' : '--', 'dashes' : dashes}
    styles = ['k', dashed]

    keys = varbar.data_vars.keys()[:2]
    data1 = varbar[keys]
    data1_styles = {nm : style for nm, style in zip(keys, styles)}
    if grp.row == grp.nrow - 1:
        xlabel = 'Latitude'
    else:
        xlabel = ''
    if grp.row == 0:
        plt.title('SASM Sector Mean')
    utils.plotyy(data1, data2=varbar['DIFF'], xname=latnm,
                 data1_styles=data1_styles,
                 xlims=xlims, xticks=xticks, ylims=None, yticks=None,
                 y2_lims=None, xlabel=xlabel, y1_label='', y2_label='',
                 legend=True, legend_kw=legend_kw, grid=False)


def ubudget_lineplot(ubudget_sector, keys, day, style, xlims=(-60, 60),
             xticks=range(-60, 61, 15), ylims=None, ylabel=None, legend=True,
             legend_kw={'fontsize' : 8, 'loc' : 'lower center', 'ncol' : 2,
                        'handlelength' : 2.5}):
    """Plot ubudget terms and winds vs latitude."""
    subset_dict = {'dayrel' : (day, day), 'lat': xlims}
    data = atm.subset(ubudget_sector[keys], subset_dict, squeeze=True)
    data = data.to_dataframe()
    data.plot(ax=plt.gca(), style=style, legend=False)
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.xticks(xticks, xticks)
    plt.gca().set_xticks(xticks, minor=True)
    plt.xlabel('Latitude')
    #plt.grid()
    if legend:
        plt.legend(**legend_kw)
    if ylabel is not None:
        plt.ylabel(ylabel)


def psi_decomposition(psi, ps, cint=10, xlims=(-60, 60),
                      xticks=range(-60, 61, 15), title='', u=None,
                      u_clr='#EE82EE'):
    xmin, xmax = xlims
    axlims = (xmin, xmax, 0, 1000)
    if u is not None:
        atm.contour_latpres(u, clev=[0], omitzero=False, colors=u_clr,
                            axlims=axlims)
    atm.contour_latpres(psi, clev=cint, topo=ps, omitzero=True, axlims=axlims)
    plt.xticks(xticks, xticks)
    #plt.grid()
    plt.title(title, fontsize=10)


# ======================================================================
# FIGURES
# ======================================================================

# Overview map with mixed layer depths and precip frac
mld = get_mld(data['mld'])
fig_kw = {'figsize' : (0.5 * figwidth, 0.35 * figwidth)}
gridspec_kw = {'left' : 0.1, 'right' : 0.95, 'bottom' : 0.1, 'top' : 0.95}
plt.subplots(1, 1, gridspec_kw=gridspec_kw, **fig_kw)
m = atm.init_latlon(0, 35, 58, 102, resolution='l', coastlines=False,
                    fillcontinents=True)
m.drawcoastlines(linewidth=0.5, color='0.5')

# Mixed layer depths
clevs = None
#clevs = np.arange(0, 71, 2.5)
climits = (10, 70)
cticks = np.arange(10, 71, 10)
mld_map(mld, m=m, month=5, cmap='hot_r', clevs=clevs, climits=climits,
        cticks=cticks)

# JJAS fraction of annual precip
precip_frac(data['gpcp']['FRAC_JJAS'], m=m)

# SASM domain
x = [lon1, lon1, lon2, lon2, lon1]
y = [lat1, lat2, lat2, lat1, lat1]
plt.plot(x, y, color='b', linewidth=2)

# ----------------------------------------------------------------------
# Daily tseries
nrow, ncol = 3, 1
fig_kw = {'figsize' : (0.5*figwidth, 0.8 * figwidth), 'sharex' : True}
gridspec_kw = {'left' : 0.16, 'right' : 0.84, 'bottom' : 0.07, 'top' : 0.96,
               'wspace' : 0.5, 'hspace' : 0.15}
legend = True
legend_kw = {'loc' : 'upper left', 'framealpha' : 0.0}
legend = True
dlist = [15]
opts = []
opts.append({'keys1' : ['MFC', pcp_nm], 'keys2' : ['CMFC'],
             'units1' : 'mm day$^{-1}$', 'units2' : 'mm',
             'ylims' : (-3.5, 9), 'legend_loc' : 'upper left' })
# opts.append({'keys1' : ['U200_0N'], 'keys2' : ['V200_15N'],
#              'units1' : '   m s$^{-1}$', 'units2' :  '   m s$^{-1}$',
#              'ylims' : (-20, 0), 'legend_loc' : 'lower left' })
opts.append({'keys1' : ['U200_0N'], 'keys2' : ['PSI500_0N'],
             'units1' : '   m s$^{-1}$', 'units2' :  '  10$^9$ kg s$^{-1}$',
             'ylims' : (-20, 0), 'legend_loc' : 'lower left' })
opts.append({'keys1' : ['T200_30N'], 'keys2' : ['T200_30S'],
             'units1' : '   K', 'units2' :  '   K',
             'ylims' : (218, 227), 'legend_loc' : 'upper left' })

grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for opt in opts:
    grp.next()
    if grp.row < grp.nrow - 1:
        xlabel = ''
    else:
        xlabel = 'Days Since Onset'
    daily_tseries(tseries, index, pcp_nm, npre, npost, grp, legend=legend,
                  ind_nm=ind_nm, dlist=dlist, xlabel=xlabel, **opt)

# Add a-d labels
labelpos = (-0.2, 1.05)
labels = ['a', 'b', 'c']
add_labels(grp, labels, labelpos, labelsize)

# ----------------------------------------------------------------------
# Lat-pres contour plots of streamfunction, U

nrow, ncol = 2, 3
advance_by = 'row'
fig_kw = {'figsize' : (figwidth, 0.7*figwidth), 'sharex' : 'col', 'sharey' : 'row'}
gridspec_kw = {'left' : 0.1, 'right' : 0.96, 'wspace' : 0.06, 'hspace' : 0.2,
               'bottom' : 0.08, 'top' : 0.95}
plotdays = [-45, -30, -15, 0, 15, 30]
xlims, xticks = (-35, 35), range(-30, 31, 10)
grp = atm.FigGroup(nrow, ncol,fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for day in plotdays:
    grp.next()
    title = 'Day %d' % day
    latpres(data_latp, day, ps=ps, xlims=xlims, xticks=xticks)
    plt.title(title, fontsize=11)
    if grp.row < grp.nrow - 1:
        plt.xlabel('')
    if grp.col > 0:
        plt.ylabel('')

# ----------------------------------------------------------------------
# Hovmoller plots (lat-day)
xticks = range(-npre, npost + 10, 30)
if ind_nm == 'onset':
    dlist = [0, index['length'].mean(dim='year')]
    d0 = 15
    xtick_labels = xticks
else:
    dlist = [-index['length'].mean(dim='year'), 0]
    d0 = None
    xtick_labels = skip_ticklabel(xticks)

keys = [pcp_nm, 'PSI500', 'U200', 'T200', 'THETA_E_LML']
clevs = {pcp_nm : 1, 'U200' : 5, 'V200' : 1, 'PSI500' : 5, 'T200' : 0.5,
         'THETA_E_LML' : 2.5, 'TLML' : 1, 'QLML' : 5e-4, 'U850' : 1,
         'MSE_LML' : 2}
cticks_dict = {pcp_nm : np.arange(0, 13, 2),
               'T200' : np.arange(208, 229, 4),
               'U200' : np.arange(-80, 81, 20),
               'U850' : np.arange(-20, 21, 4),
               'PSI500' : np.arange(-80, 81, 20),
               'THETA_E_LML' : np.arange(240, 361, 20),
               'MSE_LML' : np.arange(240, 361, 20)}
clim_dict = {pcp_nm : (0, 10), 'U200' : (-50, 50),
             'PSI500' : (-80, 80), 'T200' : (208, 227),
             'THETA_E_LML' : (260, 350), 'U850' : (-18, 18),
             'MSE_LML' : (245, 350)}
plot_latmax = False

nrow, ncol = 3, 2
fig_kw = {'figsize' : (figwidth, 0.9 * figwidth), 'sharex' : True,
          'sharey' : True}
gridspec_kw = {'left' : 0.07, 'right' : 0.99, 'bottom' : 0.07, 'top' : 0.94,
               'wspace' : 0.05}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for key in keys:
    grp.next()
    var = data_hov[key]
    clev = clevs.get(key)
    cticks = cticks_dict.get(key)
    climits = clim_dict.get(key)
    print(key, clev, climits, cticks)
    contourf_latday(var, clev=clev, cticks=cticks, climits=climits,
                    title=key.upper(), grp=grp,
                    dlist=dlist, ind_nm=ind_nm)
    if d0 is not None:
        plt.axvline(d0, color='k', linestyle='--', dashes=dashes)
    if plot_latmax and key.startswith('THETA_E'):
        latmax = annotate_latmax(var, nroll=None)

plt.xticks(xticks, xtick_labels)
plt.xlim(-npre, npost)
labels = ['a', 'b', 'c', 'd', 'e']
x1, x2, y0 = -0.15, -0.05, 1.05
pos = [(x1, y0), (x2, y0), (x1, y0), (x2, y0), (x1, y0), (x2, y0)]
add_labels(grp, labels, pos, labelsize)

# Hide axes on empty plot
ax = grp.axes[nrow - 1 , ncol - 1]
ax.axis('off')
plt.draw()

# ----------------------------------------------------------------------
# D0--D15 Lat-lon and sector line plots

nms_list = [['GPCP', 'U200', 'T200'], ['THETA_E_LML', 'TLML']]

clim_dict = {'GPCP' : (0, 12), 'U200' : (-50, 50), 'T200' : (213, 227),
             'TLML' : (260, 315), 'QLML' : (0, 0.022),
             'THETA_E_LML' : (270, 360)}

ncol = 3
gridspec_kw = {'left' : 0.12, 'right' : 0.9, 'bottom' : 0.09, 'top' : 0.93,
               'wspace' : 0.45, 'hspace' : 0.15, 'width_ratios' : [1, 1, 1.5]}

x1, x2, y0 = -0.6, -0.25, 1.05
labelpos = [(x1, y0), (x2, y0), (x2, y0)] * 3
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

for nms in nms_list:
    nrow = len(nms)
    if nrow < 3:
        height = 0.55 * figwidth
    else:
        height = 0.8 * figwidth
    fig_kw = {'figsize' : (figwidth, height), 'sharex' : 'col'}
    grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
    for nm in nms:
        latlon_and_sector(data_latlon[nm], data_diff[nm], lon1, lon2, grp,
                          clim=clim_dict[nm], clim_diff=None, dashes=dashes)
    add_labels(grp, labels[:3*nrow], labelpos[:3*nrow], labelsize)


# ----------------------------------------------------------------------
# Ubudget components at 200 hPa

style = {'ADV_MMC' : 'b', 'COR_MMC' : 'b--', 'ADV+COR' : 'r', 'DMDY' : 'r',
         'PGF' : 'k', 'CRS' : 'g',  'ADV_AVST' : 'g--',
         'ADV_STAV' : 'g-.', 'EMFC' : 'm', 'EMFC_TR' : 'm--', 'EMFC_ST' : 'm-.',
         'SUM' : 'k--', 'ACCEL' : 'c', 'ANA' : 'y', 'U' : 'k', 'V' : 'k--'}

keys_dict = collections.OrderedDict()
keys_dict['ubudget'] = ['ADV_MMC', 'COR_MMC', 'DMDY', 'PGF',
                        'CRS', 'EMFC']
keys_dict['winds'] = ['U']
keys_dict['eddies'] = ['EMFC_TR', 'EMFC_ST', 'EMFC', 'ADV_CRS']

ylabels = {}
units = '$10^{-4}$ m s$^{-2}$'
ylabels['ubudget'] = units
ylabels['eddies'] = ylabels['ubudget']
ylabels['winds'] = 'm s$^{-1}$'

ylims = {'ubudget' : (-8, 8), 'winds' : (-20, 50)}

plotdays = [-45, -30, -15, 0, 15, 30]
nrow, ncol = 2, 3
advance_by = 'row'
fig_kw = {'figsize' : (figwidth, 0.5 * figwidth),
          'sharex' : 'col', 'sharey' : 'row'}
gridspec_kw = {'left' : 0.08, 'right' : 0.99, 'wspace' : 0.09, 'hspace' : 0.1,
               'bottom' : 0.09, 'top' : 0.92, 'height_ratios' : [0.5, 1]}
legend_kw={'fontsize' : 8, 'loc' : 'upper center', 'ncol' : 2,
           'handlelength' : 2.5}
xlims, xticks = (-60, 60), range(-60, 61, 15)

grp = atm.FigGroup(nrow, ncol, advance_by, fig_kw=fig_kw,
                   gridspec_kw=gridspec_kw)
for day in plotdays:
    for nm in ['winds', 'ubudget']:
        grp.next()
        if grp.row == 0:
            plt.title('Day %d' % day)
        if grp.col == 0:
            legend = True
        else:
            legend = False
        keys = keys_dict[nm]
        ubudget_lineplot(ubudget, keys, day, style, xlims=xlims,
                         xticks=xticks, ylims=ylims[nm],
                         legend=legend, legend_kw=legend_kw,
                         ylabel=ylabels[nm])
        if nm == 'winds':
            plt.axhline(0, color='0.7', linestyle='--', dashes=[6, 1])
        if grp.row == grp.nrow - 1:
            plt.xlabel('Latitude')

# ----------------------------------------------------------------------
# Streamfunction decomposition

plotdays = [-30, 0, 30]
#plotdays = [-15, 0, 15]
keys = ['TOT', 'MMC', 'EDDY']
xlims, xticks = (-35, 35), range(-30, 31, 10)
cint = 5
nrow, ncol = len(keys), len(plotdays)
advance_by = 'col'
fig_kw = {'figsize' : (figwidth, 0.7 * figwidth), 'sharex' : True,
          'sharey' : True}
gridspec_kw = {'left' : 0.08, 'right' : 0.99, 'wspace' : 0.06, 'hspace' : 0.11,
               'bottom' : 0.08, 'top' : 0.95}
#suptitle = '%d-%dE $\psi$ components' % (lon1, lon2)
suptitle = ''
grp = atm.FigGroup(nrow, ncol, advance_by, fig_kw=fig_kw,
                   gridspec_kw=gridspec_kw, suptitle=suptitle)
for key in keys:
    for day in plotdays:
        grp.next()
        if grp.row == 0:
            title = 'Day %d' % day
            u = data_latp['U'].sel(dayrel=day)
        else:
            title = ''
            u = None
        if key == 'TOT':
            psi = data_latp['PSI'].sel(dayrel=day)
        else:
            psi = data['psi_comp'][key].sel(dayrel=day)
        psi_decomposition(psi, ps, cint, xlims, xticks, title=title,
                          u=u)
        if grp.col > 0:
            plt.ylabel('')
        if grp.row < grp.nrow - 1:
            plt.xlabel('')
        atm.text(key, (0.05, 0.88))


# ----------------------------------------------------------------------
# Energy budget - contour plots

def contour_londay(var, clev=None, grp=None,n_pref=40,
                   yticks=np.arange(-120, 201, 30)):
    lon = atm.get_coord(var, 'lon')
    days = atm.get_coord(var, 'dayrel')
    if clev is None:
        cint = atm.cinterval(var, n_pref=n_pref, symmetric=True)
        clev = atm.clevels(var, cint, symmetric=True)
    plt.contourf(lon, days, var, clev, cmap='RdBu_r', extend='both')
    plt.colorbar()
    plt.yticks(yticks)
    plt.axhline(0, color='0.5', linestyle='--', dashes=[6, 1])
    if grp is not None and grp.row == grp.nrow - 1:
        plt.xlabel('Longitude')
    if grp is not None and grp.col == 0:
        plt.ylabel('Days Since Onset')


mse_vars = {'VMSE' : 'VH', 'VCPT' : 'VFLXCPT', 'VPHI' : 'VFLXPHI',
            'VLQV' : 'VFLXLQV'}

scale = 1e9
vmse_eq = xray.Dataset({nm : ebudget_eq[mse_vars[nm]] for nm in mse_vars})
vmse_eq = vmse_eq / scale

nrow, ncol = 2, 2
fig_kw = {'figsize' : (figwidth, 0.7 * figwidth), 'sharex' : True,
          'sharey' : True}
gridspec_kw = {'left' : 0.1, 'right' : 0.99, 'bottom' : 0.07, 'top' : 0.9,
               'wspace' : 0.05}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
lonrange = (40, 120)
for nm in ['VMSE', 'VCPT', 'VPHI', 'VLQV']:
    grp.next()
    var = atm.subset(vmse_eq[nm], {'lon' : lonrange})
    contour_londay(var, grp=grp)
    plt.title(nm, fontsize=11)
plt.gca().invert_yaxis()

labels = ['a', 'b', 'c', 'd']
x1, x2, y0 = -0.15, -0.05, 1.05
pos = [(x1, y0), (x2, y0), (x1, y0), (x2, y0)]
add_labels(grp, labels, pos, labelsize)

# ----------------------------------------------------------------------
# Energy budget - sector means

# vmse_sector = xray.Dataset()
# for nm in ['VH', 'VFLXCPT', 'VFLXPHI', 'VFLXLQV']:
#     key = nm.replace('VFLX', 'V').replace('VH', 'VMSE')
#     vmse_sector[key] = ebudget_sector[nm] / scale


# Cross-equatorial flues integrated over sectors
a = atm.constants.radius_earth.values
eq_int = xray.Dataset()
lonranges = [(40, 60), (40, 100), (lon1, lon2)]
eq_int.attrs['lonranges'] = ['%d-%dE' % lonrange for lonrange in lonranges]
for lonrange in lonranges:
    lon1, lon2 = lonrange
    dist = a * np.radians(lon2 - lon1)
    for nm in vmse_eq.data_vars:
        key = nm + '_%d-%dE' % (lon1, lon2)
        eq_int[key] = atm.dim_mean(vmse_eq[nm], 'lon', lon1, lon2) * dist
# Convert to PW
eq_int = eq_int / 1e6



days = atm.get_coord(eq_int, 'dayrel')
nms = ['VMSE', 'VCPT', 'VPHI', 'VLQV']
colors = {'40-60E' : 'r', '60-100E' : 'b'}
styles = {'VMSE' : {'linewidth' : 2}, 'VPHI' : {'linestyle' : 'dotted'},
          'VCPT' : {'linestyle' : 'dashed', 'dashes' : dashes},
          'VLQV' : {'linestyle' : 'solid'}}
lonranges = ['40-60E', '60-100E']
#lonranges = eq_int.attrs['lonranges']

plt.figure(figsize=(0.7*figwidth, 0.4 * figwidth))
for lonrange in lonranges:
    for nm in nms:
        style = styles[nm]
        style['color'] = colors[lonrange]
        key = nm + '_' + lonrange
        plt.plot(days, eq_int[key], label=key, **style)
#plt.legend(loc='upper left', ncol=1, handlelength=3)
#plt.grid()
plt.xticks(np.arange(-120, 211, 30))
plt.xlim(-120, 210)
plt.axvline(0, color='0.5')
plt.xlabel('Days Since Onset')
plt.ylabel('<V*MSE> (PW)')


# nrow, ncol = 1, 2
# fig_kw = {'figsize' : (figwidth, 0.4 * figwidth), 'sharex' : True}
# gridspec_kw = {'left' : 0.07, 'right' : 0.96, 'bottom' : 0.15, 'top' : 0.9,
#                'wspace' : 0.15}
# #suptitle = 'Sector Cross-Eq <V*MSE> (%s)' % eq_int.attrs['units']
# suptitle = ''
# grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw,
#                    suptitle=suptitle)
#
# for lonrange in lonranges:
#     grp.next()
#     plt.title(lonrange, fontsize=11)
#     for nm in nms:
#         key = nm + '_' + lonrange
#         plt.plot(days, eq_int[key], label=nm, **styles[nm])
#     plt.legend(fontsize=9, loc=locs[lonrange], handlelength=3)
#     #plt.grid()
#     plt.xticks(np.arange(-120, 201, 30))
#     plt.axvline(0, color='0.5')
#     if grp.row == grp.nrow - 1:
#         plt.xlabel('Days Since Onset')
#     if grp.col == 0:
#         plt.ylabel('<V*MSE> (PW)')
