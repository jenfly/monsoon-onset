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

# *** Consolidate code from momentum-budget.py and analyze-composites.py

# ----------------------------------------------------------------------
version = 'merra2'
yearstr = '1980-2015'
datadir = atm.homedir() + 'datastore/%s/figure_data/' % version
pcp_nm = 'GPCP'
ind_nm = 'onset'
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
plev_ubudget = 200
npre, npost =  120, 200

datafiles = {}
datafiles['ubudget'] = datadir + 'merra2_ubudget_1980-2014_excl.nc'
filestr = datadir + version + '_%s_' + yearstr + '.nc'
for nm in ['latp', 'hov', 'latlon', 'tseries', 'psi_comp']:
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
data_latp = data['latp']

subset_dict = {'plev' : (plev_ubudget, plev_ubudget)}
ubudget = atm.subset(data['ubudget'], subset_dict, squeeze=True)
for nm in ['U', 'V']:
    ubudget[nm] = atm.squeeze(data_latp[nm].sel(lev=plev_ubudget))


# ----------------------------------------------------------------------
# Plotting functions and other utilities

def get_varnm(nm):
    varnms = {'U200' : 'U', 'V200' : 'V', 'T200' : 'T', 'TLML' : 'T',
              'QLML' : 'Q', 'THETA_E_LML' : 'THETA_E'}
    return varnms.get(nm)

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
            cticks=range(0, 71, 10)):
    cb_kwargs = {'ticks' : cticks, 'extend' : 'max'}
    _, _, cb = atm.pcolor_latlon(mld.sel(month=month), m=m, cmap=cmap,
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
            title=None, clev_u=5, clev_psi=5, u_clr='m', u_kw={'alpha' : 0.35},
            psi_kw={'alpha' : 0.7}):
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
    plt.grid()
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
    if var.min() < 0:
        symmetric = True
    else:
        symmetric = False
    if var.name.startswith('PCP'):
        cmap, extend = 'PuBuGn', 'max'
    else:
        cmap, extend = 'RdBu_r', 'both'
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
mld_map(mld, m=m, month=5, cmap='hot_r', climits=(0, 60))

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
opts.append({'keys1' : ['U200_0N'], 'keys2' : ['V200_15N'],
             'units1' : '   m s$^{-1}$', 'units2' :  '   m s$^{-1}$',
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

nrow, ncol = 2, 2
advance_by = 'row'
fig_kw = {'figsize' : (figwidth, 0.7*figwidth), 'sharex' : 'col', 'sharey' : 'row'}
gridspec_kw = {'left' : 0.1, 'right' : 0.96, 'wspace' : 0.06, 'hspace' : 0.2,
               'bottom' : 0.08, 'top' : 0.95}
plotdays = [-15, 0, 15, 30]
xlims, xticks = (-35, 35), range(-30, 31, 10)
grp = atm.FigGroup(nrow, ncol,fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for day in plotdays:
    grp.next()
    title = 'Day %d' % day
    latpres(data_latp, day, ps=data_latp['PS'], xlims=xlims, xticks=xticks)
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
         'THETA_E_LML' : 2.5, 'TLML' : 1, 'QLML' : 5e-4, 'U850' : 1}
cticks_dict = {pcp_nm : np.arange(0, 13, 2),
               'T200' : np.arange(208, 229, 4),
               'U200' : np.arange(-80, 81, 20),
               'U850' : np.arange(-20, 21, 4),
               'PSI500' : np.arange(-80, 81, 20),
               'THETA_E_LML' : np.arange(240, 361, 20)}
clim_dict = {pcp_nm : (0, 10), 'U200' : (-50, 50),
             'PSI500' : (-80, 80), 'T200' : (208, 227),
             'THETA_E_LML' : (260, 350), 'U850' : (-18, 18)}
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

data_latlon = {nm : data['latlon'][nm] for nm in data['latlon'].data_vars}
data_latlon['GPCP'] = data['gpcp']['PCP']

clim_dict = {'GPCP' : (0, 20), 'U200' : (-50, 50), 'T200' : (213, 227),
             'TLML' : (260, 315), 'QLML' : (0, 0.022),
             'THETA_E_LML' : (270, 360)}


# ----------------------------------------------------------------------
# Ubudget components at 200 hPa

def lineplot(ubudget_sector, keys, day, style, xlims=(-60, 60),
             xticks=range(-60, 61, 15), title=None, ylabel=None, legend=True,
             legend_kw={'fontsize' : 8, 'loc' : 'lower center', 'ncol' : 2,
                        'handlelength' : 2.5}):
    """Plot ubudget terms and winds vs latitude."""
    subset_dict = {'dayrel' : (day, day), 'lat': xlims}
    data = atm.subset(ubudget_sector[keys], subset_dict, squeeze=True)
    data = data.to_dataframe()
    data.plot(ax=plt.gca(), style=style, legend=False)
    plt.xlim(xlims)
    plt.xticks(xticks, xticks)
    plt.xlabel('Latitude')
    plt.grid()
    if legend:
        plt.legend(**legend_kw)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

style = {'ADV_AVG' : 'b', 'COR_AVG' : 'b--', 'ADV+COR' : 'r', 'DMDY' : 'r',
         'PGF_ST' : 'k', 'ADV_CRS' : 'g',  'ADV_AVST' : 'g--',
         'ADV_STAV' : 'g-.', 'EMFC' : 'm', 'EMFC_TR' : 'm--', 'EMFC_ST' : 'm-.',
         'SUM' : 'k--', 'ACCEL' : 'c', 'ANA' : 'y', 'U' : 'k', 'V' : 'k--'}

keys_dict = collections.OrderedDict()
keys_dict['ubudget'] = ['ADV_AVG', 'COR_AVG', 'DMDY', 'PGF_ST',
                        'ADV_CRS', 'EMFC']
keys_dict['winds'] = ['U', 'V']
keys_dict['eddies'] = ['EMFC_TR', 'EMFC_ST', 'EMFC', 'ADV_CRS']

ylabels = {}
units = '$10^{-4}$ m s$^{-2}$'
ylabels['ubudget'] = units
ylabels['eddies'] = ylabels['ubudget']
ylabels['winds'] = 'm/s'

plotdays = [-30, 0, 30]
nrow, ncol = 4, 3
advance_by = 'row'
fig_kw = {'figsize' : (11, 9), 'sharex' : 'col', 'sharey' : 'row'}
gridspec_kw = {'left' : 0.08, 'right' : 0.99, 'wspace' : 0.09, 'hspace' : 0.1,
               'bottom' : 0.05, 'top' : 0.92, 'height_ratios' : [1, 0.6, 1, 1]}
legend_kw={'fontsize' : 8, 'loc' : 'upper center', 'ncol' : 2,
           'handlelength' : 2.5}
suptitle = '%d-%d E U and $\psi$ contours, ubudget at 200 hPa' % (lon1, lon2)
tropics = False
if tropics:
    xlims, xticks = (-35, 35), range(-30, 31, 10)
else:
    xlims, xticks = (-60, 60), range(-60, 61, 15)

grp = atm.FigGroup(nrow, ncol, advance_by, fig_kw=fig_kw,
                   gridspec_kw=gridspec_kw, suptitle=suptitle)
for day in plotdays:
    grp.next()
    if grp.row == 0:
        title = 'Day %d' % day
    else:
        title = None
    latpres(data_latp, day, data_latp['PS'], title=title, xlims=xlims, xticks=xticks)
    for nm in ['winds', 'ubudget', 'eddies']:
        grp.next()
        if grp.col == 0:
            legend = True
        else:
            legend = False
        keys = keys_dict[nm]
        lineplot(ubudget, keys, day, style, xlims=xlims,
                 xticks=xticks, legend=legend, legend_kw=legend_kw,
                 ylabel=ylabels[nm])

# ----------------------------------------------------------------------
# Streamfunction decomposition

def psi_latpres(psi, ps, cint=10, xlims=(-60, 60), xticks=range(-60, 61, 15),
                title='', u=None):
    xmin, xmax = xlims
    axlims = (xmin, xmax, 0, 1000)
    if u is not None:
        atm.contour_latpres(u, clev=[0], colors='m', omitzero=False,
                            axlims=axlims)
    atm.contour_latpres(psi, clev=cint, topo=ps, omitzero=True, axlims=axlims)
    plt.xticks(xticks, xticks)
    plt.grid()
    plt.title(title, fontsize=10)


#plotdays = [-30, 0, 30]
plotdays = [-15, 0, 15]
keys = ['TOT', 'MMC', 'EDDY']
xlims, xticks = (-35, 35), range(-30, 31, 10)
cint = 5
nrow, ncol = len(keys), len(plotdays)
advance_by = 'col'
fig_kw = {'figsize' : (11, 7), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.08, 'right' : 0.99, 'wspace' : 0.06, 'hspace' : 0.08,
               'bottom' : 0.08, 'top' : 0.9}
suptitle = '%d-%dE $\psi$ components' % (lon1, lon2)
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
        psi_latpres(psi, data_latp['PS'], cint, xlims, xticks, title=title,
                    u=u)
        if grp.col > 0:
            plt.ylabel('')
        if grp.row < grp.nrow - 1:
            plt.xlabel('')
        atm.text(key, (0.05, 0.88))
