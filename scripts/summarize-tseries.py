import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

import atmos as atm
import merra
import precipdat
import utils

mpl.rcParams['font.size'] = 10

# ----------------------------------------------------------------------
onset_nm = 'CHP_MFC'
pcp_nm = 'CMAP'
theta_nm = 'THETA_E950'
dtheta_nm = theta_nm + '_DY'
years = np.arange(1979, 2015)

reldir = atm.homedir() + 'datastore/merra/analysis/'
datadir = atm.homedir() + 'datastore/merra/daily/'
filestr = datadir + 'merra_MFC_40E-120E_90S-90N_%d.nc'
indfiles = [filestr % yr for yr in years]
pcpfile = atm.homedir() + '/datastore/cmap/cmap.enhanced.precip.pentad.mean.nc'

lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
#npre, npost = 90, 200
npre, npost, 120, 200
pmid = 500
nroll = 7
plot_yearly = False

varnms = ['precip', 'U200', 'U850', 'V200', 'V850', 'PSI', 'T200', theta_nm,
          dtheta_nm, 'HFLUX', 'VFLXMSE', 'VFLXCPT', 'VFLXPHI',
          'VFLXLQV']
lat_extract = [-30, -15, 0, 15, 30]

relfiles = {}
yearstr = '%d-%d' % (min(years), max(years))
filestr = reldir + 'merra_%s_dailyrel_%s_%s.nc'
for nm in varnms:
    if nm == 'PSI':
        nm0 = 'V_sector_%dE-%dE' % (lon1, lon2)
    elif nm == 'VFLXLQV':
        nm0 = 'VFLXQV'
    elif nm == dtheta_nm:
        nm0 = theta_nm
    else:
        nm0 = nm
    relfiles[nm] = filestr % (nm0, onset_nm, yearstr)

# ----------------------------------------------------------------------
# Read data and calculate indices

# Precipitation
precip = precipdat.read_cmap(pcpfile, yearmin=min(years), yearmax=max(years))
pcp_box = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)
# -- Interpolate to daily resolution
days = np.arange(1, 367)
pcp_i = np.nan * np.ones((len(years), len(days)))
for y, year in enumerate(years):
    pcp_i[y] = np.interp(days, pcp_box['day'], pcp_box[y])
coords = {'day' : days, 'year' : years}
pcp = xray.DataArray(pcp_i, dims=['year', 'day'], coords=coords)

# Monsoon onset, retreat indices
index = utils.get_onset_indices(onset_nm, indfiles, years)
mfc = atm.rolling_mean(index['ts_daily'], nroll, center=True)
onset = index['onset']
ssn_length=index['length'].mean(dim='year')

data = {}
data['MFC'] = utils.daily_rel2onset(mfc, onset, npre, npost)
data[pcp_nm] = utils.daily_rel2onset(pcp, onset, npre, npost)
data['MFC_ACC'] = utils.daily_rel2onset(index['tseries'], onset, npre, npost)

for nm in varnms:
    print('Loading ' + relfiles[nm])
    with xray.open_dataset(relfiles[nm]) as ds:
        if nm == 'PSI':
            data[nm] = atm.streamfunction(ds['V'])
            psimid = atm.subset(data[nm], {'plev' : (pmid, pmid)},
                                squeeze=True)
            data['PSI%d' % pmid] = psimid
        elif nm == 'VFLXLQV':
            var = atm.dim_mean(ds['VFLXQV'], 'lon', lon1, lon2)
            data[nm] = var * atm.constants.Lv.values
        elif nm == theta_nm:
            theta = ds[nm]
            _, _, dtheta = atm.divergence_spherical_2d(theta, theta)
            data[nm] = atm.dim_mean(ds[nm], 'lon', lon1, lon2)
            data[dtheta_nm] = atm.dim_mean(dtheta, 'lon', lon1, lon2)
        elif nm == dtheta_nm:
            continue
        else:
            data[nm] = atm.dim_mean(ds[nm], 'lon', lon1, lon2)

databar = {}
for nm in data:
    if 'year' in data[nm].dims:
        databar[nm] = data[nm].mean(dim='year')
    else:
        databar[nm] = data[nm]

# ----------------------------------------------------------------------
# Extract variables at specified latitudes

tseries = xray.Dataset()
for nm in ['MFC', 'MFC_ACC', pcp_nm]:
    tseries[nm] = databar[nm]

for nm in varnms:
    print(nm)
    var = atm.subset(databar[nm], {'dayrel' : (-npre, npost)})
    lat = atm.get_coord(var, 'lat')
    if nm == 'PSI':
        var = atm.subset(var, {'lat' : (-25, 10)})
        latname = atm.get_coord(var, 'lat', 'name')
        pname = atm.get_coord(var, 'plev', 'name')
        var_out = var.max(dim=latname).max(dim=pname)
        tseries['PSIMAX'] = atm.rolling_mean(var_out, nroll, center=True)
    else:
        for lat0 in lat_extract:
            lat0_str = atm.latlon_labels(lat0, 'lat', deg_symbol=False)
            key = nm + '_' + lat0_str
            val, ind = atm.find_closest(lat, lat0)
            var_out = atm.squeeze(var[:, ind])
            tseries[key] = atm.rolling_mean(var_out, nroll, center=True)

# ----------------------------------------------------------------------
# Functions for plotting

def fmt_axes(xlims, xticks, ylims, yticks):
    if xticks is not None:
        plt.xticks(xticks)
    if xlims is not None:
        plt.xlim(xlims)
    if yticks is not None:
        plt.yticks(yticks)
    if ylims is not None:
        plt.ylim(ylims)

def clear_labels(axtype, ax=None):
    if ax is None:
        ax = plt.gca()
    if axtype.lower() == 'x':
        ax.set_xlabel('')
        ax.set_xticklabels([])
    else:
        ax.set_ylabel('')
        ax.set_yticklabels([])

def to_dataset(data):
    if isinstance(data, xray.DataArray):
        data = data.to_dataset()
    return data

def contourf_latday(var, clev=None, title='', nc_pref=40, grp=None,
                    xlims=(-120, 200), xticks=np.arange(-120, 201, 30), 
                    ylims=(-60, 60), yticks=np.arange(-60, 61, 20),
                    ssn_length=None):                    
    vals = var.values.T
    lat = atm.get_coord(var, 'lat')
    days = atm.get_coord(var, 'dayrel')    
    if var.name.lower() == 'precip':
        cmap = 'hot_r'
        extend = 'max'        
    else:
        cmap = 'RdBu_r'
        extend = 'both'
    if clev == None:
        symmetric = atm.symm_colors(vals)        
        cint = atm.cinterval(vals, n_pref=nc_pref, symmetric=symmetric)
        clev = atm.clevels(vals, cint, symmetric=symmetric)
    if var.name == 'T200':
        cticks = np.arange(-208, 227, 2)
    else:
        cticks = None
    plt.contourf(days, lat, vals, clev, cmap=cmap, extend=extend)
    plt.colorbar(ticks=cticks)
    fmt_axes(xlims, xticks, ylims, yticks)
    plt.grid()
    plt.title(title)
    plt.axvline(0, color='k')
    if ssn_length is not None:
        plt.axvline(ssn_length, color='k')
    if grp is not None and grp.row == grp.ncol - 1:
        plt.xlabel('Rel Day')
    if grp is not None and grp.col == 0:
        plt.ylabel('Latitude')
    
    

def lineplots(data1, data2=None, data1_style=None, xlims=None, xticks=None,
              ylims=None, yticks=None, length=None, legend=False,
              legend_kw={'fontsize' : 9, 'handlelength' : 2.5},
              y2_lims=None, y2_opts={'color' : 'r', 'alpha' : 0.6},
              y1_label='', y2_label=''):

    data1, data2 = to_dataset(data1), to_dataset(data2)

    for nm in data1.data_vars:
        if data1_style is None:
            plt.plot(data1['dayrel'], data1[nm], label=nm)
        else:
            plt.plot(data1['dayrel'], data1[nm], data1_style[nm], label=nm)
    fmt_axes(xlims, xticks, ylims, yticks)
    plt.grid(True)
    plt.axvline(0, color='k')
    if length is not None:
        plt.axvline(length, color='k')
    plt.xlabel('Rel Day')
    plt.ylabel(y1_label)
    axes = [plt.gca()]

    if data2 is not None:
        plt.sca(plt.gca().twinx())
        for nm in data2.data_vars:
            plt.plot(data2['dayrel'], data2[nm], label=nm, linewidth=2,
                     **y2_opts)
        if y2_lims is not None:
            plt.ylim(y2_lims)
        atm.fmt_axlabels('y', y2_label, **y2_opts)
    axes = axes + [plt.gca()]

    if legend:
        if data2 is None:
            plt.legend(**legend_kw)
        else:
            atm.legend_2ax(axes[0], axes[1], **legend_kw)

    return axes

def stdize_df(df):
    for nm in df.columns:
        df[nm] = (df[nm] - df[nm].mean())/df[nm].std()
    return df

# ----------------------------------------------------------------------
# Lat-day contour plots
keys = ['precip', 'U200', 'PSI500', 'T200']
nrow, ncol = 2, 2
fig_kw = {'figsize' : (11, 7), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.05, 'right' : 0.95, 'bottom' : 0.06, 'top' : 0.95,
               'wspace' : 0.05}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for key in keys:
    grp.next()
    contourf_latday(data[key], title=key.upper(), grp=grp, ssn_length=ssn_length)
                       


# ----------------------------------------------------------------------
# Timeseries summary plot

xlims = (-npre, npost)
xticks = range(-npre, npost + 1, 30)

# keypairs = [(['MFC', pcp_nm], ['MFC_ACC']),
#             (['U850_15N'], ['V850_15N']),
#             (['U200_0N'],['V200_15N']),            
#             (['VFLXCPT_0N', 'VFLXPHI_0N', 'VFLXLQV_0N', 'VFLXMSE_0N'], None),
#             (['T200_30N'], ['T200_30S']),
#             ([dtheta_nm + '_15N'], ['HFLUX_30N'])]
# nrow, ncol = 3, 2
keypairs = [(['MFC', pcp_nm], ['MFC_ACC']),
            (['U850_15N'], ['V850_15N']),
            (['U200_0N'],['V200_15N']),            
            (['T200_30N'], ['T200_30S'])]
opts = [('upper left', 'mm/day', 'mm'),
        ('upper left', 'm/s', 'm/s'),
        ('lower left', 'm/s', 'm/s'),
        ('upper left', 'K', 'K')]
nrow, ncol = 2, 2
fig_kw = {'figsize' : (11, 7), 'sharex' : True}
gridspec_kw = {'left' : 0.08, 'right' : 0.9, 'bottom' : 0.06, 'top' : 0.95,
               'wspace' : 0.3}
styles = ['k', 'k--', 'g', 'm']
legend_kw={'fontsize' : 9, 'handlelength' : 2.5}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw,
                   gridspec_kw=gridspec_kw)
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
    data1_style = {nm : style for (nm, style) in zip(keys1, styles)}
    lineplots(data1, data2, data1_style, xlims, xticks, legend=True,
              length=ssn_length, legend_kw=legend_kw, y1_label=y1_label,
              y2_label=y2_label)

# ----------------------------------------------------------------------
# Explore different timeseries

nrow, ncol = 2, 3
fig_kw = {'figsize' : (14, 10)}
gridspec_kw = {'left' : 0.04, 'right' : 0.94, 'bottom' : 0.06, 'top' : 0.95}

for stdize in [False, True]:
    grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
    for nm in varnms:
        grp.next()
        keys = [key for key in tseries if key.startswith(nm)]
        df = tseries[keys].to_dataframe()
        if stdize:
            df = stdize_df(df)
        df.plot(ax=grp.ax, legend=False)
        plt.legend(fontsize=9)
        plt.grid()
        plt.xlim(xlims)
        plt.xticks(xticks)

# ----------------------------------------------------------------------
# MFC and precip timeseries in individual years

if plot_yearly:
    ylims = (-5, 15)
    yticks = range(-5, 16, 5)
    y2_lims = (-350, 350)
    style = {'MFC' : 'k', pcp_nm : 'k--'}
    fig_kw = {'figsize' : (14, 10)}
    gridspec_kw = {'left' : 0.04, 'right' : 0.94, 'bottom' : 0.06, 'top' : 0.95,
                   'hspace' : 0.07, 'wspace' : 0.07}
    suptitle = 'Daily and Cumulative Precip/MFC for %s Onset' % onset_nm
    nrow, ncol = 3, 4
    grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw,
                       suptitle=suptitle)
    for y, year in enumerate(years):
        grp.next()
        if grp.row == 0 and grp.col == ncol - 1:
            legend = True
        else:
            legend = False
        data1 = xray.Dataset({'MFC' : data['MFC'][y], pcp_nm : data[pcp_nm][y]})
        data2 = data['MFC_ACC'][y]
        axes = lineplots(data1, data2, style, xlims, xticks, ylims, yticks,
                         y2_lims=y2_lims, legend=legend,
                         length=index['length'][y])
        title = '%d\n%d, %d' % (year, onset[y], index['retreat'][y])
        atm.text(title, (0.03, 0.85), fontsize=10)
        if grp.row < nrow - 1:
            for ax in axes:
                clear_labels('x', ax)

        for i, ax in enumerate(axes):
            if i == 0 and grp.col > 0:
                clear_labels('y', ax)
            if i > 0 and grp.col < ncol - 1:
                clear_labels('y', ax)
