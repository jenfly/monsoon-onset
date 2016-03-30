import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import collections

import atmos as atm
import merra
import indices
import utils

# ----------------------------------------------------------------------
yearstr = '1979-2014'
onset_nm = 'CHP_MFC'
ndays = 5
lon1, lon2 = 60, 100
plev = 200
daynm, yearnm, latname, lonname = 'dayrel', 'year', 'YDim', 'XDim'
datadir = atm.homedir() + 'datastore/merra/analysis/'
savedir = 'figs/'
filenm = 'merra_ubudget%d_dailyrel_%s_ndays%d_%dE-%dE_%s.nc'
filenm = datadir + filenm % (plev, onset_nm, ndays, lon1, lon2, yearstr)
files = {}
files['ubudget'] = filenm
varnms = ['U', 'V']
for nm in varnms:
    filenm = datadir + 'merra_%s%d_dailyrel_%s_%s.nc'
    files[nm] = filenm % (nm, plev, onset_nm, yearstr)
    filenm = datadir + 'merra_%s_sector_%dE-%dE_dailyrel_%s_%s.nc'
    files[nm + '_latp'] = filenm % (nm, lon1, lon2, onset_nm, yearstr)

# ----------------------------------------------------------------------
# Read data from each year

# Zonal momentum budget components
with xray.open_dataset(files['ubudget']) as ubudget:
    ubudget.load()


# Scaling factor for all terms in momentum budget
scale = 1e-4
ubudget.attrs['comp_units'] = '%.0e m/s2' % scale
ubudget = ubudget / scale

# Read other lat-lon variables and smooth with rolling mean
for nm in varnms:
    varnm = '%s%d' % (nm, plev)
    with xray.open_dataset(files[nm]) as ds:
        var = ds[varnm].load()
    daydim = atm.get_coord(var, coord_name=daynm, return_type='dim')
    ubudget[nm] = atm.rolling_mean(var, ndays, axis=daydim, center=True)

# Read other lat-pres variables and smooth with rolling mean
data_latp = xray.Dataset()
for nm in varnms:
    varnm = nm + '_latp'
    with xray.open_dataset(files[varnm]) as ds:
        var = ds[nm].load()
    daydim = atm.get_coord(var, coord_name=daynm, return_type='dim')
    data_latp[nm] = atm.rolling_mean(var, ndays, axis=daydim, center=True)

# Compute streamfunction
if (lon2 - lon1) < 360:
    sector_scale = (lon2 - lon1) / 360.
else:
    sector_scale = None
data_latp['PSI'] = atm.streamfunction(data_latp['V'], sector_scale=sector_scale)

# Additional metadata
ubudget.attrs['plev'] = plev
ubudget.attrs['ndays'] = ndays
ubudget.attrs['lon1'] = lon1
ubudget.attrs['lon2'] = lon2

# Topography for lat-pres contour plots
psfile = atm.homedir() + 'dynamics/python/atmos-tools/data/topo/ncep2_ps.nc'
with xray.open_dataset(psfile) as ds:
    ps = ds['ps'] / 100
    if (lon2 - lon1) < 360:
        ps = atm.dim_mean(ps, 'lon', lon1, lon2)
    else:
        ps = atm.dim_mean(ps, 'lon')

# ----------------------------------------------------------------------
# Consolidate terms together
groups = collections.OrderedDict()
groups['ADV_AVG'] = ['ADV_AVG_AVG_X', 'ADV_AVG_AVG_Y', 'ADV_AVG_AVG_P']
groups['ADV_AVST'] = ['ADV_AVG_ST_X', 'ADV_AVG_ST_Y', 'ADV_AVG_ST_P']
groups['ADV_STAV'] = ['ADV_ST_AVG_X', 'ADV_ST_AVG_Y', 'ADV_ST_AVG_P']
groups['ADV_CRS'] = ['ADV_AVST', 'ADV_STAV']
groups['EMFC_TR'] = ['EMFC_TR_X', 'EMFC_TR_Y', 'EMFC_TR_P']
groups['EMFC_ST'] = ['EMFC_ST_X', 'EMFC_ST_Y', 'EMFC_ST_P']
groups['EMFC'] = ['EMFC_TR', 'EMFC_ST']
groups['COR'] = ['COR_AVG', 'COR_ST']
groups['ADV+COR'] = ['ADV_AVG', 'COR_AVG']
groups['SUM'] = ['ADV_AVG', 'ADV_CRS', 'EMFC', 'COR', 'PGF_ST', 'ANA']

for key in groups:
    nms = groups[key]
    ubudget[key] = ubudget[nms[0]]
    for nm in nms[1:]:
        ubudget[key] = ubudget[key] + ubudget[nm]

# Tile the zonal mean values
varbig = ubudget['SUM']
for nm in ubudget.data_vars:
    if lonname not in ubudget[nm].dims:
        vals = atm.biggify(ubudget[nm], varbig, tile=True)
        ubudget[nm] = xray.DataArray(vals, coords=varbig.coords)

# Sector mean budget
ubudget_sector = atm.dim_mean(ubudget, 'lon', lon1, lon2)

# ----------------------------------------------------------------------
# Utility functions and plot formatting options

def saveclose(filestr):
    atm.savefigs(filestr, ext='pdf', merge=True)
    plt.close('all')

def get_daystr(plotdays):
    if len(atm.makelist(plotdays)) > 1:
        daystr = 'Rel Days %d to %d' % (plotdays[0], plotdays[-1])
        savestr = 'reldays%d_%d' % (plotdays[0], plotdays[-1])
    else:
        daystr = 'Rel Day %d' % plotdays
        savestr = 'relday%d' % plotdays
    return daystr, savestr

# ----------------------------------------------------------------------
# Lat-pres contours and line plots on individual days

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


style = {'ADV_AVG' : 'b', 'COR_AVG' : 'b--', 'ADV+COR' : 'r',
         'PGF_ST' : 'k', 'ADV_CRS' : 'g',  'ADV_AVST' : 'g--',
         'ADV_STAV' : 'g-.', 'EMFC' : 'm', 'EMFC_TR' : 'm--', 'EMFC_ST' : 'm-.',
         'SUM' : 'k--', 'ACCEL' : 'c', 'ANA' : 'y', 'U' : 'k', 'V' : 'k--'}

keys_dict = collections.OrderedDict()
keys_dict['ubudget'] = ['ADV_AVG', 'COR_AVG', 'ADV+COR', 'PGF_ST',
                        'ADV_CRS', 'EMFC', 'ANA', 'SUM', 'ACCEL']
keys_dict['winds'] = ['U', 'V']
keys_dict['eddies'] = ['EMFC_TR', 'EMFC_ST', 'EMFC', 'ADV_CRS']

ylabels = {}
units = '$10^{-4} m s^{-2}$'
ylabels['ubudget'] = '%d hPa ubudget (%s)' % (plev, units)
ylabels['eddies'] = ylabels['ubudget']
ylabels['winds'] = '%d hPa winds (m/s)' % plev


nrow, ncol = 4, 5
advance_by = 'row'
fig_kw = {'figsize' : (18, 12), 'sharex' : 'col', 'sharey' : 'row'}
gridspec_kw = {'left' : 0.05, 'right' : 0.99, 'wspace' : 0.06, 'hspace' : 0.08,
               'bottom' : 0.04, 'top' : 0.92, 'height_ratios' : [1, 0.6, 1, 1]}
legend_kw={'fontsize' : 8, 'loc' : 'lower center', 'ncol' : 2,
           'handlelength' : 2.5}
suptitle = '%d-%d E U and $\psi$ contours, ubudget at 200 hPa' % (lon1, lon2)
plotdays = [-30, -15, 0, 15, 30] + [-90, -45, 0, 45, 90]
for tropics in [False, True]:
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
        latpres(data_latp, day, ps, title=title, xlims=xlims, xticks=xticks)
        for nm in ['winds', 'ubudget', 'eddies']:
            grp.next()
            if grp.col == 0:
                legend = True
                if nm == 'ubudget' :
                    legend_kw['loc'] = 'lower center'
                else:
                    legend_kw['loc'] = 'upper center'
            else:
                legend = False
            keys = keys_dict[nm]
            lineplot(ubudget_sector, keys, day, style, xlims=xlims,
                     xticks=xticks, legend=legend, legend_kw=legend_kw,
                     ylabel=ylabels[nm])

saveclose(savedir + 'ubudget_sector_latpres_lineplots')

# ----------------------------------------------------------------------
# Plot groups together

keys_list = [['ADV_AVG', 'ADV_CRS', 'COR_AVG', 'COR_ST', 'EMFC', 'PGF_ST',
              'SUM', 'ACCEL'],
             ['U', 'V'],
             ['ADV_AVG', 'ADV_AVST', 'ADV_STAV', 'ADV_CRS'],
             ['COR_AVG', 'COR_ST', 'COR'],
             ['EMFC_TR', 'EMFC_ST', 'EMFC']]

def pcolor_sector(var, daynm, clims, u=None, v=None):
        days = var[daynm].values
        lat = atm.get_coord(var, 'lat')
        x, y = np.meshgrid(days, lat)
        vals = var.values.T
        vals = np.ma.masked_array(vals, mask=np.isnan(vals))
        plt.pcolormesh(x, y, vals, cmap='RdBu_r')
        plt.clim(clims)
        plt.colorbar(extend='both')
        if u is not None:
            plt.contour(x, y, u.values.T, [0], colors='k', linewidths=1.5)
        if v is not None:
            plt.contour(x, y, v.values.T, [0], colors='k', alpha=0.5)
        plt.xlim(days.min(), days.max())
        plt.xlabel('Rel Day')
        plt.ylabel('Latitude')

def plot_groups(ubudget, keys_list, daynm, plotdays=None, latlims=None):
    """Plot groups of lat-lon or lat-day plots.
    """

    if latlims is not None:
        ubudget = atm.subset(ubudget, {'lat' : latlims})

    units = ubudget.attrs['comp_units']
    plev = ubudget.attrs['plev']
    lon1, lon2 = ubudget.attrs['lon1'], ubudget.attrs['lon2']

    try:
        lon = atm.get_coord(ubudget, 'lon')
        sector = False
    except ValueError:
        sector = True

    if sector:
        suptitle = '%d-%d E Zonal Momentum Budget at %d hPa (%s)'
        suptitle = suptitle % (lon1, lon2, plev, units)
        xticks = range(-120, 201, 60)
    else:
        daystr, savestr = get_daystr(plotdays)
        suptitle = '%s Zonal Momentum Budget at %d hPa (%s)'
        suptitle = suptitle % (daystr, plev, units)
        xticks = range(40, 121, 20)

    nrow, ncol = 3, 4
    figsize = (14, 10)
    opts = {'left' : 0.05, 'right' : 0.95, 'bottom' : 0.04, 'top' : 0.92,
            'wspace' : 0.1, 'hspace' : 0.1}

    for i, keys in enumerate(keys_list):
        if sector:
            data = ubudget[keys]
        else:
            data = atm.subset(ubudget[keys], {daynm : (plotdays, None)})
            if len(atm.makelist(plotdays)) > 1:
                data = data.mean(dim=daynm)
        clims = atm.climits(data, symmetric=True)
        if sector:
            clims = 0.9 * np.array(clims)
        if i == 0 or i == 2:
            isub = 0
            plt.figure(figsize=figsize)
            plt.suptitle(suptitle)
            plt.subplots_adjust(**opts)
        for j, nm in enumerate(keys):
            isub += 1
            if 'U' in keys:
                clims = atm.climits(data[nm], symmetric=True)
            plt.subplot(nrow, ncol, isub)
            if sector:
                pcolor_sector(data[nm], daynm, clims, ubudget['U'], ubudget['V'])
            else:
                atm.pcolor_latlon(data[nm], fancy=False)
                plt.clim(clims)
            plt.title(nm, fontsize=9)
            atm.fmt_subplot(nrow, ncol, isub, xticks=xticks)
            plt.grid(True)
        # Skip to next row if necessary
        if ncol > len(keys):
            isub += ncol - len(keys)

for tropics in [True, False]:
    savestr = savedir + 'ubudget_'
    if tropics:
        savestr = savestr + 'tropics_'
        latlims = [-30, 30]
    else:
        latlims = None

    # Lat-lon maps
    for plotdays in [-90, -30, 0, 30, 60]:
        plot_groups(ubudget, keys_list, daynm, plotdays, latlims)
    saveclose(savestr + 'latlon')

    # Sector lat-day maps
    plot_groups(ubudget_sector, keys_list, daynm, None, latlims)
    saveclose(savestr + 'sector_latday')


# ----------------------------------------------------------------------
# def zerocrossings(var, latmin, latmax, smoothing=30, interp_res=0.1, nkeep=3):
#     var = atm.subset(var, {'lat' : (latmin, latmax)})
#     if smoothing is not None:
#         var = atm.rolling_mean(var, smoothing, axis=0, center=True)
#     lat = atm.get_coord(var, 'lat')
#     lat_i = np.arange(latmin, latmax + interp_res, interp_res)
#     daynm = var.dims[0]
#     days = var[daynm]
#     crossings = np.nan * np.ones((nkeep, len(days)), dtype=float)
#
#     for d, day in enumerate(days):
#         vals = var.sel(**{daynm : day})
#         if not np.isnan(vals).all():
#             vals = np.interp(lat_i, lat, vals)
#             icross = np.where(np.diff(np.sign(vals)))[0]
#             latcross = lat_i[icross]
#             n = min(nkeep, len(latcross))
#             crossings[:n, d] = latcross[:n]
#
#     coords = {'n' : np.arange(nkeep) + 1, daynm : var[daynm]}
#     crossings = xray.DataArray(crossings, name='zerolat', dims=['n', daynm],
#                                coords=coords)
#
#     return crossings
#
# def psimax_lat(psi, latmin=-30, latmax=10, pmin=300, pmax=700, nsmooth=5):
#     days_in = psi['dayrel']
#     psi = atm.subset(psi, {'lat' : (latmin, latmax), 'plev' : (pmin, pmax)},
#                      squeeze=True)
#     psi = psi[nsmooth:-nsmooth]
#     pdim = atm.get_coord(psi, 'plev', 'dim')
#     psi = psi.max(axis=pdim)
#
#     lat = atm.get_coord(psi, 'lat')
#     latdim = atm.get_coord(psi, 'lat', 'dim')
#     ilatmax = psi.argmax(axis=latdim)
#     latmax = lat[ilatmax]
#     days = atm.get_coord(psi, 'dayrel')
#     latmax = xray.DataArray(latmax, coords={'dayrel' : days})
#     latmax = latmax.reindex_like(days_in)
#     return latmax

# ----------------------------------------------------------------------
# # Line plots on individual days
#
# latmin, latmax = -40, 50
# smoothing = None
# nkeep = {'U' : 2, 'V' : 3}
# zerolats = xray.Dataset()
# for nm in nkeep:
#     n = nkeep[nm]
#     crossings = zerocrossings(ubudget_sector[nm], latmin, latmax, nkeep=n,
#                               smoothing=smoothing)
#     for i in crossings['n'].values:
#         key = nm + '%d' % i
#         zerolats[key] = crossings.sel(n=i).drop('n')
#
# check_zerolats = False
# if check_zerolats:
#     plt.figure()
#     for nm in zerolats.data_vars:
#         plt.plot(zerolats[daynm], zerolats[nm], label=nm)
#     plt.legend()
#
#
# style = {'ADV_AVG' : 'b', 'COR_AVG' : 'b--', 'ADV+COR' : 'r',
#          'PGF_ST' : 'k', 'ADV_CRS' : 'g',  'ADV_AVST' : 'g--',
#          'ADV_STAV' : 'g-.', 'EMFC' : 'm', 'EMFC_TR' : 'm--', 'EMFC_ST' : 'm-.',
#          'SUM' : 'k--', 'ACCEL' : 'c', 'ANA' : 'y', 'U' : 'k', 'V' : 'k--'}
#
# keys_dict = collections.OrderedDict()
# keys_dict['ubudget'] = ['ADV_AVG', 'COR_AVG', 'ADV+COR', 'PGF_ST',
#                         'ADV_CRS', 'EMFC', 'ANA', 'SUM', 'ACCEL']
# keys_dict['winds'] = ['U', 'V']
# keys_dict['eddies'] = ['EMFC_TR', 'EMFC_ST', 'EMFC', 'ADV_AVST', 'ADV_STAV',
#                        'ADV_CRS']
# suptitle = '%d-%d E %s at %d hPa'
# suptitles = {}
# suptitles['ubudget'] = suptitle % (lon1, lon2, 'Zonal Momentum Budget', plev)
# suptitles['eddies'] = suptitles['ubudget']
# suptitles['winds'] = suptitle % (lon1, lon2, 'Winds', plev)
# ylabels = {}
# ylabels['ubudget'] = 'ubudget (%s)' % ubudget.attrs['comp_units']
# ylabels['eddies'] = ylabels['ubudget']
# ylabels['winds'] = 'winds (m/s)'
#
# plotdays = [-90, -30, -15, 0, 15, 30, 60, 90]
# nrow, ncol = 2, 4
# figsize = (14, 10)
# lat = atm.get_coord(ubudget, 'lat')
# latname = atm.get_coord(ubudget, 'lat', 'name')
# opts = {'left' : 0.05, 'right' : 0.95, 'bottom' : 0.06, 'top' : 0.94,
#         'wspace' : 0.1, 'hspace' : 0.1}
# lg_row, lg_col, lg_loc, lg_ncol = 2, 1, 'upper center', 2
# zlat_opts = {'U1' : {'label' : 'U=0'}, 'U2' : {},
#              'V1' : {'linestyle' : 'dashed', 'label' : 'V=0'},
#              'V2' : {'linestyle' : 'dashed'}, 'V3' : {'linestyle' : 'dashed'}}
#
# for nm in keys_dict:
#     keys = keys_dict[nm]
#     suptitle, ylabel = suptitles[nm], ylabels[nm]
#     for latlims in [(-60, 60), (-35, 35)]:
#         fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True,
#                                  sharey=True)
#         plt.subplots_adjust(**opts)
#         plt.autoscale(tight=True)
#         plt.suptitle(suptitle)
#         for i, day in enumerate(plotdays):
#             row, col = atm.subplot_index(nrow, ncol, i + 1)
#             ax = axes[row - 1, col - 1]
#             subset_dict = {daynm : (day, day), latname: latlims}
#             data = atm.subset(ubudget_sector[keys], subset_dict, squeeze=True)
#             #data = data.drop(daynm).to_dataframe()
#             data = data.to_dataframe()
#             data.plot(ax=ax, style=style, legend=False)
#             # Plot vertical lines for U=0 and V=0
#             zlats = zerolats.sel(**{daynm : day})
#             for nm in zlats.data_vars:
#                 ax.axvline(zlats[nm], color='k', alpha=0.5, linewidth=1.5,
#                            **zlat_opts[nm])
#             ax.set_title('Day %d' % day, fontsize=10)
#             ax.grid(True)
#             if row == lg_row and col == lg_col:
#                 ax.legend(fontsize=9, loc=lg_loc, ncol=lg_ncol, handlelength=3)
#             if row == nrow:
#                 ax.set_xlabel('Lat')
#             if col == 1:
#                 ax.set_ylabel(ylabel)
#
# saveclose(savedir + 'ubudget_sector_lineplots')

# ----------------------------------------------------------------------
# latmax = psimax_lat(data_latp['PSI'], nsmooth=ndays, pmin=600, pmax=700)
#
# # Ubudget terms at latitude of psimax
# print('Computing ubudget terms at latitude of psimax for each day')
# days = latmax['dayrel']
# days = days[np.isfinite(latmax)]
# ubudget_psimax = xray.Dataset()
# for d, day in enumerate(days):
#     lat0 = latmax.sel(dayrel=day).values
#     ds = atm.subset(ubudget_sector, {'lat' : (lat0, lat0)}, squeeze=True)
#     ds = atm.subset(ds, {'dayrel' : (day, day)}, squeeze=False)
#     if d == 0:
#         ubudget_psimax = ds
#     else:
#         ubudget_psimax = xray.concat([ubudget_psimax, ds], dim='dayrel')
#
# keys = ['ADV_AVG', 'COR_AVG', 'ADV+COR_AVG', 'PGF_ST',
#         'ADV_CRS', 'EMFC', 'ANA', 'SUM', 'ACCEL']
#
# xticks = range(-120, 201, 30)
# xlims = [-120, 200]
# plt.figure(figsize=(8, 12))
# plt.subplot(2, 1, 1)
# plt.plot(latmax['dayrel'], latmax)
# plt.xticks(xticks)
# plt.xlim(xlims)
# plt.grid(True)
# plt.subplot(2, 1, 2)
# ubudget_psimax[keys].to_dataframe().plot(ax=plt.gca(), style=style, legend=False)
# plt.legend(fontsize=8, ncol=3)
# plt.xticks(xticks)
# plt.xlim(xlims)
# plt.grid(True)
