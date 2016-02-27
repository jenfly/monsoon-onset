import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
import collections
import pandas as pd

import atmos as atm
import precipdat
import merra
import indices
import utils

mpl.rcParams['font.size'] = 10

# ----------------------------------------------------------------------
#onset_nm = 'HOWI'
onset_nm = 'CHP_MFC'
#onset_nm = 'CHP_PCP'

# years, years2 = np.arange(1979, 2015), None
# yearstr, savestr = '%d-%d Climatology' % (years.min(), years.max()), 'clim'
#savestr = 'clim_pre1pre2'

# CHP_MFC Early/Late Years
# years = [2004, 1999, 1990, 2000, 2001]
# years2 = [1983, 1992, 1997, 2014, 2012]
# yearstr, savestr = 'Late Minus Early Anomaly', 'late_minus_early'
years, yearstr = [2004, 1999, 1990, 2000, 2001], '5 Earliest Years'
years2, savestr = None, 'early'
# years, yearstr = [1983, 1992, 1997, 2014, 2012], '5 Latest Years'
# years2, savestr = None, 'late'

# HOWI Early/Late Years
# years, yearstr = [2004, 2000, 1999, 2001, 1990], '5 Earliest Years'
# years, yearstr = [1983, 1979, 1997, 1992, 1995], '5 Latest Years'

datadir = atm.homedir() + 'datastore/merra/analysis/'
savedir = 'figs/'
run_anim = False
run_eht = False

vargroup = 'subset'

varlist = {
    'test' : ['precip', 'U200'],
    'subset' : ['precip', 'U200', 'V200', 'T200', 'H200', 'U850', 'THETA_E950'],
    'group1' : ['precip', 'U200', 'V200', 'H200', 'T200'],
    'group2' : ['U850', 'V850', 'H850', 'T850', 'QV850'],
    'group3' : ['THETA950', 'THETA_E950', 'V*THETA_E950',
                'HFLUX', 'EFLUX', 'EVAP'],
    'nearsurf' : ['T950', 'H950', 'QV950', 'V950', 'THETA950',
                  'THETA_E950', 'DSE950', 'MSE950', 'V*THETA950',
                  'V*THETA_E950', 'V*DSE950', 'V*MSE950']
}

varnms = varlist[vargroup]

remove_tricky = False
years_tricky = [2002, 2004, 2007, 2009, 2010]

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# Plotting anomalies (strong - weak, or regression coefficients)
# or climatology
if years2 is not None:
    anom_plot = True
else:
    anom_plot = False

# Day range for latitude vs. day plots
npre, npost = 120, 200

# Day ranges for lat-lon composites
comps_all = {'PRE' : range(-5, 0), 'POST' : range(15, 20),
             'PRE1' : range(-60, -45), 'PRE2' : range(-30, -15),
             'SSN' : range(0, 137), 'DIFF' : None}

compkeys = ['PRE', 'POST', 'DIFF']

def plusminus(num):
    return atm.format_num(num, ndecimals=0, plus_sym=True)

compdays = collections.OrderedDict()
comp_attrs = {key : {} for key in compkeys}
for key in compkeys:
    compdays[key] = comps_all[key]
    comp_attrs[key]['name'] = key
    if key == 'DIFF':
        comp_attrs[key]['long_name'] = '%s-%s' % (compkeys[1], compkeys[0])
        comp_attrs[key]['axis'] = 2
    else:
        d1 = plusminus(min(compdays[key]))
        d2 = plusminus(max(compdays[key]))
        comp_attrs[key]['long_name'] = 'D0%s:D0%s' % (d1, d2)
        comp_attrs[key]['axis'] = 1


# ----------------------------------------------------------------------
# List of data files

if remove_tricky:
    print('Removing tricky years')
    yearstr = yearstr + '_excl_tricky'
    print(years_tricky)
    years = list(years)
    for year in years_tricky:
        years.remove(year)
    years = np.array(years)

def get_filenames(yrs, varnms, onset_nm, datadir):
    files = collections.OrderedDict()
    filestr = datadir + 'merra_%s_dailyrel_%s_%d.nc'
    for nm in varnms:
        files[nm] = [filestr % (nm, onset_nm, yr) for yr in yrs]
    return files


datafiles = get_filenames(years, varnms, onset_nm, datadir)
if years2 is not None:
    datafiles2 = get_filenames(years2, varnms, onset_nm, datadir)
else:
    datafiles2 = None

# ----------------------------------------------------------------------
# Read data and compute averages and composites

def housekeeping(var):
    # Convert units
    unit_dict={'m' : ('km', 1e-3)}
    units_in = var.attrs['units']
    if units_in in unit_dict:
        attrs = var.attrs
        attrs['units'] = unit_dict[units_in][0]
        var = var * unit_dict[units_in][1]
        var.attrs = attrs

    # Fill Ro200 with NaNs near equator
    if var.name == 'Ro200':
        latbuf = 5
        lat = atm.get_coord(var, 'lat')
        latbig = atm.biggify(lat, var, tile=True)
        vals = var.values
        vals = np.where(abs(latbig)>latbuf, vals, np.nan)
        var.values = vals

    return var

def theta_e_latmax(var):
    lat = atm.get_coord(var, 'lat')
    coords={'year' : var['year'], 'dayrel': var['dayrel']}
    latdim = atm.get_coord(var, 'lat', 'dim')
    latmax = lat[np.nanargmax(var, axis=latdim)]
    latmax = xray.DataArray(latmax, dims=['year', 'dayrel'], coords=coords)
    latmax = atm.dim_mean(latmax, 'year')
    return latmax

def sector_mean(var, lon1, lon2, minfrac=0.5):
    var = atm.subset(var, {'lon' : (lon1, lon2)})
    londim = atm.get_coord(var, 'lon', 'dim')

    # Create mask for any point where more than minfrac fraction is missing
    missings = np.isnan(var)
    missings = missings.sum(axis=londim)
    min_num = var.shape[londim] * minfrac
    mask = missings > min_num

    # Compute mean and apply mask
    var = atm.dim_mean(var, 'lon')
    name, attrs, coords, dims = atm.meta(var)
    attrs['minfrac'] = minfrac
    kwargs = {'name' : name, 'attrs' :  attrs, 'coords' : coords, 'dims' : dims}
    vals = np.ma.masked_array(var.values, mask).filled(np.nan)
    var.values = xray.DataArray(vals, **kwargs)

    return var

def get_composites(var, compdays, comp_attrs):
    keys = compdays.keys()
    comp = xray.Dataset()
    for key in compdays:
        if key == 'DIFF':
            comp[key] = comp[keys[1]] - comp[keys[0]]
        else:
            comp[key] = var.sel(dayrel=compdays[key]).mean(dim='dayrel')

    # Add metadata
    for key in compdays:
        comp[key].attrs = comp_attrs[key]

    return comp


def all_data(datafiles, npre, npost, lon1, lon2, compdays, comp_attrs):
    # Read daily data fields aligned relative to onset day
    data = collections.OrderedDict()
    sectordata = collections.OrderedDict()
    comp = collections.OrderedDict()
    sectorcomp = collections.OrderedDict()
    sector_latmax = {}

    for varnm in datafiles:
        print('Reading daily data for ' + varnm)
        var, onset, retreat = utils.load_dailyrel(datafiles[varnm])
        var = atm.subset(var, {'dayrel' : (-npre, npost)})
        var = housekeeping(var)

        # Compute sector mean and composite averages
        sectorvar = sector_mean(var, lon1, lon2)
        compvar = get_composites(var, compdays, comp_attrs)
        sectorcompvar = get_composites(sectorvar, compdays, comp_attrs)

        # Latitude of maximum subcloud theta_e
        if varnm == 'THETA_E950':
            sector_latmax[varnm] = theta_e_latmax(sectorvar)

        # Compute regression or take the climatology
        if 'year' in var.dims:
            var = atm.dim_mean(var, 'year')
            sectorvar = atm.dim_mean(sectorvar, 'year')
            compvar = atm.dim_mean(compvar, 'year')
            sectorcompvar = atm.dim_mean(sectorcompvar, 'year')

        # Pack everything into dicts for output
        data[varnm], sectordata[varnm] = var, sectorvar
        comp[varnm], sectorcomp[varnm] = compvar, sectorcompvar

    return data, sectordata, sector_latmax, comp, sectorcomp


everything = all_data(datafiles, npre, npost, lon1, lon2, compdays, comp_attrs)
data, sectordata, sector_latmax, comp, sectorcomp = everything

if years2 is not None:
    everything2 = all_data(datafiles2, npre, npost, lon1, lon2, compdays,
                           comp_attrs)
    data2, sectordata2, sector_latmax2, comp2, sectorcomp2 = everything2
    for nm in data:
        data[nm] = data2[nm] - data[nm]
        sectordata[nm] = sectordata2[nm] - sectordata[nm]
        comp[nm] = comp2[nm] - comp[nm]
        sectorcomp[nm] = sectorcomp2[nm] - sectorcomp[nm]

# ----------------------------------------------------------------------
# Housekeeping
#
# # Add extra dimension for year if necessary (i.e. if plotting difference
# # between two sets of years or plotting regression field)
# if years2 is not None:
#     year_coord = xray.DataArray([-1], name='year', coords={'year' : [-1]})
#     for varnm in data:
#         name, attrs, coords, dims = atm.meta(data[varnm])
#         dims = ['year'] + list(dims)
#         coords = atm.odict_insert(coords, 'year', year_coord)
#         vals = np.expand_dims(data[varnm], axis=0)
#         data[varnm] = xray.DataArray(vals, name=name, attrs=attrs, dims=dims,
#                                      coords=coords)


#
#
# # Latitude of maximum theta_e in sector mean
# sector_latmax = collections.OrderedDict()
# varnm = 'THETA_E950'
# if varnm in sectordata and not anom_plot:
#     var = sectordata[varnm]
#     lat = var[latname].values
#     coords={'year' : var['year'], 'dayrel': var['dayrel']}
#     # Yearly
#     latmax = lat[np.nanargmax(var, axis=2)]
#     sector_latmax[varnm] = xray.DataArray(latmax, dims=['year', 'dayrel'],
#                                           coords=coords)
#     # Climatology
#     latmax = lat[np.nanargmax(var.mean(dim='year'), axis=1)]
#     key = varnm + '_CLIM'
#     sector_latmax[key] = xray.DataArray(latmax, coords={'dayrel' : var['dayrel']})


# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Plotting params and utilities

axlims = (-60, 60, 40, 120)

def get_colormap(varnm, anom_plot):
    if varnm == 'precip' and not anom_plot:
        cmap = 'hot_r'
    else:
        cmap = 'RdBu_r'
    return cmap



def annotate_theta_e(days, latmax):
    latmax_0 = latmax.sel(dayrel=0)
    plt.plot(days, latmax, 'k', linewidth=2, label='Latitude of Max')
    plt.legend(loc='lower left')
    ax = plt.gca()
    s = atm.latlon_labels(latmax_0, latlon='lat', fmt='%.1f')
    ax.annotate(s, xy=(0, latmax_0), xycoords='data',
                xytext=(-50, 50), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"))

def lineplot(sectors, ax1=None, y1_label='', y2_label='',
             legend_opts = {'fontsize' : 8, 'loc' : 'lower center',
                            'handlelength' : 3, 'frameon' : False},
             ax2_color='r', ax2_alpha=0.5):
    if ax1 is None:
        ax1 = plt.gca()

    ax1_fmts = [{'color' : 'k', 'linestyle' : 'dashed'}, {'color' : 'k'},
                {'color' : 'k', 'linewidth' : 1.5}]
    ax2_fmts = [{'linewidth' : 2, 'alpha' : ax2_alpha, 'color' : ax2_color}]
    lat = atm.get_coord(sectors, 'lat')
    ax1.set_xlabel('Lat')
    ax1.set_ylabel(y1_label)
    ax1.grid(True)
    i1, i2 = 0, 0
    for key in sectors.data_vars:
        var = sectors[key]
        if var.attrs['axis'] == 1:
            ax, fmts = ax1, ax1_fmts[i1]
            i1 += 1
        else:
            if i2 == 0:
                ax2 = ax1.twinx()
            ax, fmts = ax2, ax2_fmts[i2]
            i2 += 1
        ax.plot(lat, var, label=key, **fmts)

    if legend_opts is not None:
        legend_opts['ncol'] = i1
        ax1.legend(**legend_opts)
    if i2 > 0:
        ax2.set_ylabel(y2_label, color=ax2_color, alpha=ax2_alpha)
        for t1 in ax2.get_yticklabels():
            t1.set_color(ax2_color)
    plt.draw()
    return None

# ----------------------------------------------------------------------
# Latitude-time contour plot

figsize = (10, 12)
nrow, ncol = (2, 1)
keys = sectordata.keys()
for i, varnm in enumerate(keys):
    if i % nrow == 0:
        plt.figure(figsize=figsize)
    plt.subplot(nrow, ncol, 1 + i % nrow)
    plotdata = sectordata[varnm]
    lat = atm.get_coord(plotdata, 'lat')
    days = atm.get_coord(plotdata, coord_name='dayrel')
    cmap = get_colormap(varnm, anom_plot)
    title = '%d-%dE ' %(lon1, lon2) + varnm + ' - ' + yearstr
    utils.contourf_lat_time(lat, days, plotdata, title, cmap, onset_nm)
    plt.ylim(axlims[0], axlims[1])
    if i % nrow == 0:
        plt.xlabel('')
    # Add latitudes of maxima
    if varnm in ['THETA_E950'] and not anom_plot:
        latmax = sector_latmax[varnm]
        annotate_theta_e(days, latmax)

filestr = 'sector_%d-%dE-onset_%s-%s-%s'
filestr = filestr % (lon1, lon2, onset_nm, savestr, vargroup)
atm.savefigs(savedir + filestr, 'pdf', merge=True)
plt.close('all')


# ----------------------------------------------------------------------
# Composite averages

# Plot lat-lon maps and sector means of pre/post onset composite averages
climits = {'precip' : (0, 20), 'U200' : (-50, 50), 'V200' : (-10, 10),
           'Ro200' : (-1, 1), 'rel_vort200' : (-4e-5, 4e-5),
           'abs_vort200' : (-2e-4, 2e-4), 'T200' : (213, 227),
           'H200' : (11.2, 12.6), 'U850' : (-20, 20), 'V850' : (-10, 10),
           'rel_vort850' : (-3e-5, 3e-5), 'abs_vort850' : (-1.5e-4, 1.5e-4),
           'EMFD200' : (-2e-4, 2e-4),
           'H850' : (1.1, 1.6), 'T850' : (260, 305),
           'QV850' : (0, 0.015),
           'THETA975' : (260, 315), 'THETA_E975' : (260, 370),
           'DSE975' : (2.6e5, 3.2e5), 'MSE975' : (2.5e5, 3.5e5),
           'V*DSE975' : (-6e6,6e6), 'V*MSE975' : (-9e6, 9e6),
           'THETA950' : (260, 315), 'THETA_E950' : (260, 365),
           'DSE950' : (2.6e5, 3.2e5), 'MSE950' : (2.5e5, 3.5e5),
           'V*DSE950' : (-4.5e6,4.5e6), 'V*MSE950' : (-5e6, 5e6),
           'V*THETA950' : (-4500, 4500), 'V*THETA_E950' : (-4900, 4900),
           'HFLUX' : (-125, 125), 'EFLUX' : (-200, 200), 'EVAP' : (-8, 8)}

keys = compdays.keys()
y1_label = ', '.join([key for key in compdays if comp_attrs[key]['axis'] == 1])
y2_label = ', '.join([key for key in compdays if comp_attrs[key]['axis'] == 2])
subset_dict = {'lat' : (axlims[0], axlims[1]), 'lon' : (axlims[2], axlims[3])}
nrow, ncol = 2, 4
fig_opts = {'figsize' : (12, 7), 'sharex' : 'col'}
subplot_opts = {'left' : 0.06, 'right' : 0.95}

for varnm in comp:
    dat = {key : atm.subset(comp[varnm][key], subset_dict)
           for key in keys}
    if anom_plot:
        cmax = max([abs(dat[key]).max().values for key in keys])
        cmin = -cmax
    else:
        cmin, cmax = climits[varnm][0], climits[varnm][1]

    fig, axes = plt.subplots(nrow, ncol, **fig_opts)
    plt.subplots_adjust(**subplot_opts)
    plt.suptitle('%s Composites Relative to %s Onset Day - %s' %
                 (varnm.upper(), onset_nm, yearstr))

    # Lat-lon maps of composites
    for i, key in enumerate(keys):
        if comp_attrs[key]['axis'] == 1:
            cmap = get_colormap(varnm, anom_plot)
        else:
            cmap = 'RdBu_r'
        plt.subplot(nrow, ncol, i + 1)
        atm.pcolor_latlon(dat[key], axlims=axlims, cmap=cmap, fancy=False)
        if comp_attrs[key]['axis'] == 1:
            plt.clim(cmin, cmax)
        else:
            symmetric = atm.symm_colors(dat[key])
            if symmetric:
                cmax = np.nanmax(abs(dat[key]))
                plt.clim(-cmax, cmax)
        plt.title(key.upper())

    # Line plots of sector averages
    ax = axes[0, ncol - 1]
    lineplot(sectorcomp[varnm], ax, y1_label, y2_label)

filestr = 'comp-onset_%s-%s-%s' % (onset_nm, savestr, vargroup)
atm.savefigs(savedir + filestr, 'pdf', merge=True)
plt.close('all')
# ----------------------------------------------------------------------
    #
    # # Lat-lon map of difference between composites
    # plt.subplot(2, 3, 3)
    # atm.pcolor_latlon(dat[key2] - dat[key1], axlims=axlims, cmap='RdBu_r')
    # symmetric = atm.symm_colors(dat[key2] - dat[key1])
    # if symmetric:
    #     cmax = np.nanmax(abs(dat[key2] - dat[key1]))
    #     plt.clim(-cmax, cmax)
    # plt.title('Difference (%s-%s)' % (key2.upper(), key1.upper()))
    #
    # # Line plot of sector mean
    # sector1 = sectorcomp[varnm][key1].mean(dim='year')
    # sector2 = sectorcomp[varnm][key2].mean(dim='year')
    # lat = atm.get_coord(sector1, 'lat')
    # plt.subplot(2, 2, 3)
    # plt.plot(lat, sector1, 'b', label=key1.upper())
    # plt.plot(lat, sector2, 'r', label=key2.upper())
    # plt.title('%d-%d E Composites' % (lon1, lon2))
    # if varnm in ['precip', 'Ro_200', 'rel_vort200']:
    #     legend_loc = 'upper right'
    # elif varnm in ['V850']:
    #     legend_loc = 'lower left'
    # else:
    #     legend_loc = 'lower right'
    # plt.legend(loc=legend_loc)
    # plt.subplot(2, 2, 4)
    # plt.plot(lat, sector2 - sector1)
    # plt.title('%d-%d E Difference (%s-%s)' % (lon1, lon2, key2.upper(), key1.upper()))
    # for i in [3, 4]:
    #     plt.subplot(2, 2, i)
    #     plt.xlim(axlims[0], axlims[1])
    #     plt.xlabel('Latitude')
    #     plt.ylabel(varnm)
    #     plt.grid()

# ----------------------------------------------------------------------
# Cross-equatorial atmospheric heat fluxes

if run_eht:
    keys = ['V*DSE950','V*MSE950']
    eht = {key : data[key] for key in keys}
    lat0 = 0.625
    for key in eht:
        eht[key] = atm.squeeze(atm.subset(eht[key], {'lat' : (lat0, lat0)}))
        eht[key] = eht[key].mean(dim='year')

    # Plot longitude-time contours
    figsize = (10, 10)
    ncont = 20
    cmap = 'RdBu_r'
    for key in eht:
        plt.figure(figsize=figsize)
        ehtplot = eht[key]
        days = ehtplot['dayrel'].values
        lon = ehtplot['XDim'].values
        plt.contourf(lon, days, ehtplot, ncont, cmap=cmap)
        plt.title('Cross-Equatorial ' + key)
        plt.xlabel('Longitude')
        plt.ylabel('Relative Day')
        cb = plt.colorbar()
        cmax = abs(cb.boundaries).max()
        plt.clim(-cmax, cmax)
        plt.gca().invert_yaxis()

# ----------------------------------------------------------------------
# Animation of daily data relative to onset

if run_anim:
    nframes = npre + npost + 1
    fps = 4

    climits = {'precip' : (0, 20),
               'U200' : (-50, 50),
               'V200' : (-10, 10),
               'Ro200' : (-1, 1),
               'rel_vort200' : (-4e-5, 4e-5),
               'T200' : (213, 227),
               'H200' : (11.2e3, 12.6e3),
               'U850' : (-20, 20),
               'V850' : (-10, 10)}

    def animate(i):
        plt.clf()
        m, _ = atm.pcolor_latlon(animdata[i], axlims=axlims, cmap=cmap)
        plt.clim(cmin, cmax)
        day = animdata[daynm + 'rel'].values[i]
        plt.title('%s %s RelDay %d' % (varnm, yearstr, day))
        return m

    for varnm in data:
        savefile = savedir + 'latlon_%s_%s.mp4' % (varnm, yearstr)
        animdata = data[varnm].mean(dim='year')
        cmap = get_colormap(varnm)
        cmin, cmax = climits[varnm]
        fig = plt.figure()
        anim = animation.FuncAnimation(fig, animate, frames=nframes)
        print('Saving to ' + savefile)
        anim.save(savefile, writer='mencoder', fps=fps)
        plt.close()

    # Animated line plots of 60-100E sector mean
    ylimits = {'precip' : (0, 12),
               'U200' : (-20, 50),
               'V200' : (-8.5, 6.5),
               'rel_vort200' : (-3e-5, 4e-5),
               'Ro200' : (-1, 1),
               'H200' : (11.5e3, 12.6e3),
               'T200' : (212, 228),
               'U850' : (-10, 18),
               'V850' : (-8.5, 3.5)}

    def animate2(i):
        plt.clf()
        plt.plot(animdata[latname], animdata[i])
        plt.ylim(ylim1, ylim2)
        plt.grid(True)
        day = animdata[daynm + 'rel'].values[i]
        plt.xlabel('Latitude')
        plt.title('%d-%dE %s %s RelDay %d' % (lon1, lon2, varnm, yearstr, day))

    for varnm in sectordata:
        savefile = savedir + 'sector_%d-%dE_%s_%s.mp4' % (lon1, lon2, varnm, yearstr)
        animdata = sectordata[varnm].mean(dim='year')
        ylim1, ylim2 = ylimits[varnm]
        fig = plt.figure()
        anim = animation.FuncAnimation(fig, animate2, frames=nframes)
        print('Saving to ' + savefile)
        anim.save(savefile, writer='mencoder', fps=fps)
        plt.close()
