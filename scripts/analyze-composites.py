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

mpl.rcParams['font.size'] = 11

# ----------------------------------------------------------------------
#onset_nm = 'HOWI'
onset_nm = 'CHP_MFC'
#onset_nm = 'CHP_PCP'

years, years2 = np.arange(1979, 2015), None
yearstr, savestr = '%d-%d Climatology' % (years.min(), years.max()), 'clim'
#savestr = 'clim_pre1pre2'

# CHP_MFC Early/Late Years
# years = [2004, 1999, 1990, 2000, 2001]
# years2 = [1983, 1992, 1997, 2014, 2012]
# yearstr, savestr = 'Late Minus Early Anomaly', 'late_minus_early'
# years, yearstr = [2004, 1999, 1990, 2000, 2001], '5 Earliest Years'
# years2, savestr = None, 'early'
# years, yearstr = [1983, 1992, 1997, 2014, 2012], '5 Latest Years'
# years2, savestr = None, 'late'

datadir = atm.homedir() + 'datastore/merra/analysis/'
savedir = 'figs/'
run_anim = False
run_eht = False

vargroup = 'pres'

varlist = {
    'test' : ['precip', 'U200'],
    'pres' : ['precip', 'U200', 'T200', 'TLML', 'QLML', 'THETA_E_LML'],
    'group1' : ['precip', 'U200', 'V200', 'T200', 'H200', 'U850', 'V850',
                'H850'],
    'group2' : ['T850', 'QV850', 'THETA950', 'THETA_E950', 'V*THETA_E950',
                'HFLUX', 'EFLUX', 'EVAP'],
    'group3' : ['VFLXCPT', 'VFLXPHI', 'VFLXQV', 'VFLXMSE'],
    'nearsurf' : ['T950', 'H950', 'QV950', 'V950', 'THETA950',
                  'THETA_E950', 'DSE950', 'MSE950', 'V*THETA950',
                  'V*THETA_E950', 'V*DSE950', 'V*MSE950']
}

varnms = varlist[vargroup]

# Day ranges for lat-lon composites
comps_all = {'PRE' : range(-5, 0), 'POST' : range(15, 20),
             'PRE1' : range(-60, -45), 'PRE2' : range(-30, -15),
             'SSN' : range(0, 137), 'DIFF' : None,
             'D0' : [0], 'D15' : [15]}

compkeys = ['D0', 'D15', 'DIFF']
#compkeys = ['PRE', 'POST', 'DIFF']
#compkeys = ['PRE1', 'PRE2', 'SSN']

# Day range for latitude vs. day plots
npre, npost = 120, 200

remove_tricky = False
years_tricky = [2002, 2004, 2007, 2009, 2010]

# Longitude sector
lon1, lon2 = 60, 100

# Plotting anomalies (strong - weak, or regression coefficients)
# or climatology
if years2 is not None:
    anom_plot = True
else:
    anom_plot = False

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
# Set up parameters and metadata for composites

def init_comp(compkeys, comps_all):

    def plusminus(num):
        if num == 0:
            s = ''
        else:
            s = atm.format_num(num, ndecimals=0, plus_sym=True)
        return s

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

    return compdays, comp_attrs

compdays, comp_attrs = init_comp(compkeys, comps_all)

# ----------------------------------------------------------------------
# Read data and compute averages and composites

def housekeeping(var):
    # Convert units
    unit_dict={'m' : ('km', 1e-3)}
    units_in = var.attrs.get('units')
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

def subtract_fields(var1, var2):
    var = var2 - var1
    var.attrs = var1.attrs
    if isinstance(var, xray.Dataset):
        for nm in var.data_vars:
            var[nm].attrs = var1[nm].attrs
    return var

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
        sectorvar = atm.dim_mean(var, 'lon', lon1, lon2)
        compvar = get_composites(var, compdays, comp_attrs)
        sectorcompvar = get_composites(sectorvar, compdays, comp_attrs)

        # Latitude of maximum subcloud theta_e
        if varnm == 'THETA_E950' or varnm == 'THETA_E_LML':
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
        data[nm] = subtract_fields(data[nm], data2[nm])
        sectordata[nm] = subtract_fields(sectordata[nm], sectordata2[nm])
        comp[nm] = subtract_fields(comp[nm], comp2[nm])
        sectorcomp[nm] = subtract_fields(sectorcomp[nm], sectorcomp2[nm])

# ----------------------------------------------------------------------
# Plotting params and utilities

axlims = (-60, 60, 40, 120)

def get_colormap(varnm, anom_plot):
    if varnm == 'precip' and not anom_plot:
        cmap = 'hot_r'
    else:
        cmap = 'RdBu_r'
    return cmap


def annotate_theta_e(days, latmax, ax=None, nroll=7):
    if ax is None:
        ax = plt.gca()
    if nroll is not None:
        latmax = atm.rolling_mean(latmax, nroll, center=True)
    latmax_0 = latmax.sel(dayrel=0)
    ax.plot(days, latmax, 'k', linewidth=2, label='Latitude of Max')
    ax.legend(loc='lower right', fontsize=10)
    s = atm.latlon_labels(latmax_0, latlon='lat', fmt='%.1f')
    ax.annotate(s, xy=(0, latmax_0), xycoords='data',
                xytext=(-50, 50), textcoords='offset points',
                arrowprops=dict(arrowstyle="->"))

def lineplot(sectors, ax1=None, y1_label='', y2_label='', title='',
             latmin=None, latmax=None,
             legend_opts = {'fontsize' : 9, 'loc' : 'lower center',
                            'handlelength' : 2.5, 'frameon' : False},
             ax2_color='r', ax2_alpha=0.5, row=1, nrow=1, y1_lims=None):
    if ax1 is None:
        ax1 = plt.gca()

    ax1_fmts = [{'color' : 'k', 'linestyle' : 'dashed'}, {'color' : 'k'},
                {'color' : 'k', 'linewidth' : 1.5}]
    ax2_fmts = [{'linewidth' : 2, 'alpha' : ax2_alpha, 'color' : ax2_color}]
    lat = atm.get_coord(sectors, 'lat')
    ax1.set_title(title)
    if row < nrow - 1:
        ax1.set_xticklabels([])
    else:
        ax1.set_xlabel('Latitude')
    if y1_lims is not None:
        ax1.set_ylim(y1_lims)
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

    if latmin is not None:
        ax1.set_xlim(latmin, latmax)

    if legend_opts is not None:
        legend_opts['ncol'] = i1 + i2
        if i2 > 0:
            atm.legend_2ax(ax1, ax2, **legend_opts)
        else:
            ax1.legend(**legend_opts)
    if i2 > 0:
        # ax2.set_ylabel(y2_label, color=ax2_color, alpha=ax2_alpha)
        for t1 in ax2.get_yticklabels():
            t1.set_color(ax2_color)

    plt.draw()
    return None

# ----------------------------------------------------------------------
# Latitude-time contour plot

fig_kw = {'figsize' : (11, 7), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.07, 'right' : 0.99, 'bottom' : 0.07, 'top' : 0.95,
               'wspace' : 0.05}
# fig_kw = {'figsize' : (14, 10), 'sharex' : True, 'sharey' : True}
# gridspec_kw = {'left' : 0.05, 'right' : 0.98, 'bottom' : 0.05,
#                'top' : 0.92, 'wspace' : 0.01, 'hspace' : 0.1}
suptitle = '%d-%dE ' %(lon1, lon2) + yearstr
nrow, ncol = (2, 2)
keys = sectordata.keys()
grp = atm.FigGroup(nrow, ncol, advance_by='col', fig_kw=fig_kw,
                   gridspec_kw=gridspec_kw, suptitle=suptitle)
for varnm in keys:
    grp.next()
    plotdata = sectordata[varnm]
    lat = atm.get_coord(plotdata, 'lat')
    days = atm.get_coord(plotdata, coord_name='dayrel')
    cmap = get_colormap(varnm, anom_plot)
    utils.contourf_lat_time(lat, days, plotdata, title=varnm, cmap=cmap,
                            onset_nm=onset_nm)
    plt.ylim(axlims[0], axlims[1])
    if grp.row < nrow - 1:
        plt.xlabel('')
    if grp.col > 0:
        plt.ylabel('')
    # Add latitudes of maxima
    if varnm in ['THETA_E950', 'THETA_E_LML'] and not anom_plot:
        latmax = sector_latmax[varnm]
        annotate_theta_e(days, latmax)
        plt.title('THETA_EB', fontsize=11)
        plt.axvline(0, color='k')


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
           'H850' : (1.1, 1.6), 'T850' : (260, 305), 'QV850' : (0, 0.015),
           'TLML' : (260, 315), 'QLML' : (0, 0.022),
           'THETA_LML' : (260, 315), 'THETA_E_LML' : (270, 360),
           'THETA975' : (260, 315), 'THETA_E975' : (260, 370),
           'DSE975' : (2.6e5, 3.2e5), 'MSE975' : (2.5e5, 3.5e5),
           'V*DSE975' : (-6e6,6e6), 'V*MSE975' : (-9e6, 9e6),
           'THETA950' : (260, 315), 'THETA_E950' : (260, 365),
           'DSE950' : (2.6e5, 3.2e5), 'MSE950' : (2.5e5, 3.5e5),
           'V*DSE950' : (-4.5e6,4.5e6), 'V*MSE950' : (-5e6, 5e6),
           'V*THETA950' : (-4500, 4500), 'V*THETA_E950' : (-4900, 4900),
           'HFLUX' : (-125, 125), 'EFLUX' : (-200, 200), 'EVAP' : (-8, 8),
           'VFLXCPT' : (-1.3e10, 1.3e10), 'VFLXPHI' : (-5e9, 5e9),
           'VFLXQV' : (-350, 350), 'VFLXMSE' : (-1.6e10, 1.6e10)
           }
y1_lims = {'THETA_E_LML' : (275, 360)}
keys = compdays.keys()
y1_label = ''
y2_label = ', '.join([key for key in compdays if comp_attrs[key]['axis'] == 2])
compnms = ['%s (%s)' % (s, comp_attrs[s]['long_name']) for s in compkeys]
suptitle = 'Composites Relative to %s Onset Day - %s\n' % (onset_nm, yearstr)
suptitle = suptitle + ', '.join(compnms)
subset_dict = {'lat' : (axlims[0], axlims[1]), 'lon' : (axlims[2], axlims[3])}
#nrow, ncol, figsize = 4, 4, (12, 14)
#gridspec_kw = {'width_ratios' : [1, 1, 1, 1.5], 'left' : 0.03, 'right' : 0.94,
#               'wspace' : 0.3, 'hspace' : 0.2, 'bottom' : 0.06}
nrow, ncol, figsize = 3, 4, (11, 9)
gridspec_kw = {'width_ratios' : [1, 1, 1, 1.5], 'left' : 0.03, 'right' : 0.96,
               'wspace' : 0.35, 'hspace' : 0.2, 'bottom' : 0.06, 'top' : 0.9}
fig_kw = {'figsize' : figsize}
legend_opts = {'fontsize' : 9, 'handlelength' : 2.5, 'frameon' : False}
grp = atm.FigGroup(nrow, ncol, advance_by='col', fig_kw=fig_kw,
                   gridspec_kw=gridspec_kw, suptitle=suptitle)
for varnm in comp:
    if varnm == 'THETA_E_LML':
        varstr = 'THETA_EB'
    elif varnm.endswith('LML'):
        varstr = varnm.replace('LML', '_EB')
    else:
        varstr = varnm.upper()
    dat = {key : atm.subset(comp[varnm][key], subset_dict)
           for key in keys}
    if anom_plot:
        cmax = max([abs(dat[key]).max().values for key in keys])
        cmin = -cmax
    else:
        cmin, cmax = climits[varnm][0], climits[varnm][1]
    # Lat-lon maps of composites
    for j, key in enumerate(keys):
        grp.next()
        if comp_attrs[key]['axis'] == 1:
            cmap = get_colormap(varnm, anom_plot)
        else:
            cmap = 'RdBu_r'
        atm.pcolor_latlon(dat[key], axlims=axlims, cmap=cmap, fancy=False)
        plt.xticks(range(40, 121, 20))
        if comp_attrs[key]['axis'] == 1:
            plt.clim(cmin, cmax)
        else:
            symmetric = atm.symm_colors(dat[key])
            if symmetric:
                cmax = np.nanmax(abs(dat[key]))
                plt.clim(-cmax, cmax)
        plt.title(varstr + ' ' + key.upper(), fontsize=11)
        if grp.col > 0:
            plt.gca().set_yticklabels([])
        if grp.row < nrow - 1:
            plt.gca().set_xticklabels([])
    # Line plots of sector averages
    grp.next()
    if varnm == 'precip':
        legend_opts['loc'] = 'upper center'
    else:
        legend_opts['loc'] = 'lower center'
    title = '%s %d-%dE' % (varstr, lon1, lon2)
    lineplot(sectorcomp[varnm], plt.gca(), y1_label, y2_label,
             latmin=axlims[0], latmax=axlims[1], row=grp.row, nrow=nrow,
             legend_opts=legend_opts, y1_lims=y1_lims.get(varnm))
    plt.title(title, fontsize=11)

filestr = 'comp-onset_%s-%s-%s' % (onset_nm, savestr, vargroup)
atm.savefigs(savedir + filestr, 'pdf', merge=True)
plt.close('all')

# ======================================================================
# OLD STUFF
# ======================================================================
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
