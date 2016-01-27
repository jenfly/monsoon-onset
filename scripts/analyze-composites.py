import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation
import collections
import pandas as pd
import atmos as atm
import precipdat
import merra
import indices
import utils
from utils import daily_rel2onset

# ----------------------------------------------------------------------
#onset_nm = 'HOWI'
onset_nm = 'CHP_MFC'
#onset_nm = 'CHP_PCP'

years = range(1979, 2015)
yearstr = '%d-%d Climatology' % (years[0], years[-1])

# CHP_MFC Early/Late Years
# years, yearstr = [2004, 1999, 1990, 2000, 2001], '5 Earliest Years'
# years, yearstr = [1983, 1992, 1997, 2014, 2012], '5 Latest Years'

# HOWI Early/Late Years
# years, yearstr = [2004, 2000, 1999, 2001, 1990], '5 Earliest Years'
# years, yearstr = [1983, 1979, 1997, 1992, 1995], '5 Latest Years'

datadir = atm.homedir() + 'datastore/merra/daily/'
savedir = 'figs/'
run_anim = False
run_eht = False

varnms = ['precip', 'U200', 'V200', 'rel_vort200', 'Ro200', 'abs_vort200',
          'H200', 'T200']
# varnms = ['U850', 'V850', 'rel_vort850', 'abs_vort850', 'H850', 'T850',
#           'QV850']
# varnms = ['T950', 'H950', 'QV950', 'V950', 'THETA950', 'THETA_E950', 'DSE950',
#           'MSE950', 'V*THETA950', 'V*THETA_E950', 'V*DSE950', 'V*MSE950']
# varnms = ['T950', 'H950', 'QV950', 'V950', 'THETA950', 'THETA_E950',
#           'V*THETA950', 'V*THETA_E950']

keys_remove = ['T950', 'H950', 'QV950', 'V950',  'DSE950',
                'MSE950', 'V*DSE950', 'V*MSE950']

datafiles = {}
datafiles['vimt'] = [datadir + 'merra_vimt_ps-300mb_%d.nc' % yr for yr in years]
datafiles['mfc'] = [datadir + 'merra_MFC_ps-300mb_%d.nc' % yr for yr in years]
datafiles['precip'] = [datadir + 'merra_precip_%d.nc' % yr for yr in years]

def yrlyfile(var, plev, year, subset=''):
    return 'merra_%s%d_40E-120E_60S-60N_%s%d.nc' % (var, plev, subset, year)

for plev in [200, 850]:
    files = [datadir + yrlyfile('uv', plev, yr) for yr in years]
    for key in ['U', 'V', 'Ro', 'rel_vort', 'abs_vort']:
        datafiles['%s%d' % (key, plev)] = files
    for key in ['T', 'H', 'QV']:
        key2 = '%s%d' % (key, plev)
        datafiles[key2] = [datadir + yrlyfile(key, plev, yr) for yr in years]

for plev in [950, 975]:
    for key in ['T', 'H','QV', 'V']:
        files = [datadir + yrlyfile(key, plev, yr) for yr in years]
        datafiles['%s%d' % (key, plev)] = files

ensofile = atm.homedir() + 'dynamics/calc/ENSO/enso_oni.csv'
enso_ssn = 'JJA'

remove_tricky = False
years_tricky = [2002, 2004, 2007, 2009, 2010]

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# Monsoon onset day and index timeseries

if onset_nm == 'HOWI':
    maxbreak = 10
    npts = 100
    ds = atm.combine_daily_years(['uq_int', 'vq_int'],datafiles['vimt'], years,
                                 yearname='year')
    index, _ = indices.onset_HOWI(ds['uq_int'], ds['vq_int'], npts, maxbreak=maxbreak)
    index.attrs['title'] = 'HOWI (N=%d)' % npts
elif onset_nm == 'CHP_MFC':
    mfc = atm.combine_daily_years('MFC', datafiles['mfc'], years, yearname='year')
    mfcbar = atm.mean_over_geobox(mfc, lat1, lat2, lon1, lon2)
    mfc_acc = np.cumsum(mfcbar, axis=1)
    index = indices.onset_changepoint(mfc_acc)
elif onset_nm == 'CHP_PCP':
    subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}
    precip = atm.combine_daily_years('PRECTOT', datafiles['precip'], years,
                                      yearname='year', subset_dict=subset_dict)
    precip = atm.precip_convert(precip, precip.attrs['units'], 'mm/day')
    precipbar = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)
    precip_acc = np.cumsum(precipbar, axis=1)
    index = indices.onset_changepoint(precip_acc)

# Array of onset days
onset = index['onset']

# Tile the climatology to each year
if 'tseries_clim' in index:
    tseries_clim = index['tseries_clim']
else:
    tseries_clim = index['tseries'].mean(dim='year')
vals = atm.biggify(tseries_clim.values, index['tseries'].values, tile=True)
_, _, coords, dims = atm.meta(index['tseries'])
tseries_clim = xray.DataArray(vals, name=tseries_clim.name, coords=coords,
                              dims=dims)

# Daily timeseries for each year
tseries = xray.Dataset()
tseries[onset_nm] = index['tseries']
tseries[onset_nm + '_clim'] = tseries_clim

# ----------------------------------------------------------------------
# ENSO
enso = pd.read_csv(ensofile, index_col=0)
enso = enso[enso_ssn].loc[years]
enso_sorted = enso.copy()
enso_sorted.sort(ascending=False)
enso = xray.DataArray(enso).rename({'Year' : 'year'})

nyrs = 5
print('El Nino Top %d' % nyrs)
print(enso_sorted[:nyrs])
print('La Nina Top %d' % nyrs)
print(enso_sorted[-1:-nyrs-1:-1])


# ----------------------------------------------------------------------
# Read daily data fields and align relative to onset day

#npre, npost = 30, 30
npre, npost = 90, 90
yearnm, daynm = 'year', 'day'

def var_type(varnm):
    keys = ['THETA', 'MSE', 'DSE', 'V*', 'abs_vort']
    test =  [varnm.startswith(key) for key in keys]
    if np.array(test).any():
        vtype = 'calc'
    else:
        vtype = 'basic'
    return vtype

def read_data(varnm, data, onset, npre, npost):
    daymin, daymax = onset.values.min() - npre, onset.values.max() + npost
    if varnm != 'precip':
        plev = int(varnm[-3:])
        varid = varnm[:-3]
    if varnm == 'precip':
        subset_dict = {'day' : (daymin, daymax),
                       'lon' : (40, 120),
                       'lat' : (-60, 60)}
        var = atm.combine_daily_years('PRECTOT', datafiles['precip'], years,
                                      yearname='year', subset_dict=subset_dict)
        var = atm.precip_convert(var, var.attrs['units'], 'mm/day')
    elif var_type(varnm) == 'calc':
        pres = atm.pres_convert(plev, 'hPa', 'Pa')
        Tnm = 'T%d' % plev
        Hnm = 'H%d' % plev
        QVnm = 'QV%d' % plev
        print('Computing ' + varid)
        if varid == 'THETA':
            var = atm.potential_temp(data[Tnm], pres)
        elif varid == 'THETA_E':
            var = atm.equiv_potential_temp(data[Tnm], pres, data[QVnm])
        elif varid == 'DSE':
            var = atm.dry_static_energy(data[Tnm], data[Hnm])
        elif varid == 'MSE':
            var = atm.moist_static_energy(data[Tnm], data[Hnm], data[QVnm])
        elif varid.startswith('V*'):
            varid2 = '%s%d' % (varid[2:], plev)
            var = data['V%d' % plev] * data[varid2]
            var.name = varid
        elif varid == 'abs_vort':
            rel_vort = data['rel_vort%d' % plev]
            lat = atm.get_coord(rel_vort, 'lat')
            f = atm.coriolis(lat)
            f = atm.biggify(f, rel_vort, tile=True)
            var = rel_vort + f
            var.name = varid
    else:
        var = atm.combine_daily_years(varid, datafiles[varnm], years,
                                      subset_dict={'Day' : (daymin, daymax)})
        var = var.rename({'Year' : 'year', 'Day' : 'day'})
        var = atm.squeeze(var)

    return var


data = collections.OrderedDict()
for varnm in varnms:
    print('Reading daily data for ' + varnm)
    var = read_data(varnm, data, onset, npre, npost)
    if var_type(varnm) == 'basic':
        print('Aligning data relative to onset day')
        data[varnm] = daily_rel2onset(var, onset, npre, npost, yearnm=yearnm,
                                      daynm=daynm)
    else:
        data[varnm] = var

# Remove data that I don't want to include in plots
keys = data.keys()
for key in keys_remove:
    if key in keys:
        data = atm.odict_delete(data, key)

# Fill Ro200 with NaNs near equator
varnm = 'Ro200'
if varnm in data:
    latbuf = 5
    lat = atm.get_coord(data[varnm], 'lat')
    latbig = atm.biggify(lat, data[varnm], tile=True)
    vals = data[varnm].values
    vals = np.where(abs(latbig)>latbuf, vals, np.nan)
    data[varnm].values = vals

# ----------------------------------------------------------------------
# Remove tricky years before calculating composites

if remove_tricky:
    print('Removing tricky years')
    yearstr = yearstr + '_excl_tricky'
    print(years_tricky)
    years = list(years)
    for year in years_tricky:
        years.remove(year)
    years = np.array(years)

    # Remove tricky years from all indices and data variables
    onset = atm.subset(onset, {yearnm : (years, None)})
    mfcbar = atm.subset(mfcbar, {yearnm : (years, None)})
    enso = atm.subset(enso, {yearnm : (years, None)})

    for nm in data.keys():
        print(nm)
        data[nm] = atm.subset(data[nm], {yearnm : (years, None)})

# ----------------------------------------------------------------------
# Sector mean data

lonname, latname = 'XDim', 'YDim'
sectordata = collections.OrderedDict()
for varnm in data:
    var = atm.subset(data[varnm], {lonname : (lon1, lon2)})
    sectordata[varnm] = var.mean(dim=lonname)

# ----------------------------------------------------------------------
# Latitude of maximum of each field in sector mean

sector_latmax = collections.OrderedDict()
for varnm in sectordata:
    var = sectordata[varnm]
    lat = var[latname].values
    coords={'year' : var['year'], 'dayrel': var['dayrel']}
    # Yearly
    latmax = lat[np.nanargmax(var, axis=2)]
    sector_latmax[varnm] = xray.DataArray(latmax, dims=['year', 'dayrel'],
                                          coords=coords)

    # Climatology
    latmax = lat[np.nanargmax(var.mean(dim='year'), axis=1)]
    key = varnm + '_CLIM'
    sector_latmax[key] = xray.DataArray(latmax, coords={'dayrel' : var['dayrel']})

# ----------------------------------------------------------------------
# Plotting params and utilities

axlims = (-60, 60, 40, 120)

def symm_colors(plotdata):
    if plotdata.min() * plotdata.max() > 0:
        symmetric = False
    else:
        symmetric = True
    return symmetric

def get_colormap(varnm):
    if varnm == 'precip':
        cmap = 'hot_r'
    else:
        cmap = 'RdBu_r'
    return cmap

def plot_colorbar(symmetric, orientation='vertical'):
    if symmetric:
        atm.colorbar_symm(orientation=orientation)
    else:
        plt.colorbar(orientation=orientation)

# ----------------------------------------------------------------------
# Latitude-time contour plot

def contourf_lat_time(lat, days, plotdata, title, cmap, onset_nm,
                      zero_line=False):
    vals = plotdata.values.T
    vals = np.ma.array(vals, mask=np.isnan(vals))
    ncont = 40
    symmetric = symm_colors(plotdata)
    cint = atm.cinterval(vals, n_pref=ncont, symmetric=symmetric)
    clev = atm.clevels(vals, cint, symmetric=symmetric)
    plt.contourf(days, lat, vals, clev, cmap=cmap)
    plot_colorbar(symmetric)
    if symmetric and zero_line:
        plt.contour(days, lat, vals, [0], colors='k')
    plt.grid(True)
    plt.ylabel('Latitude')
    plt.xlabel('Day Relative to %s Onset' % onset_nm)
    plt.title(title)
    xmin, xmax = plt.gca().get_xlim()
    if xmax > 60:
        plt.xticks(range(int(xmin), int(xmax) + 1, 30))

keys = sectordata.keys()
for varnm in keys:
    if 'year' in sectordata[varnm].dims:
        plotdata = sectordata[varnm].mean(dim='year')
    else:
        plotdata = sectordata[varnm]
    lat = plotdata[latname].values
    days = plotdata['dayrel'].values
    cmap = get_colormap(varnm)
    title = '%d-%dE ' %(lon1, lon2) + varnm + ' - ' + yearstr
    plt.figure(figsize=(12, 8))
    contourf_lat_time(lat, days, plotdata, title, cmap, onset_nm)
    # Add latitudes of maxima
    if varnm in ['THETA_E950']:
        latmax = sector_latmax[varnm].mean(dim='year')
        latmax_0 = latmax.sel(dayrel=0)
        plt.plot(days, latmax, 'k', linewidth=2, label='Latitude of Max')
        plt.legend(loc='lower left')
        ax = plt.gca()
        s = atm.latlon_labels(latmax_0, latlon='lat', fmt='%.1f')
        ax.annotate(s, xy=(0, latmax_0), xycoords='data',
                    xytext=(-50, 50), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->"))


atm.savefigs(savedir + 'sector_%d-%dE_onset_%s_' % (lon1, lon2, onset_nm), 'pdf')
plt.close('all')

# ----------------------------------------------------------------------
# Composite averages
#compdays = utils.comp_days_centered(5)
#compdays = utils.comp_days_centered(5, offset=3)
compdays = {'pre' : np.arange(-5, 0), 'post' : np.arange(15, 20)}
#compdays = {'pre' : np.array([-10]), 'post' : np.array([0])}

def plusminus(num):
    if num < 0:
        numstr = '%d' % num
    else:
        numstr = '+%d' % num
    return numstr

compnms = {}
for key in compdays:
    d1 = plusminus(compdays[key].min())
    d2 = plusminus(compdays[key].max())
    compnms[key] = 'D0%s:D0%s' % (d1, d2)


print('Computing composites of lat-lon and sector data')
comp = collections.OrderedDict()
sectorcomp = collections.OrderedDict()
for varnm in data:
    print varnm
    compdat = utils.composite(data[varnm], compdays, daynm='dayrel',
                              return_avg=True)
    compsec = utils.composite(sectordata[varnm], compdays, daynm='dayrel',
                              return_avg=True)
    comp[varnm] = {}
    sectorcomp[varnm] = {}
    for key in compdat:
        comp[varnm][key] = compdat[key]
        sectorcomp[varnm][key] = compsec[key]


# Plot lat-lon maps and sector means of pre/post onset composite averages
axlims = (-60, 60, 40, 120)
climits = {'precip' : (0, 20), 'U200' : (-50, 50), 'V200' : (-10, 10),
           'Ro200' : (-1, 1), 'rel_vort200' : (-4e-5, 4e-5),
           'abs_vort200' : (-2e-4, 2e-4), 'T200' : (213, 227),
           'H200' : (11.2e3, 12.6e3), 'U850' : (-20, 20), 'V850' : (-10, 10),
           'rel_vort850' : (-3e-5, 3e-5), 'abs_vort850' : (-1.5e-4, 1.5e-4),
           'H850' : (1100, 1600), 'T850' : (260, 305),
           'QV850' : (0, 0.015),
           'THETA975' : (260, 315), 'THETA_E975' : (260, 370),
           'DSE975' : (2.6e5, 3.2e5), 'MSE975' : (2.5e5, 3.5e5),
           'V*DSE975' : (-6e6,6e6), 'V*MSE975' : (-9e6, 9e6),
           'THETA950' : (260, 315), 'THETA_E950' : (260, 365),
           'DSE950' : (2.6e5, 3.2e5), 'MSE950' : (2.5e5, 3.5e5),
           'V*DSE950' : (-4.5e6,4.5e6), 'V*MSE950' : (-5e6, 5e6),
           'V*THETA950' : (-4500, 4500), 'V*THETA_E950' : (-4900, 4900)}

key1, key2 = 'pre', 'post'
for varnm in comp:
    plt.figure(figsize=(12, 10))
    plt.subplots_adjust(left=0.06, right=0.95)
    plt.suptitle('%s Composites Relative to %s Onset Day - %s' %
                 (varnm.upper(), onset_nm, yearstr))

    # Lat-lon maps of composites
    for i, key in enumerate([key1, key2]):
        dat = comp[varnm][key].mean(dim='year')
        plt.subplot(2, 3, i + 1)
        atm.pcolor_latlon(dat, axlims=axlims, cmap=get_colormap(varnm))
        plt.clim(climits[varnm][0], climits[varnm][1])
        plt.title(key.upper() + ' ' + compnms[key])

    # Lat-lon map of difference between composites
    plt.subplot(2, 3, 3)
    dat = comp[varnm][key2].mean(dim='year') - comp[varnm][key1].mean(dim='year')
    atm.pcolor_latlon(dat, axlims=axlims, cmap='RdBu_r')
    symmetric = symm_colors(dat)
    if symmetric:
        cmax = np.nanmax(abs(dat))
        plt.clim(-cmax, cmax)
    plt.title('Difference (%s-%s)' % (key2.upper(), key1.upper()))

    # Line plot of sector mean
    sector1 = sectorcomp[varnm][key1].mean(dim='year')
    sector2 = sectorcomp[varnm][key2].mean(dim='year')
    lat = atm.get_coord(sector1, 'lat')
    plt.subplot(2, 2, 3)
    plt.plot(lat, sector1, label=key1.upper())
    plt.plot(lat, sector2, label=key2.upper())
    plt.title('%d-%d E Composites' % (lon1, lon2))
    if varnm in ['precip', 'Ro_200', 'rel_vort200']:
        legend_loc = 'upper right'
    elif varnm in ['V850']:
        legend_loc = 'lower left'
    else:
        legend_loc = 'lower right'
    plt.legend(loc=legend_loc)
    plt.subplot(2, 2, 4)
    plt.plot(lat, sector2 - sector1)
    plt.title('%d-%d E Difference (%s-%s)' % (lon1, lon2, key2.upper(), key1.upper()))
    for i in [3, 4]:
        plt.subplot(2, 2, i)
        plt.xlabel('Latitude')
        plt.ylabel(varnm)
        plt.grid()

atm.savefigs(savedir + 'comp_clim_onset_%s_' % onset_nm, 'pdf')
plt.close('all')


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
