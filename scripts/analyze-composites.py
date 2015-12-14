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
from utils import daily_rel2onset, comp_days_centered, composite

# ----------------------------------------------------------------------
years = range(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/daily/'
savedir = 'mp4/'
onsetfile = datadir + 'merra_vimt_ps-300mb_apr-sep_1979-2014.nc'
precipfile = datadir + 'merra_precip_40E-120E_60S-60N_days91-274_1979-2014.nc'
uvstr = 'merra_uv%d_40E-120E_60S-60N_%d.nc'
uvfiles = {}
for plev in [200, 850]:
    uvfiles[plev] = [datadir + uvstr % (plev, yr) for yr in years]

ensofile = atm.homedir() + 'dynamics/calc/ENSO/enso_oni.csv'
enso_ssn = 'JJA'

remove_tricky = False
years_tricky = [2002, 2004, 2007, 2009, 2010]

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# Monsoon onset day and index timeseries
onset_nm = 'HOWI'
maxbreak = 10
npts = 100
with xray.open_dataset(onsetfile) as ds:
    uq_int = ds['uq_int'].load()
    vq_int = ds['vq_int'].load()
    howi, _ = indices.onset_HOWI(uq_int, vq_int, npts, maxbreak=maxbreak)
    howi.attrs['title'] = 'HOWI (N=%d)' % npts

# Array of onset days
onset = howi['onset']

# Tile the climatology to each year
tseries_clim = howi['tseries_clim']
vals = atm.biggify(tseries_clim.values, howi['tseries'].values, tile=True)
_, _, coords, dims = atm.meta(howi['tseries'])
tseries_clim = xray.DataArray(vals, name=tseries_clim.name, coords=coords,
                              dims=dims)

# Daily timeseries for each year
tseries = xray.Dataset()
tseries[onset_nm] = howi['tseries']
tseries[onset_nm + '_clim'] = tseries_clim

# ----------------------------------------------------------------------
# MFC over SASM region
print('Calculating MFC')
mfc = atm.moisture_flux_conv(ds['uq_int'], ds['vq_int'], already_int=True)
mfcbar = atm.mean_over_geobox(mfc, lat1, lat2, lon1, lon2)

nroll = 7
mfcbar = atm.rolling_mean(mfcbar, nroll, axis=-1, center=True)
tseries['MFC'] = mfcbar


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
# Daily timeseries plots together (day of year)
style = {onset_nm : 'k', onset_nm + '_clim' : 'k--', 'MFC' : 'b'}
onset_style = {onset_nm : 'k'}
d_onset = {onset_nm : onset.values}
indices.plot_tseries_together(tseries, onset=d_onset, data_style=style,
                              onset_style=onset_style, show_days=True)

# ----------------------------------------------------------------------
# Daily timeseries composites relative to onset day
keys = [onset_nm, 'MFC']
npre, npost = 30, 90
tseries_rel = xray.Dataset()
for key in keys:
    tseries_rel[key] = daily_rel2onset(tseries[key], onset, npre, npost,
                                       yearnm='year', daynm='day')
dayrel = tseries_rel['dayrel'].values

offset, factor = {}, {}
for key in keys:
    offset[key] = -np.nanmean(tseries[key].values.ravel())
    factor[key] = np.nanstd(tseries[key].values.ravel())


def plot_tseries(dayrel, ind, std, clr, key, xlabel, ylabel):
    plt.plot(dayrel, ind, clr, label=key)
    plt.fill_between(dayrel, ind-std, ind+std, color=clr, alpha=0.2)
    plt.legend(loc='lower right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.autoscale(tight=True)

clrs = {onset_nm : 'b', 'MFC' : 'g'}
plt.figure(figsize=(8, 10))
for i, key in enumerate(keys):
    ind = tseries_rel[key].mean(dim='year')
    std = tseries_rel[key].std(dim='year')

    # Individual timeseries
    plt.subplot(3, 1, i + 1)
    plot_tseries(dayrel, ind, std, clrs[key], key, '', 'Timeseries')
    if i == 0:
        plt.title('1979-2014 Climatological Composites')

    # Standardized timeseries together
    ind = (ind + offset[key]) / factor[key]
    std = std / factor[key]
    plt.subplot(3, 1, 3)
    xlabel = 'Day of year relative to ' + onset_nm + ' onset day'
    ylabel = 'Standardized Timeseries'
    plot_tseries(dayrel, ind, std, clrs[key], key, xlabel, ylabel)


# ----------------------------------------------------------------------
# Read daily data fields and align relative to onset day

npre, npost = 30, 30
yearnm, daynm = 'year', 'day'

def read_data(varnm):
    daymin, daymax = 91, 274
    if varnm == 'precip':
        with xray.open_dataset(precipfile) as ds:
            var = ds['PRECTOT'].load()
    elif varnm in ['U200', 'V200', 'Ro200', 'rel_vort200',
                   'U850', 'V850', 'Ro850', 'rel_vort850']:
        plev = int(varnm[-3:])
        varid = varnm[:-3]
        var = atm.combine_daily_years(varid, uvfiles[plev], years,
                                      subset1=('Day', daymin, daymax))
        var = var.rename({'Year' : 'year', 'Day' : 'day'})
        var = atm.squeeze(var)
        
    return var


varnms = ['precip', 'U200', 'V200', 'rel_vort200', 'Ro200', 'U850', 'V850']
data = {}
for varnm in varnms:
    print('Reading daily data for ' + varnm)
    var = read_data(varnm)
    print('Aligning data relative to onset day')
    data[varnm] = daily_rel2onset(var, onset, npre, npost, yearnm=yearnm, 
                                  daynm=daynm)        

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
# Sector mean data

lonname, latname = 'XDim', 'YDim'
sectordata = {}
for varnm in data.keys():
    var = atm.subset(data[varnm], lonname, lon1, lon2)
    sectordata[varnm] = var.mean(dim=lonname)
    
# ----------------------------------------------------------------------
# Remove tricky years before calculating composites

if remove_tricky:
    print('Removing tricky years')
    print(years_tricky)
    years = list(years)
    for year in years_tricky:
        years.remove(year)
    years = np.array(years)

    # Remove tricky years from all indices and data variables
    onset = atm.subset(onset, yearnm, years)
    mfcbar = atm.subset(mfcbar, yearnm, years)
    enso = atm.subset(enso, yearnm, years)
 
    for nm in data.keys():
        print(nm)
        data[nm] = atm.subset(data[nm], yearnm, years)
    
 
# ----------------------------------------------------------------------
# Animation of daily data relative to onset

nframes = npre + npost + 1
fps = 4
axlims = (-60, 60, 40, 120)

climits = {'precip' : (0, 20),
           'U200' : (-50, 50),
           'V200' : (-10, 10),
           'Ro200' : (-1, 1),
           'rel_vort200' : (-4e-5, 4e-5),
           'U850' : (-20, 20),
           'V850' : (-10, 10)}

def get_colormap(varnm):
    if varnm == 'precip':
        cmap = 'hot_r'
    else:
        cmap = 'RdBu_r'
    return cmap
    
def animate(i):
    plt.clf()
    m, _ = atm.pcolor_latlon(animdata[i], axlims=axlims, cmap=cmap)
    plt.clim(cmin, cmax)
    day = animdata[daynm + 'rel'].values[i]
    plt.title('%s %s RelDay %d' % (varnm, yearstr, day))
    return m

yearstr = '%d-%d' % (years[0], years[-1])
if remove_tricky:
    yearstr = yearstr + '_excl_tricky'
        
for varnm in data.keys():
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
           'Ro200' : (-2.5, 2.8),
           'U850' : (-10, 18),
           'V850' : (-8.5, 3.5)}

def animate2(i):
    plt.clf()
    plt.plot(animdata[latname], animdata[i])
    plt.ylim(ylim1, ylim2)
    plt.grid(True)
    day = animdata[daynm + 'rel'].values[i]
    plt.title('%d-%dE %s %s RelDay %d' % (lon1, lon2, varnm, yearstr, day))

for varnm in sectordata.keys():
    savefile = savedir + 'sector_%d-%dE_%s_%s.mp4' % (lon1, lon2, varnm, yearstr)
    animdata = sectordata[varnm].mean(dim='year')
    ylim1, ylim2 = ylimits[varnm]
    fig = plt.figure()
    anim = animation.FuncAnimation(fig, animate2, frames=nframes)
    print('Saving to ' + savefile)
    anim.save(savefile, writer='mencoder', fps=fps)
    plt.close()

# Latitude-time contour plot

def pcolor_lat_time(lat, days, plotdata, title, cmap):
    # Use a masked array so that pcolormesh displays NaNs properly
    vals = plotdata.values
    vals = np.ma.array(vals, mask=np.isnan(vals))
    plt.pcolormesh(lat, days, vals, cmap=cmap)
    plt.colorbar(orientation='vertical')
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.xlabel('Latitude')
    plt.ylabel('RelDay')
    plt.title(title)
    
for varnm in sorted(sectordata.keys()):
    plotdata = sectordata[varnm].mean(dim='year')
    lat = plotdata[latname].values
    days = plotdata['dayrel'].values
    cmap = get_colormap(varnm)
    title = '%d-%dE ' %(lon1, lon2) + varnm + ' ' + yearstr
    plt.figure(figsize=(10, 10))
    pcolor_lat_time(lat, days, plotdata, title, cmap)

atm.savefigs(savedir + 'sector_%d-%dE_' % (lon1, lon2), 'pdf')
plt.close('all')

# ----------------------------------------------------------------------
