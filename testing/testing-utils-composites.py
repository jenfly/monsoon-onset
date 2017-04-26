import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xarray as xray
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
datadir = atm.homedir() + 'datastore/merra/daily/'
onsetfile = datadir + 'merra_u850_40E-120E_60S-60N_apr-sep_1979-2014.nc'
datafile = datadir + 'merra_precip_40E-120E_60S-60N_days91-274_1979-2014.nc'

# Onset day calcs
with xray.open_dataset(onsetfile) as ds:
    u = ds['U'].load()
oci = indices.onset_OCI(u)
d_onset = oci['onset']

# Daily data fields
var = 'precip'
yearnm, daynm = 'year', 'day'
cmap = 'hot_r'
axlims = (-10, 50, 40, 120)
with xray.open_dataset(datafile) as ds:
    data = ds['PRECTOT'].load()

# Align daily data relative to onset day
ndays = 30
data = daily_rel2onset(data, d_onset, ndays, ndays, yearnm=yearnm, daynm=daynm)
years = data[yearnm].values
databar = data.mean(dim=yearnm)

# Days for pre/onset/post composites
compdays = comp_days_centered(5)
#compdays = comp_days_centered(1, 9)

# Calculate composites
comp = composite(data, compdays, daynm=daynm + 'rel')
compbar = collections.OrderedDict()
for key in comp:
    compbar[key] = comp[key].mean(dim=yearnm)

# Plot pre/onset/post composites - climatology
cmin, cmax = 0, 20
plt.figure(figsize=(12,10))
for i, key in enumerate(comp):
    plt.subplot(2, 2, i+1)
    atm.pcolor_latlon(compbar[key], axlims=axlims, cmap=cmap)
    plt.clim(cmin, cmax)
    plt.title(var + ' ' + key)

# Plot pre/onset/post composites - individual years
cmin, cmax = 0, 30
nrow, ncol = 3, 4
figsize = (14, 10)
for key in comp.keys():
    suptitle = var + ' ' + key
    for y, year in enumerate(years):
        if y % (nrow * ncol) == 0:
            fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
            plt.subplots_adjust(left=0.08, right=0.95, wspace=0, hspace=0.2)
            plt.suptitle(suptitle)
            yplot = 1
        else:
            yplot += 1

        plt.subplot(nrow, ncol, yplot)
        atm.pcolor_latlon(comp[key][y], axlims=axlims, cmap=cmap)
        plt.clim(cmin, cmax)
        plt.title(year)

# ----------------------------------------------------------------------
# Animation of daily data relative to onset

nframes = 2*ndays + 1
fps = 4

def animate(i):
    plt.clf()
    m, _ = atm.pcolor_latlon(animdata[i], axlims=axlims, cmap=cmap)
    plt.clim(cmin, cmax)
    day = animdata[daynm + 'rel'].values[i]
    plt.title('%s %s RelDay %d' % (var, yearstr, day))
    return m

# Climatology
cmin, cmax = 0, 20
yearstr = '%d-%d' % (years[0], years[-1])
animdata = databar
savefile = 'mp4/' + var + '_climatology.mp4'
fig = plt.figure()
anim = animation.FuncAnimation(fig, animate, frames=nframes)
anim.save(savefile, writer='mencoder', fps=fps)

# Individual years
cmin, cmax = 0, 60
for y, year in enumerate(years):
    print(year)
    animdata = data[y]
    yearstr = '%d' % year
    savefile = 'mp4/%s_%s.mp4' % (var, yearstr)
    plt.close('all')
    fig = plt.figure()
    anim = animation.FuncAnimation(fig, animate, frames=nframes)
    anim.save(savefile, writer='mencoder', fps=fps)

# # ----------------------------------------------------------------------
# # Define monsoon onset index
#
# datadir = atm.homedir() + 'datastore/cmap/'
# cmap_file = datadir + 'cmap.precip.pentad.mean.nc'
#
# # Read CMAP data and average over box
# lon1, lon2 = 60, 100
# lat1, lat2 = 10, 30
# titlestr = 'CMAP %d-%dE, %d-%dN ' % (lon1, lon2, lat1, lat2)
# precip = precipdat.read_cmap(cmap_file)
# precipbar = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)
# nyears, npentad = precipbar.shape
# years = precipbar.year.values
#
# # Smooth with truncated Fourier series and calculate onset index
# # with Wang & LinHo method
# kmax = 12
# threshold = 5.0
# pcp_sm, Rsq = atm.fourier_smooth(precipbar, kmax)
# p_onset = np.zeros(nyears)
# for y, year in enumerate(years):
#     p_onset[y], _, _ = indices.onset_WLH_1D(pcp_sm[y])
#
# # Convert onset index from pentad to day of year
# d_onset = np.zeros(nyears)
# for y in range(nyears):
#     d_onset[y] = atm.pentad_to_jday(p_onset[y], pmin=0)
#
# # ----------------------------------------------------------------------
# # Read daily mean MERRA data
#
# datadir = atm.homedir() + 'datastore/merra/daily/'
# filestr = datadir + 'merra_uv200_40E-120E_60S-60N_%d.nc'
# files = []
# for yr in years:
#     files.append(filestr % yr)
#
# ds = atm.combine_daily_years(['U', 'V', 'Ro'], files, years)
# u = ds['U']
# v = ds['V']
# Ro = ds['Ro']



# ndays = 10
# varlist = ['U', 'V', 'Ro']
# comp = {}
# for y, yr in enumerate(years):
#     filn = filestr % yr
#     print('Loading ' + filn)
#     with xray.open_dataset(filn) as ds:
#         for varnm in varlist:
#             print(varnm)
#             var = ds[varnm]
#             var.coords['Year'] = yr
#             comp_in = composite_premidpost(var, d_onset[y], ndays)
#             if y == 0:
#                 comp[varnm] = comp_in
#             else:
#                 for key in comp_in:
#                     comp[varnm] = xray.concat((comp[varnm][key], comp_in[key]),
#                                               dim='Year')
# print('Done!')
