import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xarray as xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import atmos as atm
import precipdat
from indices import onset_WLH, onset_WLH_1D

datadir = atm.homedir() + 'datastore/cmap/'
cmap_file = datadir + 'cmap.precip.pentad.mean.nc'

def all_WLH(cmap_file, yearmin=None, yearmax=None, climatology=False,
            kmax=12, threshold=5.0, onset_min=20):
    precip = precipdat.read_cmap(cmap_file, yearmin, yearmax)
    lat = atm.get_coord(precip, 'lat')
    lon = atm.get_coord(precip, 'lon')
    years = precip.year
    if climatology:
        precip = precip.mean(dim='year')
        axis = 0
    else:
        axis = 1
    wlh = onset_WLH(precip, axis, kmax, threshold, onset_min)
    wlh['precip'] = precip
    wlh['lat'] = lat
    wlh['lon'] = lon
    wlh['years'] = years
    wlh['climatology'] = climatology
    return wlh

def plot_all_WLH(wlh, y=0, axlims=(0,50,50,180), cmap='jet', clines=True):
    onset = wlh['onset']
    Rsq = wlh['Rsq']
    lat = wlh['lat']
    lon = wlh['lon']
    kmax = wlh['smoothing_kmax']
    years=wlh['years']
    if wlh['climatology']:
        titlestr = 'CMAP %d-%d Climatology' % (years.min(), years.max())
    else:
        onset = onset[y]
        Rsq = Rsq[y]
        titlestr = 'CMAP %d' % years[y]

    # Note:  add 1 to pentad indices to index from 1-73 for comparison
    # with Wang & LinHo
    onset = onset + 1

    # Calculate onset dates from pentads
    nlat, nlon = onset.shape
    onset_date = np.nan * np.ones((nlat, nlon))
    for i in range(nlat):
        for j in range(nlon):
            jday = atm.pentad_to_jday(onset[i, j], pmin=1)
            mon, day = atm.jday_to_mmdd(jday)
            onset_date[i, j] = 100*mon + day
    # -- Smooth with cubic spline
    # lat_i = np.arange(-89.5, 90, 0.5)
    # lon_i = np.arange(0, 360, 0.5)
    lat_i = np.arange(-90, 90, 4.)
    lon_i = np.arange(0, 360, 4.)
    onset_date = atm.interp_latlon(onset_date, lat_i, lon_i, lat, lon, order=3)
    clev_date = [501, 511, 521, 601, 611, 621, 701, 711]
    clev_label = {}
    for d in clev_date:
        clev_label[d] = '%02d-%02d' % (d//100, d%100)

    # Plot maps
    manual = False
    plt.figure(figsize=(14,10))
    cmin, cmax = 20, 55
    plt.subplot(211)
    _, pc = atm.pcolor_latlon(onset, lat, lon, cmap=cmap, axlims=axlims)
    pc.set_clim(cmin, cmax)
    if clines:
        _, cs = atm.contour_latlon(onset_date, lat_i, lon_i, clev=clev_date,
                                   axlims=axlims)
        plt.clabel(cs, clev_date, fmt=clev_label, manual=manual)
        plt.title(titlestr + ' Onset Pentad & Day')
    else:
        plt.title(titlestr + ' Onset Pentad')

    plt.subplot(212)
    clev = np.arange(0, 1.1, 0.025)
    _, pc = atm.pcolor_latlon(Rsq, lat, lon, cmap=cmap, axlims=axlims)
    pc.set_clim(0., 1.)
    plt.title('$R^2$ for kmax = %d' % kmax)


def plot_single_WLH(pcp, pcp_sm, pcp_ann, Rsq, Rsq_ann, i_onset, i_retreat,
                    i_peak, kmax, kann, titlestr):
    # Note:  add 1 to pentad indices to index from 1-73 for comparison
    # with Wang & LinHo
    sz = 8
    fnt = 8
    loc = 'upper left'
    label_ann = 'kmax=%d, $R^2$=%.2f' % (kann, Rsq_ann)
    label = 'kmax=%d, $R^2$=%.2f' % (kmax, Rsq)
    x = np.arange(1, 74)
    plt.plot(x, pcp, color='grey', label='unsmoothed')
    plt.plot(x, pcp_ann, 'k--', label=label_ann)
    plt.plot(x, pcp_sm, 'k', linewidth=2, label=label)
    for ind in [i_onset, i_peak, i_retreat]:
        if not np.isnan(ind):
            ind = int(ind)
            plt.plot(ind + 1, pcp_sm[ind], 'ro', markersize=sz)
    plt.grid()
    plt.legend(loc=loc, fontsize=fnt)
    plt.title(titlestr, fontsize=fnt+2)
    plt.xlim(0, 74)

def single_WLH(cmap_file, yrmin, yrmax, lat0, lon0, loc_nm, kmax, kann,
               onset_min=20):
    # Single grid point and single year/climatology
    precip = precipdat.read_cmap(cmap_file, yrmin, yrmax)
    if yrmax > yrmin:
        precip = precip.mean(axis=0)
        titlestr = 'CMAP %d-%d' % (yrmin, yrmax)
    else:
        precip = precip[0]
        titlestr = 'CMAP %d' % yrmin

    latval, ilat0 = atm.find_closest(precip.lat, lat0)
    lonval, ilon0 = atm.find_closest(precip.lon, lon0)
    d = 1.25
    latstr = atm.latlon_labels(latval+d, 'lat', '%.1f', deg_symbol=False)
    lonstr = atm.latlon_labels(lonval+d, 'lon', '%.1f', deg_symbol=False)
    titlestr = '%s %s (%s, %s)' % (titlestr, loc_nm, latstr, lonstr)

    pcp = precip[:, ilat0, ilon0]
    pcp_sm, Rsq = atm.fourier_smooth(pcp, kmax)
    pcp_ann, Rsq_ann = atm.fourier_smooth(pcp, kann)
    i_onset, i_retreat, i_peak = onset_WLH_1D(pcp_sm, threshold, onset_min)

    plot_single_WLH(pcp, pcp_sm, pcp_ann, Rsq, Rsq_ann, i_onset, i_retreat, i_peak, kmax, kann, titlestr)

    return pcp, pcp_sm, pcp_ann, Rsq, Rsq_ann, i_onset, i_retreat, i_peak


# ----------------------------------------------------------------------
# Compare climatology with Wang and LinHo

yrmin, yrmax = 1979, 2014
#yrmin, yrmax = 1979, 1997
climatology = True

# Smoothing parameters and threshold for onset criteria
kmax = 12
kann = 4
threshold = 5.0

wlh = all_WLH(cmap_file, yrmin, yrmax, climatology, kmax, threshold)
plot_all_WLH(wlh)

d = 1.25
pts =[('Arabian Sea', 12.5-d, 70-d),
      ('Bay of Bengal', 12.5-d, 90-d),
      ('South China Sea', 12.5-d, 115-d),
      ('Western North Pacific', 15-d, 140-d)]
plt.figure(figsize=(8,10))
for i, pt in enumerate(pts):
    plt.subplot(4, 1, i+1)
    loc_nm, lat0, lon0 = pt
    single_WLH(cmap_file, yrmin, yrmax, lat0, lon0, loc_nm, kmax, kann)
    plt.ylim(-6, 24)
    plt.yticks(np.arange(-6, 25, 6))
    if i < 3:
        plt.xticks(np.arange(0, 74, 4), [])
    else:
        plt.xticks(np.arange(0, 74, 4))
        plt.xlabel('Pentad')

# ----------------------------------------------------------------------
# Individual years

yrmin, yrmax = None, None
climatology = False

# Smoothing parameters and threshold for onset criteria
kmax = 12
kann = 4
threshold = 5.0

wlh = all_WLH(cmap_file, yrmin, yrmax, climatology, kmax, threshold)

y = 35 # Index of year to plot
years = wlh['years']
plot_all_WLH(wlh, y, clines=False)

ylim1, ylim2 = -10, 50
yticks = np.arange(-10, 51, 10)
d = 1.25
pts =[('Arabian Sea', 12.5-d, 70-d),
      ('Bay of Bengal', 12.5-d, 90-d),
      ('South China Sea', 12.5-d, 115-d),
      ('Western North Pacific', 15-d, 140-d)]
plt.figure(figsize=(8,10))
for i, pt in enumerate(pts):
    plt.subplot(4, 1, i+1)
    loc_nm, lat0, lon0 = pt
    single_WLH(cmap_file, years[y], years[y], lat0, lon0, loc_nm, kmax, kann)
    plt.ylim(ylim1, ylim2)
    plt.yticks(yticks)
    if i < 3:
        plt.xticks(np.arange(0, 74, 4), [])
    else:
        plt.xticks(np.arange(0, 74, 4))
        plt.xlabel('Pentad')

# ----------------------------------------------------------------------
# # Average over box
# lon1, lon2 = 60, 100
# lat1, lat2 = 10, 30
# precip = precipdat.read_cmap(cmap_file)
#
# precipbar = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)
# nyears, npentad = precipbar.shape
# years = precipbar.year.values
# pentads = precipbar.pentad
#
# pcp_sm, Rsq = atm.fourier_smooth(precipbar, kmax)
# pcp_ann, Rsq_ann = atm.fourier_smooth(precipbar, kann)
# i_onset = np.zeros(nyears)
# i_retreat = np.zeros(nyears)
# i_peak = np.zeros(nyears)
# for y, year in enumerate(years):
#     i_onset[y], i_retreat[y], i_peak[y] = onset_WLH_1D(pcp_sm[y])
#
# iplot = 0
# for y in range(len(years)):
#     if y % 4 == 0:
#         plt.figure(figsize=(8, 10))
#     iplot = y % 4 + 1
#     plt.subplot(4, 1, iplot)
#     plot_single_WLH(precipbar[y], pcp_sm[y], pcp_ann[y], Rsq[y], Rsq_ann[y],
#         i_onset[y], i_retreat[y], i_peak[y], kmax, kann, years[y])
#     if iplot < 4:
#         plt.xticks(np.arange(0, 74, 4), [])
#     else:
#         plt.xticks(np.arange(0, 74, 4))
#         plt.xlabel('Pentad')

# ----------------------------------------------------------------------
# Average over box

def get_onset(years, pcp_sm):
    i_onset = np.zeros(nyears)
    i_retreat = np.zeros(nyears)
    i_peak = np.zeros(nyears)
    for y, year in enumerate(years):
        i_onset[y], i_retreat[y], i_peak[y] = onset_WLH_1D(pcp_sm[y])
    return i_onset, i_retreat, i_peak


def plot_single_WLH(pcp, pcp_sm, label_sm, i_onset, i_retreat, i_peak,
                    titlestr, pcp_ann=None, label_ann=None,):
    # Note:  add 1 to pentad indices to index from 1-73 for comparison
    # with Wang & LinHo
    sz = 8
    fnt = 8
    loc = 'upper left'
    xlim1, xlim2 = 0, 74
    ylim1, ylim2 = -2, 14

    x = np.arange(1, 74)
    plt.plot(x, pcp, color='grey', label='unsmoothed')
    if pcp_ann is not None:
        plt.plot(x, pcp_ann, 'k--', label=label_ann)
    plt.plot(x, pcp_sm, 'k', linewidth=2, label=label_sm)
    for ind in [i_onset, i_peak, i_retreat]:
        if not np.isnan(ind):
            ind = int(ind)
            plt.plot(ind + 1, pcp_sm[ind], 'ro', markersize=sz)
    plt.grid()
    plt.legend(loc=loc, fontsize=fnt)
    plt.title(titlestr, fontsize=fnt+2)
    plt.xlim(xlim1, xlim2)
    plt.ylim(ylim1, ylim2)


def plot_years(years, pcp, pcp_sm, label_sm, i_onset, i_peak, i_retreat,
               titlestr, pcp_ann=None, label_ann=None):
    iplot = 0
    for y in range(len(years)):
        if y % 4 == 0:
            plt.figure(figsize=(8, 10))
        iplot = y % 4 + 1
        plt.subplot(4, 1, iplot)
        if pcp_ann is not None:
            plot_single_WLH(pcp[y], pcp_sm[y], label_sm[y], i_onset[y],
                            i_retreat[y], i_peak[y], titlestr + str(years[y]),
                            pcp_ann[y], label_ann[y])
        else:
            plot_single_WLH(pcp[y], pcp_sm[y], label_sm[y], i_onset[y],
                            i_retreat[y], i_peak[y], titlestr + str(years[y]))
        if iplot < 4:
            plt.xticks(np.arange(0, 74, 4), [])
        else:
            plt.xticks(np.arange(0, 74, 4))
            plt.xlabel('Pentad')


# Read data and average over box
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
titlestr = 'CMAP %d-%dE, %d-%dN ' % (lon1, lon2, lat1, lat2)
precip = precipdat.read_cmap(cmap_file)
precipbar = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)
nyears, npentad = precipbar.shape
years = precipbar.year.values
nyears = len(years)
pentads = precipbar.pentad
onset = {}

# Threshold for onset criteria
threshold = 5.0

# Smooth with truncated Fourier series
kmax = 12
kann = 4
pcp_sm, Rsq = atm.fourier_smooth(precipbar, kmax)
pcp_ann, Rsq_ann = atm.fourier_smooth(precipbar, kann)
label_sm, label_ann = [], []
for y in range(nyears):
    label_sm.append('kmax=%d, $R^2$=%.2f' % (kmax, Rsq[y]))
    label_ann.append('kmax=%d, $R^2$=%.2f' % (kann, Rsq_ann[y]))

i_onset, i_retreat, i_peak = get_onset(years, pcp_sm)

plot_years(years, precipbar, pcp_sm, label_sm, i_onset, i_peak, i_retreat,
            titlestr, pcp_ann, label_ann)
onset['fourier'] = i_onset

# Smooth with rolling mean
nroll = 3
pcp_sm = np.zeros(precipbar.shape)
label_sm = []
for y in range(nyears):
    pcp_sm[y] = pd.rolling_mean(precipbar[y].values, nroll, center=True)
    label_sm.append('%d-pentad rolling' % nroll)

i_onset, i_retreat, i_peak = get_onset(years, pcp_sm)

plot_years(years, precipbar, pcp_sm, label_sm, i_onset, i_peak, i_retreat,
            titlestr)
onset['rolling'] = i_onset

# Compare the smoothing methods
plt.figure(figsize=(8,10))
plt.subplot(311)
plt.scatter(years, onset['fourier'])
plt.title('Fourier smoothing kmax=%d' % kmax)
plt.xlabel('Year')
plt.ylabel('Onset Index')
plt.subplot(312)
plt.scatter(years, onset['rolling'])
plt.title('Smoothing with %d-pentad rolling mean' % nroll)
plt.xlabel('Year')
plt.ylabel('Onset Index')
plt.subplot(313)
plt.scatter(onset['fourier'], onset['rolling'])
plt.xlabel('Onset from Fourier smoothing')
plt.ylabel('Onset from rolling mean smoothing')

# ----------------------------------------------------------------------
# Composites
# p_onset = onset['fourier']
#
# ncomp = 10
# inds = slice_premidpost(p_onset, ncomp)
#
# comp = {}
# for y, yr in enumerate(years):
#     pre = precip[y, inds['pre'][y]].mean(axis=0)
#     ons = precip[y, inds['mid'][y]].mean(axis=0)
#     post = precip[y, inds['post'][y]].mean(axis=0)
#     if y == 0:
#         comp['pre-onset'] = pre
#         comp['onset'] = ons
#         comp['post-onset'] = post
#     else:
#         comp['pre-onset'] = xray.concat((comp['pre-onset'], pre), dim='year')
#         comp['onset'] = xray.concat((comp['onset'], ons), dim='year')
#         comp['post-onset'] = xray.concat((comp['post-onset'], post),
#                                           dim='year')
#
# cmap = 'hot_r'
# clim1, clim2 = 0, 20
# iplot = 1
# plt.figure(figsize=(7,10))
# for key in ['pre-onset', 'onset', 'post-onset']:
#     plt.subplot(3, 1, iplot)
#     _, pc = atm.pcolor_latlon(comp[key].mean(axis=0), cmap=cmap)
#     pc.set_clim(clim1, clim2)
#     plt.title(key)
#     iplot += 1
