import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import atmos as atm
import precipdat
from indices import onset_WLH, onset_WLH_1D

datadir = atm.homedir() + 'datastore/cmap/'
cmap_file = datadir + 'cmap.precip.pentad.mean.nc'

# Smoothing parameters and threshold for onset criteria
kmax = 12
kann = 4
threshold = 5.0

def all_WLH(cmap_file, yearmin=None, yearmax=None, climatology=False,
            kmax=12, threshold=5.0):
    precip = precipdat.read_cmap(cmap_file, yearmin, yearmax)
    lat = atm.get_coord(precip, 'lat')
    lon = atm.get_coord(precip, 'lon')
    years = precip.year
    if climatology:
        precip = precip.mean(dim='year')
        axis = 0
    else:
        axis = 1
    wlh = onset_WLH(precip, axis)
    wlh['precip'] = precip
    wlh['lat'] = lat
    wlh['lon'] = lon
    wlh['years'] = years
    wlh['climatology'] = climatology
    return wlh

def plot_all_WLH(wlh, y=0, axlims=(0,50,50,180), cmap='jet'):
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
        titlestr = 'CMAP %d' % years[0]

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
    clev_date = [501, 511, 521, 601, 611, 621, 701, 711, 721]
    clev_label = {}
    for d in clev_date:
        clev_label[d] = '%02d-%02d' % (d//100, d%100)
    
    # Plot maps
    plt.figure(figsize=(10,8))
    clev = np.arange(20, 60)
    plt.subplot(211)
    atm.contourf_latlon(onset, lat, lon, clev=clev, cmap=cmap,
                        axlims=axlims, symmetric=False)
    _, cs = atm.contour_latlon(onset_date, lat, lon, clev=clev_date,
                               axlims=axlims)
    plt.clabel(cs, clev_date, fmt=clev_label, manual=True)
    plt.title(titlestr + ' Onset Pentad & Day')
    
    plt.subplot(212)
    clev = np.arange(0, 1.1, 0.025)
    atm.contourf_latlon(Rsq, lat, lon, clev=clev, cmap=cmap, axlims=axlims,
                        symmetric=False)
    plt.title('$R^2$ for kmax = %d' % kmax)

def single_WLH(cmap_file, yrmin, yrmax, lat0, lon0, loc_nm, kmax, kann):
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
    i_onset, i_retreat, i_peak = onset_WLH_1D(pcp_sm, threshold)
    i_onset = int(i_onset)
    i_retreat = int(i_retreat)
    i_peak = int(i_peak)

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
    plt.plot(i_onset+1, pcp_sm[i_onset], 'ro', markersize=sz)
    plt.plot(i_peak+1, pcp_sm[i_peak], 'ro', markersize=sz)
    plt.plot(i_retreat+1, pcp_sm[i_retreat], 'ro', markersize=sz)
    plt.grid()
    plt.legend(loc=loc, fontsize=fnt)
    plt.title(titlestr, fontsize=fnt+2)
    plt.xlim(0, 74)

    return pcp, pcp_sm, pcp_ann, Rsq, Rsq_ann, i_onset, i_retreat, i_peak


# ----------------------------------------------------------------------
# Compare 1979-1997 climatology with Wang and LinHo
yrmin, yrmax = 1979, 1997
climatology = True
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
wlh = all_WLH(cmap_file, yrmin, yrmax, climatology, kmax, threshold)

y = 35 # Index of year to plot
years = wlh['years']
plot_all_WLH(wlh)

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
