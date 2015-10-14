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


datadir = '/home/jennifer/datastore/cmap/'
#datadir = '/home/jwalker/datastore/cmap/'
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
        precip = cmap
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
        Rsq = wlh[y]
        titlestr = 'CMAP %d' % years[0]

    plt.figure(figsize=(10,8))
    clev = np.arange(20, 60)
    plt.subplot(211)
    atm.contourf_latlon(onset, lat, lon, clev=clev, cmap=cmap,
                        axlims=axlims, symmetric=False)
    plt.title(titlestr + ' Onset Pentad')
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
        titlestr = 'CMAP %d' % yrmin

    latval, ilat0 = atm.find_closest(precip.lat, lat0)
    lonval, ilon0 = atm.find_closest(precip.lon, lon0)
    latstr = atm.latlon_labels([latval], 'lat', '%.1f', deg_symbol=False)[0]
    lonstr = atm.latlon_labels([lonval], 'lon', '%.1f', deg_symbol=False)[0]
    titlestr = '%s %s (%s, %s)' % (titlestr, loc_nm, latstr, lonstr)

    pcp = precip[:, ilat0, ilon0]
    pcp_sm, Rsq = atm.fourier_smooth(pcp, kmax)
    pcp_ann, Rsq_ann = atm.fourier_smooth(pcp, kann)
    i_onset, i_retreat, i_peak = onset_WLH_1D(pcp_sm, threshold)
    i_onset = int(i_onset)
    i_retreat = int(i_retreat)
    i_peak = int(i_peak)

    sz = 8
    fnt = 8
    loc = 'upper left'
    label_ann = 'kmax=%d, $R^2$=%.2f' % (kann, Rsq_ann)
    label = 'kmax=%d, $R^2$=%.2f' % (kmax, Rsq)
    plt.plot(pcp, color='grey', label='unsmoothed')
    plt.plot(pcp_ann, 'k--', label=label_ann)
    plt.plot(pcp_sm, 'k', linewidth=2, label=label)
    plt.plot(i_onset, pcp_sm[i_onset], 'ro', markersize=sz)
    plt.plot(i_peak, pcp_sm[i_peak], 'ro', markersize=sz)
    plt.plot(i_retreat, pcp_sm[i_retreat], 'ro', markersize=sz)
    plt.grid()
    plt.legend(loc=loc, fontsize=fnt)
    plt.title(titlestr, fontsize=fnt+2)
    plt.xlim(0, 72)

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
    if i < 3:
        plt.xticks(np.arange(0, 72, 10), [])
    else:
        plt.xlabel('Pentad')

# ----------------------------------------------------------------------
# Individual years
