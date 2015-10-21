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


# ----------------------------------------------------------------------
# Average over box

# Threshold for onset criteria
threshold = 5.0

# Read data and average over box
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
titlestr = 'CMAP %d-%dE, %d-%dN ' % (lon1, lon2, lat1, lat2)
precip = precipdat.read_cmap(cmap_file)
precipbar = atm.mean_over_geobox(precip, lat1, lat2, lon1, lon2)
nyears, npentad = precipbar.shape
years = precipbar.year.values
pentads = precipbar.pentad
onset = {}

# Smooth with truncated Fourier series
kmax = 12
kann = 4
pcp_sm, Rsq = atm.fourier_smooth(precipbar, kmax)
pcp_ann, Rsq_ann = atm.fourier_smooth(precipbar, kann)
label_sm, label_ann = [], []
for y, year in enumerate(years):
    label_sm.append('kmax=%d, $R^2$=%.2f' % (kmax, Rsq[y]))
    label_ann.append('kmax=%d, $R^2$=%.2f' % (kann, Rsq_ann[y]))

i_onset, i_retreat, i_peak = get_onset(years, pcp_sm)

plot_years(years, precipbar, pcp_sm, label_sm, i_onset, i_peak, i_retreat,
            titlestr, pcp_ann, label_ann)
onset['fourier'] = i_onset + 1

# Smooth with rolling mean
nroll = 4
pcp_sm = np.zeros(precipbar.shape)
label_sm = []
for y in range(len(years)):
    pcp_sm[y] = pd.rolling_mean(precipbar[y].values, nroll)
    label_sm.append('%d-pentad rolling' % nroll)

i_onset, i_retreat, i_peak = get_onset(years, pcp_sm)

plot_years(years, precipbar, pcp_sm, label_sm, i_onset, i_peak, i_retreat,
            titlestr)
onset['rolling'] = i_onset + 1




plt.figure()
plt.scatter(years, i_onset)
