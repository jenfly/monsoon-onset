import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
from indices import onset_changepoint

# ----------------------------------------------------------------------
# Onset changepoint
years = range(1979, 2015)
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
varnms = ['MFC', 'precip']
datadir = atm.homedir() + 'datastore/merra/daily/'
isave = True

data = {}
data_acc = {}
chp = {}
for varnm in varnms:
    if varnm == 'MFC':
        varid, filestr = 'MFC', 'MFC_ps-300mb'
    elif varnm == 'precip':
        varid, filestr = 'PRECTOT', 'precip'

    files = [datadir + 'merra_%s_%d.nc' % (filestr, year) for year in years]
    data[varnm] = atm.combine_daily_years(varid, files, years, yearname='year',
                                          subset1=('lat', lat1, lat2),
                                          subset2=('lon', lon1, lon2))
    data[varnm] = atm.mean_over_geobox(data[varnm], lat1, lat2, lon1, lon2)
    data[varnm] = atm.precip_convert(data[varnm], data[varnm].attrs['units'],
                                    'mm/day')

    # Accumulated precip or MFC
    data_acc[varnm] = np.cumsum(data[varnm], axis=1)

    # Compute onset and retreat days
    chp[varnm] = onset_changepoint(data_acc[varnm])

def plotyear(chp, y, xlims, ylims):
    days = chp['day'].values
    d_onset = chp['onset'][y]
    d_retreat = chp['retreat'][y]
    plt.plot(days, chp['tseries'][y])
    plt.plot(days, chp['tseries_fit_onset'][y])
    plt.plot(days, chp['tseries_fit_retreat'][y])
    plt.plot([d_onset, d_onset], ylims, 'k')
    plt.plot([d_retreat, d_retreat], ylims, 'k')
    plt.xlim(xlims[0], xlims[1])
    plt.ylim(ylims[0], ylims[1])
    plt.title(years[y])
    atm.text('Onset: %d\nRetreat: %d' % (d_onset, d_retreat), (0.03, 0.83))
    plt.grid()

nrow, ncol = 3, 4
figsize = (14, 10)
xlims = 0, 366
ylims = {'MFC' : (-350, 400), 'precip' : (0, 1700)}

for varnm in varnms:
    suptitle = 'Onset/Retreat from Accumulated %s Changepoints' % varnm.upper()
    for y, year in enumerate(years):
        if y % (nrow * ncol) == 0:
            fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
            plt.subplots_adjust(left=0.08, right=0.95, wspace=0.0, hspace=0.2)
            plt.suptitle(suptitle)
            yplot = 1
        else:
            yplot += 1

        plt.subplot(nrow, ncol, yplot)
        plotyear(chp[varnm], y, xlims, ylims[varnm])
        row, col = atm.subplot_index(nrow, ncol, yplot)
        if row == nrow:
            plt.xlabel('Day')
        else:
            plt.gca().set_xticklabels('')
        if col > 1:
            plt.gca().set_yticklabels('')
    if isave:
        atm.savefigs('onset_changepoint_' + varnm.upper(), 'pdf')
        plt.close('all')

df = pd.DataFrame()
for key in ['onset', 'retreat']:
    for varnm in varnms:
        df['%s_%s' % (key, varnm.upper())] = chp[varnm][key].to_series()
atm.scatter_matrix(df, incl_p=True, incl_line=True, annotation_pos=(0.05, 0.7))

plt.figure(figsize=(8, 10))
for i, key in enumerate(['onset', 'retreat']):
    plt.subplot(2, 1, i + 1)
    for varnm in varnms:
        plt.plot(years, df['%s_%s' % (key, varnm.upper())], label=varnm.upper())
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Day of Year')
    plt.title(key.capitalize() + ' from Changepoint Method')
    plt.grid()
    plt.autoscale(tight=True)

if isave:
    atm.savefigs('onset_changepoint_compare_', 'pdf')
