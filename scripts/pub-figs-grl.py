import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import collections
import pandas as pd

import atmos as atm
import indices
import utils

mpl.rcParams['font.size'] = 11

# ----------------------------------------------------------------------
version = 'merra2'
years = np.arange(1980, 2016)
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
onset_nm = 'CHP_MFC'
onset_nms = ['CHP_MFC', 'MOK', 'HOWI', 'OCI']
pts_nm = 'CHP_CMAP'
pcp_nm = 'PRECTOT'
varnms = ['PRECTOT', 'U200', 'V200', 'U850', 'V850']
lat_extract = {'U200' : 0, 'V200' : 15, 'U850' : 15, 'V850' : 15}
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
nroll = 7 # n-day rolling averages for smoothing daily timeseries
npre, npost = 120, 200

yearstr = '%d-%d.nc' % (min(years), max(years))
filestr = datadir + version + '_index_%s_' + yearstr
indfiles = {nm : filestr % nm for nm in ['CHP_MFC', 'HOWI', 'OCI']}
indfiles['MOK'] = atm.homedir() + 'dynamics/python/monsoon-onset/data/MOK.dat'
filestr2 = datadir + version + '_%s_dailyrel_' + onset_nm + '_' + yearstr
datafiles = {nm : filestr2 % nm for nm in varnms}
datafiles['CMAP'] = datadir + 'cmap_dailyrel_' + onset_nm + '_1980-2014.nc'
ptsfile = datadir + version + '_index_pts_%s_' % pts_nm
if pts_nm == 'CHP_CMAP':
    ptsfile = ptsfile + '1980-2014.nc'
else:
    ptsfile = ptsfile + yearstr

# ----------------------------------------------------------------------
# Read data

# Large-scale onset/retreat indices
index_all = {}
for nm in indfiles:
    print('Loading ' + indfiles[nm])
    if nm == 'MOK':
        mok = indices.onset_MOK(indfiles['MOK'], yearsub=years)
        index_all['MOK'] =  xray.Dataset({'onset' : mok})
    else:
        with xray.open_dataset(indfiles[nm]) as ds:
            index_all[nm] = ds.load()

index = index_all[onset_nm]
index['length'] = index['retreat'] - index['onset']

onset_all = pd.DataFrame()
for nm in index_all:
    onset_all[nm] = index_all[nm]['onset'].to_series()

# Onset/retreat at grid points
print('Loading ' + ptsfile)
with xray.open_dataset(ptsfile) as index_pts:
    index_pts.load()

# Dailyrel climatology
keys_dict = {'PRECTOT' : 'PRECTOT', 'CMAP' : 'precip', 'U200' : 'U',
             'U850' : 'U', 'V200' : 'V', 'V850' : 'V'}
data = {}
for nm in datafiles:
    print('Loading ' + datafiles[nm])
    with xray.open_dataset(datafiles[nm]) as ds:
        if 'year' in ds.dims:
            ds = ds.mean(dim='year')
        data[nm] = ds[keys_dict[nm]].load()

# ENSO indices


# ----------------------------------------------------------------------
# Daily timeseries

tseries = xray.Dataset()
tseries['MFC'] = utils.daily_rel2onset(index_all['CHP_MFC']['daily_ts'],
                                       index['onset'], npre, npost)
tseries['MFC_ACC'] = utils.daily_rel2onset(index_all['CHP_MFC']['tseries'],
                                           index['onset'], npre, npost)
for nm in ['CMAP', 'PRECTOT']:
    tseries[nm] = atm.mean_over_geobox(data[nm], lat1, lat2, lon1, lon2)

# Extract variables at specified latitudes
for nm, lat0 in lat_extract.iteritems():
    var = atm.dim_mean(data[nm], 'lon', lon1, lon2)
    lat = atm.get_coord(var, 'lat')
    lat0_str = atm.latlon_labels(lat0, 'lat', deg_symbol=False)
    key = nm + '_' + lat0_str
    lat_closest, _ = atm.find_closest(lat, lat0)
    print '%s %.2f %.2f' % (nm, lat0, lat_closest)
    tseries[key] = atm.subset(var, {'lat' : (lat_closest, None)}, squeeze=True)

# Compute climatology and smooth with rolling mean
if 'year' in tseries.dims:
    tseries = tseries.mean(dim='year')
if nroll is not None:
    for nm in tseries.data_vars:
        tseries[nm] = atm.rolling_mean(tseries[nm], nroll, center=True)

# ----------------------------------------------------------------------
# Plotting functions

def yrly_index(onset_all, legend=True):
    """Plot onset day vs. year for different onset definitions."""

    corr = onset_all.corr()[onset_nm]
    print(corr)

    labels = {nm : nm for nm in onset_all.columns}
    for nm in onset_all.columns:
        if nm != onset_nm:
            labels[nm] = labels[nm] + ' %.2f' % corr[nm]

    styles = {'CHP_MFC' : {'color' : 'k', 'linewidth' : 2},
              'OCI' : {'color' : 'r'}, 'HOWI' : {'color' : 'b'},
              'MOK' : {'color' : 'g'}}
    xticks = np.arange(1980, 2016, 5)
    xticklabels = [1980, '', 1990, '', 2000, '', 2010, '']

    for nm in onset_all.columns:
        plt.plot(years, onset_all[nm], label=labels[nm], **styles[nm])
    if legend:
        plt.legend(fontsize=9, loc='upper left', ncol=2)
    plt.grid()
    plt.xlim(min(years) - 1, max(years) + 1)
    plt.xticks(xticks, xticklabels)
    plt.xlabel('Year')
    plt.ylabel('Day of Year')
    plt.title('Onset')


# ----------------------------------------------------------------------

# Plot daily tseries
xlims = (-npre, npost)
xticks = range(-npre, npost + 1, 30)
x0 = [0, index['length'].mean(dim='year')]
keypairs = [(['MFC', 'CMAP'], ['MFC_ACC']),
            (['U850_15N'], ['V850_15N']),
            (['U200_0N'],['V200_15N'])]
opts = [('upper left', 'mm/day', 'mm'),
        ('upper left', 'm/s', 'm/s'),
        ('lower left', 'm/s', 'm/s')]
y2_opts={'color' : 'r', 'alpha' : 0.6}
nrow, ncol = 2, 2
fig_kw = {'figsize' : (11, 7)}
gridspec_kw = {'left' : 0.06, 'right' : 0.93, 'bottom' : 0.06, 'top' : 0.95,
               'wspace' : 0.35, 'hspace' : 0.2}
styles = ['k', 'k--', 'g', 'm']
legend_kw={'fontsize' : 11, 'handlelength' : 2.5}
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, gridspec_kw=gridspec_kw)
for pair, opt in zip(keypairs, opts):
    grp.next()
    keys1, keys2 = pair
    legend_kw['loc'] = opt[0]
    y1_label = opt[1]
    y2_label = opt[2]
    data1 = tseries[keys1]
    if keys2 is not None:
        data2 = tseries[keys2]
    else:
        data2 = None
    data1_styles = {nm : style for (nm, style) in zip(keys1, styles)}
    utils.plotyy(data1, data2, xname='dayrel', data1_styles=data1_styles,
                 y2_opts=y2_opts, xlims=xlims, xticks=xticks, xlabel='Rel Day',
                 y1_label=y1_label, y2_label=y2_label, legend=True,
                 legend_kw=legend_kw, x0_axvlines=x0)

# Plot yearly tseries
grp.next()
yrly_index(onset_all, legend=True)

# ----------------------------------------------------------------------
# Table of summary stats on onset/retreat/length

nms = ['Mean', 'Std', 'Max', 'Min']
for nm in ['onset', 'retreat', 'length']:
    ind = index[onset_nm][nm].values
    series = pd.Series([ind.mean(), ind.std(), ind.max(), ind.min()], index=nms)
    if nm == 'onset':
        stats = series.to_frame(name=nm.capitalize())
    else:
        stats[nm.capitalize()] = series
stats = stats.T

def daystr(day):
    day = round(day)
    mm, dd = atm.jday_to_mmdd(day)
    mon = atm.month_str(mm)
    return '%.0f (%s-%.0f)' % (day, mon, dd)

for nm1 in stats.columns:
    for nm2 in stats.index:
        if nm1 != 'Std' and nm2 != 'Length':
            stats[nm1].loc[nm2] = daystr(stats[nm1].loc[nm2])
        else:
            stats[nm1].loc[nm2] = '%.0f' % stats[nm1].loc[nm2]

print(stats.to_latex())

# ----------------------------------------------------------------------
# Table of correlations between detrended indices
# onset-retreat-length-ENSO


# ----------------------------------------------------------------------
