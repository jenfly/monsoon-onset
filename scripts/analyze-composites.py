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
datadir = atm.homedir() + 'datastore/merra/daily/'
onsetfile = datadir + 'merra_vimt_ps-300mb_apr-sep_1979-2014.nc'
ensofile = atm.homedir() + 'dynamics/calc/ENSO/enso_oni.csv'
enso_ssn = 'JJA'

remove_tricky = False
years = range(1979, 2015)
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


def plot_tseries(dayrel, ind, std, clr, key, xlim1, xlim2, xlabel, ylabel):
    plt.plot(dayrel, ind, clr, label=key)
    plt.fill_between(dayrel, ind-std, ind+std, color=clr, alpha=0.2)
    plt.legend(loc='lower right')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.xlim(xlim1, xlim2)

clrs = {onset_nm : 'b', 'MFC' : 'g'}
plt.figure(figsize=(8, 10))
for i, key in enumerate(keys):
    ind = tseries_rel[key].mean(dim='year')
    std = tseries_rel[key].std(dim='year')
    
    # Individual timeseries
    plt.subplot(3, 1, i + 1)
    plot_tseries(dayrel, ind, std, clrs[key], key, -npre, npost, '', 'Timeseries')
    if i == 0:
        plt.title('1979-2014 Climatological Composites')
    
    # Standardized timeseries together
    ind = (ind + offset[key]) / factor[key]
    std = std / factor[key]
    plt.subplot(3, 1, 3)
    plot_tseries(dayrel, ind, std, clrs[key], key, -npre, npost, 
                 'Day of year relative to onset day', 'Standardized Timeseries')
    

# ----------------------------------------------------------------------
# ENSO
enso = pd.read_csv(ensofile, index_col=0)
enso = enso[enso_ssn].loc[years]
enso_sorted = enso.copy()
enso_sorted.sort(ascending=False)

nyrs = 5
print('El Nino Top %d' % nyrs)
print(enso_sorted[:nyrs])
print('La Nina Top %d' % nyrs)
print(enso_sorted[-1:-nyrs-1:-1])



# ----------------------------------------------------------------------
if remove_tricky:
    for year in years_tricky:
        years.remove(year)
years = np.array(years)
