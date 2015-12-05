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
style = {onset_nm : 'k', onset_nm + '_clim' : 'k--', 'MFC' : 'b'}
onset_style = {onset_nm : 'k'}
d_onset = {onset_nm : onset.values}
indices.plot_tseries_together(tseries, onset=d_onset, data_style=style,
                              onset_style=onset_style, show_days=True)

# ----------------------------------------------------------------------
if remove_tricky:
    for year in years_tricky:
        years.remove(year)
years = np.array(years)
