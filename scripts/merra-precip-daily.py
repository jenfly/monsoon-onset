import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import collections
import pandas as pd
import atmos as atm
import precipdat
import merra

datadir = atm.homedir() + 'datastore/merra/daily/'
years = np.arange(1979, 2015)

# Day range to extract (inclusive)
# Apr 1 - Oct 1 non-leap year | Mar 31 - Sep 30 leap year
daymin = 91
daymax = 274

# lat, lon range to extract
lon1, lon2 = 40, 120
lat1, lat2 = -60, 60
lons = atm.latlon_labels([lon1, lon2], 'lon', deg_symbol=False)
lats = atm.latlon_labels([lat1, lat2], 'lat', deg_symbol=False)
latlon = '%s-%s_%s-%s' % (lons[0], lons[1], lats[0], lats[1])

savefile = datadir + ('merra_precip_%s_days%d-%d_%d-%d.nc' % 
                      (latlon, daymin, daymax, years.min(), years.max()))

for y, year in enumerate(years):
    datafile = datadir + 'merra_precip_%d.nc' % year
    print('Loading ' + datafile)
    with xray.open_dataset(datafile) as ds:
        precip1 = atm.subset(ds['PRECTOT'], 'day', daymin, daymax)
        precip1 = atm.subset(precip1, 'lat', lat1, lat2, 'lon', lon1, lon2)
        precip1 = precip1.load()
        precip1.coords['year'] = year
    if y == 0:
        precip = precip1
    else:
        precip = xray.concat((precip, precip1), dim='year')

print('Converting to mm/day')
precip.values = atm.precip_convert(precip, precip.units, 'mm/day')
precip.attrs['units'] = 'mm/day'

print('Saving to ' + savefile)
atm.save_nc(savefile, precip)
     
