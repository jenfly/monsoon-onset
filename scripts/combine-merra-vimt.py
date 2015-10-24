import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray

import atmos as atm
import merra

# ----------------------------------------------------------------------
# Daily vertically integrated moisture transport cf. Fasullo and Webster 2003
# Combine daily data from individual years, months

datadir = atm.homedir() + 'datastore/merra/daily/'

def datafile(datadir, year, mon):
    filn = datadir + 'merra_vimt_%d%02d.nc' % (year, mon)
    return filn

def savefile(datadir, years, months, pmin):
    mon1 = atm.month_str(months[0]).lower()
    mon2 = atm.month_str(months[-1]).lower()
    yr1 = years[0]
    yr2 = years[-1]
    filn = datadir + 'merra_vimt_ps-%.0fmb_%s-%s_%d-%d.nc'
    filn = filn % (pmin/100, mon1, mon2, yr1, yr2)
    return filn

years = range(1979, 2015)
months = [5, 6, 7, 8, 9]

# Read daily data from each year and month and concatenate together
for y, year in enumerate(years):
    for m, mon in enumerate(months):
        filn = datafile(datadir, year, mon)
        print('Loading ' + filn)
        with xray.open_dataset(filn) as ds:
            ds.load()
            if m == 0:
                dsyr = ds
            else:
                dsyr = xray.concat((dsyr, ds), dim='day')
    dsyr.coords['year'] = year
    if atm.isleap(year):
        # Standardize to non-leap year day numbers
        dsyr['day'] = dsyr['day'] - 1
    if y == 0:
        ds_all = dsyr
    else:
        ds_all = xray.concat((ds_all, dsyr), dim='year')

# Save to file
filn = savefile(datadir, years, months, ds_all['uq_int'].attrs['pmin'])
print('Saving to ' + filn)
ds_all.to_netcdf(filn)
