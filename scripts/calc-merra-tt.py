import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra

# ----------------------------------------------------------------------
# Daily 200-600mb vertical mean T cf. Goswami et al. 2006

def savefile(year, mon, pmin, pmax):
    savedir = atm.homedir() + 'datastore/merra/daily/'
    filn = savedir + 'merra_T%d-%d_%d%02d.nc' % (pmin/100, pmax/100, year, mon)
    print('Saving to ' + filn)
    return filn

lon1, lon2 = 40, 100
lat1, lat2 = -15, 35
varlist = ['T']
pmin, pmax = 200e2, 600e2
years = range(1979, 2015)
months = [4, 5, 6, 7, 8, 9]

g = atm.constants.g.values
nperday = 8

for year in years:
    for mon in months:
        dayvals = atm.season_days(atm.month_str(mon), atm.isleap(year))
        T = merra.read_daily(varlist, year, mon, subset1=('lon', lon1, lon2),
                              subset2=('lat', lat1, lat2))

        # Daily means of 3-hourly data
        T = atm.daily_from_subdaily(T, nperday, dayvals=dayvals)

        # Vertical integral
        T = atm.int_pres(T, pmin=pmin, pmax=pmax)
        T = T * g / (pmax - pmin)
        T.name='T'
        T.attrs['long_name'] = 'Vertical mean atmospheric temperature'
        T.attrs['pmin'] = pmin
        T.attrs['pmax'] = pmax

        # Save to file
        atm.save_nc(savefile(year, mon, pmin, pmax), T)
