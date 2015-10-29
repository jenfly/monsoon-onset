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
# Daily vertically integrated moisture transport cf. Fasullo and Webster 2003

def savefile(year, mon):
    savedir = atm.homedir() + 'datastore/merra/daily/'
    filn = savedir + 'merra_vimt_%d%02d.nc' % (year, mon)
    print('Saving to ' + filn)
    return filn

lon1, lon2 = 40, 100
lat1, lat2 = -20, 30
varlist = ['u', 'v', 'q']
pmin = 300e2
years = range(1979, 2015)
#months = [5, 6, 7, 8]
months = [4]

for year in years:
    for mon in months:
        dayvals = atm.season_days(atm.month_str(mon), atm.isleap(year))
        ds = merra.read_daily(varlist, year, mon, subset1=('lon', lon1, lon2),
                              subset2=('lat', lat1, lat2))
        uq = ds['U'] * ds['QV']
        vq = ds['V'] * ds['QV']

        # Daily means of 3-hourly data
        nperday = 8
        uq = atm.daily_from_subdaily(uq, nperday, dayvals=dayvals)
        vq = atm.daily_from_subdaily(vq, nperday, dayvals=dayvals)

        # Vertical integral
        uq_int = atm.int_pres(uq, pmin=pmin)
        vq_int = atm.int_pres(vq, pmin=pmin)
        uq_int.name='uq_int'
        uq_int.attrs['pmin'] = pmin
        vq_int.name='vq_int'
        vq_int.attrs['pmin'] = pmin

        # Save to file
        atm.save_nc(savefile(year, mon), uq_int, vq_int)
