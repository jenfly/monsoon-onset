import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import collections

import atmos as atm
import merra
import indices
import utils

# ----------------------------------------------------------------------
datadir = atm.homedir() + 'datastore/merra/daily/'
savedir = atm.homedir() + 'datastore/merra/analysis/'
years = np.arange(1979, 2015)
ndays = 5      # Rolling pentad
lon1, lon2 = 60, 100

def savefile(savedir, year, ndays, lon1, lon2):
    lonstr = atm.latlon_labels([lon1, lon2], 'lon', deg_symbol=False)
    lonstr = '-'.join(lonstr)
    filenm = 'merra_ubudget200_ndays%d_%s_%d.nc' % (ndays, lonstr, year)
    return savedir + filenm

for year in years:
    print(year)
    files = collections.OrderedDict()
    files['U'] = datadir + 'merra_uv200_40E-120E_60S-60N_%d.nc' % year
    files['V'] = datadir + 'merra_uv200_40E-120E_60S-60N_%d.nc' % year
    files['DUDP'] = datadir + 'merra_DUDP200_40E-120E_90S-90N_%d.nc' % year
    files['H'] = datadir + 'merra_H200_40E-120E_60S-60N_%d.nc' % year
    files['OMEGA'] = datadir + 'merra_OMEGA200_40E-120E_90S-90N_%d.nc' % year
    files['DOMEGADP'] = datadir + 'merra_DOMEGADP200_40E-120E_90S-90N_%d.nc' % year

    # Read data and calculate momentum budget
    ubudget, data = utils.calc_ubudget(files, ndays, lon1, lon2)
    filenm = savefile(savedir, year, ndays, lon1, lon2)
    print('Saving to ' + filenm)
    ubudget.to_netcdf(filenm)
