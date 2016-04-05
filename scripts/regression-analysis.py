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
import utils

mpl.rcParams['font.size'] = 10

# ----------------------------------------------------------------------


onset_nm = 'CHP_MFC'
years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/analysis/'

varnms = ['U200', 'V200', 'T200', 'H200', 'U850', 'V850', 'T850', 'H850',
          'THETA950', 'THETA_E950', 'QV950', 'T950', 'HFLUX', 'EFLUX',
          'EVAP', 'precip']
regdays = [-60, -30, 0, 30, 60]
seasons = ['JJAS', 'SSN']
lon1, lon2 = 60, 100
nroll = 5

def get_filenames(datadir, varnms, onset_nm, years, lon1, lon2, nroll):
    lonstr = atm.latlon_str(lon1, lon2, 'lon')
    yearstr = '_%d-%d.nc' % (min(years), max(years))
    filestr = datadir + 'merra_%s' 
    if nroll is not None:
        filestr = filestr + '_nroll%d' % nroll
    filestr = filestr + '_reg_%s_onset_' + onset_nm + yearstr
    filenames = {}
    for varnm in varnms:
        filenames[varnm] = {'latlon' : filestr % (varnm, 'latlon'),
                            'sector' : filestr % (varnm, lonstr)}        
    return filenames

datafiles = get_filenames(datadir, varnms, onset_nm, years, lon1, lon2, nroll)
reg = {}

for varnm in varnms:
    for key in ['latlon', 'sector']:
        with xray.open_dataset(datafiles[varnm][key]) as ds:
            reg[varnm + '_' + key] = ds.load()



filenm = datadir + 'merra_U200_nroll7_reg_60E-100E_onset_CHP_MFC_1979-2014.nc'
with xray.open_dataset(filenm) as ds:
    ds.load()

lat = atm.get_coord(ds, 'lat')
days = atm.get_coord(ds, 'dayrel')
var = ds['r']
pts_mask = (ds['p'] >= 0.05)
xsample, ysample = 4, 2

xname, yname = 'dayrel', 'YDim'

plt.figure()
utils.contourf_lat_time(lat, days, var)
atm.stipple_pts(pts_mask, xname, yname, xsample, ysample)
