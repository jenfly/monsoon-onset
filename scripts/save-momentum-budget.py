import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')
sys.path.append('/home/jwalker/dynamics/python/monsoon-onset')

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
version = 'merra2'
datadir = atm.homedir() + 'datastore/%s/daily/' % version
savedir = atm.homedir() + 'datastore/%s/analysis/' % version
years = np.arange(1980, 2016)
plevs = [1000,925,850,775,700,600,500,400,300,250,200,150,100,70,50,30,20]
dp, ana = True, True
ndays = 5      # Rolling pentad
lon1, lon2 = 60, 100


def datafiles(version, datadir, year, plev, dp=True, ana=True):
    latlonstr = '40E-120E_90S-90N'
    filestr = datadir + '/%d/' % year + version  + '_%s%d_%s_%d.nc'
    nms = ['U', 'V', 'H', 'OMEGA']
    if dp:
        nms = nms + ['DUDP', 'DOMEGADP']
    if ana:
        nms = nms + ['DUDTANA']
    files = collections.OrderedDict()
    for nm in nms:
        files[nm] = filestr % (nm, plev, latlonstr, year)
    return files


def savefile(version, savedir, year, ndays, lon1, lon2, plev):
    lonstr =  atm.latlon_str(lon1, lon2, 'lon')
    filenm = savedir + version + '_ubudget%d_ndays%d_%s_%d.nc'
    filenm = filenm % (plev, ndays, lonstr, year)
    return filenm

for year in years:
    print(year)
    for plev in plevs:
        print(plev)
        files = datafiles(version, datadir, year, plev, dp, ana)

        # Read data and calculate momentum budget
        ubudget, data = utils.calc_ubudget(files, ndays, lon1, lon2, plev)
        filenm = savefile(version, savedir, year, ndays, lon1, lon2, plev)
        print('Saving to ' + filenm)
        ubudget.to_netcdf(filenm)
