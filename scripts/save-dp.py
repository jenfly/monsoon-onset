"""
Calculate d/dp from downloaded pressure level variable and save to files
"""
import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import collections

import atmos as atm

# ----------------------------------------------------------------------
datadir = atm.homedir() + 'datastore/merra/daily/'
savedir = atm.homedir() + 'datastore/merra/analysis/'
#years = np.arange(1979, 2015)
years = np.arange(1979, 2006)
plevs = [1000,925,850,775,700,600,500,400,300,250,200,150,100,70,50,30,20]
pdim = 1
varnms = ['U', 'OMEGA']

def datafile(datadir, varnm, plev, year):
    latlonstr = '40E-120E_90S-90N'
    filenm = datadir + 'merra_%s%d_%s_%d.nc' % (varnm, plev, latlonstr, year)
    return filenm

def concat_plevs(datadir, year, varnm, plevs, pdim):
    pname = 'Height'
    for i, plev in enumerate(plevs):
        filenm = datafile(datadir, varnm, plev, year)
        print('Reading ' + filenm)
        with xray.open_dataset(filenm) as ds:
            var_in = ds[varnm].load()
            var_in = atm.expand_dims(var_in, pname, plev, axis=1)
        if i == 0:
            var = var_in
        else:
            var = xray.concat([var, var_in], dim=pname)
    return var

def calc_dp(var, plev):
    """Extract subset of pressure levels and calculate d/dp."""
    plevs = atm.get_coord(var, 'plev')
    pname = atm.get_coord(var, 'plev', 'name')
    pdim = atm.get_coord(var, 'plev', 'dim')
    ind = (list(plevs)).index(plev)
    i1 = max(0, ind - 1)
    i2 = min(len(plevs) - 1, ind + 1) + 1
    psub = plevs[i1:i2]
    varsub = var.sel(**{pname : psub})
    pres = atm.pres_convert(psub, 'hPa', 'Pa')
    atm.disptime()
    print('Computing d/dp for pressure level %d' % plev)
    dvar = atm.gradient(varsub, pres, axis=pdim)
    dvar = dvar.sel(**{pname : plev})
    dvar.name = 'D%sDP' % var.name
    atm.disptime()
    return dvar

# Compute d/dp and save
for year in years:
    for varnm in varnms:
        var = concat_plevs(datadir, year, varnm, plevs, pdim)
        for plev in plevs:
            dvar = calc_dp(var, plev)
            filenm = datafile(savedir, 'D%sDP' % varnm, plev, year)
            print('Saving to ' + filenm)
            atm.save_nc(filenm, dvar)
