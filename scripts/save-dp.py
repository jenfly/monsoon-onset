"""
Calculate d/dp from downloaded pressure level variable and save to files
"""
import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xarray as xray
import pandas as pd
import collections

import atmos as atm

# ----------------------------------------------------------------------
version = 'merra2'
datadir = atm.homedir() + 'datastore/%s/daily/' % version
savedir = datadir
years = np.arange(1980, 2010)
plevs = [1000,925,850,775,700,600,500,400,300,250,200,150,100,70,50,30,20]
pdim = 1
varnms = ['U', 'OMEGA']

def datafile(datadir, varnm, plev, year, version):
    latlonstr = '40E-120E_90S-90N'
    filenm = '%s_%s%d_%s_%d.nc' % (version, varnm, plev, latlonstr, year)
    filenm = datadir + filenm
    return filenm

def concat_plevs(datadir, year, varnm, plevs, pdim, version):
    pname = 'Height'
    for i, plev in enumerate(plevs):
        filenm = datafile(datadir, varnm, plev, year, version)
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
        var = concat_plevs(datadir, year, varnm, plevs, pdim, version)
        for plev in plevs:
            dvar = calc_dp(var, plev)
            filenm = datafile(savedir, 'D%sDP' % varnm, plev, year, version)
            print('Saving to ' + filenm)
            atm.save_nc(filenm, dvar)
