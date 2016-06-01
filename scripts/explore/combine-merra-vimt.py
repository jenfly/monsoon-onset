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

years = range(1979, 2015)
months = range(1, 13)
monthstr = ''
#months = [4, 5, 6, 7, 8, 9]
#monthstr='apr-sep_'

def datafile(datadir, year, mon):
    filn = datadir + 'merra_vimt_%d%02d.nc' % (year, mon)
    return filn

def savefile(datadir, varnm, year, monthstr, pmin):
    filn = datadir + 'merra_%s_ps-%.0fmb_%s%d.nc'
    filn = filn % (varnm, pmin/100, monthstr, year)
    return filn

# Read daily data from each year and month and concatenate together
for y, year in enumerate(years):
    files = [datafile(datadir, year, mon) for mon in months]
    ds = atm.load_concat(files, concat_dim='day')
    pmin = ds['uq_int'].attrs['pmin']
    filn = savefile(datadir, 'vimt', year, monthstr, pmin)
    print('Saving VIMT to ' + filn)
    ds.to_netcdf(filn)

    # Compute moisture flux convergence and save to files
    print('Calculating MFC')
    mfc = atm.moisture_flux_conv(ds['uq_int'], ds['vq_int'], already_int=True)
    mfc.attrs['long_name'] = mfc.name
    mfc.name = 'MFC'
    for key in ds['uq_int'].attrs:
        mfc.attrs[key] = ds['uq_int'].attrs[key]
    filn = savefile(datadir, 'MFC', year, monthstr, pmin)
    print('Saving MFC to ' + filn)
    atm.save_nc(filn, mfc)
