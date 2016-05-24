import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')
sys.path.append('/home/jwalker/dynamics/python/monsoon-onset')

import xray
import numpy as np
import collections
import pandas as pd

import atmos as atm
import indices
import utils
from utils import get_data_rel, load_dailyrel

# ----------------------------------------------------------------------
version = 'merra2'
years = np.arange(1980, 2016)
onset_nm = 'CHP_MFC'
ind_nm, npre, npost = 'onset', 120, 200
#ind_nm, npre, npost = 'retreat', 200, 60

datadir = atm.homedir() + 'datastore/%s/daily/' % version
savedir = atm.homedir() + 'eady/datastore/%s/analysis/' % version
yearstr = '%d-%d' % (min(years), max(years))
indfile = savedir + version + '_index_%s_%s.nc' % (onset_nm, yearstr)

varnms = ['PRECTOT', 'U200', 'V200',  'T200', 'H200', 'OMEGA500',
          'U850', 'V850', 'T850', 'H850', 'QV850', 'TLML', 'QLML', 'PS',
          'EFLUX', 'EVAP', 'HFLUX', 'VFLXCPT', 'VFLXPHI', 'VFLXQV',
          'UFLXCPT', 'UFLXPHI', 'UFLXQV']
sector_varnms = ['U', 'V', 'OMEGA', 'T', 'H', 'QV']
keys_remove = ['H950', 'V950',  'DSE950', 'MSE950', 'V*DSE950', 'V*MSE950']

# Lat-lon box for MFC / precip onset calcs
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# List of data files

def yrlyfile(version, datadir, varnm, year, subset1='40E-120E_90S-90N',
             subset2=''):
    filenm = datadir + '%d/%s_%s_%s_%s%d.nc'
    filenm = filenm % (year, version, varnm, subset1, subset2, year)
    return filenm

def get_savefile(version, savedir, varnm, onset_nm, ind_nm, year):
    filenm = savedir + version + '_%s_dailyrel' % varnm
    if ind_nm == 'retreat':
        filenm = filenm + '_retreat'
    filenm = filenm + '_%s_%d.nc' % (onset_nm, year)
    return filenm

# Lat-lon data files
datafiles = {}
for nm in varnms:
    datafiles[nm] = [yrlyfile(version, datadir, nm, yr) for yr in years]

# Zonal and sector mean data
for lonlims in [(0, 360), (60, 100)]:
    lonstr = atm.latlon_str(lonlims[0], lonlims[1], 'lon')
    subset1 = 'sector_' + lonstr
    for nm in sector_varnms:
        key = nm + '_' + subset1
        datafiles[key] = [yrlyfile(version, datadir, nm, yr, subset1)
                          for yr in years]

# ----------------------------------------------------------------------
# Onset index for each year

with xray.open_dataset(indfile) as index:
    index.load()
onset = index['onset']
retreat = index['retreat']

# ----------------------------------------------------------------------
# Get daily data

def get_varid(varnm):
    if varnm.find('sector') >= 0:
        varid = varnm.split('_')[0]
    else:
        varid = varnm
    return varid


# Save daily data for each year to file
for y, year in enumerate(years):
    files = {key : [datafiles[key][y]] for key in datafiles}
    d_onset = int(onset[y].values)
    d_retreat = int(retreat[y].values)
    d0 = int(index[ind_nm][y].values)
    coords = {'year' : [year]}
    onset_var = xray.DataArray([d_onset], name='D_ONSET', coords=coords)
    retreat_var = xray.DataArray([d_retreat], name='D_RETREAT', coords=coords)
    for varnm in files:
        print('Reading daily data for ' + varnm)
        varid = get_varid(varnm)
        var = get_data_rel(varid, year, files.get(varnm), data, d0, npre, npost)
        var.attrs = atm.odict_delete(var.attrs, 'd_onset')
        var.name = varid
        savefile = get_savefile(version, savedir, varnm, onset_nm, ind_nm, year)
        print('Saving to ' + savefile)
        atm.save_nc(savefile, var, onset_var, retreat_var)

# ----------------------------------------------------------------------
# Compute climatologies and save
relfiles = {}
for key in datafiles:
    relfiles[key] = [get_savefile(version, savedir, key, onset_nm, ind_nm, yr)
                     % yr for yr in years]

for varnm in relfiles:
    varid = get_varid(varnm)
    var, onset, retreat = load_dailyrel(relfiles[varnm])
    ds = xray.Dataset()
    ds[varid], ds['D_ONSET'], ds['D_RETREAT'] = var, onset, retreat
    print('Computing climatological mean')
    ds = ds.mean(dim=yearnm)
    ds[varid].attrs = var.attrs
    ds[varid].attrs['years'] = years
    filn = relfiles[varnm][0].replace('%d' % years[0], yearstr)
    print('Saving to ' + filn)
    ds.to_netcdf(filn)
