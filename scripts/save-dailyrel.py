import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import animation
import collections
import pandas as pd

import atmos as atm
import precipdat
import merra
import indices
import utils
from utils import get_data_rel, load_dailyrel

# ----------------------------------------------------------------------
onset_nm = 'CHP_MFC'

#years = np.arange(1979, 2015)
years = np.arange(1979, 1995)

datadir = atm.homedir() + 'datastore/merra/daily/'
savedir = atm.homedir() + 'datastore/merra/analysis/'

# Number of days before and after onset to include
npre, npost = 120, 200

varnms = ['U_sector_0E-360E', 'U_sector_60E-100E', 'V_sector_0E-360E',
          'V_sector_60E-100E']

# varnms = ['VFLXPHI', 'VFLXCPT', 'VFLXQV', 'VFLXMSE']
# varnms = ['precip', 'U200', 'V200', 'rel_vort200', 'Ro200',
#           'abs_vort200', 'H200', 'T200',
#           'U850', 'V850', 'H850', 'T850', 'QV850',
#           'T950', 'H950', 'QV950', 'V950', 'THETA950', 'THETA_E950',
#           'V*THETA_E950', 'EFLUX', 'HFLUX', 'EVAP', 'VFLXPHI', 'VFLXCPT',
#           'VFLXQV', 'VFLXMSE']

keys_remove = ['H950', 'V950',  'DSE950', 'MSE950', 'V*DSE950', 'V*MSE950']

# Lat-lon box for MFC / precip onset calcs
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# List of data files

def yrlyfile(var, plev, year, subset1='40E-120E_60S-60N', subset2=''):
    if plev is None:
        varid = var
    else:
        varid = '%s%d' % (var, plev)
    return 'merra_%s_%s_%s%d.nc' % (varid, subset1, subset2, year)

def get_savefile(savedir, varnm, onset_nm, year):
    return savedir + 'merra_%s_dailyrel_%s_%d.nc' % (varnm, onset_nm, year)

def get_filenames(years, datadir):
    datafiles = {}
    datafiles['HOWI'] = ['merra_vimt_ps-300mb_%d.nc' % yr for yr in years]
    #datafiles['CHP_MFC'] = ['merra_MFC_ps-300mb_%d.nc' % yr for yr in years]
    subset1 = '40E-120E_90S-90N'
    datafiles['CHP_MFC'] = [yrlyfile('MFC', None, yr, subset1) for yr in years]
    datafiles['CHP_PCP'] = ['merra_precip_%d.nc' % yr for yr in years]
    datafiles['precip'] = datafiles['CHP_PCP']

    for plev in [200, 850]:
        files = [yrlyfile('uv', plev, yr) for yr in years]
        for key in ['U', 'V', 'Ro', 'rel_vort', 'abs_vort']:
            datafiles['%s%d' % (key, plev)] = files
        for key in ['T', 'H', 'QV']:
            key2 = '%s%d' % (key, plev)
            datafiles[key2] = [yrlyfile(key, plev, yr) for yr in years]

    for plev in [950, 975]:
        for key in ['T', 'H','QV', 'V']:
            files = [yrlyfile(key, plev, yr) for yr in years]
            datafiles['%s%d' % (key, plev)] = files

    for key in ['EFLUX', 'HFLUX', 'EVAP', 'VFLXPHI', 'VFLXCPT', 'VFLXQV']:
        subset1 = '40E-120E_90S-90N'
        datafiles[key] = [yrlyfile(key, None, yr, subset1) for yr in years]

    # Zonal and sector mean data
    for lonlims in [(0, 360), (60, 100)]:
        lonstr = atm.latlon_str(lonlims[0], lonlims[1], 'lon')
        subset1 = 'sector_' + lonstr
        for nm in ['U', 'V', 'OMEGA', 'T', 'H', 'QV']:
            key = nm + '_' + subset1
            datafiles[key] = [yrlyfile(nm, None, yr, subset1) for yr in years]

    for varnm in datafiles:
        files = datafiles[varnm]
        datafiles[varnm] = [datadir + filenm for filenm in files]

    return datafiles

datafiles = get_filenames(years, datadir)

# ----------------------------------------------------------------------
# Onset index for each year

index = utils.get_onset_indices(onset_nm, datafiles[onset_nm], years)
onset = index['onset']
retreat = index['retreat']

# ----------------------------------------------------------------------
# Get daily data

def housekeeping(data, keys_remove):
    # Remove intermediate data that I don't want to keep
    keys = data.keys()
    for key in keys_remove:
        if key in keys:
            data = atm.odict_delete(data, key)

    # Fill Ro200 with NaNs near equator
    varnm = 'Ro200'
    if varnm in data:
        latbuf = 5
        lat = atm.get_coord(data[varnm], 'lat')
        latbig = atm.biggify(lat, data[varnm], tile=True)
        vals = data[varnm].values
        vals = np.where(abs(latbig)>latbuf, vals, np.nan)
        data[varnm].values = vals
    return data


def get_varid(varnm):
    if varnm.find('sector') >= 0:
        varid = varnm.split('_')[0]
    else:
        varid = varnm
    return varid



# Save daily data for each year to file
yearnm, daynm = 'year', 'day'
relfiles = {}
for y, year in enumerate(years):
    files = {key : [datafiles[key][y]] for key in datafiles}
    d_onset = int(onset[y].values)
    d_retreat = int(retreat[y].values)
    coords = {yearnm : [year]}
    onset_var = xray.DataArray([d_onset], name='D_ONSET', coords=coords)
    retreat_var = xray.DataArray([d_retreat], name='D_RETREAT', coords=coords)
    data = {}
    for varnm in varnms:
        print('Reading daily data for ' + varnm)
        varid = get_varid(varnm)
        var = get_data_rel(varid, year, files.get(varnm), data, d_onset,
                           npre, npost, yearnm, daynm)
        var.attrs = atm.odict_delete(var.attrs, 'd_onset')
        var.name = varid
        data[varnm] = var
    data = housekeeping(data, keys_remove)
    for varnm in data:
        savefile = get_savefile(savedir, varnm, onset_nm, year)
        if y == 0:
            relfiles[varnm] = [savefile]
        else:
            relfiles[varnm] = relfiles[varnm] + [savefile]
        print('Saving to ' + savefile)
        atm.save_nc(savefile, data[varnm], onset_var, retreat_var)

# ----------------------------------------------------------------------
# Compute climatologies and save

yearstr = '%d-%d' % (years.min(), years.max())
for varnm in relfiles:
    var, onset, retreat = load_dailyrel(relfiles[varnm])
    ds = xray.Dataset()
    ds[varnm], ds['D_ONSET'], ds['D_RETREAT'] = var, onset, retreat
    print('Computing climatological mean')
    ds = ds.mean(dim=yearnm)
    ds[varnm].attrs = var.attrs
    ds[varnm].attrs['years'] = years
    filn = savedir + 'merra_%s_dailyrel_%s_%s.nc' % (varnm, onset_nm, yearstr)
    print('Saving to ' + filn)
    ds.to_netcdf(filn)
