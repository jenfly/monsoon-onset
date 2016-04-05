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

datafiles = collections.OrderedDict()
filestr = datadir + 'merra_%s_dailyrel_%s_%d.nc'
for nm in varnms:
    datafiles[nm] = [filestr % (nm, onset_nm, yr) for yr in years]

# ----------------------------------------------------------------------
# For each variable, read data, compute regressions and save to files

def get_filenames(datadir, varnm, onset_nm, years, lon1, lon2, nroll):
    if nroll is not None:
        varnm = varnm + '_nroll%d' % nroll
    lonstr = atm.latlon_str(lon1, lon2, 'lon')
    yearstr = '_%d-%d.nc' % (min(years), max(years))
    filestr = datadir + 'merra_%s_reg_%s_onset_' + onset_nm + yearstr
    filenames = {}
    filenames['latlon'] = filestr % (varnm, 'latlon')
    filenames['sector'] = filestr % (varnm, lonstr)
    return filenames

def season_days(season, year, d_onset, d_retreat):
    if season == 'SSN':
        days = range(0, d_retreat - d_onset + 1)
    else:
        days = atm.season_days(season, atm.isleap(year)) - d_onset
    return days

def ssn_average(var, onset, retreat, season):
    years = var['year'].values
    for y, year in enumerate(years):
        days = season_days(season, year, onset.values[y], retreat.values[y])
        var_yr = atm.subset(var, {'year' : (year, year)}, squeeze=False)
        var_yr = var_yr.sel(dayrel=days).mean(dim='dayrel')
        if y == 0:
            var_out = var_yr
        else:
            var_out = xray.concat([var_out, var_yr], dim='year')
    return var_out

def get_data(varnm, datafiles, regdays, seasons, lon1, lon2, nroll=None):
    var, onset, retreat = utils.load_dailyrel(datafiles[varnm])
    if nroll is not None:
        var = atm.rolling_mean(var, nroll, axis=1, center=True)

    # Seasonal averages and daily lat-lon data
    data = xray.Dataset()
    for season in seasons:
        key = varnm + '_' + season
        data[key] = ssn_average(var, onset, retreat, season)
    # Daily data on regdays
    data[varnm + '_DAILY'] = var.sel(dayrel=regdays)

    # Sector mean data
    var_sector = atm.dim_mean(var, 'lon', lon1, lon2)

    alldata = {'data_latlon' : data, 'var_sector' : var_sector,
              'onset' : onset, 'retreat' : retreat}

    return alldata

def regress_data(alldata, indname='onset', axis=0):
    data = alldata['data_latlon']
    var_sector = alldata['var_sector']
    index = alldata[indname]
    print('Computing regressions')
    reg_data = xray.Dataset()
    for nm in data.data_vars:
        print(nm)
        reg = atm.regress_field(data[nm], index, axis)
        for nm2 in reg.data_vars:
            reg_data[nm + '_' + nm2] = reg[nm2]

    reg_sector = atm.regress_field(var_sector, index, axis)
    reg_out = {'latlon' : reg_data, 'sector' : reg_sector}
    return reg_out

def process_one(varnm, datafiles, years, regdays, seasons, lon1, lon2,
                datadir, indname='onset', reg_axis=0, nroll=None):
    print('Processing ' + varnm)
    alldata = get_data(varnm, datafiles, regdays, seasons, lon1, lon2)
    reg_data = regress_data(alldata, indname=indname, axis=reg_axis)
    savefiles = get_filenames(datadir, varnm, onset_nm, years, lon1, lon2,
                              nroll)
    for nm in reg_data:
        filenm = savefiles[nm]
        print('Saving to ' + filenm)
        ds = reg_data[nm]
        ds.to_netcdf(filenm)
    return alldata, reg_data

# Iterate over each variable
for varnm in varnms:
    data, reg_data = process_one(varnm, datafiles, years, regdays, seasons,
                                 lon1, lon2, datadir, indname='onset',
                                 reg_axis=0, nroll=nroll)
