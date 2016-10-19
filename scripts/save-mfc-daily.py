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
version = 'merra2'
onset_nm = 'CHP_MFC'

years = np.arange(1980, 2016)
datadir = atm.homedir() + 'datastore/%s/daily/' % version
datafiles = collections.OrderedDict()
filestr = datadir + '%d/%s_%s_40E-120E_90S-90N_%d.nc'
datafiles['MFC'] = [filestr % (yr, version, 'MFC', yr) for yr in years]
datafiles['PCP'] = [filestr % (yr, version, 'PRECTOT', yr) for yr in years]

years_gpcp = np.arange(1997, 2015)
filestr2 = atm.homedir() + 'datastore/gpcp/gpcp_daily_%d.nc'
datafiles['GPCP'] = [filestr2 % yr for yr in years_gpcp]
years_dict = {'MFC' : years, 'PCP' : years, 'GPCP' : years_gpcp}
varnms = {'MFC' : 'MFC', 'PCP' : 'PRECTOT', 'GPCP' : 'PREC'}

savedir = atm.homedir() + 'eady/datastore/merra2/analysis/'
savefile = savedir + 'merra2_gpcp_mfc_box_daily.nc'

# Lat-lon box for averaging
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# Read data

def get_data(files, yrs, varnm, lat1, lat2, lon1, lon2):
    subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}
    var = atm.combine_daily_years(varnm, files, yrs, yearname='year',
                                  subset_dict=subset_dict)
    var = atm.precip_convert(var, var.attrs.get('units'), 'mm/day')
    var = atm.mean_over_geobox(var, lat1, lat2, lon1, lon2)
    return var

data = xray.Dataset()

for nm in datafiles:
    var = get_data(datafiles[nm], years_dict[nm], varnms[nm], lat1, lat2,
                   lon1, lon2)
    for nm2 in ['subset_lons', 'subset_lats', 'area_weighted', 'land_only']:
        var.attrs = atm.odict_delete(var.attrs, nm2)
    data[nm] = var

# Save to file
data.to_netcdf(savefile)    

