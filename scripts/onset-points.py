import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import atmos as atm
import indices

# ----------------------------------------------------------------------
# Changepoint onset at individual points

datadir = atm.homedir() + 'datastore/merra/daily/'
savedir = atm.homedir() + 'datastore/merra/analysis/'
filestr = datadir + 'merra_precip_%d.nc'
savestr = savedir + 'merra_onset_pts_CHP_PCP_%d.nc'
years = np.arange(1979, 2015)
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
subset_dict =  {'lon' : (lon1, lon2), 'lat' : (lat1, lat2)}

def calc_points(pcp_acc):
    onset = (np.nan * pcp_acc[0]).drop('day')
    retreat = (np.nan * pcp_acc[0]).drop('day')
    lat = atm.get_coord(pcp_acc, 'lat')
    lon = atm.get_coord(pcp_acc, 'lon')
    for i, lat0 in enumerate(lat):
        for j, lon0 in enumerate(lon):
            print('%.1f E, %.1f N' % (lon0, lat0))
            chp = indices.onset_changepoint(pcp_acc[:, i, j])
            onset.values[i, j] = chp.onset.values
            retreat.values[i, j] = chp.retreat.values
    return onset, retreat      


for year in years:
    filenm = filestr % year
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        pcp = atm.subset(ds['PRECTOT'], subset_dict)
        pcp.load()
    pcp = atm.precip_convert(pcp, pcp.attrs['units'], 'mm/day')
    pcp_acc = np.cumsum(pcp, axis=0)
    onset, retreat = calc_points(pcp_acc)
    savefile = savestr % year
    print('Saving to ' + savefile)
    atm.save_nc(savefile, onset, retreat)

