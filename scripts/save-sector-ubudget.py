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
onset_nm = 'CHP_MFC'
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
years = np.arange(1980, 2016)
yearstr = '%d-%d' % (min(years), max(years))
plevs = [1000,925,850,775,700,600,500,400,300,250,200,150,100,70,50,30,20]
lon1, lon2 = 60, 100
ndays = 5     # n-day rolling mean for smoothing
scale = 1e-4  # Scaling factor for all terms in momentum budget
eqbuf = 5.0   # Latitude buffer around equator for psi decomposition

filestr = (datadir + version + '_ubudget%d_dailyrel_' + onset_nm +
           '_ndays%d_%dE-%dE_%s.nc' % (ndays, lon1, lon2, yearstr))
datafiles = [filestr % plev for plev in plevs]
savefile_ubudget = datafiles[0].replace('%d' % plevs[0], '_sector')

nms_latp = ['U', 'V']
files_latp = {}
for nm in nms_latp:
    filenm = datadir + version + '_%s_sector_%dE-%dE_dailyrel_%s_%s.nc'
    files_latp[nm] = filenm % (nm, lon1, lon2, onset_nm, yearstr)
savefile_latp = savefile_ubudget.replace('ubudget', 'latp')
savefile_psicomp = savefile_ubudget.replace('ubudget', 'psicomp')

# ----------------------------------------------------------------------
# Read ubudget data and save

def consolidate(ds):
    # Consolidate terms in ubudget
    groups = collections.OrderedDict()
    # groups['ADV_AVG'] = ['ADV_AVG_AVG_X', 'ADV_AVG_AVG_Y', 'ADV_AVG_AVG_P']
    # groups['ADV_AVST'] = ['ADV_AVG_ST_X', 'ADV_AVG_ST_Y', 'ADV_AVG_ST_P']
    # groups['ADV_STAV'] = ['ADV_ST_AVG_X', 'ADV_ST_AVG_Y', 'ADV_ST_AVG_P']
    # groups['ADV_CRS'] = ['ADV_AVST', 'ADV_STAV']
    # groups['EMFC_TR'] = ['EMFC_TR_X', 'EMFC_TR_Y', 'EMFC_TR_P']
    # groups['EMFC_ST'] = ['EMFC_ST_X', 'EMFC_ST_Y', 'EMFC_ST_P']
    groups['ADV_AVG'] = ['ADV_AVG_AVG_X', 'ADV_AVG_AVG_Y']
    groups['ADV_AVST'] = ['ADV_AVG_ST_X', 'ADV_AVG_ST_Y']
    groups['ADV_STAV'] = ['ADV_ST_AVG_X', 'ADV_ST_AVG_Y']
    groups['ADV_CRS'] = ['ADV_AVST', 'ADV_STAV']
    groups['EMFC_TR'] = ['EMFC_TR_X', 'EMFC_TR_Y']
    groups['EMFC_ST'] = ['EMFC_ST_X', 'EMFC_ST_Y']

    groups['EMFC'] = ['EMFC_TR', 'EMFC_ST']
    groups['COR'] = ['COR_AVG', 'COR_ST']
    groups['ADV+COR'] = ['ADV_AVG', 'COR_AVG']
    groups['SUM'] = ['ADV_AVG', 'ADV_CRS', 'EMFC', 'COR', 'PGF_ST', 'ANA']

    print('Consolidating ubudget terms')
    for key in groups:
        nms = groups[key]
        ds[key] = ds[nms[0]]
        for nm in nms[1:]:
            ds[key] = ds[key] + ds[nm]

    return ds


def process_file(filenm, plev, lon1, lon2, pname='Height'):
    # Process data for a single file (one pressure level)
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        ds = consolidate(ds)
        print('Computing sector mean')
        ds = atm.dim_mean(ds, 'lon', lon1, lon2)
        ds.load()
    for nm in ds.data_vars:
        ds[nm] = atm.expand_dims(ds[nm], pname, plev, axis=1)
    return ds

# Read ubudget at each plevel and concatenate
for i, filenm in enumerate(datafiles):
    plev = plevs[i]
    ds = process_file(filenm, plev, lon1, lon2)
    if i == 0:
        ubudget = ds
    else:
        ubudget = xray.concat([ubudget, ds], dim='Height')

ubudget['Height'].attrs['units'] = 'hPa'

# Apply scaling
ubudget = ubudget / scale
ubudget.attrs['units'] = '%.0e m/s2' % scale
for nm in ubudget.data_vars:
    ubudget[nm].attrs['units'] = '%.0e m/s2' % scale

# Additional metadata
ubudget.attrs['ndays'] = ndays
ubudget.attrs['lon1'] = lon1
ubudget.attrs['lon2'] = lon2

# Save to file
ubudget.to_netcdf(savefile_ubudget)

#************************ TEMPORARY TROUBLESHOOTING ******************
filestr = ('/home/jwalker/datastore/merra2/analysis/merra2_ubudget%d_' +
           'dailyrel_CHP_MFC_ndays5_60E-100E_%d.nc')

for year in years:
    with open('troubleshooting_%d.txt' % year, 'w') as f1:
        for plev in plevs:
            filenm = filestr % (plev, year)
            print(filenm)
            f1.write('------------------------------------------\n')
            f1.write(filenm + '\n')
            f1.write('Year %d, pressure level %.0f' % (year, plev))
            with xray.open_dataset(filenm) as ds:
                vals = ds.max()
            biggest = vals.to_array().values.max()
            f1.write('%.e\n' % biggest)
            if biggest > 10:
                for nm in vals.data_vars:
                    f1.write('%s\t%.e\n' % (nm, vals[nm]))



# ----------------------------------------------------------------------
# Calculate streamfunction components from ubudget

print('Computing streamfunction components')
sector_scale = (lon2 - lon1) / 360.0
v = utils.v_components(ubudget, scale=scale, eqbuf=eqbuf)
psi_comp = xray.Dataset()
for nm in v.data_vars:
    psi_comp[nm] = atm.streamfunction(v[nm], sector_scale=sector_scale)


# ----------------------------------------------------------------------
# Read sector mean U, V and calculate streamfunction

data_latp = xray.Dataset()
for nm in nms_latp:
    filenm = files_latp[nm]
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        var = ds[nm].load()
    daydim = atm.get_coord(var, coord_name='dayrel', return_type='dim')
    data_latp[nm] = atm.rolling_mean(var, ndays, axis=daydim, center=True)

# Compute streamfunction
print('Computing streamfunction')
if (lon2 - lon1) < 360:
    sector_scale = (lon2 - lon1) / 360.
else:
    sector_scale = None
data_latp['PSI'] = atm.streamfunction(data_latp['V'], sector_scale=sector_scale)

# Topography for lat-pres contour plots
print('Loading topography')
psfile = atm.homedir() + 'dynamics/python/atmos-tools/data/topo/ncep2_ps.nc'
with xray.open_dataset(psfile) as ds:
    ps = ds['ps'] / 100
    if (lon2 - lon1) < 360:
        ps = atm.dim_mean(ps, 'lon', lon1, lon2)
    else:
        ps = atm.dim_mean(ps, 'lon')
