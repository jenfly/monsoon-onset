import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')
sys.path.append('/home/jwalker/dynamics/python/monsoon-onset')

import xray
import numpy as np
import collections
import pandas as pd
import scipy

import atmos as atm
import indices
import utils
import precipdat

# ----------------------------------------------------------------------
version, yearstr = 'merra2', '1980-2015'
onset_nm = 'CHP_MFC'
years = np.arange(1980, 2015) # CMAP last full year is 2014

datadir = atm.homedir() + 'datastore/%s/analysis/' % version
pcpfile = atm.homedir() + 'datastore/cmap/cmap.enhanced.precip.pentad.mean.nc'
indfile = atm.homedir() + ('datastore/%s/analysis/%s_index_%s_%s.nc' %
                           (version, version, onset_nm, yearstr))
ind_nm, npre, npost = 'onset', 119, 200
#ind_nm, npre, npost = 'retreat', 200, 50
savefile = datadir + 'cmap_dailyrel_%s_%s-%d.nc' % (onset_nm, min(years),
                                                    max(years))
if ind_nm == 'retreat':
    savefile = savefile.replace('dailyrel', 'dailyrel_retreat')
# ----------------------------------------------------------------------
# Data and calcs

# Onset index
with xray.open_dataset(indfile) as index:
    index.load()
index = index.sel(year=years)
d0 = index[ind_nm].values

# Precip data
pcp = precipdat.read_cmap(pcpfile, yearmin=min(years), yearmax=max(years))
name, attrs, coords, dimnames = atm.meta(pcp)

# Interpolate to daily resolution
days = np.arange(3, 364)
interp_func = scipy.interpolate.interp1d(pcp['day'], pcp, axis=1)
vals = interp_func(days)
coords['day'] = xray.DataArray(days, coords={'day' : days})
pcp_i = xray.DataArray(vals, dims=dimnames, coords=coords, name=name,
                       attrs=attrs)

# Daily relative to onset/withdrawal
pcp_rel = utils.daily_rel2onset(pcp_i, d0, npre, npost)
print('Saving to ' + savefile)
atm.save_nc(savefile, pcp_rel)
