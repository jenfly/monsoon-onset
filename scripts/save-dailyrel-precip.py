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
datadir = atm.homedir() + 'datastore/%s/analysis/' % version
onset_nm = 'CHP_MFC'
indfile = atm.homedir() + ('datastore/%s/analysis/%s_index_%s_%s.nc' %
                           (version, version, onset_nm, yearstr))
#ind_nm, npre, npost = 'onset', 120, 200
ind_nm, npre, npost = 'retreat', 270, 100

pcp_nm = 'gpcp'
years = np.arange(1997, 2016)
#years = np.arange(1980, 2015) # CMAP last full year is 2014
#pcpfiles = atm.homedir() + 'datastore/cmap/cmap.enhanced.precip.pentad.mean.nc'
pcpfiles = [atm.homedir() + 'datastore/gpcp/gpcp_daily_%d.nc' % yr
            for yr in years]

savefile = datadir + '%s_dailyrel_%s_%s-%d.nc' % (pcp_nm, onset_nm, min(years),
                                                  max(years))
if ind_nm == 'retreat':
    savefile = savefile.replace('dailyrel', 'dailyrel_retreat')

subset_dict = {'lon' : (40, 120)}

# ----------------------------------------------------------------------
# Data and calcs

# Onset index
with xray.open_dataset(indfile) as index:
    index.load()
index = index.sel(year=years)
d0 = index[ind_nm].values

# Precip data
if pcp_nm == 'cmap':
    pcp = precipdat.read_cmap(pcpfiles, yearmin=min(years), yearmax=max(years))

    # Interpolate to daily resolution
    name, attrs, coords, dimnames = atm.meta(pcp)
    days = np.arange(3, 364)
    interp_func = scipy.interpolate.interp1d(pcp['day'], pcp, axis=1)
    vals = interp_func(days)
    coords['day'] = xray.DataArray(days, coords={'day' : days})
    pcp = xray.DataArray(vals, dims=dimnames, coords=coords, name=name,
                         attrs=attrs)
else:
    pcp = atm.combine_daily_years(None, pcpfiles, years, yearname='year',
                                  subset_dict=subset_dict)


# Wrap from following year to get extended daily range
daymin = min(d0) - npre
daymax = max(d0) + npost
pcp = utils.wrapyear_all(pcp, daymin=daymin, daymax=daymax)

# Daily relative to onset/withdrawal
pcp_rel = utils.daily_rel2onset(pcp, d0, npre, npost)
print('Saving to ' + savefile)
atm.save_nc(savefile, pcp_rel)
