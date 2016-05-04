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

years = np.arange(1979, 2015)
datadir = atm.homedir() + 'datastore/merra/analysis/'
onset_nm, pts_nm = 'CHP_MFC', 'CHP_PCP'

yearstr = '%d-%d.nc' % (min(years), max(years))
filestr = datadir + 'merra_index_%s_' + yearstr
indfile = filestr % onset_nm
datafile = filestr % ('pts_' + pts_nm)

with xray.open_dataset(indfile) as index:
    index.load()
with xray.open_dataset(datafile) as data:
    data.load()
    
