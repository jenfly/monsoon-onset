import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
import indices

# ----------------------------------------------------------------------
onset_nm = 'CHP_MFC'
enso_keys = ['ONI_MAM', 'ONI_JJA', 'MEI_MARAPR', 'MEI_JULAUG']
years = range(1979, 2015)

ensodir = atm.homedir() + 'dynamics/calc/ENSO/'
datadir = atm.homedir() + 'datastore/merra/daily/'
ensofiles = {'MEI' : ensodir + 'enso_mei.csv',
             'ONI' : ensodir + 'enso_oni.csv'}
datafiles = {}
datafiles['vimt'] = [datadir + 'merra_vimt_ps-300mb_%d.nc' % yr for yr in years]
datafiles['mfc'] = [datadir + 'merra_MFC_ps-300mb_%d.nc' % yr for yr in years]
datafiles['precip'] = [datadir + 'merra_precip_%d.nc' % yr for yr in years]

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# ENSO indices
enso_in = {}
for key in ensofiles:
    enso_in[key] = pd.read_csv(ensofiles[key], index_col=0)

enso = pd.DataFrame()
for key in enso_in:
    for ssn in enso_in[key]:
        enso[key + '_' + ssn] = enso_in[key][ssn]

enso = enso.loc[enso.index.isin(years)]
enso = enso[enso_keys]

# ----------------------------------------------------------------------
# Monsoon onset/retreat indices

if onset_nm == 'HOWI':
    maxbreak = 10
    npts = 100
    ds = atm.combine_daily_years(['uq_int', 'vq_int'],datafiles['vimt'], years,
                                 yearname='year')
    index, _ = indices.onset_HOWI(ds['uq_int'], ds['vq_int'], npts, maxbreak=maxbreak)
    index.attrs['title'] = 'HOWI (N=%d)' % npts
elif onset_nm == 'CHP_MFC':
    mfc = atm.combine_daily_years('MFC', datafiles['mfc'], years, yearname='year')
    mfcbar = atm.mean_over_geobox(mfc, lat1, lat2, lon1, lon2)
    mfc_acc = np.cumsum(mfcbar, axis=1)
    index = indices.onset_changepoint(mfc_acc)

# Dataframe of onset, retreat, length
index = index[['onset', 'retreat']].to_dataframe()
index['length'] = index['retreat'] - index['onset']

# ----------------------------------------------------------------------
# Detrend

def detrend(df):
    df_detrend = df.copy()
    x = df.index.values
    for col in df.columns:
        y = df[col].values
        reg = atm.Linreg(x, y)
        df_detrend[col] = df[col] - reg.predict(x)
    return df_detrend

# ----------------------------------------------------------------------
# Scatter plots and correlations between ENSO and monsoon indices

figsize = (12, 9)
suptitle = onset_nm + ' Monsoon Indices vs. ENSO'
atm.scatter_matrix_pairs(enso, index, figsize, suptitle)
suptitle = suptitle + ' (Detrended)'
atm.scatter_matrix_pairs(detrend(enso), detrend(index), figsize, suptitle)

# ----------------------------------------------------------------------
# Correlations between onset/retreat/length

opts = {'figsize' : (12, 9), 'annotation_pos' : (0.05, 0.75), 'incl_p' : True,
        'incl_line' : True, 'pmax_bold' : 0.05}
atm.scatter_matrix(index, suptitle=onset_nm, **opts)
atm.scatter_matrix(detrend(index), suptitle=onset_nm + ' (Detrended)', **opts)

# ----------------------------------------------------------------------
# Plot ENSO indices
keys = ['ONI_JJA', 'MEI_JULAUG']

def threshold(key):
    if key.startswith('ONI'):
        thresh = 0.5
    else:
        thresh = 1.0
    return thresh

plt.figure(figsize=(12, 10))
nrows = len(keys)
for i, key in enumerate(keys):
    data = enso[key]
    plt.subplot(nrows, 1, i + 1)
    plt.bar(data.index, data.values, color='k', alpha=0.3)
    iwarm = data.values > threshold(key)
    icold = data.values < -threshold(key)
    plt.bar(data[iwarm].index, data[iwarm].values, color='r')
    plt.bar(data[icold].index, data[icold].values, color='b')
    plt.grid()
    plt.title('ENSO_' + key)


data = enso['ONI_JJA'].copy().loc[1979:2014]
data.sort(ascending=False)
nyrs = 5
print('El Nino Top %d' % nyrs)
print(data[:nyrs])
print('La Nina Top %d' % nyrs)
print(data[-1:-nyrs-1:-1])
