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
import utils

# ----------------------------------------------------------------------
onset_nm = 'CHP_MFC'
enso_keys = ['ONI_MAM', 'ONI_JJA', 'MEI_MARAPR', 'MEI_JULAUG']
plot_enso_ind = False

years = np.arange(1979, 2015)
datafiles = {}
datafiles['HOWI'] = [datadir + 'merra_vimt_ps-300mb_%d.nc' % yr for yr in years]
datafiles['MFC'] = [datadir + 'merra_MFC_ps-300mb_%d.nc' % yr for yr in years]
datafiles['PCP'] = [datadir + 'merra_precip_%d.nc' % yr for yr in years]

# Lat-lon box for MFC / precip
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30

# ----------------------------------------------------------------------
# Read data

# MFC and precip over SASM region
nroll = 7
tseries = utils.get_mfc_box(datafiles['MFC'], datafiles['PCP'], years,
                            nroll, lat1, lat2, lon1, lon2)

# Monsoon onset/retreat indices
if onset_nm.startswith('CHP'):
    # --- Use precip/MFC already loaded
    data = tseries[onset_nm.split('_')[1] + '_ACC']
    files = None
else:
    data = None
    files = datafiles[onset_nm]
index = utils.get_onset_indices(onset_nm, files, years, data)
index = index[['onset', 'retreat']].to_dataframe()
index['length'] = index['retreat'] - index['onset']

# ENSO
enso = utils.get_enso_indices(years, enso_keys)

# Monsoon strength
mfc = tseries['MFC_UNSM']
precip = tseries['PCP_UNSM']
ssn = utils.get_strength_indices(years, tseries['MFC_UNSM'],
                                 tseries['PCP_UNSM'], index['onset'],
                                 index['retreat'])

# ----------------------------------------------------------------------
# Helper functions

def detrend(df):
    df_detrend = df.copy()
    x = df.index.values
    for col in df.columns:
        y = df[col].values
        reg = atm.Linreg(x, y)
        df_detrend[col] = df[col] - reg.predict(x)
    return df_detrend

def line_plus_reg(years, ssn, key, clr):
    reg = atm.Linreg(years, ssn[key].values)
    plt.plot(years, ssn[key], clr, label=key)
    plt.plot(years, reg.predict(years), clr + '--')

# ----------------------------------------------------------------------
# Line plots of indices vs. year with trends

plt.figure(figsize=(12, 10))
clrs = ['b', 'g', 'r', 'c']
keys = ['MFC_JJAS_', 'MFC_LRS_', 'PCP_JJAS_', 'PCP_LRS_']
for i, nm in enumerate(['TOT', 'AVG']):
    plt.subplot(2, 2, i + 1)
    for j, varnm in enumerate(keys):
        key = varnm + nm
        line_plus_reg(years, ssn, key, clrs[j])
    if nm == 'TOT':
        plt.ylabel('Total (mm)')
    else:
        plt.ylabel('Avg (mm/day)')
plt.subplot(2, 2, 3)
line_plus_reg(years, ssn, 'onset', clrs[0])
line_plus_reg(years, ssn, 'length', clrs[1])
plt.ylabel('Day of Year')
plt.subplot(2, 2, 4)
line_plus_reg(years, ssn, 'retreat', clrs[0])
plt.ylabel('Day of Year')
for i in range(1, 5):
    plt.subplot(2, 2, i)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True)
    plt.xlabel('Year')
    plt.xlim(years.min(), years.max())
plt.suptitle(onset_nm + ' Monsoon Onset/Retreat')

# ----------------------------------------------------------------------
# Correlations between indices

opts = {'figsize' : (12, 9), 'annotation_pos' : (0.05, 0.75), 'incl_p' : True,
        'incl_line' : True, 'pmax_bold' : 0.05}

for i_detrend in [True, False]:

    # Correlations between onset/retreat/length
    if i_detrend:
        suptitle = onset_nm + ' (Detrended Indices)'
        atm.scatter_matrix(detrend(index), suptitle=suptitle, **opts)
    else:
        atm.scatter_matrix(index, suptitle=onset_nm, **opts)

    # Cumulative and average rainfall over monsoon season
    df1 = ssn[['onset', 'retreat', 'length']]
    nms = ['MFC_JJAS', 'MFC_LRS', 'PCP_JJAS', 'PCP_LRS']
    suptitle = 'Season Totals (%s Monsoon Onset/Retreat)' % onset_nm
    for key in ['_TOT', '_AVG']:
        keys = [nm + key for nm in nms]
        df2 = ssn[keys]
        if i_detrend:
            atm.scatter_matrix_pairs(detrend(df1), detrend(df2),
                                     suptitle=suptitle + ' ( Detrended)')
        else:
            atm.scatter_matrix_pairs(df1, df2, suptitle=suptitle)

    # Scatter plots and correlations between ENSO and monsoon indices
    suptitle = onset_nm + ' Monsoon Indices vs. ENSO'
    if i_detrend:
        atm.scatter_matrix_pairs(detrend(enso), detrend(index), figsize,
                                 suptitle + ' (Detrended)')
    else:
        atm.scatter_matrix_pairs(enso, index, opts['figsize'], suptitle)

# ----------------------------------------------------------------------
# Plot ENSO indices

if plot_enso_ind:
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
