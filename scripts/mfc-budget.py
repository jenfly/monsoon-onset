import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xarray as xray
import pandas as pd
import matplotlib.pyplot as plt

import atmos as atm
import merra
import indices
import utils

# ----------------------------------------------------------------------
# version, years = 'merra', np.arange(1979, 2015)
version, years = 'merra2', np.arange(1980, 2016)

datadir = atm.homedir() + 'datastore/%s/daily/' % version
files = {}
filestr = version + '_%s_40E-120E_90S-90N_%d.nc'
files['MFC'] = [datadir + filestr % ('MFC', yr) for yr in years]
if version == 'merra':
    files['PCP'] = [datadir + 'merra_precip_%d.nc' % yr for yr in years]
else:
    files['PCP'] = [datadir + filestr % ('PRECTOT', yr) for yr in years]
files['EVAP'] = [datadir + filestr % ('EVAP', yr) for yr in years]
files['ANA'] = [datadir + filestr % ('DQVDT_ANA', yr) for yr in years]
files['W'] = [datadir + filestr % ('TQV', yr) for yr in years]

# Lat-lon box for MFC budget
lon1, lon2 = 60, 100
lat1, lat2 = 10, 30
latlonstr = '%d-%dE, %d-%dN' % (lon1, lon2, lat1, lat2)

# ----------------------------------------------------------------------
# Read data

nroll = 7
days_ssn = atm.season_days('JJAS')
subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}

ts = xray.Dataset()
ssn = xray.Dataset()
varnms = {'PCP' : 'PRECTOT', 'MFC' : 'MFC', 'EVAP' : 'EVAP', 'W' : 'TQV',
          'ANA' : 'DQVDT_ANA'}

for nm in files:
    var = atm.combine_daily_years(varnms[nm], files[nm], years, yearname='year',
                                  subset_dict=subset_dict)
    var = atm.mean_over_geobox(var, lat1, lat2, lon1, lon2)
    units = var.attrs.get('units')
    if units in ['kg/m2/s', 'kg m-2 s-1']:
        var = atm.precip_convert(var, units, 'mm/day')
    var_sm = atm.rolling_mean(var, nroll, axis=-1, center=True)
    ts[nm] = var_sm
    if nm == 'W':
        # Difference in precipitable water from beginning to end of season
        var_ssn = var_sm.sel(day=days_ssn)
        ssn['d' + nm] = (var_ssn[:,-1] - var_ssn[:, 0]) / len(days_ssn)
        # dW/dt
        dvar = np.nan * np.ones(var.shape)
        for y, year in enumerate(years):
            dvar[y] = np.gradient(var[y])
        ts['d%s/dt' % nm] = xray.DataArray(dvar, coords=var.coords)
    else:
        # Seasonal mean and daily timeseries
        ssn[nm] = var.sel(day=days_ssn).mean(dim='day')

# Net precip and residuals
ts['P-E+dW/dt-ANA'] = ts['PCP'] - ts['EVAP'] + ts['dW/dt'] - ts['ANA']
ts['RESID'] = ts['MFC'] - ts['P-E+dW/dt-ANA']

# Standardize the seasonal averages for easier comparison
ssn = ssn.to_dataframe()
ssn['P-E+dW-ANA'] = ssn['PCP'] - ssn['EVAP'] + ssn['dW'] - ssn['ANA']
ssn_st = pd.DataFrame()
for key in ssn.columns:
    ssn_st[key] = (ssn[key] - ssn[key].mean()) / ssn[key].std()

# ----------------------------------------------------------------------
# Plot daily tseries of MFC budget components

def plot_tseries(ts, year=None, ax=None):
    keys = ['PCP', 'EVAP', 'ANA', 'dW/dt', 'P-E+dW/dt-ANA', 'MFC', 'RESID']
    tseries = ts[keys]
    if year is None:
        tseries = tseries.mean(dim='year')
        title = 'Climatological Mean'
    else:
        tseries = tseries.sel(year=year).drop('year')
        title = year
    tseries = tseries.to_dataframe()
    tseries.plot(ax=ax, grid=True, legend=False)
    ax.set_title(title, loc='left', fontsize=11)

# Plot climatology and a few individual years
plotyears = [None, years[0], years[1], years[2]]
plt.figure(figsize=(12, 9))
suptitle = version.upper() + ' MFC Budget (%s) - Daily Tseries' % latlonstr
plt.suptitle(suptitle)
nrow, ncol = 2, 2
for y, year in enumerate(plotyears):
    ax = plt.subplot(nrow, ncol, y + 1)
    plot_tseries(ts, year, ax=ax)
    if y == 0:
        ax.legend(loc='upper left', fontsize=9)
        s = 'RESID = MFC-P+E-dW/dt-ANA'
        atm.text(s, (0.03, 0.5), fontsize=9)

# ----------------------------------------------------------------------
# Plot daily timeseries of W

W = ts['W']
d1, d2 = days_ssn[0], days_ssn[-1]
W1 = W.sel(day=d1)
W2 = W.sel(day=d2)
plotyears = years[:4]
clrs = ['b', 'r', 'g', 'k']
plt.figure()
for y, year in enumerate(plotyears):
    plt.plot(W['day'], W[y], clrs[y], label=year)
    plt.plot([d1, d2], [W1[y], W2[y]], clrs[y], marker='.')
plt.grid()
plt.legend(fontsize=8)
plt.xlabel('Day')
plt.ylabel('Precipitable Water (mm)')

# ----------------------------------------------------------------------
# Average values over JJAS and LRS - interannual variability

keys = ['PCP', 'EVAP', 'ANA', 'dW', 'P-E+dW-ANA', 'MFC']
fmts = {'PCP' : 'b', 'EVAP' : 'g', 'ANA' : 'r', 'dW' : 'c',
        'P-E+dW-ANA' : 'm', 'MFC' : 'k--'}

ssn[keys].plot(figsize=(10, 10), style=fmts, legend=False, grid=True)
plt.legend(loc='upper left', fontsize=10)
plt.ylabel('mm/day')
plt.title('JJAS Mean')


plt.figure(figsize=(7, 10))
plt.suptitle('JJAS mean standardized timeseries')
nrow, ncol = len(keys), 1
for i, col in enumerate(ssn_st[keys].columns):
    plt.subplot(nrow, ncol, i + 1)
    plt.plot(years, ssn_st[col], 'k')
    plt.title(col, loc='left', fontsize=10)
    plt.autoscale(tight=True)
    plt.grid(True)
    if i < nrow - 1:
        plt.gca().set_xticklabels([])
    else:
        plt.xlabel('Year')

atm.scatter_matrix(ssn[keys], figsize=(14, 10), pmax_bold=0.05, incl_p=True,
                   annotation_pos=(0.05, 0.6), incl_line=True, suptitle='JJAS Mean')
