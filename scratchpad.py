import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import atmos as atm

# ----------------------------------------------------------------------

# x-y data
regdays = [-60, -30, 0, 30, 60]
plotdays = [-60, -30]
clev_r = np.arange(-1.0, 1.01, 0.05)
for nm in varnms:
    print(nm)
    var = data[nm].sel(dayrel=regdays)
    reg_daily = atm.regress_field(var, onset, axis=0)
    for day in plotdays:
        reg = reg_daily.sel(dayrel=day)
        title = '%s day %d vs. Onset ' % (var.name, day)
        cint_m = atm.cinterval(reg.m)
        clev_m = atm.clevels(reg.m, cint_m, symmetric=True)
        plt.figure(figsize=(11, 8))
        plt.subplot(1, 2, 1)
        atm.contourf_latlon(reg['r'], clev=clev_r, cmap='RdBu_r')
        plt.title(title + ' - Corr Coeff')
        plt.subplot(1, 2, 2)
        atm.contourf_latlon(reg['m'], clev=clev_m, cmap='RdBu_r')
        plt.title(title + ' - Reg Coeff')


# ----------------------------------------------------------------------
# For later, when combining plevel data:
def func(var, pname='Height', axis=1):
    pres = var.attrs[pname]
    var = atm.expand_dims(var, pname, pres, axis=axis)
    return var


# ----------------------------------------------------------------------
# Streamfunction and zonal wind from dailyrel climatology

datadir = atm.homedir() + 'datastore/merra/analysis/'
lon1, lon2 = 60, 100
lonstr = atm.latlon_str(lon1, lon2, 'lon')
filestr = datadir + 'merra_%s_sector_' + lonstr + '_dailyrel_CHP_MFC_%d.nc'
years = np.arange(1979, 1995)

files = {}
for nm in ['U', 'V']:
    files[nm] = [filestr % (nm, yr) for yr in years]

data = xray.Dataset()
for nm in files:
    data[nm] = atm.combine_daily_years(nm, files[nm], years, yearname='year')

# Climatological mean
databar = data.mean(dim='year')

# Streamfunction
if (lon2 - lon1) < 360:
    sector_scale = (lon2 - lon1) / 360.
else:
    sector_scale = None
databar['PSI'] = atm.streamfunction(databar['V'], sector_scale = sector_scale)

# Topography
psfile = atm.homedir() + 'dynamics/python/atmos-tools/data/topo/ncep2_ps.nc'
with xray.open_dataset(psfile) as ds:
    ps = ds['ps'] / 100
    if (lon2 - lon1) < 360:
        ps = atm.dim_mean(ps, 'lon', lon1, lon2)
    else:
        ps = atm.dim_mean(ps, 'lon')


# Finding latitude of max psi
# psi = atm.subset(databar['PSI'], {'plev' : (700, 700), 'lat' : (-30, 10)},
#                   squeeze=True)
psi = atm.subset(databar['PSI'], {'lat' : (-30, 10), 'plev' : (100, 800)},
                 squeeze=True)
psi = psi.max(axis=1)

lat = atm.get_coord(psi, 'lat')
ilatmax = psi.argmax(axis=1)
latmax = lat[ilatmax]
days = atm.get_coord(psi, 'dayrel')
latmax = xray.DataArray(latmax, coords={'dayrel' : days})
plt.figure()
plt.plot(latmax['dayrel'], latmax)


# Lat-pres plots on days
clev_u, clev_psi = 5, 5
clims = [-50, 50]
omitzero = True
days = [-30]
u = databar['U'].sel(dayrel=days).mean(dim='dayrel')
psi = databar['PSI'].sel(dayrel=days).mean(dim='dayrel')
latm = latmax.sel(dayrel=days).mean(dim='dayrel')

plt.figure()
atm.contourf_latpres(u, clev=clev_u, topo=ps)
plt.clim(clims)
atm.contour_latpres(psi, clev=clev_psi, omitzero=omitzero)
plt.grid()
plt.axvline(latm, color='m', linewidth=2)
plt.title('Days ' + str(days))



# ----------------------------------------------------------------------
# Double and single westerly jets for group meeting presentation

yearstr = '1979-2015'
varnms = ['U', 'V']
datadir = atm.homedir() + 'datastore/merra/monthly/'
filestr = datadir + 'merra_%s_%s.nc'
files = {nm : filestr % (nm, yearstr) for nm in varnms}
ssns = ['DJF', 'JJA']
sector_ssn = 'JJA'
data_str = 'MERRA %s' % yearstr

data = xray.Dataset()
for nm in varnms:
    with xray.open_dataset(files[nm]) as ds:
        data[nm] = ds[nm].load()

data['PSI'] = atm.streamfunction((data['V']).mean(dim='XDim'), pdim=-2)

keys = data.data_vars.keys()
for ssn in ssns:
    for nm in keys:
        months = atm.season_months(ssn)
        data[nm + '_' + ssn] = (data[nm]).sel(month=months).mean(dim='month')

lat = atm.get_coord(data, 'lat')
lon = atm.get_coord(data, 'lon')

psfile = atm.homedir() + 'dynamics/python/atmos-tools/data/topo/ncep2_ps.nc'
ps = atm.get_ps_clim(lat, lon, psfile)
ps = ps / 100
psbar = atm.dim_mean(ps, 'lon')

# Mean over sectors:
def calc_sectors(var):
    sectors = xray.Dataset()
    name = var.name
    lon = atm.get_coord(var, 'lon')
    sector_lons = {'Atlantic' : lon[(lon >= -90) & (lon <= 0)],
                   'Pacific' : lon[(lon >= 135) | (lon <= -100)],
                   'Indian' : lon[(lon >= 40) & (lon <= 120)]}
    sector_lims = {'Atlantic' : (-75, 0), 'Pacific' : (135, -100),
                   'Indian' : (40, 120)}
    for nm in sector_lons:
        lon_sub = sector_lons[nm]
        var_sub = atm.subset(var, {'lon' : (lon_sub, None)})
        var_sub.attrs['lon_lims'] = sector_lims[nm]
        sectors[nm] = atm.dim_mean(var_sub, 'lon')
    return sectors

usectors = calc_sectors(data['U_' + sector_ssn])

# DJF and JJA zonal mean zonal wind and streamfunction
def plot_contours(data, varnm, ssn, psbar, row, col, xticks):
    key = varnm + '_' + ssn
    var = data[key]
    if 'XDim' in var.dims:
        var = var.mean(dim='XDim')
    clev = {'U' : 5, 'PSI' : 10}[varnm]
    omitzero = {'U' : False, 'PSI' : True}[varnm]
    atm.contour_latpres(var, clev=clev, topo=psbar, omitzero=omitzero)
    plt.xticks(xticks, [])
    plt.xlabel('')
    name = {'PSI' : '$\psi$', 'U' : 'U'}[varnm]
    sz = {'PSI' : 16, 'U' : 14}[varnm]
    wt = {'PSI' : 'bold', 'U' : 'normal'}[varnm]
    atm.text(name, (0.02, 0.88), fontsize=sz, fontweight=wt)
    if row == 1:
        plt.title(ssn, fontsize=12, fontweight='bold')
    if col > 1:
        plt.ylabel('')
        plt.gca().set_yticklabels([])

plot_psi = True
if plot_psi:
    nr, nc, figsize = 3, 2, (11, 8)
    nms = ['PSI', 'U']
    suptitle = 'Zonal Mean Streamfunction and U (%s)' % data_str
else:
    nr, nc, figsize = 2, 2, (11, 7)
    nms = ['U']
    suptitle = 'Zonal Mean U (%s)' % data_str
xticks = range(-90, 91, 30)
ylims = (-10, 45)
gridspec_kw = {'left' : 0.07, 'right' : 0.97, 'wspace' : 0.05, 'hspace' :0.08,
               'top' : 0.92, 'bottom' : 0.08}
fig, axes = plt.subplots(nr, nc, figsize=figsize, gridspec_kw=gridspec_kw)
plt.suptitle(suptitle, fontsize=12)
for i, ssn in enumerate(['DJF', 'JJA']):
    col = i + 1
    for j, nm in enumerate(nms):
        row = j + 1
        plt.sca(axes[j, col - 1])
        plot_contours(data, nm, ssn, psbar, row, col, xticks)

    plt.sca(axes[nr - 1, col -1])
    key = 'U_%s' % ssn
    u850 = atm.dim_mean(data[key].sel(Height=850), 'lon')
    u200 = atm.dim_mean(data[key].sel(Height=200), 'lon')
    plt.plot(lat, u200, 'k', label='200mb')
    plt.plot(lat, u850, 'k--', label='850mb')
    plt.legend(fontsize=10)
    plt.xticks(xticks)
    plt.grid(True)
    plt.xlim(-90, 90)
    plt.ylim(ylims)
    plt.xlabel('Latitude')
    atm.text('U', (0.02, 0.88), fontsize=14)
    if col == 1:
        plt.ylabel('Zonal Wind (m/s)')
    else:
        plt.gca().set_yticklabels([])

# Lat-lon maps and sector line plots
ssn = sector_ssn
gridspec_kw = {'left' : 0.02, 'right' : 0.98, 'wspace' : 0.3, 'hspace' : 0.2,
               'bottom' : 0.08, 'top' : 0.92, 'width_ratios' : [2, 1]}
nr, nc = 2, 2
style = {'Indian' : 'm', 'Atlantic' : 'k--', 'Pacific' : 'g'}
climits = {200 : (-50, 50), 850 : (-16, 16)}
iplot = 0
fig, axes = plt.subplots(nr, nc, figsize=(11,8), gridspec_kw=gridspec_kw)
plt.suptitle('%s Zonal Wind' % ssn, fontsize=14)
for i, plev in enumerate([200, 850]):
    iplot += 1
    row, col = atm.subplot_index(nr, nc, iplot)
    u = atm.subset(data['U_' + ssn], {'plev' : (plev, plev)}, squeeze=True)
    usec = atm.subset(usectors, {'plev' : (plev, plev)}, squeeze=True)
    plt.sca(axes[row - 1, col - 1])
    atm.pcolor_latlon(u)
    plt.title('%d hPa' % plev, fontsize=12)
    plt.xlabel('')
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.clim(climits[plev])
    for nm in usec.data_vars:
        for lon0 in usec[nm].attrs['lon_lims']:
            plt.axvline(lon0, color=style[nm][0], linewidth=2)
    iplot += 1
    row, col = atm.subplot_index(nr, nc, iplot)
    plt.sca(axes[row - 1, col - 1])
    df = usec.to_dataframe()
    df.plot(ax=plt.gca(), style=style, legend=False, linewidth=1.5)
    plt.legend(fontsize=10, handlelength=2.5)
    plt.xticks(xticks)
    plt.ylabel('U (m/s)')
    if row == nr:
        plt.xlabel('Latitude')
    else:
        plt.xlabel('')
        plt.gca().set_xticklabels([])
    plt.title('%d hPa' % plev, fontsize=12)
    plt.grid(True)


# ----------------------------------------------------------------------
# Calculate monthly U, V climatology

years = np.arange(1979, 2016)
datadir = atm.homedir() + '/datastore/merra/monthly/'
varnms = ['U', 'V']
months = range(1, 13)
filestr = datadir + 'merra_%s_1979-2015_%02d.nc'
filestr2 = datadir + 'merra_%s_1979-2015.nc'

for nm in varnms:
    files = [datadir + 'merra_%s_%d.nc' % (nm, yr) for yr in years]
    for month in months:
        var = atm.load_concat(files, nm, concat_dim='year',
                               subset_dict={'month' : (month, month)},
                               squeeze=False)
        var = atm.dim_mean(var, 'year')
        filenm = filestr % (nm, month)
        print('Saving to ' + filenm)
        atm.save_nc(filenm, var)

    # Concatenate months together
    files = [filestr % (nm, month) for month in months]
    var = atm.load_concat(files, nm, concat_dim='month')
    filenm = filestr2 % nm
    print('Saving to ' + filenm)
    atm.save_nc(filenm, var)

# ----------------------------------------------------------------------
# EMFD
datadir = atm.homedir() + 'datastore/merra/daily/'
ds = xray.open_dataset(datadir + 'merra_uv200_40E-120E_60S-60N_1979.nc')
u = atm.squeeze(ds['U'])
v = atm.squeeze(ds['V'])

nroll = 7
u_tr = u - atm.rolling_mean(u, nroll, axis=0)
v_tr = v - atm.rolling_mean(v, nroll, axis=0)

emfd_tr, emfd_tr_x, emfd_tr_y = atm.divergence_spherical_2d(u_tr * u_tr,
                                                            u_tr * v_tr)

# ----------------------------------------------------------------------
# Lat-pres streamfunction
v = merra.read_daily('V', 1979, 7, days=range(1,6),
                     subset_dict={'lon' : (60, 100)})
v = v.mean(dim='TIME')
psi = atm.streamfunction(v)
psibar = psi.mean(dim='XDim')
plt.figure()
atm.contourf_latpres(psibar)

# ----------------------------------------------------------------------
# 01/14/2016 Plots for Simona

# Load data from compare-indices.py

keys = ['HOWI_100', 'OCI', 'SJKE', 'TT',  'WLH_MERRA_PRECIP_nroll7']
shortkeys = ['HOWI', 'OCI', 'SJKE', 'TT',  'WLH']
#shortkeys = [short[key] for key in keys]

years = index[keys[0]].year.values
onset = np.reshape(index[keys[0]].onset.values, (len(years), 1))
for key in keys[1:]:
    ind = np.reshape(index[key].onset.values, (len(years), 1))
    onset = np.concatenate([onset, ind], axis=1)
onset = pd.DataFrame(onset, index=years, columns=shortkeys)

# Add monsoon strength index
ind_comp = onset.copy()
ind_comp['JJAS_MFC'] = strength['MERRA_DET']

# Box plots of onset days
plt.figure()
onset.boxplot()
plt.xlabel('Onset Index')
plt.ylabel('Day of Year')

# ----------------------------------------------------------------------
# Model level MERRA data


varnm = 'T'

xsub = '[330:2:450]'
ysub = '[60:2:301]'
tsub = '[0:1:3]'
lev = 71
zsub = '[%d:1:%d]' % (lev, lev)

def datafile(year, mon, day, varnm, xsub, ysub, zsub, tsub):
    url = ('http://goldsmr3.sci.gsfc.nasa.gov:80/opendap/MERRA/MAI6NVANA.5.2.0/'
           '%d/%02d/MERRA100.prod.assim.inst6_3d_ana_Nv.%d%02d%02d.hdf'
           '?%s%s%s%s%s,XDim%s,YDim%s,Height%s,TIME%s') % (year, mon, year, mon,
           day, varnm, tsub, zsub, ysub, xsub, xsub, ysub, zsub, tsub)
    return url

year = 1979
month = 4
#jdays = atm.season_days(atm.month_str(month), atm.isleap(year))
days = range(1, atm.days_this_month(year, month) + 1)
urls = [datafile(year, month, day, varnm, xsub, ysub, zsub, tsub) for day
        in days]
savedir = atm.homedir() + '/datastore/merra/daily/'
savefile = '%smerra_%s_ML%02d_40-120E_60S-60N_%d%02d.nc' % (savedir, varnm, lev,
                                                            year, month)
var = atm.load_concat(urls, varnm, 'TIME')
print('Saving to ' + savefile)
atm.save_nc(savefile, var)

# for d, day in enumerate(days):
#     url = datafile(year, month, day, varnm, xsub, ysub, zsub, tsub)
#     print('Reading %s' % url)
#     ds = xray.open_dataset(url)
#     var_in = atm.squeeze(ds[varnm])
#     # Daily mean:
#     var_in = var_in.mean(dim='TIME')
#     var_in.coords['Day'] = day
#     if d == 0:
#         var = var_in
#     else:
#         var = xray.concat([var, var_in], dim='Day')
#
#
# T = ds['T']
# T = T[0]


# ----------------------------------------------------------------------

#datadir = '/home/jennifer/datastore/merra/daily/'
datadir = '/home/jwalker/eady/datastore/merra/daily/'
filestr = 'merra_uv200_40E-120E_60S-60N_'
pathstr = datadir + filestr
years = np.arange(1979, 2015)
lon1, lon2 = 60, 100
#lat1, lat2 = 10, 30
lat1, lat2 = 10.625, 10.625
#lat1, lat2 = 0.625, 0.625
#lat1, lat2 = -5.625, -5.625
#lat1, lat2 = -10.625, -10.625

# ----------------------------------------------------------------------
def timeseries_allyears(pathstr, years, lat1, lat2, lon1, lon2):
    """Return the mean_over_geobox of daily data from selected years."""

    def get_year(ds, year, lat1, lat2, lon1, lon2):
        """Get daily data for this year, calculate mean_over_geobox,
        and add NaN if applicable so that non-leap and leap years
        can be concatenated together."""
        dsbar = xray.Dataset()
        nan = np.nan*np.ones((1,1))
        days = np.arange(1,367)
        for nm in ds.data_vars:
            print(nm)
            var = atm.mean_over_geobox(ds[nm], lat1, lat2, lon1, lon2)
            vals = var.values
            if not atm.isleap(year):
                vals = np.concatenate([vals, nan])
            coords = {'Day' : days, 'Height': var.coords['Height'], 'Year': year}
            dsbar[nm] = xray.DataArray(vals, name=var.name, dims=var.dims,
                                       attrs=var.attrs, coords=coords)
        return dsbar

    for i, year in enumerate(years):
        filename = '%s%d.nc' % (pathstr, year)
        print('Loading ' + filename)
        with xray.open_dataset(filename) as ds:
            data = get_year(ds, year, lat1, lat2, lon1, lon2)
        if i == 0:
            dsbar = data
        else:
            dsbar = xray.concat([dsbar, data], dim='Year')
    return dsbar


def plot_timeseries_year(dsbar, year, nroll=None):
    iplot = {'U' : 1, 'V' : 2, 'rel_vort' : 3, 'Ro' : 4}
    plt.figure(figsize=(12, 9))
    plt.suptitle(year)
    for nm in dsbar.data_vars:
        var = dsbar[nm].sel(Year=year)
        plt.subplot(2, 2, iplot[nm])
        plt.plot(var.Day, var, color='gray')
        if nroll is not None:
            data = pd.rolling_mean(np.squeeze(var.values), nroll)
            plt.plot(var.Day, data, color='black')
        plt.title(nm)


# ----------------------------------------------------------------------

dsbar = timeseries_allyears(pathstr, years, lat1, lat2, lon1, lon2)

nroll = 10
for year in [1979, 1980, 1981, 1982]:
    plot_timeseries_year(dsbar, year, nroll)


# ----------------------------------------------------------------------
