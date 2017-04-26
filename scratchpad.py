import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xarray as xray
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import atmos as atm
import merra
import indices


# ----------------------------------------------------------------------
# JJAS precip and fraction of annual totals

datadir = atm.homedir() + 'datastore/merra2/figure_data/'
filenm = datadir + 'gpcp_dailyrel_1997-2015.nc'
with xray.open_dataset(filenm) as ds:
    pcp_jjas = ds['PCP_JJAS'].load()
    pcp_frac = ds['FRAC_JJAS'].load()

axlims = (-20, 35, 50, 115)
xticks = range(40, 121, 10)
clev = np.arange(0, 10.5, 1)
plt.figure(figsize=(8, 6))
m = atm.contourf_latlon(pcp_jjas, clev=clev, axlims=axlims, cmap='PuBuGn',
                        extend='max')
plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
atm.contour_latlon(pcp_frac, clev=[0.5], m=m, colors='m', linewidths=1)
atm.geobox(10, 30, 60, 100, m=m, color='b')
plt.xlim(axlims[2], axlims[3])


# ----------------------------------------------------------------------
# Map of monsoon region

m = atm.init_latlon(-50, 50, 40, 120, coastlines=False)
m.shadedrelief(scale=0.3)
yticks = range(-45, 46, 15)
xticks = range(40, 121, 20)
plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
plt.yticks(yticks, atm.latlon_labels(yticks, 'lat'))

atm.geobox(10, 30, 60, 100, m=m, color='k')
plt.savefig('figs/map_box.png', dpi=200)
# ----------------------------------------------------------------------
# Animation of precip and 850 mb winds

datadir = atm.homedir() + 'datastore/merra2/analysis/'

files = {'PREC' : datadir + 'gpcp_dailyrel_CHP_MFC_1997-2015.nc'}
for nm in ['U', 'V']:
    files[nm] = datadir + 'merra2_%s850_dailyrel_CHP_MFC_1980-2015.nc' % nm

ndays = 10
data = {}
for nm in files:
    filenm = files[nm]
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        var = ds[nm].load()
        if 'year' in var:
            var = var.mean(dim='year')
        daydim = atm.get_coord(var, 'dayrel', 'dim')
        data[nm] = atm.rolling_mean(var, ndays, axis=daydim)


def animate(data, day, axlims=(-30, 45, 40, 120), dx=5, dy=5, climits=(-5, 15),
            cmap='BuPu', d0=138, clev=np.arange(5, 15.5, 1),
            cticks=np.arange(5, 16, 2.5)):
    lat1, lat2, lon1, lon2 = axlims
    subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}
    xticks = range(40, 121, 20)
    mm, dd = atm.jday_to_mmdd(day + d0)
    title = (atm.month_str(mm)).capitalize() + ' %d' % dd

    u = atm.subset(data['U'].sel(dayrel=day), subset_dict)
    v = atm.subset(data['V'].sel(dayrel=day), subset_dict)
    u = u[::dy, ::dx]
    v = v[::dy, ::dx]
    #spd = np.sqrt(u**2 + v**2)
    pcp = data['PREC'].sel(dayrel=day)
    lat = atm.get_coord(u, 'lat')
    lon = atm.get_coord(u, 'lon')

    plt.clf()
    m = atm.init_latlon(lat1, lat2, lon1, lon2, coastlines=False)
    m.drawcoastlines(color='k', linewidth=0.5)
    m.shadedrelief(scale=0.3)
    atm.contourf_latlon(pcp, clev=clev, axlims=axlims, m=m, cmap=cmap,
                        extend='max', cb_kwargs={'ticks' : cticks})
    #atm.pcolor_latlon(pcp, axlims=axlims, cmap=cmap, cb_kwargs={'extend' : 'max'})
    plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
    plt.clim(climits)
    #plt.quiver(lon, lat, u, v, linewidths=spd.values.ravel())
    plt.quiver(lon, lat, u, v)
    plt.title(title)
    plt.draw()

# Need to scale arrows in quiver plot so that they are consistent across
# different days

days = range(-90, 201, 1)
for i, day in enumerate(days):
    animate(data, day)
    filenm = 'figs/anim/frame%03d.png' % i
    print('Saving to ' + filenm)
    plt.savefig(filenm)

# ----------------------------------------------------------------------
years = np.arange(1980, 1999)
datadir = atm.homedir() + 'datastore/merra2/dailyrad/'
files = [datadir + 'merra2_RAD_%d.nc4' % yr for yr in years]

 ds = atm.mean_over_files(files)


# ----------------------------------------------------------------------
from pydap.client import open_url

authfile = atm.homedir() + '.netrc'

with open(authfile) as f:
    lines = f.readlines()

username = lines[1].split()[1]
password = lines[2].split()[1]

url = ('https://%s:%s@' % (username, password) +
       'goldsmr5.sci.gsfc.nasa.gov/opendap/MERRA2/M2I3NPASM.5.12.4/'
       '1986/01/MERRA2_100.inst3_3d_asm_Np.19860101.nc4.nc4')

ds = open_url(url)

# ----------------------------------------------------------------------
plev = 200
filestr = '/home/jennifer/datastore/merra2/analysis/merra2_H%d_dailyrel_CHP_MFC_1980-2015.nc'
filenm = filestr % plev
with xray.open_dataset(filenm) as ds:
    ds.load()

lon1, lon2 = 60, 100
hgt = ds['H']
hgt = hgt - atm.dim_mean(hgt, 'lon', lon1, lon2)

if plev == 20:
    climits = (-80, 80)
else:
    climits = (-40, 40)

plotdays = [-30, 0, 30]
xticks = range(40, 121, 20)
axlims = (-60, 60, 40, 120)
nrow, ncol = 1, 3
fig_kw = {'figsize' : (11, 5), 'sharex' : True, 'sharey' : True}
gridspec_kw = {'left' : 0.07, 'right' : 0.9, 'bottom' : 0.07, 'top' : 0.9,
               'wspace' : 0.3}
suptitle = 'H* at %d hPa' % plev
grp = atm.FigGroup(nrow, ncol, fig_kw=fig_kw, suptitle=suptitle,
                   gridspec_kw=gridspec_kw)

for day in plotdays:
    grp.next()
    var = hgt.sel(dayrel=day)
    atm.pcolor_latlon(var, axlims=axlims, cb_kwargs={'extend' : 'both'})
    plt.clim(climits)
    plt.title('Day %d' % day)
    plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
    plt.axvline(lon1, color='k', dashes=[6,1])
    plt.axvline(lon2, color='k', dashes=[6,1])


# ----------------------------------------------------------------------
# JJAS precip and fraction of annual totals

datadir = atm.homedir() + 'datastore/merra2/figure_data/'
filenm = datadir + 'gpcp_dailyrel_1997-2015.nc'
with xray.open_dataset(filenm) as ds:
    pcp_jjas = ds['PCP_JJAS'].load()
    pcp_frac = ds['FRAC_JJAS'].load()

axlims = (-20, 35, 50, 115)
xticks = range(40, 121, 10)
clev = np.arange(0, 10.5, 1)
plt.figure(figsize=(8, 6))
m = atm.contourf_latlon(pcp_jjas, clev=clev, axlims=axlims, cmap='PuBuGn',
                        extend='max')
plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
atm.contour_latlon(pcp_frac, clev=[0.5], m=m, colors='m', linewidths=1)
atm.geobox(10, 30, 60, 100, m=m, color='b')
plt.xlim(axlims[2], axlims[3])


# ----------------------------------------------------------------------
# Map of monsoon region

m = atm.init_latlon(-50, 50, 40, 120, coastlines=False)
m.shadedrelief(scale=0.3)
yticks = range(-45, 46, 15)
xticks = range(40, 121, 20)
plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
plt.yticks(yticks, atm.latlon_labels(yticks, 'lat'))

atm.geobox(10, 30, 60, 100, m=m, color='k')
plt.savefig('figs/map_box.png', dpi=200)
# ----------------------------------------------------------------------
# Animation of precip and 850 mb winds

datadir = atm.homedir() + 'datastore/merra2/analysis/'

files = {'PREC' : datadir + 'gpcp_dailyrel_CHP_MFC_1997-2015.nc'}
for nm in ['U', 'V']:
    files[nm] = datadir + 'merra2_%s850_dailyrel_CHP_MFC_1980-2015.nc' % nm

ndays = 10
data = {}
for nm in files:
    filenm = files[nm]
    print('Loading ' + filenm)
    with xray.open_dataset(filenm) as ds:
        var = ds[nm].load()
        if 'year' in var:
            var = var.mean(dim='year')
        daydim = atm.get_coord(var, 'dayrel', 'dim')
        data[nm] = atm.rolling_mean(var, ndays, axis=daydim)


def animate(data, day, axlims=(-30, 45, 40, 120), dx=5, dy=5, climits=(-5, 15),
            cmap='BuPu', d0=138, clev=np.arange(5, 15.5, 1),
            cticks=np.arange(5, 16, 2.5)):
    lat1, lat2, lon1, lon2 = axlims
    subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}
    xticks = range(40, 121, 20)
    mm, dd = atm.jday_to_mmdd(day + d0)
    title = (atm.month_str(mm)).capitalize() + ' %d' % dd

    u = atm.subset(data['U'].sel(dayrel=day), subset_dict)
    v = atm.subset(data['V'].sel(dayrel=day), subset_dict)
    u = u[::dy, ::dx]
    v = v[::dy, ::dx]
    #spd = np.sqrt(u**2 + v**2)
    pcp = data['PREC'].sel(dayrel=day)
    lat = atm.get_coord(u, 'lat')
    lon = atm.get_coord(u, 'lon')

    plt.clf()
    m = atm.init_latlon(lat1, lat2, lon1, lon2, coastlines=False)
    m.drawcoastlines(color='k', linewidth=0.5)
    m.shadedrelief(scale=0.3)
    atm.contourf_latlon(pcp, clev=clev, axlims=axlims, m=m, cmap=cmap,
                        extend='max', cb_kwargs={'ticks' : cticks})
    #atm.pcolor_latlon(pcp, axlims=axlims, cmap=cmap, cb_kwargs={'extend' : 'max'})
    plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
    plt.clim(climits)
    #plt.quiver(lon, lat, u, v, linewidths=spd.values.ravel())
    plt.quiver(lon, lat, u, v)
    plt.title(title)
    plt.draw()

# Need to scale arrows in quiver plot so that they are consistent across
# different days

days = range(-90, 201, 1)
for i, day in enumerate(days):
    animate(data, day)
    filenm = 'figs/anim/frame%03d.png' % i
    print('Saving to ' + filenm)
    plt.savefig(filenm)

# ----------------------------------------------------------------------
years = np.arange(1980, 1999)
datadir = atm.homedir() + 'datastore/merra2/dailyrad/'
files = [datadir + 'merra2_RAD_%d.nc4' % yr for yr in years]

 ds = atm.mean_over_files(files)


# ----------------------------------------------------------------------
from pydap.client import open_url

authfile = atm.homedir() + '.netrc'

with open(authfile) as f:
    lines = f.readlines()

username = lines[1].split()[1]
password = lines[2].split()[1]

url = ('https://%s:%s@' % (username, password) +
       'goldsmr5.sci.gsfc.nasa.gov/opendap/MERRA2/M2I3NPASM.5.12.4/'
       '1986/01/MERRA2_100.inst3_3d_asm_Np.19860101.nc4.nc4')

ds = open_url(url)

# ----------------------------------------------------------------------
# 11/2/2016 Using pydap and xray to try reading MERRA2_100

from pydap_auth import install_basic_client
 
install_basic_client()

from pydap.client import open_url

url = ('http://goldsmr4.sci.gsfc.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/' +
        '2016/06/MERRA2_400.tavg1_2d_slv_Nx.20160601.nc4.nc4')
ds = open_url(url)


###################

from pydap_cas import install_cas_client
install_cas_client()

from pydap.client import open_url
import xarray

url = ('http://jenfly29:Mozart1981@goldsmr5.sci.gsfc.nasa.gov/opendap/' +
    'MERRA2/M2I3NPASM.5.12.4/1986/01/MERRA2_100.inst3_3d_asm_Np.19860101.nc4.nc4')

ds1 = open_url(url)    # Works but data isn't in xarray format
ds2 = xarray.open_dataset(url, engine='pydap')    # Error message, see attached


# ----------------------------------------------------------------------
# 11/1/2016 MSE budget terms from monthly data

years = range(1980, 1983)
months = range(1, 13)
lon1, lon2 = 60, 100
datadir = '/home/jwalker/datastore/merra2/monthly/'
filestr = datadir + 'MERRA2_100.tavgM_2d_rad_Nx.%d%02d.nc4'
datafiles = {yr : [filestr % (yr, m) for m in months] for yr in years}

def net_rad(rad, weights):
    for i, nm in enumerate(weights):
        if i == 0:
            net = rad[nm] * weights[nm]
        else:
            net = net + rad[nm] * weights[nm]
    net.attrs['long_name'] = 'net_longwave_and_shortwave_into_column'
    return net

def get_year(files, year, months=range(1,13)):
    weights = {'SWTNT' : 1.0, 'LWTUP' : -1.0, 'SWGNT' : -1.0, 'LWGNT' : -1.0}
    nms = weights.keys()

    for i, filenm in enumerate(files):
        month = months[i]
        print('Loading ' + filenm)
        with xray.open_dataset(filenm) as ds:
            rad = atm.squeeze(ds[nms])
            # Change 'time' dimension to 'month' and add 'year' dimension
            for nm in rad.data_vars:
                rad[nm] = atm.expand_dims(rad[nm], 'month', month, axis=0)
                rad[nm] = atm.expand_dims(rad[nm], 'year', year, axis=0)
            rad['NETRAD'] = net_rad(rad, weights)
        if i == 0:
            data = rad
        else:
            data = xray.concat([data, rad], dim='month')
    return data

for i, year in enumerate(years):
    files = datafiles[year]
    ds = get_year(files, year)
    if i == 0:
        data = ds
    else:
        data = xray.concat([data, ds], dim='year')


# Mean along sector and within 2 degrees of equator
latmin, latmax = -2, 2
data_eq = atm.dim_mean(data, 'lon', lon1, lon2)
data_eq = atm.dim_mean(data_eq, 'lat', latmin, latmax)

plotyear = 1980
plt.figure(figsize=(6, 8))
plt.suptitle('%d-%dE RAD (W/m2) at equator' % (lon1, lon2))
plt.subplot(2, 1, 1)
for nm in data_eq.data_vars:
    plt.plot(months, data_eq[nm].sel(year=plotyear), label=nm)
plt.legend(fontsize=8)
plt.title(plotyear)
plt.subplot(2, 1, 2)
for year in years:
    plt.plot(months, data_eq['NETRAD'].sel(year=year), label=year)
plt.plot(months, data_eq['NETRAD'].mean(dim='year'), 'k', linewidth=2,
         label='CLIM')
plt.legend(fontsize=8)
plt.title('NETRAD')
plt.xlabel('Month')

# ----------------------------------------------------------------------
# 10/30/2016 Temporary ubudget climatologies with problem years removed
version = 'merra2'
years = range(1980, 1984) + [1985] + range(1987, 1995) + [1996]
years = years + range(1998, 2008) + range(2009, 2012) + [2014]

onset_nm = 'CHP_MFC'
plevs = [1000,925,850,775,700,600,500,400,300,250,200,150,100,70,50,30,20]

datadir = atm.homedir() + 'datastore/%s/analysis/' % version
savedir = atm.homedir() + 'eady/datastore/%s/analysis/ubudget_temp/' % version
filestr = (version + '_ubudget%d_dailyrel_' + onset_nm +
           '_ndays5_60E-100E')
datafiles = {}
for plev in plevs:
    datafiles[plev] = [datadir + filestr % plev + '_%d.nc' % yr for yr in years]

# Compute climatologies and save
yearstr = '_%d-%d_excl.nc' % (min(years), max(years))
for plev in plevs:
    relfiles = datafiles[plev]
    savefile = savedir + filestr % plev + yearstr
    ds = atm.mean_over_files(relfiles)
    ds.attrs['years'] = years
    print('Saving to ' + savefile)
    ds.to_netcdf(savefile)

#************************ TEMPORARY TROUBLESHOOTING ******************
# filestr = ('/home/jwalker/datastore/merra2/analysis/merra2_ubudget%d_' +
#            'dailyrel_CHP_MFC_ndays5_60E-100E_%d.nc')
#
# for year in years:
#     with open('troubleshooting_%d.txt' % year, 'w') as f1:
#         for plev in plevs:
#             filenm = filestr % (plev, year)
#             print(filenm)
#             f1.write('------------------------------------------\n')
#             f1.write(filenm + '\n')
#             f1.write('Year %d, pressure level %.0f' % (year, plev))
#             with xray.open_dataset(filenm) as ds:
#                 vals = ds.max()
#             biggest = vals.to_array().values.max()
#             f1.write('%.e\n' % biggest)
#             if biggest > 10:
#                 for nm in vals.data_vars:
#                     f1.write('%s\t%.e\n' % (nm, vals[nm]))


# ----------------------------------------------------------------------
# 10/20/2016 Read India state boundaries from geojson file
filenm = 'data/india_state.geojson'
with open(filenm) as f:
    data = json.load(f)

i_region, i_poly = 17, 44
poly = data['features'][i_region]['geometry']['coordinates'][i_poly][0]
arr = np.array(poly)
x, y = arr[:, 0], arr[:, 1]

# Cut out wonky bits
i1, i2 = 8305, 19200
x = np.concatenate((x[:i1], x[i2:]))
y = np.concatenate((y[:i1], y[i2:]))

plt.figure()
atm.init_latlon(5, 20, 70, 85, resolution='l')
plt.plot(x, y)


# ----------------------------------------------------------------------
# 7/13/2016 MERRA2 radiation data

years = 1980
months = 7
#opts = {'vertical' : 'X', 'res' : 'N', 'time_kind' : 'T', 'kind' : 'RAD'}

url_dict = merra.get_urls(years, months=months, version='merra2',
                          varnm='SWGNT', monthly=True)

weights = {'SWTNT' : 1.0, 'LWTUP' : -1.0, 'SWGNT' : -1.0, 'LWGNT' : -1.0}
nms = weights.keys()

def net_rad(rad, weights):
    for i, nm in enumerate(weights):
        if i == 0:
            net = rad[nm] * weights[nm]
        else:
            net = net + rad[nm] * weights[nm]
    return net

url = url_dict.values()[0]
with xray.open_dataset(url) as ds:
    rad = atm.squeeze(ds[nms])
    rad['NET'] = net_rad(rad, weights)

url_dict2 = merra.get_urls(years, months=months, version='merra2',
                           varnm='EFLUX', monthly=True)
url2 = url_dict2.values()[0]
with xray.open_dataset(url2) as ds:
    Fnet = atm.squeeze(ds[['EFLUX', 'HFLUX']])

Fnet['RAD'] = rad['NET']
Fnet['TOT'] = Fnet['EFLUX'] + Fnet['HFLUX'] + Fnet['RAD']

plt.figure()
for i, nm in enumerate(Fnet.data_vars):
    plt.subplot(2, 2, i + 1)
    atm.pcolor_latlon(Fnet[nm])
    plt.title(nm)

h_nms = ['UFLXCPT', 'UFLXPHI', 'UFLXQV', 'VFLXCPT', 'VFLXPHI', 'VFLXQV']
Lv = atm.constants.Lv.values
urls = merra.get_urls(years, months=months, version='merra2', monthly=True,
                      varnm='UFLXCPT')
url3 = urls.values()[0]
with xray.open_dataset(url3) as ds:
    mse = atm.squeeze(ds[h_nms])
for nm in ['UFLXQV', 'VFLXQV']:
    key = nm.replace('QV', 'LQV')
    mse[key] = mse[nm] * Lv
    mse[key].attrs['units'] = mse[nm].attrs['units'].replace('kg', 'J')
mse['UFLXTOT'] = mse['UFLXCPT'] + mse['UFLXPHI'] + mse['UFLXLQV']
mse['VFLXTOT'] = mse['VFLXCPT'] + mse['VFLXPHI'] + mse['VFLXLQV']

mse_div, mse_div_x, mse_div_y = atm.divergence_spherical_2d(mse['UFLXTOT'],
                                                            mse['VFLXTOT'])

var = atm.subset(mse['VFLXTOT'], {'lat' : (-80, 80)})
dvar = atm.subset(mse_div_y, {'lat' : (-80, 80)})
lon0 = 10
val, ind = atm.find_closest(var.lon, lon0)
var0, dvar0 = var[:, ind], dvar[:, ind]

lat = var0.lat.values
lat_rad = np.radians(lat)
coslat = np.cos(lat_rad)
a = atm.constants.radius_earth.values
dy = np.gradient(lat_rad)

plt.figure()
plt.subplot(2, 2, 1)
plt.plot(lat, var0)
plt.subplot(2, 2, 2)
plt.plot(lat, var0 * coslat)
plt.subplot(2, 2, 3)
plt.plot(lat, np.gradient(var0 * coslat, dy))
plt.subplot(2, 2, 4)
plt.plot(lat, np.gradient(var0 * coslat, dy) / (coslat*a))
plt.plot(lat, dvar0, 'r')


plt.figure()
plt.subplot(2, 1, 1)
plt.plot(var.lat, var0)
plt.grid()
plt.subplot(2, 1, 2)
plt.plot(var.lat, dvar0)
plt.grid()

# v . grad(phi)
nms2 = ['U', 'V', 'H']
urls = merra.get_urls(years, months=months, version='merra2', monthly=True,
                      varnm='U')
url4 = urls.values()[0]
with xray.open_dataset(url4) as ds:
    phi_vars = atm.squeeze(ds[nms2])
phi = phi_vars['H'] * atm.constants.g.values


# ----------------------------------------------------------------------
# GPCP daily climatology

years = range(1997, 2015)
datadir = atm.homedir() + 'datastore/gpcp/'
files = [datadir + 'gpcp_daily_%d.nc' % yr for yr in years]
savefile = datadir + 'gpcp_daily_%d-%d.nc' % (min(years), max(years))

pcp = atm.combine_daily_years('PREC', files, years, yearname='year')
pcp = pcp.mean(dim='year')
print('Saving to ' + savefile)
atm.save_nc(savefile, pcp)

day1 = atm.mmdd_to_jday(6, 1)
day2 = atm.mmdd_to_jday(9, 30)
pcp_ssn = atm.subset(pcp, {'day' : (day1, day2)})
pcp_frac = pcp_ssn.sum(dim='day') / pcp.sum(dim='day')

# ----------------------------------------------------------------------
# Data-wrangling for ENSO indices

datadir = atm.homedir() + 'dynamics/python/data/ENSO/'
datafile = datadir + 'enso_sst_monthly.txt'
df = pd.read_table(datafile, skiprows=8, index_col=[0, 1],
                   delim_whitespace=True)
savestr = datadir + 'enso_sst_monthly_%s.csv'
for key in ['NINO1+2', 'NINO3', 'NINO3.4', 'NINO4']:
    savefile = savestr % key.lower().replace('.', '').replace('+', '')
    enso = df.unstack()[key]
    enso.columns = [(atm.month_str(m)).capitalize() for m in range(1, 13)]
    enso.to_csv(savefile)


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
