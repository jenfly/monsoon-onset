import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xarray as xray
import matplotlib.pyplot as plt

import atmos as atm

lon1, lon2 = 60, 100
filenm = '/home/jwalker/datastore/merra2/analysis/merra2_U200_dailyrel_CHP_MFC_1980-2015.nc'
with xray.open_dataset(filenm) as ds:
    u = ds['U'].load()

u = atm.dim_mean(u, 'lon', lon1, lon2)
lat = atm.get_coord(u, 'lat')


def angular_momentum(lat, u):
    omega = atm.constants.Omega.values
    a = atm.constants.radius_earth.values
    coslat = np.cos(np.radians(lat))
    m = omega * (a**2) * (coslat**2) + a * u * coslat
    return m


def model(lat, phi0=0, u0=0):
    omega = atm.constants.Omega.values
    a = atm.constants.radius_earth.values
    coslat = np.cos(np.radians(lat))
    cosphi0 = np.cos(np.radians(phi0))
    um = omega * a * (cosphi0**2 - coslat**2) / coslat
    um = um + u0 * cosphi0 / coslat
    return um


m = angular_momentum(lat, u)

# Plot angular momentum vs. latitude on various days
fmts = ['b--', 'b', 'r--', 'r', 'k--', 'k']
plotdays = [-60, -30, 0, 30, 60, 90]
latmin, latmax = -30, 30

plt.figure()
for day, fmt in zip(plotdays, fmts):
    var = atm.subset(m, {'dayrel' : (day, None), 'lat' : (latmin, latmax)})
    plt.plot(var['lat'], var, fmt, label='Day %d' % day)
plt.grid()
plt.legend(fontsize=10, handlelength=3)
plt.title('200 hPa Angular Momentum')
plt.xlabel('Latitude')
plt.ylabel('Angular Momentum ($m^2 s^{-1}$)')


# Plot angular momentum conserving wind for various values of phi0, u0
plotday = 30
latmin, latmax = -30, 30
fmts = ['b--', 'b', 'k--', 'k']
phi0_list = [0, 10, 20, 30]

uplot = atm.subset(u, {'dayrel' : (plotday, None), 'lat' : (latmin, latmax)})
lat_in = var['lat']
plt.figure()
plt.plot(lat_in, uplot, 'r', label='U Day %d' % plotday)
u0 = 0
for phi0, fmt in zip(phi0_list, fmts):
    label = 'U$_M : \phi_0$ = %d, u$_0$ = %d' % (phi0, u0)
    plt.plot(lat_in, model(lat_in, phi0, u0), fmt, label=label)
plt.legend(fontsize=10, handlelength=3, loc='upper center')
plt.grid()
plt.xlabel('Latitude')
plt.ylabel('Zonal Wind (m/s)')
plt.title('Angular Momentum Conserving Winds')

u0_list = [-5, 0, 5, 10]
phi0 = 12
plt.figure()
plt.plot(lat_in, uplot, 'r', label='U Day %d' % plotday)
for u0, fmt in zip(u0_list, fmts):
    label = 'U$_M : \phi_0$ = %d, u$_0$ = %d' % (phi0, u0)
    plt.plot(lat_in, model(lat_in, phi0, u0), fmt, label=label)
plt.legend(fontsize=10, handlelength=3, loc='upper center')
plt.grid()
plt.xlabel('Latitude')
plt.ylabel('Zonal Wind (m/s)')
plt.title('Angular Momentum Conserving Winds')

