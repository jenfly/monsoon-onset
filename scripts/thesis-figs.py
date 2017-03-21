import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')
sys.path.append('/home/jwalker/dynamics/python/monsoon-onset')

import numpy as np
import xray
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation
import collections

import atmos as atm
import merra
import indices
import utils

figwidth = 12
style = atm.homedir() + 'dynamics/python/mpl-styles/presentation.mplstyle'
plt.style.use(style)
fontsize = mpl.rcParams['font.size']

# ----------------------------------------------------------------------
pcpfile = '/home/jwalker/datastore/gpcp/gpcp_daily_1997-2014.nc'
datadir = atm.homedir() + 'datastore/merra2/analysis/'
files = {'PREC' : datadir + 'gpcp_dailyrel_CHP_MFC_1997-2015.nc'}
for nm in ['U', 'V']:
    files[nm] = datadir + 'merra2_%s850_dailyrel_CHP_MFC_1980-2015.nc' % nm
mldfile = atm.homedir() + 'datastore/mld/ifremer_mld_DT02_c1m_reg2.0.nc'
indfile = datadir + 'merra2_index_CHP_MFC_1980-2015.nc'

lon1, lon2 = 60, 100
ndays = 5

with xray.open_dataset(pcpfile) as ds:
    pcp = atm.subset(ds, {'day' : (1, 365)})
    pcp.load()

for ssn in ['JAN', 'JUL', 'JJAS']:
    days = atm.season_days(ssn)
    pcp[ssn] = atm.dim_mean(pcp['PREC'], 'day', min(days), max(days))

pcp['ANN'] = pcp['PREC'].sum(dim='day')
pcp_jjas = pcp['PREC'].sel(day=atm.season_days('JJAS')).sum(dim='day')
pcp['FRAC'] = pcp_jjas / pcp['ANN']
pcp['PREC'] = atm.rolling_mean(pcp['PREC'], ndays, axis=0, center=True)
pcp['SECTOR'] = atm.dim_mean(pcp['PREC'], 'lon', lon1, lon2)

# Composites relative to onset day
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

# Mixed layer depths
imonth = 4  # Index for month of May
with xray.open_dataset(mldfile, decode_times=False) as ds:
    mld = ds['mld'][imonth].load()
dims, coords = mld.dims, mld.coords
missval = mld.attrs['mask_value']
vals = mld.values
vals = np.ma.masked_array(vals, vals==missval)
vals = np.ma.filled(vals, np.nan)
mld = xray.DataArray(vals, dims=dims, coords=coords)

# Onset/retreat indices and timeseries
with xray.open_dataset(indfile) as index:
    index.load()
    

# ----------------------------------------------------------------------

# Global precip maps in winter/summer
def precip_global(precip, clev=np.arange(0, 16.5, 1), cmap='hot_r'):
    cticks = range(0, 17, 2)
    m = atm.contourf_latlon(precip, clev=clev, cmap=cmap, extend='max',
                            colorbar=False)
    cb = m.colorbar(ticks=cticks, size='3%')
    cb.ax.set_title('mm/day', fontsize=12)

ssn_dict = {'JAN' : 'January', 'JUL' : 'July'}
fig_kw = {'figsize' : (0.75 * figwidth, 0.8 * figwidth)}
grp = atm.FigGroup(2, 1, fig_kw=fig_kw)
for ssn in ['JAN', 'JUL']:
    grp.next()
    precip_global(pcp[ssn])
    plt.title(ssn_dict[ssn])

# Hovmoller plot of sector mean precip
def hovmoller(precip, clev=np.arange(0, 12.5, 1), cticks=np.arange(0, 12.5, 2),
              cmap='hot_r', ylimits=(-40, 40)):
    lat = atm.get_coord(precip, 'lat')
    days = atm.get_coord(precip, 'day')
    plt.contourf(days, lat, precip.T, clev, cmap=cmap, extend='max')
    cb = plt.colorbar(ticks=cticks)
    cb.ax.set_title('mm/day', fontsize=12)
    plt.ylim(ylimits)
    plt.xlim(2, 365)
    plt.ylabel('Latitude')
    plt.xlabel('Day of Year')

plt.figure(figsize=(0.8 * figwidth, 0.4*figwidth))
hovmoller(pcp['SECTOR'])

# Map of monsoon region
plt.figure(figsize=(0.4*figwidth, 0.6*figwidth))
m = atm.init_latlon(-50, 50, 40, 120, coastlines=False)
m.shadedrelief(scale=0.3)
yticks = range(-45, 46, 15)
xticks = range(40, 121, 20)
plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
plt.yticks(yticks, atm.latlon_labels(yticks, 'lat'))
#atm.geobox(10, 30, 60, 100, m=m, color='k')


# JJAS precip and fraction of annual totals
axlims = (-15, 35, 50, 115)
xticks = range(40, 121, 10)
clev = np.arange(0, 18.5, 1)
plt.figure(figsize=(0.8*figwidth, 0.5*figwidth))
m = atm.init_latlon(axlims[0], axlims[1], axlims[2], axlims[3], resolution='l')
atm.contourf_latlon(pcp['JJAS'], clev=clev, m=m, cmap='hot_r', extend='max')
plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
_, cs = atm.contour_latlon(pcp['FRAC'], clev=[0.5], m=m, colors='b',
                           linewidths=2)
label_locs = [(65, 12)]
cs_opts = {'fmt' : '%.1f', 'fontsize' : fontsize, 'manual' : label_locs}
plt.clabel(cs, **cs_opts)
atm.geobox(10, 30, 60, 100, m=m, color='g')
plt.xlim(axlims[2], axlims[3])

# Mixed layer depths
def mld_map(mld, cmap='Blues', axlims=(0, 35, 58, 102), climits=(10, 60),
            cticks=range(10, 71, 10), clevs=None):
    cb_kwargs = {'ticks' : cticks, 'extend' : 'both'}
    m = atm.init_latlon(axlims[0], axlims[1], axlims[2], axlims[3],
                        resolution='l', coastlines=False,
                        fillcontinents=True)
    m.drawcoastlines(linewidth=0.5, color='0.5')
    atm.pcolor_latlon(mld, m=m, cmap=cmap, cb_kwargs=cb_kwargs)
    plt.clim(climits)

lat0 = 15.5
plt.figure(figsize=(0.5*figwidth, 0.35*figwidth))
mld_map(mld)
plt.axhline(lat0, color='k')


# ------------------------------------------------------------------------
# Animation of precip and winds
def animate(i):
    days = range(-136, 227, 1)
    day = days[i]
    axlims=(-30, 45, 40, 120)
    dx, dy = 5, 5
    climits=(0, 20)
    cmap = 'hot_r'
    d0 = 138
    cticks=np.arange(4, 21, 2)
    scale = 250
    clev=np.arange(4, 20.5, 1)
    lat1, lat2, lon1, lon2 = axlims
    subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}
    xticks = range(40, 121, 20)
    yticks = range(-20, 41, 10)
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
    plt.yticks(yticks, atm.latlon_labels(yticks, 'lat'))
    plt.clim(climits)
    #plt.quiver(lon, lat, u, v, linewidths=spd.values.ravel())
    plt.quiver(lon, lat, u, v, scale=scale, pivot='middle')
    plt.title(title)
    plt.draw()

fig = plt.figure()
days = range(-136, 227, 1)
#anim = animation.FuncAnimation(fig, animate,  frames=len(days),
#                              interval=20, blit=True)
#anim = animation.FuncAnimation(fig, animate,  frames=len(days))
anim = animation.FuncAnimation(fig, animate,  frames=30)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
writer=animation.FFMpegWriter(bitrate=500)
print('Saving animation')
anim.save('figs/anim/test.mp4', writer=writer, fps=30)
print('Done')

# --------------------------------------------------------------------------
# def animate(data, day, axlims=(-30, 45, 40, 120), dx=5, dy=5, climits=(0, 20),
#             cmap='hot_r', d0=138, clev=np.arange(4, 20.5, 1),
#             cticks=np.arange(4, 21, 2), scale=250):
#     lat1, lat2, lon1, lon2 = axlims
#     subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}
#     xticks = range(40, 121, 20)
#     yticks = range(-20, 41, 10)
#     mm, dd = atm.jday_to_mmdd(day + d0)
#     title = (atm.month_str(mm)).capitalize() + ' %d' % dd
#
#     u = atm.subset(data['U'].sel(dayrel=day), subset_dict)
#     v = atm.subset(data['V'].sel(dayrel=day), subset_dict)
#     u = u[::dy, ::dx]
#     v = v[::dy, ::dx]
#     #spd = np.sqrt(u**2 + v**2)
#     pcp = data['PREC'].sel(dayrel=day)
#     lat = atm.get_coord(u, 'lat')
#     lon = atm.get_coord(u, 'lon')
#
#     plt.clf()
#     m = atm.init_latlon(lat1, lat2, lon1, lon2, coastlines=False)
#     m.drawcoastlines(color='k', linewidth=0.5)
#     m.shadedrelief(scale=0.3)
#     atm.contourf_latlon(pcp, clev=clev, axlims=axlims, m=m, cmap=cmap,
#                         extend='max', cb_kwargs={'ticks' : cticks})
#     #atm.pcolor_latlon(pcp, axlims=axlims, cmap=cmap, cb_kwargs={'extend' : 'max'})
#     plt.xticks(xticks, atm.latlon_labels(xticks, 'lon'))
#     plt.yticks(yticks, atm.latlon_labels(yticks, 'lat'))
#     plt.clim(climits)
#     #plt.quiver(lon, lat, u, v, linewidths=spd.values.ravel())
#     plt.quiver(lon, lat, u, v, scale=scale, pivot='middle')
#     plt.title(title)
#     plt.draw()
#
#
# days = range(-136, 227, 1)
# plt.figure()
# for i, day in enumerate(days):
#     animate(data, day)
#     filenm = 'figs/anim/frame%03d.png' % i
#     print('Saving to ' + filenm)
#     plt.savefig(filenm)
