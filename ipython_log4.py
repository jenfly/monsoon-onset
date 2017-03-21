# IPython log file

get_ipython().magic(u'paste ')
get_ipython().magic(u'logstart ipython_log4.py')
filenm = '/home/jwalker/datastore/scratch/merra2_U1000_40E-120E_90S-90N_1997.nc'
import xarray as xray
ds = xray.open_dataset(filenm)
ds
ds.max()
ds.min()
var = ds['U']
var.shape
var[0].max()
var[0].min()
var[0].min().values
var[1].max()
var
var[1].max()
for i, day in enumerate(var['day']):
    vals = var[i].values
    vmax = abs(vals).max()
    if vmax > 200:
        print day, vmax
        
for i, day in enumerate(var['day'].values):
    vals = var[i].values
    vmax = abs(vals).max()
    if vmax > 200:
        print day, vmax
        
var.shape
vals = var[:, 100, 50].values
vals
vals.max()
np.nanmax(vals)
np.nanmin(vals)
var
d0 = 273
var0 = var.sel(day=d0)
var0
var1 = var.sel(day=(d0+1))
var1
var1.shape
var1.pcolor_latlon()
atm.pcolor_latlon(var1)
plt.figure()
atm.pcolor_latlon(var0)
var0.shape
var0[0,:]
var0a = atm.subset(var0, {'lat' : (-85, 85)})
plt.figure()
atm.pcolor_latlon(var0a)
plt.clim(-100, 100)
var0[1,:]
var0[2,:]
var0[10,:]
var0[20,:]
var0[40,:]
var0[60,:]
var0[100,:]
var0[150,:]
var0[15var0,:]
var
var.lat
var.lon
filenm
var.max()
var.min()
var[0]
var[1]
var[273].max()
var[272].max()
test = var.copy()
test[272] = test[273]
test
test[272]
test[273]
test[272].max()
test[273].max()
exit()
