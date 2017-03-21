# IPython log file

import xarray as xray
xray.backends.api._get_default_engine
get_ipython().magic(u'pinfo xray.backends.api._get_default_engine')
xray.backends.api._get_default_engine()
url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/2016/06/MERRA2_400.tavg1_2d_slv_Nx.20160601.nc4'
xray.backends.api._get_default_engine(url)
url = 'http://iridl.ldeo.columbia.edu/SOURCES/.OSU/.PRISM/.monthly/dods'
xray.backends.api._get_default_engine(url)
ds = xray.open_dataset(url)
ds = xray.open_dataset(url, decode_times=False)
ds
ds.max()
ds['tdmax']
ds
ds['tmax']
ds.close()
ds = xray.open_dataset(url, decode_times=False, engine='pydap')
xray.backends.api._get_default_engine(url)
ds = xray.open_dataset(url, decode_times=False, decode_cf=False, engine='pydap')
get_ipython().magic(u'logstart ipython_log3.py')
pydap
import pydap
get_ipython().magic(u'pinfo pydap.client.open_url')
get_ipython().magic(u'pinfo pydap.cas.urs.setup_session')
import pydap.cas
pydap.cas.urs
import pydap.cas.urs
get_ipython().magic(u'pinfo pydap.cas.urs.setup_session')
import collections
collections.Mapping
get_ipython().magic(u'pinfo collections.Mapping')
url = "https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/hyrax/MERRA2/M2I3NVASM.5.12.4/2015/08/MERRA2_400.inst3_3d_asm_Nv.20150802.nc4"
session
get_ipython().magic(u'paste ')
from pydap.client import open_url
from pydap.cas.urs import setup_session
netrc_file = '/home/jwalker/.netrc'
with open(netrc_file) as f:
    lines = f.readlines()
    
username = lines[1].replace('\n','').split()[1]
password = lines[2].replace('\n','').split()[1]
username
password
session = setup_session(username, password)
ds = open_url(url, session=session)
ds
ds['lat']
ds['lat'][:10]
var = ds['lat']
var
var.data
var.shape
var[0:10]
ds
ds.iteritems()
ds.attributes
ds
ds.children
ds.children()
for nm in ds.children():
    print nm
    
ds.data
ds
var = ds['T']
var.shape
var[:, 0, 0, 0]
ds.keys()
var
var.dimensions
var.maps
ds.maps
var.dimensions
ds.keys()
ds = open_url(url, engine='pydap', session=session)
var
xray.backends.PydapDataStore.open_store_variable(var)
url = 'http://iridl.ldeo.columbia.edu/SOURCES/.OSU/.PRISM/.monthly/dods'
xds = xray.open_dataset(url)
xds = xray.open_dataset(url, decode_times=False)
xds
xds.close()
xds = xray.open_dataset(url, decode_times=False, engine='pydap')
mydict = {'a' : 1, 'b' : 'kittens', 'c' : 'puppies'}
for k, v in mydict.iteritems():
    print k, v
    
mydict.keys()
ds = open_url(url)
ds
var = ds['T']
var
var.data
var.children
var.children()
ds.keys()
xds
xds = xray.open_dataset(url, decode_times=False)
for k, v in xds.iteritems():
    print k, v
    
xds.keys()
ds.keys()
for k, v in mydict.iteritems():
    print k, v
    
ds.keys()
logstatus
log
logstat
get_ipython().magic(u'logstart ipython_log4.py')
import atmos as atm
get_ipython().magic(u'cd ../atmos-tools/')
import atmos as atm
atm.homedir()
exit()
