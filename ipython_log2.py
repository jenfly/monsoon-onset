# IPython log file

url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T1NXSLV.5.12.4/2016/06/MERRA2_400.tavg1_2d_slv_Nx.20160601.nc4'
get_ipython().magic(u'logstart ipython_log2.py')
get_ipython().magic(u'paste ')
from pydap.client import open_url
from pydap.cas.urs import setup_session
session = setup_session("your_username", "your_pw")
with open('~/.netrc') as f:
    contents = f.read()
    
get_ipython().magic(u'pwd ')
netrc_file = atm.homedir() + '.netrc'
netrc_file = '/home/jwalker/.netrc'
f = open(netrc_file)
f.contents()
with open(netrc_file) as f:
    contents = f.read()
    
contents
contents.split()
with open(netrc_file) as f:
    lines = f.readlines()
    
lines
url
lines[1]
lines[1].replace('\n','')
lines[1].replace('\n','').split()
lines[1].replace('\n','').split()[1]
username = lines[1].replace('\n','').split()[1]
password = lines[2].replace('\n','').split()[1]
username
password
get_ipython().magic(u'pinfo setup_session')
session = setup_session(username, password)
ds = open_url(url, session=session)
ds
ds.keys()
ds['T250']
var = ds['T250']
var.data
dat = var.data
dat.shape
dat[10, 10, 10]
dat[:, 0, 0]
import xarray as xray
get_ipython().magic(u'pinfo xray.open_dataset')
dat.values
dat[:, 0, 0]
vals = dat[:, 0, 0]
vals.flatten()
get_ipython().magic(u'pinfo xray.open_dataset')
ds = xray.open_dataset(url, session=session)
ds = xray.open_dataset(url, engine='pydap')
url
dat[10, 10, 10]
dat[:, 10, 10]
