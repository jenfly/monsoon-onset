import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import pandas as pd
import xray
import matplotlib.pyplot as plt
import collections

import atmos as atm
import merra
import indices

# ----------------------------------------------------------------------
def daily_rel2onset(data, d_onset, npre, npost, daynm='Day', yearnm='Year'):
    """Return subset of daily data aligned relative to onset day.

    Parameters
    ----------
    data : xray.DataArray
        Daily data.
    d_onset : ndarray
        Array of onset date (day of year) for each year.
    npre, npost : int
        Number of days before and after onset to extract.
    daynm, yearnm : str, optional
        Name of day and year dimensions in data.

    Returns
    -------
    data_out : xray.DataArray
        Subset of N days of daily data for each year, where
        N = npre + npost + 1 and the day dimension is
        dayrel = day - d_onset.
    """

    name, attrs, coords, dimnames = atm.meta(data)
    years = atm.get_coord(data, coord_name=yearnm)

    if isinstance(d_onset, xray.DataArray):
        d_onset = d_onset.values

    dayrel = np.arange(-npre, npost + 1)
    relnm = daynm + 'rel'

    for y, year in enumerate(years):
        dmin, dmax = d_onset[y] - npre, d_onset[y] + npost
        subset_dict = {yearnm : (year, None), daynm : (dmin, dmax)}
        sub = atm.subset(data, subset_dict)
        sub = sub.rename({daynm : relnm})
        sub[relnm] = dayrel
        sub[relnm].attrs['long_name'] = 'Day of year relative to onset day'
        if y == 0:
            data_out = sub
        else:
            data_out = xray.concat([data_out, sub], dim=yearnm)

    data_out.attrs['d_onset'] = d_onset

    return data_out


# ----------------------------------------------------------------------
def comp_days_centered(ndays, offset=0):
    """Return days for pre/onset/post composites centered on onset.

    Parameters
    ----------
    ndays : int
        Number of days to average in each composite.
    offset : int, optional
        Number of offset days between pre/onset and onset/post
        day ranges.

    Returns
    -------
    reldays : dict of arrays
        Components are 'pre', 'onset',  and 'post', arrays of days
        of the year relative to onset day, for each composite.
    """

    ndays = int(ndays)
    n1 = int(ndays // 2)
    n2 = ndays - n1

    reldays = collections.OrderedDict()
    reldays['pre'] = np.arange(-offset - n1 - ndays, -offset - n1)
    reldays['onset'] = np.arange(-n1, n2)
    reldays['post'] = np.arange(offset + n2, offset + n2 + ndays)

    return reldays


# ----------------------------------------------------------------------
def composite(data, compdays, return_avg=True, daynm='Dayrel'):
    """Return composite data fields for selected days.

    Parameters
    ----------
    data : xray.DataArray
        Daily data to composite.
    compdays: dict of arrays or lists
        Lists of days to include in each composite.
    return_avg : bool, optional
        If True, return the mean of the selected days, otherwise
        return the extracted individual days for each composite.
    daynnm : str, optional
        Name of day dimension in data.

    Returns
    -------
    comp : dict of xray.DataArrays
        Composite data fields for each key in compdays.keys().
    """

    comp = collections.OrderedDict()
    _, attrs, _, _ = atm.meta(data)

    for key in compdays:
        comp[key] = atm.subset(data, {daynm : (compdays[key], None)})
        if return_avg:
            comp[key] = comp[key].mean(dim=daynm)
            comp[key].attrs = attrs
            comp[key].attrs[daynm] = compdays[key]

    return comp


# ----------------------------------------------------------------------
def get_mfc_box(mfcfiles, precipfiles, years, nroll, lat1, lat2, lon1, lon2):
    """Return daily tseries MFC and precip averaged over lat-lon box.
    """
    subset_dict = {'lat' : (lat1, lat2), 'lon' : (lon1, lon2)}

    databox = {}
    if mfcfiles is not None:
        mfc = atm.combine_daily_years('MFC', mfcfiles, years, yearname='year',
                                       subset_dict=subset_dict)
        databox['MFC'] = mfc
    if precipfiles is not None:
        pcp = atm.combine_daily_years('PRECTOT', precipfiles, years, yearname='year',
                                      subset_dict=subset_dict)
        databox['PCP'] = pcp

    nms = databox.keys()
    for nm in nms:
        var = databox[nm]
        var = atm.precip_convert(var, var.attrs['units'], 'mm/day')
        var = atm.mean_over_geobox(var, lat1, lat2, lon1, lon2)
        databox[nm + '_UNSM'] = var
        databox[nm + '_ACC'] = np.cumsum(var, axis=1)
        if nroll is None:
            databox[nm] = var
        else:
            databox[nm] = atm.rolling_mean(var, nroll, axis=-1, center=True)

    tseries = xray.Dataset(databox)
    return tseries


# ----------------------------------------------------------------------
def get_onset_indices(onset_nm, datafiles, years, data=None):
    """Return monsoon onset/retreat/length indices.
    """

    # Options for CHP_MFC and CHP_PCP
    lat1, lat2 = 10, 30
    lon1, lon2 = 60, 100
    chp_opts = [None, lat1, lat2, lon1, lon2]

    if onset_nm == 'HOWI':
        maxbreak = 10
        npts = 100
        ds = atm.combine_daily_years(['uq_int', 'vq_int'],vimtfiles, years,
                                     yearname='year')
        index, _ = indices.onset_HOWI(ds['uq_int'], ds['vq_int'], npts,
                                      maxbreak=maxbreak)
        index.attrs['title'] = 'HOWI (N=%d)' % npts
    elif onset_nm == 'CHP_MFC':
        if data is None:
            tseries = get_mfc_box(datafiles, None, years, *chp_opts)
            data = tseries['MFC_ACC']
        index = indices.onset_changepoint(data)
    elif onset_nm == 'CHP_PCP':
        if data is None:
            tseries = get_mfc_box(None, datafiles, years, *chp_opts)
            data = tseries['PCP_ACC']
        index = indices.onset_changepoint(data)

    # Monsoon retreat and length indices
    if 'retreat' in index:
        index['length'] = index['retreat'] - index['onset']
    else:
        index['retreat'] = np.nan * index['onset']
        index['length'] = np.nan * index['onset']

    return index

# ----------------------------------------------------------------------
def get_enso_indices(years,
                     inds=['ONI_MAM', 'ONI_JJA', 'MEI_MARAPR', 'MEI_JULAUG'],
                     ensofiles=None):
    """Return ENSO indices.
    """

    if ensofiles is None:
        ensodir = atm.homedir() + 'dynamics/calc/ENSO/'
        ensofiles = {'MEI' : ensodir + 'enso_mei.csv',
                     'ONI' : ensodir + 'enso_oni.csv'}


    enso_in = {}
    for key in ensofiles:
        enso_in[key] = pd.read_csv(ensofiles[key], index_col=0)

    enso = pd.DataFrame()
    for key in enso_in:
        for ssn in enso_in[key]:
            enso[key + '_' + ssn] = enso_in[key][ssn]

    enso = enso.loc[enso.index.isin(years)]
    enso = enso[inds]

    return enso
















----------------
