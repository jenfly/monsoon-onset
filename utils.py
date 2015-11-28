import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import matplotlib.pyplot as plt

import atmos as atm
import merra

# ----------------------------------------------------------------------
def daily_prepost_onset(data, d_onset, npre, npost, daynm='Day', yearnm='Year'):
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
        sub = atm.subset(data, yearnm, year, None, daynm, dmin, dmax)
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
def comp_days_centered(ndays):
    """Return days for pre/onset/post composites centered on onset.

    Output days are day of year relative to onset day.
    """

    ndays = int(ndays)
    n1 = int(ndays // 2)
    n2 = ndays - n1

    reldays = {}
    reldays['pre'] = np.arange(-n1 - ndays, -n1)
    reldays['onset'] = np.arange(-n1, n2)
    reldays['post'] = np.arange(n2, n2 + ndays)

    return reldays
