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


# # ----------------------------------------------------------------------
# def days_premidpost(dmid, ndays):
#     """Return days of year for pre, mid, and post-onset composites.
#
#     Parameters
#     ----------
#     dmid : int
#         Day of year of monsoon onset.
#     ndays : int
#         Size of range (number of days) to define composites.
#
#     Returns
#     -------
#     days : dict of arrays
#         Keys are 'pre', 'mid', and 'post'.  Values are arrays of days
#         for each composite.
#     """
#
#     n1 = ndays // 2
#     n2 = ndays - n1
#
#     d0 = int(dmid - n1 - ndays)
#     d1 = int(dmid - n1)
#     d2 = int(dmid + n2)
#     d3 = int(dmid + n2 + ndays)
#
#     days = {}
#     days['pre'] = np.arange(d0, d1)
#     days['mid'] = np.arange(d1, d2)
#     days['post'] = np.arange(d2, d3)
#
#     return days
#
#
# # ----------------------------------------------------------------------
# def composite_premidpost(data, d_onset, ndays, daynm='Day', return_mean=False):
#     """Return pre-, mid- and post-onset composite data fields.
#
#     Parameters
#     ----------
#     data : xray.DataArray
#         Daily data to calculate composites.
#     d_onset : inset
#         Day of year of monsoon onset.
#     ndays : int
#         Size of range (number of days) to define composites.
#     daynm : str, optional
#         Name of dimension in data corresponding to day of year.
#     return_mean : bool, optional
#         If True, return mean of all days in each composite.  Otherwise
#         return the daily values.
#
#     Returns
#     -------
#     comp : dict of xray.DataArrays
#         Keys are 'pre', 'mid', and 'post' and values are DataArrays of
#         the composite data fields.
#     """
#
#     days = days_premidpost(d_onset, ndays)
#
#     comp = {}
#     for key in days:
#         comp[key] = atm.subset(data, daynm, days[key])
#         if return_mean:
#             comp[key] = comp[key].mean(dim=daynm)
#
#     return comp
