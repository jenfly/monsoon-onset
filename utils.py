import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import matplotlib.pyplot as plt

import atmos as atm
import merra

# ----------------------------------------------------------------------
def days_premidpost(dmid, ndays):
    """Return days of year for pre, mid, and post-onset composites.

    Parameters
    ----------
    dmid : int
        Day of year of monsoon onset.
    ndays : int
        Size of range (number of days) to define composites.

    Returns
    -------
    days : dict of arrays
        Keys are 'pre', 'mid', and 'post'.  Values are arrays of days
        for each composite.
    """

    n1 = ndays // 2
    n2 = ndays - n1

    d0 = int(dmid - n1 - ndays)
    d1 = int(dmid - n1)
    d2 = int(dmid + n2)
    d3 = int(dmid + n2 + ndays)

    days = {}
    days['pre'] = np.arange(d0, d1)
    days['mid'] = np.arange(d1, d2)
    days['post'] = np.arange(d2, d3)

    return days


# ----------------------------------------------------------------------
def composite_premidpost(data, d_onset, ndays, daynm='Day', return_mean=False):
    """Return pre-, mid- and post-onset composite data fields.

    Parameters
    ----------
    data : xray.DataArray
        Daily data to calculate composites.
    d_onset : inset
        Day of year of monsoon onset.
    ndays : int
        Size of range (number of days) to define composites.
    daynm : str, optional
        Name of dimension in data corresponding to day of year.
    return_mean : bool, optional
        If True, return mean of all days in each composite.  Otherwise
        return the daily values.

    Returns
    -------
    comp : dict of xray.DataArrays
        Keys are 'pre', 'mid', and 'post' and values are DataArrays of
        the composite data fields.
    """

    days = days_premidpost(d_onset, ndays)

    comp = {}
    for key in days:
        comp[key] = atm.subset(data, daynm, days[key])
        if return_mean:
            comp[key] = comp[key].mean(dim=daynm)

    return comp
