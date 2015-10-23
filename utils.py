import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import numpy as np
import xray
import matplotlib.pyplot as plt

import atmos as atm
import merra

# ----------------------------------------------------------------------
def days_premidpost(dmid, n):
    """Return days of year for pre, mid, and post-onset composites.

    Parameters
    ----------
    dmid : int
        Day of year of monsoon onset.
    n : int
        Size of range (number of days) to define composites.

    Returns
    -------
    days : dict of arrays
        Keys are 'pre', 'mid', and 'post'.  Values are arrays of days
        for each composite.
    """

    n1 = n // 2
    n2 = n - n1

    d0 = int(dmid - n1 - n)
    d1 = int(dmid - n1)
    d2 = int(dmid + n2)
    d3 = int(dmid + n2 + n)

    days = {}
    days['pre'] = np.arange(d0, d1)
    days['mid'] = np.arange(d1, d2)
    days['post'] = np.arange(d2, d3)

    return days


# ----------------------------------------------------------------------
def composite_premidpost(data, i_onset, n):
    """Return composite data fields relative to onset index.

    """

    # Get metadata from xray.DataArray

    dims = list(data.shape)
    dims[1] = n
    comp = {}
    comp['pre'] = np.zeros(dims, dtype=float)
    comp['mid'] = np.zeros(dims, dtype=float)
    comp['post'] = np.zeros(dims, dtype=float)

    for y, i in enumerate(i_onset):
        inds = slice_premidpost(i, n)
        for key in comp:
            comp[key][y] = data[y, inds[key]]
    for key in comp:
        comp[key] = comp[key].mean(axis=1)

    # Pack metadata

    return comp
