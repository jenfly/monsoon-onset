from __future__ import division
import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import atmos as atm


def onset_WLH_1D(precip_sm, threshold=5.0):
    """Return monsoon onset index computed by Wang & LinHo 2002 method.

    For a single pentad timeseries (e.g. one year of pentads at one grid
    point).

    Parameters
    ----------
    precip : 1-D array
        Smoothed pentad precipitation data.
    threshold : float, optional
        Threshold for onset/withdrawal criteria.  Same units as precip.

    Returns
    -------
    i_onset, i_retreat, i_peak : float
        Pentad index of monsoon onset, retreat and peak, or np.nan if
        data does not fit the criteria for monsoon.  Indexed from 0.
    """
    # January mean precip
    weights = np.zeros(precip_sm.shape, dtype=float)
    weights[:6] = 5.0 / 31
    weights[6] = 1.0 / 31
    precip_jan = np.mean(precip_sm * weights)

    precip_rel = precip_sm - precip_jan

    above = (precip_rel > threshold).any()
    below = (precip_rel < threshold).any()
    if not above or not below:
        i_onset, i_retreat, i_peak = np.nan, np.nan, np.nan
    else:
        # Onset index is first pentad exceeding the threshold
        i_onset = np.where(precip_rel > threshold)[0][0]

        # Retreat index is first pentad after onset below the threshold
        inds = np.where(precip_rel <= threshold)[0]
        if len(inds) == 0:
            i_retreat = np.nan
        else:
            ind2 = (inds > i_onset).argmax()
            i_retreat = inds[ind2]

        # Peak rainfall rate
        i_peak = precip_rel.argmax()

    return i_onset, i_retreat, i_peak


def onset_WLH(precip, axis=1, kmax=12, threshold=5.0):
    """Return monsoon onset index computed by Wang & LinHo 2002 method.


    Smooths multi-dimensional pentad precipitation data and computes
    onset indices at each point.

    Parameters
    ----------
    precip : ndarray
        Pentad precipitation data with pentad as the first or second
        dimension. Maximum 4D: [year, pentad, lat, lon].
    axis : {0, 1}, optional
        Axis corresponding to pentad dimension.
    kmax : int, optional
        Maximum Fourier harmonic for smoothing the input data.
    threshold : float, optional
        Threshold for onset/withdrawal criteria.  Same units as precip.

    Returns
    -------
    output : dict
        Dict with the following fields:
          precip_sm : ndarray, smoothed precip data
          onset : ndarray, pentad index of onset
          retreat : ndarray, pentad index of retreat
          peak : ndarray, pentad index of peak rainfall
          smoothing_kmax, threshold : values used in computation
        Pentads are indexed 0-72.
    """
    nmax = 4
    ndim = precip.ndim
    if ndim > nmax:
        raise ValueError('Too many dimensions in precip. Max %dD' % nmax)
    if axis == 0:
        precip = np.expand_dims(precip, 0)
    elif axis != 1:
        raise ValueError('Invalid axis %d. Must be 0 or 1.' % axis)

    precip_fft = atm.Fourier(precip, axis=1)
    precip_sm = precip_fft.smooth(kmax)

    # Add singleton dimension for looping
    while precip_sm.ndim < nmax:
        precip_sm = np.expand_dims(precip_sm, -1)

    # Calculate indices for each year and grid point
    dims = precip_sm.shape
    dims_out = list(dims)
    dims_out.pop(1)
    onset = np.nan * np.ones(dims_out)
    retreat = np.nan * np.ones(dims_out)
    peak = np.nan * np.ones(dims_out)
    for y in range(dims[0]):
        for i in range(dims[2]):
            for j in range(dims[3]):
                inds = onset_WLH_1D(precip_sm[y,:,i,j], threshold)
                onset[y,i,j] = inds[0]
                retreat[y,i,j] = inds[1]
                peak[y,i,j] = inds[2]

    # Pack everything into a dict
    output = {}
    output['precip_sm'] = precip_sm
    output['onset'] = onset
    output['retreat'] = retreat
    output['peak'] = peak

    # Collapse any extra dimensions that were added
    if axis == 0:
        for key in output:
            output[key] = atm.collapse(0, output[key])
    while onset.ndim > ndim:
        for key in output:
            output[key] = atm.collapse(-1, output[key])

    # Add some metadata
    output['smoothing_kmax'] = kmax
    output['threshold'] = threshold

    return output

# ----------------------------------------------------------------------
