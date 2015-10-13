from __future__ import division
import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import atmos as atm


def onset_WLH(precip, axis=1, dt=5.0/365, kann=4, kmax=12, threshold=5.0):
    """Return monsoon onset index computed by Wang & LinHo 2002 method.

    Parameters
    ----------
    precip : ndarray or xray.DataArray
        Pentad precipitation data with pentad as the first or second
        dimension. Maximum 4D: [year, pentad, lat, lon].
    axis : {0, 1}, optional
        Axis corresponding to pentad dimension.
    dt : float, optional
        Time spacing of data.  Default 5/365 corresponds to time units
        of one year.
    kann : int, optional
        Maximum Fourier harmonic to include for smoothed annual cycle.
    kmax : int, optional
        Maximum Fourier harmonic to include for smoothed data used
        to calculate the onset.
    threshold : float, optional
        Threshold for onset/withdrawal criteria.  Same units as precip.

    Returns
    -------

    """
    nmax = 4
    ndim = precip.ndim
    if ndim > nmax:
        raise ValueError('Too many dimensions in precip. Max %dD' % nmax)
    if axis == 0:
        precip = np.expand_dims(precip, 0)
    elif axis != 1:
        raise ValueError('Invalid axis %d. Must be 0 or 1.' % axis)

    precip_fft = atm.Fourier(precip, dt, axis=1)
    precip_ann = precip_fft.smooth(kann)
    precip_sm = precip_fft.smooth(kmax)

    # January mean precip
    weights = np.zeros(precip_sm.shape, dtype=float)
    weights[:, :6] = 5.0 / 31
    weights[:, 6] = 1.0 / 31
    precip_jan = np.mean(precip_sm * weights, axis=1)

    # Relative rainfall rate = rainfall this pentad minus January mean
    precip_rel = precip_sm - np.expand_dims(precip_jan, 1)

    # Add singleton dimension for looping
    while precip_rel.ndim < nmax:
        precip_rel = np.expand_dims(precip_rel, -1)

    # Process a single year and grid point
    def process_one(pcp, threshold):

        if not (pcp > threshold).any():
            i_onset, i_retreat = np.nan, np.nan
        else:
            # Onset index is first pentad exceeding the threshold
            i_onset = np.where(pcp > threshold)[0][0]

            # Retreat index is first pentad after onset below the threshold
            inds = np.where(pcp <= threshold)[0]
            if len(inds) == 0:
                i_retreat = np.nan
            else:
                ind2 = (inds > i_onset).argmax()
                i_retreat = inds[ind2]

        # Peak rainfall rate
        i_peak = pcp.argmax()
        return i_onset, i_retreat, i_peak

    # Calculate indices for each year and grid point
    dims = precip_rel.shape
    dims_out = list(dims)
    dims_out.pop(1)
    onset = np.nan * np.ones(dims_out)
    retreat = np.nan * np.ones(dims_out)
    peak = np.nan * np.ones(dims_out)
    for y in range(dims[0]):
        for i in range(dims[2]):
            for j in range(dims[3]):
                i_onset, i_retreat, i_peak = process_one(precip_rel[y,:,i,j],
                                                         threshold)

                # Onset, retreat, peak pentads (indexed from 1)
                onset[y,i,j] = i_onset + 1
                retreat[y,i,j] = i_retreat + 1
                peak[y,i,j] = i_peak + 1

    # Pack everything into a dict
    output = {}
    output['precip_ann'] = precip_ann
    output['precip_sm'] = precip_sm
    output['precip_rel'] = precip_rel
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

    # Add some more data for diagnostics
    output['precip_fft'] = precip_fft
    output['attrs'] = {}
    output['attrs']['dt'] = dt
    output['attrs']['kann'] = kann
    output['attrs']['kmax'] = kmax
    output['attrs']['threshold'] = threshold

    return output

# ----------------------------------------------------------------------
