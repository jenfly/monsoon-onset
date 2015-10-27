from __future__ import division

import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import atmos as atm
import precipdat

def onset_WLH_1D(precip_sm, threshold=5.0, onset_min=20):
    """Return monsoon onset index computed by Wang & LinHo 2002 method.

    For a single pentad timeseries (e.g. one year of pentads at one grid
    point).

    Parameters
    ----------
    precip : 1-D array
        Smoothed pentad precipitation data.
    threshold : float, optional
        Threshold for onset/withdrawal criteria.  Same units as precip.
    onset_min: int, optional
        Minimum pentad index allowed for onset.

    Returns
    -------
    i_onset, i_retreat, i_peak : float
        Pentad index of monsoon onset, retreat and peak, or np.nan if
        data does not fit the criteria for monsoon.  Indexed from 0.

    Reference
    ---------
    Wang, B., & LinHo. (2002). Rainy Season of the Asian-Pacific
        Summer Monsoon. Journal of Climate, 15(4), 386-398.
    """
    # January mean precip
    weights = np.zeros(precip_sm.shape, dtype=float)
    weights[:6] = 5.0 / 31
    weights[6] = 1.0 / 31
    weights = np.ma.masked_array(weights, np.isnan(precip_sm))
    weights = weights / np.sum(weights)
    precip_jan = np.mean(precip_sm * weights)
    precip_rel = precip_sm - precip_jan
    precip_rel = np.ma.masked_array(precip_rel, np.isnan(precip_rel))

    pentads = np.arange(len(precip_sm))
    above = (precip_rel > threshold) & (pentads >= onset_min)
    below = (precip_rel < threshold) & (pentads >= onset_min)
    if not above.any() or not below.any():
        i_onset, i_retreat, i_peak = np.nan, np.nan, np.nan
    else:
        # Onset index is first pentad exceeding the threshold
        i_onset = np.where(above)[0][0]

        # Peak rainfall rate
        i_peak = precip_rel.argmax()

        # Retreat index is first pentad after peak below the threshold
        inds = np.where((precip_rel < threshold) & (pentads > i_peak))[0]
        if len(inds) == 0:
            i_retreat = np.nan
        else:
            ind2 = (inds > i_onset).argmax()
            i_retreat = inds[ind2]

    return i_onset, i_retreat, i_peak


# ----------------------------------------------------------------------
def onset_WLH(precip, axis=1, kmax=12, threshold=5.0, onset_min=20):
    """Return monsoon onset index computed by Wang & LinHo 2002 method.

    Smoothes multi-dimensional pentad precipitation data and computes
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
    onset_min: int, optional
        Minimum pentad index allowed for onset.

    Returns
    -------
    output : dict
        Dict with the following fields:
          precip_sm : ndarray, smoothed precip data
          Rsq : R-squared for smoothed data
          onset : ndarray, pentad index of onset
          retreat : ndarray, pentad index of retreat
          peak : ndarray, pentad index of peak rainfall
          smoothing_kmax, threshold : values used in computation
        Pentads are indexed 0-72.

    Reference
    ---------
    Wang, B., & LinHo. (2002). Rainy Season of the Asian-Pacific
        Summer Monsoon. Journal of Climate, 15(4), 386-398.
    """
    nmax = 4
    ndim = precip.ndim
    if ndim > nmax:
        raise ValueError('Too many dimensions in precip. Max %dD' % nmax)

    # Smooth with truncated Fourier series
    precip_sm, Rsq = atm.fourier_smooth(precip, kmax, axis=axis)

    # Add singleton dimension for looping
    if axis == 0:
        precip_sm = np.expand_dims(precip_sm, 0)
    elif axis != 1:
        raise ValueError('Invalid axis %d. Must be 0 or 1.' % axis)
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
                inds = onset_WLH_1D(precip_sm[y,:,i,j], threshold, onset_min)
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
            if key != 'Rsq':
                output[key] = atm.collapse(-1, output[key])

    # Some more data to output
    output['Rsq'] = Rsq
    output['smoothing_kmax'] = kmax
    output['threshold'] = threshold

    return output


# ----------------------------------------------------------------------
def onset_HOWI(uq_int, vq_int, npts=50, nroll=7, days_pre=range(138, 145),
               days_post=range(159, 166), yearnm='year', daynm='day'):
    """Return monsoon Hydrologic Onset/Withdrawal Index.

    Parameters
    ----------
    uq_int, vq_int : xray.DataArrays
    npts : int, optional
    nroll : int, optional
    days_pre, days_post : list of ints, optional
        Default values correspond to May 18-24 and June 8-14 (numbered
        as non-leap year).
    yearnm, daynm : str, optional

    Returns
    -------

    Reference
    ---------
    J. Fasullo and P. J. Webster, 2003: A hydrological definition of
        Indian monsoon onset and withdrawal. J. Climate, 16, 3200-3211.
    """

    _, _, coords, _ = atm.meta(uq_int)
    latnm = atm.get_coord(uq_int, 'lat', 'name')
    lonnm = atm.get_coord(uq_int, 'lon', 'name')

    ds = xray.Dataset()
    ds['uq'] = uq_int
    ds['vq'] = vq_int
    ds['vimt'] = np.sqrt(ds['uq']**2 + ds['vq']**2)

    # Climatological moisture fluxes
    dsbar = ds.mean(dim=yearnm)
    ds['uq_bar'], ds['vq_bar'] = dsbar['uq'], dsbar['vq']
    ds['vimt_bar'] = np.sqrt(ds['uq_bar']**2 + ds['vq_bar']**2)

    # Pre- and post- monsoon climatology composites
    dspre = atm.subset(dsbar, daynm, days_pre).mean(dim=daynm)
    dspost = atm.subset(dsbar, daynm, days_post).mean(dim=daynm)
    dsdiff = dspost - dspre
    ds['uq_bar_pre'], ds['vq_bar_pre'] = dspre['uq'], dspre['vq']
    ds['uq_bar_post'], ds['vq_bar_post'] = dspost['uq'], dspost['vq']
    ds['uq_bar_diff'], ds['vq_bar_diff'] = dsdiff['uq'], dsdiff['vq']

    # Magnitude of vector difference
    vimt_bar_diff = np.sqrt(dsdiff['uq']**2 + dsdiff['vq']**2)
    ds['vimt_bar_diff'] = vimt_bar_diff

    # Top N difference vectors
    def top_n(data, n):
        """Return a mask with the highest n values in 2D array."""
        vals = data.copy()
        mask = np.ones(vals.shape, dtype=bool)
        for k in range(n):
            i, j = np.unravel_index(np.nanargmax(vals), vals.shape)
            mask[i, j] = False
            vals[i, j] = np.nan
        return mask

    # Mask to extract top N points
    mask = top_n(vimt_bar_diff, npts)
    ds['mask'] = xray.DataArray(mask, coords={latnm: coords[latnm],
                                              lonnm: coords[lonnm]})

    # Apply mask to DataArrays
    def applymask(data, mask):
        _, _, coords, _ = atm.meta(data)
        maskbig = atm.biggify(mask, data, tile=True)
        vals = np.ma.masked_array(data, maskbig).filled(np.nan)
        data_out = xray.DataArray(vals, coords=coords)
        return data_out

    ds['vimt_bar_masked'] = applymask(ds['vimt_bar'], mask)
    ds['vimt_bar_diff_masked'] = applymask(vimt_bar_diff, mask)
    ds['uq_masked'] = applymask(ds['uq'], mask)
    ds['vq_masked'] = applymask(ds['vq'], mask)
    ds['vimt_masked'] = np.sqrt(ds['uq_masked']**2 + ds['vq_masked']**2)

    # Timeseries data averaged over selected N points
    ds['howi_clim_raw'] = ds['vimt_bar_masked'].mean(dim=latnm).mean(dim=lonnm)
    ds['howi_raw'] = ds['vimt_masked'].mean(dim=latnm).mean(dim=lonnm)

    # Normalize
    howi_min = ds['howi_clim_raw'].min().values
    howi_max = ds['howi_clim_raw'].max().values
    def applynorm(data):
        return 2 * (data - howi_min) / (howi_max - howi_min) - 1
    ds['howi_norm'] = applynorm(ds['howi_raw'])
    ds['howi_clim_norm'] = applynorm(ds['howi_clim_raw'])

    # Apply n-day rolling mean
    def rolling(data, nroll):
        center = True
        _, _, coords, _ = atm.meta(data)
        dims = data.shape
        vals = np.zeros(dims)
        if len(dims) > 1:
            nyears = dims[0]
            for y in range(nyears):
                vals[y] = pd.rolling_mean(data.values[y], nroll, center=center)
        else:
            vals = pd.rolling_mean(data.values, nroll, center=center)
        data_out = xray.DataArray(vals, coords=coords)
        return data_out

    ds['howi_norm_roll'] = rolling(ds['howi_norm'], nroll)
    ds['howi_clim_norm_roll'] = rolling(ds['howi_clim_norm'], nroll)

    # Index timeseries dataset
    howi = xray.Dataset()
    howi['tseries'] = ds['howi_norm_roll']
    howi['tseries_clim'] = ds['howi_clim_norm_roll']

    # Find zero crossings for onset and withdrawal indices
    nyears = len(howi[yearnm])
    onset = np.zeros(nyears, dtype=int)
    retreat = np.zeros(nyears, dtype=int)
    for y in range(nyears):
        #monsoon = howi[daynm].values[howi['tseries'][y].values > 0]
        pos = howi[daynm].values[howi['tseries'][y].values > 0]

        # In case of extra zero crossings, find the longest set of days
        # with positive index
        splitpos = atm.splitdays(pos)
        lengths = np.array([len(v) for v in splitpos])
        monsoon = splitpos[lengths.argmax()]
        onset[y] = monsoon[0]
        retreat[y] = monsoon[-1] + 1

        # # Retreat = first day where index falls below zero
        # consec = np.diff(monsoon) == 1
        # if consec.all():
        #     retreat[y] = monsoon[-1]
        # else:
        #     retreat[y] = monsoon[consec.argmin()] + 1
    howi['onset'] = xray.DataArray(onset, coords={yearnm : howi[yearnm]})
    howi['retreat'] = xray.DataArray(retreat, coords={yearnm : howi[yearnm]})
    howi.attrs['npts'] = npts
    howi.attrs['nroll'] = nroll

    return howi, ds
