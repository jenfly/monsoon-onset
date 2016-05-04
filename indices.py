from __future__ import division

import sys
sys.path.append('/home/jwalker/dynamics/python/atmos-tools')
sys.path.append('/home/jwalker/dynamics/python/atmos-read')

import xray
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import collections
import pandas as pd
import atmos as atm
import precipdat

def onset_WLH_1D(precip_sm, threshold=5.0, onset_min=20, precip_jan=None):
    """Return monsoon onset index computed by Wang & LinHo 2002 method.

    For a single pentad timeseries (e.g. one year of pentads at one grid
    point). Can be used on a daily timeseries, but the January mean
    precip for each year must be specified in the input (the code here
    calculates January mean assuming pentad data).

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
    if precip_jan is None:
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
            output[key] = atm.collapse(output[key], 0)
    while onset.ndim > ndim:
        for key in output:
            if key != 'Rsq':
                output[key] = atm.collapse(output[key], -1)

    # Some more data to output
    output['Rsq'] = Rsq
    output['smoothing_kmax'] = kmax
    output['threshold'] = threshold

    return output


# ----------------------------------------------------------------------
def onset_HOWI(uq_int, vq_int, npts=50, nroll=7, days_pre=range(138, 145),
               days_post=range(159, 166), yearnm='year', daynm='day',
               maxbreak=7):
    """Return monsoon Hydrologic Onset/Withdrawal Index.

    Parameters
    ----------
    uq_int, vq_int : xray.DataArrays
        Vertically integrated moisture fluxes.
    npts : int, optional
        Number of points to use to define HOWI index.
    nroll : int, optional
        Number of days for rolling mean.
    days_pre, days_post : list of ints, optional
        Default values correspond to May 18-24 and June 8-14 (numbered
        as non-leap year).
    yearnm, daynm : str, optional
        Name of year and day dimensions in DataArray
    maxbreak:
        Maximum number of days with index <=0 to consider a break in
        monsoon season rather than end of monsoon season.

    Returns
    -------
    howi : xray.Dataset
        HOWI daily timeseries for each year and monsoon onset and retreat
        days for each year.

    Reference
    ---------
    J. Fasullo and P. J. Webster, 2003: A hydrological definition of
        Indian monsoon onset and withdrawal. J. Climate, 16, 3200-3211.

    Notes
    -----
    In some years the HOWI index can give a bogus onset or bogus retreat
    when the index briefly goes above or below 0 for a few days.  To deal
    with these cases, I'm defining the monsoon season as the longest set
    of consecutive days with HOWI that is positive or has been negative
    for no more than `maxbreak` number of days (monsoon break).
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
    dspre = atm.subset(dsbar, {daynm : (days_pre, None)}).mean(dim=daynm)
    dspost = atm.subset(dsbar, {daynm : (days_post, None)}).mean(dim=daynm)
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
        # List of days with positive HOWI index
        pos = howi[daynm].values[howi['tseries'][y].values > 0]

        # In case of extra zero crossings, find the longest set of days
        # with positive index
        splitpos = atm.splitdays(pos)
        lengths = np.array([len(v) for v in splitpos])
        imonsoon = lengths.argmax()
        monsoon = splitpos[imonsoon]

        # In case there is a break in the monsoon season, check the
        # sets of days before and after and add to monsoon season
        # if applicable
        if imonsoon > 0:
            predays = splitpos[imonsoon - 1]
            if monsoon.min() - predays.max() <= maxbreak:
                predays = np.arange(predays.min(), monsoon.min())
                monsoon = np.concatenate([predays, monsoon])
        if imonsoon < len(splitpos) - 1:
            postdays = splitpos[imonsoon + 1]
            if postdays.min() - monsoon.max() <= maxbreak:
                postdays = np.arange(monsoon.max() + 1, postdays.max() + 1)
                monsoon = np.concatenate([monsoon, postdays])

        # Onset and retreat days
        onset[y] = monsoon[0]
        retreat[y] = monsoon[-1] + 1

    howi['onset'] = xray.DataArray(onset, coords={yearnm : howi[yearnm]})
    howi['retreat'] = xray.DataArray(retreat, coords={yearnm : howi[yearnm]})
    howi.attrs = {'npts' : npts, 'nroll' : nroll, 'maxbreak' : maxbreak,
                  'days_pre' : days_pre, 'days_post' : days_post}

    return howi, ds


# ----------------------------------------------------------------------
def onset_OCI(u, latlon = (5, 15, 40, 80), mmdd_thresh=(6,1),
              ndays=7, yearnm='Year', daynm='Day'):
    """Return monsoon Onset Circulation Index.

    Parameters
    ----------
    u : xray.DataArray
        850 hPa zonal wind.
    latlon : 4-tuple of floats, optional
        Tuple of (lat1, lat2, lon1, lon2) defining South Arabian Sea
        region to average over.
    mmdd_thres : 2-tuple of ints, optional
        Tuple of (month, day) defining climatological mean onset date
        to use for threshold value of u.
    ndays : int, optional
        Number of consecutive days threshold must be exceeded to
        define onset.
    yearnm, daynm : str, optional
        Name of year and day dimensions in DataArray

    Returns
    -------
    oci : xray.Dataset
        OCI daily timeseries for each year and monsoon onset day for
        each year.

    Reference
    ---------
    Wang, B., Ding, Q., & Joseph, P. V. (2009). Objective Definition
        of the Indian Summer Monsoon Onset. Journal of Climate, 22(12),
        3303-3316.
    """

    days = atm.get_coord(u, coord_name=daynm)
    years = atm.get_coord(u, coord_name=yearnm)
    nyears = len(years)

    # Average over South Arabian Sea region
    lat1, lat2, lon1, lon2 = latlon
    ubar = atm.mean_over_geobox(u, lat1, lat2, lon1, lon2)

    # Find values at climatological onset
    m0, d0 = mmdd_thresh
    d0 = [atm.mmdd_to_jday(m0, d0, year) for year in years]
    u0 = [ubar.sel(**{daynm : day, yearnm : year}).values
          for year, day in zip(years, d0)]
    u0 = np.array(u0).flatten()
    uthreshold = np.mean(u0)

    # Find first day when OCI exceeds threshold and stays above the
    # threshold for consecutive ndays
    def onset_day(tseries, uthreshold, ndays, daynm):
        above = (tseries.values > uthreshold)
        d0 = above.argmax()
        while not above[d0:d0+ndays].all():
            d0 += 1
        return tseries[daynm].values[d0]

    # Find onset day for each year
    onset = [onset_day(ubar[y], uthreshold, ndays, daynm)
             for y in range(nyears)]

    # Pack into dataset
    oci = xray.Dataset()
    oci['tseries'] = ubar
    oci['onset'] = xray.DataArray(onset, coords={yearnm : years})
    oci.attrs['latlon'] = latlon
    oci.attrs['mmdd_thresh'] = mmdd_thresh
    oci.attrs['ndays'] = ndays
    return oci

# ----------------------------------------------------------------------
def onset_SJKE(u, v, latlon = (-5, 20, 50, 70), ndays=3, yearnm='Year',
             daynm='Day', thresh_std=1.0):
    """Return monsoon onset based on Somali Jet kinetic energy.

    Parameters
    ----------
    u, v : xray.DataArray
        850 hPa zonal and meridional wind.
    latlon : 4-tuple of floats, optional
        Tuple of (lat1, lat2, lon1, lon2) defining Somali jet region
        to average over.
    ndays : int, optional
        Number of consecutive days threshold must be exceeded to
        define onset.
    yearnm, daynm : str, optional
        Name of year and day dimensions in DataArray
    thresh_std : float, optional
        Number of standard deviations excursion to use as onset threshold.

    Returns
    -------
    sjke : xray.Dataset
        Somali jet index daily timeseries for each year and monsoon
        onset day for each year.

    Reference
    ---------
    Boos, W. R., & Emanuel, K. A. (2009). Annual intensification of the
        Somali jet in a quasi-equilibrium framework : Observational
        composites. Quarterly Journal of the Royal Meteorological
        Society, 135, 319-335.
    """

    days = atm.get_coord(u, coord_name=daynm)
    years = atm.get_coord(u, coord_name=yearnm)
    nyears = len(years)

    # Kinetic energy index
    ke = np.sqrt(u**2 + v**2)

    # Average over Somali jet region
    lat1, lat2, lon1, lon2 = latlon
    ke = atm.mean_over_geobox(ke, lat1, lat2, lon1, lon2)
    ke.attrs['title'] = 'KE'
    ke.attrs['long_name'] = 'sqrt(u**2 + v**2)'

    # Threshold for onset date
    vals = ke.values.flatten()
    keclim = np.nanmean(vals)
    kestd = np.nanstd(vals)
    threshold = keclim + thresh_std * kestd

    # Find first day when KE exceeds threshold and stays above the
    # threshold for consecutive ndays
    def onset_day(tseries, threshold, ndays, daynm):
        above = (tseries.values > threshold)
        d0 = above.argmax()
        while not above[d0:d0+ndays].all():
            d0 += 1
        return tseries[daynm].values[d0]

    # Find onset day for each year
    onset = [onset_day(ke[y], threshold, ndays, daynm)
             for y in range(nyears)]

    # Pack into dataset
    sjke = xray.Dataset()
    sjke['tseries'] = ke
    sjke['onset'] = xray.DataArray(onset, coords={yearnm : years})
    sjke.attrs = {'latlon' : latlon, 'thresh_std' : thresh_std,
                  'threshold' : threshold, 'ndays' : ndays}

    return sjke

# ----------------------------------------------------------------------
def onset_TT(T, north=(5, 35, 40, 100), south=(-15, 5, 40, 100),
             yearnm='year', daynm='day'):
    """Return monsoon onset index based on tropospheric temperature.

    Parameters
    ----------
    T : xray.DataArray
        Air temperature 200-600 hPa vertical mean.
    north, south : 4-tuples of floats, optional
        Tuple of (lat1, lat2, lon1, lon2) defining northern and
        southern regions to average over.
    yearnm, daynm : str, optional
        Name of year and day dimensions in DataArray

    Returns
    -------
    tt : xray.Dataset
        TT daily timeseries for each year and monsoon onset day for
        each year.

    Reference
    ---------
    Goswami, B. N., Wu, G., & Yasunari, T. (2006). The annual cycle,
        intraseasonal oscillations, and roadblock to seasonal
        predictability of the Asian summer monsoon. Journal of Climate,
        19, 5078-5099.
    """

    ttn = atm.mean_over_geobox(T, north[0], north[1], north[2], north[3])
    tts = atm.mean_over_geobox(T, south[0], south[1], south[2], south[3])
    tseries = ttn - tts

    # Onset day is the first day that ttn-tts becomes positive
    years = tseries[yearnm]
    onset = np.zeros(years.shape)
    for y in range(len(years)):
        pos = (tseries.values[y] > 0)
        if not pos.any():
            onset[y] = np.nan
        else:
            onset[y] = tseries[daynm][pos.argmax()]

    tt = xray.Dataset()
    tt['ttn'] = ttn
    tt['tts'] = tts
    tt['tseries'] = tseries
    tt['onset'] = xray.DataArray(onset, coords={yearnm : years})
    tt.attrs['north'] = north
    tt.attrs['south'] = south

    return tt


# ----------------------------------------------------------------------
def onset_changepoint(precip_acc, onset_range=(1, 250),
                      retreat_range=(201, 366), order=1, yearnm='year',
                      daynm='day'):
    """Return monsoon onset/retreat based on changepoint in precip.

    Uses piecewise least-squares fit of data to detect changepoints.

    Parameters
    ----------
    precip_acc : xray.DataArray
        Accumulated precipitation.
    onset_range, retreat_range : 2-tuple of ints, optional
        Range of days to use when calculating onset / retreat.
    order : int, optional
        Order of polynomial to fit.
    yearnm, daynm : str, optional
        Name of year and day dimensions in precip_acc.

    Returns
    -------
    chp : xray.Dataset
        Onset/retreat days, daily timeseries, piecewise polynomial
        fits, and rss values.

    Reference
    ---------
    Cook, B. I., & Buckley, B. M. (2009). Objective determination of
    monsoon season onset, withdrawal, and length. Journal of Geophysical
    Research, 114(D23), D23109. doi:10.1029/2009JD012795
    """

    def split(x, n):
        return x[:n], x[n:]

    def piecewise_polyfit(x, y, n, order=1):
        y = np.ma.masked_array(y, np.isnan(y))
        x1, x2 = split(x, n)
        y1, y2 = split(y, n)
        p1 = np.ma.polyfit(x1, y1, order)
        p2 = np.ma.polyfit(x2, y2, order)
        if np.isnan(p1).any() or np.isnan(p2).any():
            raise ValueError('NaN for polyfit coeffs. Check data.')
        ypred1 = np.polyval(p1, x1)
        ypred2 = np.polyval(p2, x2)
        ypred = np.concatenate([ypred1, ypred2])
        rss = np.sum((y - ypred)**2)
        return ypred, rss

    def find_changepoint(x, y, order=1):
        rss = np.nan * x
        for n in range(2, len(x)- 2):
            _, rssval = piecewise_polyfit(x, y, n, order)
            rss[n] = rssval
        n0 = np.nanargmin(rss)
        x0 = x[n0]
        ypred, _ = piecewise_polyfit(x, y, n0)
        return x0, ypred, rss

    if yearnm not in precip_acc.dims:
        precip_acc = atm.expand_dims(precip_acc, yearnm, -1, axis=0)
    years = precip_acc[yearnm].values
    chp = xray.Dataset()
    chp['tseries'] = precip_acc

    for key, drange in zip(['onset', 'retreat'], [onset_range, retreat_range]):
        print('Calculating ' + key)
        print(drange)
        dmin, dmax = drange
        precip_sub = atm.subset(precip_acc, {daynm : (dmin, dmax)})
        dsub = precip_sub[daynm].values

        d_cp = np.nan * np.ones(years.shape)
        pred = np.nan * np.ones(precip_sub.shape)
        rss = np.nan * np.ones(precip_sub.shape)
        for y, year in enumerate(years):
            print (year)
            results = find_changepoint(dsub, precip_sub[y], order)
            d_cp[y], pred[y,:], rss[y,:] = results
        chp[key] = xray.DataArray(d_cp, dims=[yearnm], coords={yearnm : years})
        chp['tseries_fit_' + key] = xray.DataArray(
            pred, dims=[yearnm, daynm], coords={yearnm : years, daynm : dsub})
        chp['rss_' + key] = xray.DataArray(
            rss, dims=[yearnm, daynm], coords={yearnm : years, daynm : dsub})

    chp.attrs['order'] = order
    chp.attrs['onset_range'] = onset_range
    chp.attrs['retreat_range'] = retreat_range

    return chp


# ----------------------------------------------------------------------
def onset_changepoint_merged(precip_acc, order=1, yearnm='year',
                             daynm='day'):
    """Return monsoon onset/retreat based on changepoint in precip.

    Uses piecewise least-squares fit of data to detect changepoints.
    I've modified the original method by doing a 3-piecewise fit
    of the entire year, rather than a 2-piecewise fit of an onset
    range and retreat range.

    Parameters
    ----------
    precip_acc : xray.DataArray
        Accumulated precipitation.
    order : int, optional
        Order of polynomial to fit.
    yearnm, daynm : str, optional
        Name of year and day dimensions in precip_acc.

    Returns
    -------
    chp : xray.Dataset
        Onset/retreat days, daily timeseries, piecewise polynomial
        fits, and rss values.

    Reference
    ---------
    Cook, B. I., & Buckley, B. M. (2009). Objective determination of
    monsoon season onset, withdrawal, and length. Journal of Geophysical
    Research, 114(D23), D23109. doi:10.1029/2009JD012795
    """

    def split(x, n1, n2):
        return x[:n1], x[n1:n2], x[n2:]

    def piecewise_polyfit(x, y, n1, n2, order):
        y = np.ma.masked_array(y, np.isnan(y))
        x1, x2, x3 = split(x, n1, n2)
        y1, y2, y3 = split(y, n1, n2)
        p1 = np.ma.polyfit(x1, y1, order)
        p2 = np.ma.polyfit(x2, y2, order)
        p3 = np.ma.polyfit(x3, y3, order)
        if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
            raise ValueError('NaN for polyfit coeffs. Check data.')
        ypred1 = np.polyval(p1, x1)
        ypred2 = np.polyval(p2, x2)
        ypred3 = np.polyval(p3, x3)
        ypred = np.concatenate([ypred1, ypred2, ypred3])
        rss = np.sum((y - ypred)**2)
        return ypred, rss

    def find_changepoint(x, y, order):
        rss = {}
        for n1 in range(2, len(x)- 4):
            print(n1)
            for n2 in range(n1 + 2, len(x) - 2):
                _, rssval = piecewise_polyfit(x, y, n1, n2, order)
                rss[(n1, n2)] = rssval
        keys = rss.keys()
        rssvec = np.nan * np.ones(len(keys))
        for i, key in enumerate(keys):
            rssvec[i] = rss[key]
        i0 = np.nanargmin(rssvec)
        n1, n2 = keys[i0]
        x1, x2 = x[n1], x[n2]
        ypred, _ = piecewise_polyfit(x, y, n1, n2, order)
        return x1, x2, ypred, rss

    if yearnm not in precip_acc.dims:
        precip_acc = atm.expand_dims(precip_acc, yearnm, -1, axis=0)
    years = precip_acc[yearnm].values
    days = precip_acc[daynm].values
    chp = xray.Dataset()
    chp['tseries'] = precip_acc

    onset = np.nan * np.ones(years.shape)
    retreat = np.nan * np.ones(years.shape)
    pred = np.nan * np.ones(precip_acc.shape)
    for y, year in enumerate(years):
        print (year)
        results = find_changepoint(days, precip_acc[y], order)
        onset[y], retreat[y], pred[y,:], _ = results
    chp['onset'] = xray.DataArray(onset, dims=[yearnm], coords={yearnm : years})
    chp['retreat'] = xray.DataArray(retreat, dims=[yearnm], coords={yearnm : years})
    chp['tseries_fit'] = xray.DataArray(pred, dims=[yearnm, daynm],
                                        coords={yearnm : years, daynm : days})
    chp.attrs['order'] = order

    return chp


# ----------------------------------------------------------------------
def plot_hist(ind, binwidth=5, incl_daystr=True, ax=None, pos=(0.05, 0.7),
              kw={'alpha' : 0.3, 'color' : 'k'}):
    """Plot histogram of onset days.
    """
    if ax is None:
        ax = plt.gca()

    def daystr(day):
        day = round(day)
        mm, dd = atm.jday_to_mmdd(day)
        mon = atm.month_str(mm)
        return '%.0f (%s-%.0f)' % (day, mon, dd)

    if isinstance(ind, pd.Series) or isinstance(ind, xray.DataArray):
        ind = ind.values

    b1 = np.floor(np.nanmin(ind) / binwidth) * binwidth
    b2 = np.ceil(np.nanmax(ind) / binwidth) * binwidth
    bin_edges = np.arange(b1, b2 + 1, binwidth)
    n, bins, _ = ax.hist(ind, bin_edges, **kw)
    ax.set_xlabel('Day of Year')
    ax.set_ylabel('Num of Occurrences')
    if incl_daystr:
        dmean = daystr(np.nanmean(ind))
        dmin = daystr(np.nanmin(ind))
        dmax = daystr(np.nanmax(ind))
    else:
        dmean = '%.0f' % np.nanmean(ind)
        dmin = '%.0f' % np.nanmin(ind)
        dmax = '%.0f' % np.nanmax(ind)
    s = 'Mean %s\n' % dmean + 'Std %.0f\n' % np.nanstd(ind)
    s = s + 'Min %s\n' % dmin + 'Max %s' % dmax
    x0, y0 = pos
    atm.text(s, (x0, y0), ax=ax, horizontalalignment='left')


# ----------------------------------------------------------------------
def summarize_indices(years, onset, retreat=None, indname='', binwidth=5,
                      figsize=(16, 10)):
    """Summarize monsoon onset/retreat days in timeseries and histogram.
    """

    if isinstance(onset, xray.DataArray):
        onset = onset.values
    if retreat is None:
        nrows, ncols = 2, 1
        figsize = (7, 10)
    else:
        if isinstance(retreat, xray.DataArray):
            retreat = retreat.values
            length = retreat - onset
        nrows, ncols = 2, 3

    plt.figure(figsize=figsize)
    plt.subplot(nrows, ncols, 1)
    plt.plot(years, onset)
    plt.xlabel('Year')
    plt.ylabel('Onset Day')
    plt.title('Onset')
    plt.grid()

    plt.subplot(nrows, ncols, ncols + 1)
    plot_hist(onset, binwidth)
    plt.title('Onset')

    if retreat is not None:
        plt.subplot(nrows, ncols, 2)
        plt.plot(years, retreat)
        plt.xlabel('Year')
        plt.ylabel('Retreat Day')
        plt.title('Retreat')
        plt.grid()

        plt.subplot(nrows, ncols, ncols + 2)
        plot_hist(retreat, binwidth)
        plt.title('Retreat')

        plt.subplot(nrows, ncols, 3)
        plt.plot(years, length)
        plt.xlabel('Year')
        plt.ylabel('# Days')
        plt.title('Monsoon Length')
        plt.grid()

        plt.subplot(nrows, ncols, ncols + 3)
        plot_hist(length, binwidth, incl_daystr=False)
        plt.xlabel('# Days')
        plt.title('Monsoon Length')

    plt.suptitle(indname)


# ----------------------------------------------------------------------
def plot_index_years(index, years=None, figsize=(14,10), nrow=3, ncol=4,
                     suptitle='', yearnm='year', daynm='day',
                     vertline=False):
    """Plot daily timeseries of monsoon index/onset/retreat each year.
    """

    days = index[daynm]
    if years is None:
        # All years
        years = index[yearnm].values

    tseries = atm.subset(index['tseries'], {yearnm : (years, None)})
    if 'onset' in index.data_vars:
        onset = atm.subset(index['onset'], {yearnm : (years, None)}).values
    else:
        onset = np.nan * years
    if 'retreat' in index.data_vars:
        retreat = atm.subset(index['retreat'], {yearnm : (years, None)}).values
    else:
        retreat = np.nan * years

    # Earliest/latest onset/retreat, shortest/longest seasons
    length = retreat - onset
    yrs_ex, nms_ex = [], []
    if not np.isnan(onset).all():
        yrs_ex.extend([years[onset.argmin()], years[onset.argmax()]])
        nms_ex.extend(['Earliest Onset', 'Latest Onset'])
    if not np.isnan(retreat).all():
        yrs_ex.extend([years[retreat.argmin()], years[retreat.argmax()]])
        nms_ex.extend(['Earliest Retreat', 'Latest Retreat'])
    if not np.isnan(length).all():
        yrs_ex.extend([years[length.argmin()], years[length.argmax()]])
        nms_ex.extend(['Shortest Monsoon', 'Longest Monsoon'])
    yrs_extreme = collections.defaultdict(str)
    for yr, nm in zip(yrs_ex, nms_ex):
        yrs_extreme[yr] = yrs_extreme[yr] + ' - ' + nm

    # Monsoon index with onset and retreat in individual years
    def line_or_point(d, ind, daynm, vertline, ax, label, clr):
        d = int(d)
        val = atm.subset(ind, {daynm : (d, None)})
        if vertline:
            ax.plot([d, d], ax.get_ylim(), clr, linewidth=2, label=label)
        ax.plot(d, val, clr + 'o', label=label)

    def onset_tseries(days, ind, d_onset, d_retreat, daynm, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.plot(days, ind)
        if d_onset is not None and not np.isnan(d_onset):
            line_or_point(d_onset, ind, daynm, vertline, ax, label='onset',
                          clr='r')
        if d_retreat is not None and not np.isnan(d_retreat):
            line_or_point(d_retreat, ind, daynm, vertline, ax, label='retreat',
                          clr='b')
        ax.grid()
        ax.set_xlim(days.min() - 1, days.max() + 1)

    # Plot each year
    for y, year in enumerate(years):
        if y % (nrow * ncol) == 0:
            fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
            plt.subplots_adjust(left=0.08, right=0.95, wspace=0.1, hspace=0.2)
            plt.suptitle(suptitle)
            yplot = 1
        else:
            yplot += 1
        i, j = atm.subplot_index(nrow, ncol, yplot)
        ax = axes[i-1, j-1]
        onset_tseries(days, tseries[y], onset[y], retreat[y], daynm, ax)
        if year in yrs_extreme.keys():
            titlestr = str(year) + yrs_extreme[year]
        else:
            titlestr = str(year)
        ax.set_title(titlestr)
        if i == nrow:
            ax.set_xlabel('Day')

    return yrs_extreme


# ----------------------------------------------------------------------
def plot_tseries_together(data, onset=None, years=None, suptitle='',
                          figsize=(14,10), legendsize=10,
                          legendloc='lower right', nrow=3, ncol=4,
                          yearnm='year', daynm='day', standardize=True,
                          label_attr=None, data_style=None, onset_style=None,
                          show_days=False):
    """Plot multiple daily timeseries together each year.

    Parameters
    ----------
    data : xray.Dataset
        Dataset of timeseries variables to plot together.
    onset : ndarray or dict of ndarrays, optional
        Array of onset day for each year, or dict of onset arrays (e.g.
        to compare onset days from different methods).
    years : ndarray, optional
        Subset of years to include.  If omitted, all years are included.
    suptitle : str, optional
        Supertitle for plot.
    figsize : 2-tuple, optional
        Size of each figure.
    legendsize : int, optional
        Font size for legend
    legendloc : str, optional
        Legend location
    nrow, ncol : int, optional
        Number of rows, columns in each figure.
    yearnm, daynm : str, optional
        Name of year and day dimensions in data.
    standardize : bool, optional
        If True, standardize each timeseries by dividing by its
        standard deviation.
    label_attr : str, optional
        Attribute of each data variable to use for labels.  If omitted,
        then the variable name is used.
    data_style, onset_style : list or dict, optional
        Matlab-style strings for each data variable or onset index.
    show_days : bool, optional
        If True, annotate each subplot with a textbox showing the
        onset days.
    """

    if years is None:
        # All years
        years = data[yearnm].values
    data = atm.subset(data, {yearnm : (years, None)})

    if label_attr is not None:
        labels = {nm : data[nm].attrs[label_attr] for nm in data.data_vars}

    if onset is not None:
        if isinstance(onset, dict):
            if onset_style is None:
                onset_style = {key : 'k' for key in onset.keys()}
        else:
            onset = {'onset' : onset}
            if onset_style is None:
                onset_style = {'onset' : 'k'}
        textpos = {key : (0.05, 0.9 - 0.1*i) for i, key in enumerate(onset)}

    # Plot each year
    for y, year in enumerate(years):
        df = atm.subset(data, {yearnm : (year, None)}).to_dataframe()
        df.drop(yearnm, axis=1, inplace=True)
        if label_attr is not None:
            df.rename(columns=labels, inplace=True)

        if standardize:
            for key in df.columns:
                df[key] = (df[key] - np.nanmean(df[key])) / np.nanstd(df[key])
            ylabel = 'Standardized Timeseries'
        else:
            ylabel = 'Timeseries'

        if y % (nrow * ncol) == 0:
            fig, axes = plt.subplots(nrow, ncol, figsize=figsize, sharex=True)
            plt.subplots_adjust(left=0.08, right=0.95, wspace=0.2, hspace=0.2)
            plt.suptitle(suptitle)
            yplot = 1
        else:
            yplot += 1

        i, j = atm.subplot_index(nrow, ncol, yplot)
        ax = axes[i-1, j-1]
        df.plot(ax=ax, style=data_style)
        ax.grid()
        if yplot == 1:
            ax.legend(fontsize=legendsize, loc=legendloc)
        else:
            ax.legend_.remove()
        if onset is not None:
            for key in onset:
                d0 = onset[key][y]
                ax.plot([d0, d0], ax.get_ylim(), onset_style[key])
                if show_days:
                    atm.text(d0, textpos[key], ax=ax, color=onset_style[key])
        if j == 1:
            ax.set_ylabel(ylabel)
        if i == nrow:
            ax.set_xlabel('Day')
        else:
            ax.set_xlabel('')
        ax.set_title(year)
