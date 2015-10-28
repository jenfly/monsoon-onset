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
    howi.attrs['npts'] = npts
    howi.attrs['nroll'] = nroll

    return howi, ds


# ----------------------------------------------------------------------
def summarize_indices(years, onset, retreat, indname='', binwidth=5,
                      figsize=(16, 10)):
    """Summarize monsoon onset/retreat days in timeseries and histogram.
    """

    if isinstance(onset, xray.DataArray):
        onset = onset.values
    if isinstance(retreat, xray.DataArray):
        retreat = retreat.values
    length = retreat - onset

    def daystr(day):
        day = round(day)
        mm, dd = atm.jday_to_mmdd(day)
        mon = atm.month_str(mm)
        return '%.0f (%s-%.0f)' % (day, mon, dd)

    def plot_hist(ind, binwidth, incl_daystr=True):
        b1 = np.floor(ind.min() / binwidth) * binwidth
        b2 = np.ceil(ind.max() / binwidth) * binwidth
        bin_edges = np.arange(b1, b2 + 1, binwidth)
        n, bins, _ = plt.hist(ind, bin_edges)
        plt.xlabel('Day of Year')
        plt.ylabel('Num of Occurrences')
        x1 = bins[0] + 0.03 * (bins[-1] - bins[0])
        y1 = n.max() * 0.9
        if incl_daystr:
            dmean = daystr(ind.mean())
            dmin = daystr(ind.min())
            dmax = daystr(ind.max())
        else:
            dmean = '%.0f' % ind.mean()
            dmin = '%.0f' % ind.min()
            dmax = '%.0f' % ind.max()
        plt.text(x1, y1, 'Mean %s' % dmean)
        plt.text(x1, y1 * 0.9, 'Std %.0f' % ind.std())
        plt.text(x1, y1 * 0.8, 'Min %s' % dmin)
        plt.text(x1, y1 * 0.7, 'Max %s' % dmax)

    plt.figure(figsize=(16,10))
    plt.subplot(231)
    plt.plot(years, onset)
    plt.xlabel('Year')
    plt.ylabel('Onset Day')
    plt.title(indname + ' Onset')
    plt.grid()

    plt.subplot(234)
    plot_hist(onset, binwidth)
    plt.title(indname + ' Onset')

    plt.subplot(232)
    plt.plot(years, retreat)
    plt.xlabel('Year')
    plt.ylabel('Retreat Day')
    plt.title(indname + ' Retreat')
    plt.grid()

    plt.subplot(235)
    plot_hist(retreat, binwidth)
    plt.title(indname + ' Retreat')

    plt.subplot(233)
    plt.plot(years, length)
    plt.xlabel('Year')
    plt.ylabel('# Days')
    plt.title('Monsoon Length')
    plt.grid()

    plt.subplot(236)
    plot_hist(length, binwidth, incl_daystr=False)
    plt.xlabel('# Days')
    plt.title('Monsoon Length')

    
# ----------------------------------------------------------------------    
def plot_index_years(index, years=None, figsize=(12,10), nrow=2, ncol=2):
    """Plot daily timeseries of monsoon index/onset/retreat each year.
    """
    
    days = index.day
    if years is None:
        # All years
        tseries = index.tseries
        years = tseries.year.values
        onset = index.onset.values
        retreat = index.retreat.values
    else:
        # Subset of years
        tseries = index.tseries.sel(year=years)
        onset = index.onset.sel(year=years).values
        retreat = index.retreat.sel(year=years).values
    
    # Earliest/latest onset/retreat, shortest/longest seasons
    length = retreat - onset
    yrs_extreme = collections.defaultdict(str)
    yrs_ex = [years[onset.argmin()], years[onset.argmax()],
              years[retreat.argmin()], years[retreat.argmax()],
              years[length.argmin()], years[length.argmax()]]
    nms_ex = ['Earliest Onset', 'Latest Onset', 
              'Earliest Retreat', 'Latest Retreat',
              'Shortest Monsoon', 'Longest Monsoon']
    for yr, nm in zip(yrs_ex, nms_ex):
        yrs_extreme[yr] = yrs_extreme[yr] + ' - ' + nm
    
    # Monsoon index with onset and retreat in individual years
    def onset_tseries(days, ind, d_onset, d_retreat):
        plt.plot(days, ind)
        plt.plot(d_onset, ind.sel(day=d_onset), 'ro', label='onset')
        plt.plot(d_onset-1, ind.sel(day=d_onset-1), 'k.', label='onset-1')
        plt.plot(d_retreat, ind.sel(day=d_retreat), 'bo', label='retreat')
        plt.plot(d_retreat-1, ind.sel(day=d_retreat-1), 'k.', label='retreat')
        plt.grid()

    # Plot each year
    for y, year in enumerate(years):
        if y % (nrow * ncol) == 0:
            plt.figure(figsize=figsize)
            yplot = 1
        else:
            yplot += 1
        plt.subplot(nrow, ncol, yplot)
        onset_tseries(days, tseries[y], onset[y], retreat[y])
        if year in yrs_extreme.keys():
            titlestr = str(year) + yrs_extreme[year]
        else:
            titlestr = str(year)
        plt.title(titlestr)
        
    return yrs_extreme
    
    


