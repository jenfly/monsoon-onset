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
    years = atm.makelist(atm.get_coord(data, coord_name=yearnm))

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
def get_mfc_box(mfcfiles, precipfiles, evapfiles, years, nroll, lat1, lat2,
                lon1, lon2):
    """Return daily tseries MFC, precip and evap averaged over lat-lon box.
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
    if evapfiles is not None:
        evap = atm.combine_daily_years('EVAP', evapfiles, years, yearname='year',
                                        subset_dict=subset_dict)
        databox['EVAP'] = evap


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
        ds = atm.combine_daily_years(['uq_int', 'vq_int'], datafiles, years,
                                     yearname='year')
        index, _ = indices.onset_HOWI(ds['uq_int'], ds['vq_int'], npts,
                                      maxbreak=maxbreak)
        index.attrs['title'] = 'HOWI (N=%d)' % npts
    elif onset_nm == 'CHP_MFC':
        if data is None:
            tseries = get_mfc_box(datafiles, None, None, years, *chp_opts)
            data = tseries['MFC_ACC']
        index = indices.onset_changepoint(data)
    elif onset_nm == 'CHP_PCP':
        if data is None:
            tseries = get_mfc_box(None, datafiles, None, years, *chp_opts)
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


# ----------------------------------------------------------------------
def get_strength_indices(years, mfc, precip, onset, retreat, yearnm='year',
                         daynm='day', varnm1='MFC', varnm2='PCP'):
    """Return various indices of the monsoon strength.

    Inputs mfc and precip are the unsmoothed daily values averaged over
    the monsoon area.
    """

    ssn = xray.Dataset()
    coords = {yearnm : years}
    ssn['onset'] = xray.DataArray(onset, coords=coords)
    ssn['retreat'] = xray.DataArray(retreat, coords=coords)
    ssn['length'] = ssn['retreat'] - ssn['onset']

    data_in = {}
    if mfc is not None:
        data_in[varnm1] = mfc
    if precip is not None:
        data_in[varnm2] = precip

    for key in data_in:
        for key2 in ['_JJAS_AVG', '_JJAS_TOT', '_LRS_AVG', '_LRS_TOT']:
            ssn[key + key2] = xray.DataArray(np.nan * np.ones(len(years)),
                                             coords=coords)

    for key in data_in:
        for y, year in enumerate(years):
            d1 = int(onset.values[y])
            d2 = int(retreat.values[y] - 1)
            days_jjas = atm.season_days('JJAS', atm.isleap(year))
            data = atm.subset(data_in[key], {yearnm : (year, None)})
            data_jjas = atm.subset(data, {daynm : (days_jjas, None)})
            data_lrs = atm.subset(data, {daynm : (d1, d2)})
            ssn[key + '_JJAS_AVG'][y] = data_jjas.mean(dim=daynm).values
            ssn[key + '_LRS_AVG'][y] = data_lrs.mean(dim=daynm).values
            ssn[key + '_JJAS_TOT'][y] = ssn[key + '_JJAS_AVG'][y] * len(days_jjas)
            ssn[key + '_LRS_TOT'][y] = ssn[key + '_LRS_AVG'][y] * ssn['length'][y]

    ssn = ssn.to_dataframe()
    return ssn

# ----------------------------------------------------------------------
def var_type(varnm):
    keys = ['THETA', 'MSE', 'DSE', 'V*', 'abs_vort', 'EMFD']
    test =  [varnm.startswith(key) for key in keys]
    if np.array(test).any():
        vtype = 'calc'
    else:
        vtype = 'basic'
    return vtype


# ----------------------------------------------------------------------
def get_data_rel(varnm, years, datafiles, data, onset, npre, npost,
                 yearnm='year', daynm='day'):
    """Return daily data relative to onset date.

    Data is read from datafiles if varnm is a basic variable.
    If varnm is a calculated variable (e.g. potential temperature),
    the base variables for calculation are provided in the dict data.
    """

    years = atm.makelist(years)
    onset = atm.makelist(onset)
    daymin, daymax = min(onset) - npre, max(onset) + npost
    if varnm not in ['precip', 'EVAP', 'EFLUX', 'HFLUX']:
        plev = int(varnm[-3:])
        varid = varnm[:-3]
    else:
        varid, plev = varnm, None
    if varnm == 'precip':
        subset_dict = {'day' : (daymin, daymax),
                       'lon' : (40, 120),
                       'lat' : (-60, 60)}
        var = atm.combine_daily_years('PRECTOT', datafiles, years,
                                      yearname=yearnm, subset_dict=subset_dict)
    elif var_type(varnm) == 'calc':
        pres = atm.pres_convert(plev, 'hPa', 'Pa')
        Tnm = 'T%d' % plev
        Hnm = 'H%d' % plev
        QVnm = 'QV%d' % plev
        Unm = 'U%d' % plev
        Vnm = 'V%d' % plev
        print('Computing ' + varid)
        if varid == 'THETA':
            var = atm.potential_temp(data[Tnm], pres)
        elif varid == 'THETA_E':
            var = atm.equiv_potential_temp(data[Tnm], pres, data[QVnm])
        elif varid == 'DSE':
            var = atm.dry_static_energy(data[Tnm], data[Hnm])
        elif varid == 'MSE':
            var = atm.moist_static_energy(data[Tnm], data[Hnm], data[QVnm])
        elif varid.startswith('V*'):
            varid2 = '%s%d' % (varid[2:], plev)
            var = data['V%d' % plev] * data[varid2]
            var.name = varid
        elif varid == 'abs_vort':
            rel_vort = data['rel_vort%d' % plev]
            lat = atm.get_coord(rel_vort, 'lat')
            f = atm.coriolis(lat)
            f = atm.biggify(f, rel_vort, tile=True)
            var = rel_vort + f
            var.name = varid
        elif varid == 'EMFD':
            nroll = 7
            u_tr = data[Unm] - atm.rolling_mean(data[Unm], nroll, axis=1)
            v_tr = data[Vnm] - atm.rolling_mean(data[Vnm], nroll, axis=1)

            _, _, var = atm.divergence_spherical_2d(u_tr * u_tr,
                                                          u_tr * v_tr)
            var.name = varid
            var.attrs['long_name'] = 'Transient EMFD_y'
    else:
        with xray.open_dataset(datafiles[0]) as ds:
            daynm_in = ds[varid].dims[0]
        var = atm.combine_daily_years(varid, datafiles, years, yearname=yearnm,
                                      subset_dict={daynm_in : (daymin, daymax)})
        var = atm.squeeze(var)

    # Convert precip and evap to mm/day
    if varnm in ['precip', 'EVAP']:
        var = atm.precip_convert(var, var.attrs['units'], 'mm/day')

    # Align relative to onset day:
    if var_type(varnm) == 'basic':
        print('Aligning data relative to onset day')
        var = var.rename({var.dims[0] : daynm})
        if len(years) == 1:
            var = atm.expand_dims(var, yearnm, years[0], axis=0)
        var = daily_rel2onset(var, onset, npre, npost, yearnm=yearnm,
                              daynm=daynm)

    return var


# ----------------------------------------------------------------------
def load_dailyrel(datafiles, yearnm='year', onset_varnm='D_ONSET',
                  retreat_varnm='D_RETREAT'):

    ds = atm.load_concat(datafiles, concat_dim=yearnm)
    varnms = ds.data_vars.keys()
    if onset_varnm is not None:
        onset = ds[onset_varnm]
        varnms.remove(onset_varnm)
    else:
        onset = np.nan * ds[yearnm]
    if retreat_varnm is not None:
        retreat = ds[retreat_varnm]
        varnms.remove(retreat_varnm)
    else:
        retreat = np.nan * ds[yearnm]

    # Remaining data variable is the data field
    varnm = varnms[0]
    data = ds[varnm]

    # Copy attributes from the first file in the list
    with xray.open_dataset(datafiles[0]) as ds0:
        data.attrs = ds0[varnm].attrs

    return data, onset, retreat


# ----------------------------------------------------------------------
def plot_colorbar(symmetric, orientation='vertical'):
    if symmetric:
        atm.colorbar_symm(orientation=orientation)
    else:
        plt.colorbar(orientation=orientation)


# ----------------------------------------------------------------------
def contourf_lat_time(lat, days, plotdata, title, cmap, onset_nm,
                      zero_line=False):
    vals = plotdata.values.T
    vals = np.ma.array(vals, mask=np.isnan(vals))
    ncont = 40
    symmetric = atm.symm_colors(plotdata)
    cint = atm.cinterval(vals, n_pref=ncont, symmetric=symmetric)
    clev = atm.clevels(vals, cint, symmetric=symmetric)
    plt.contourf(days, lat, vals, clev, cmap=cmap)
    plot_colorbar(symmetric)
    if symmetric and zero_line:
        plt.contour(days, lat, vals, [0], colors='k')
    plt.grid(True)
    plt.ylabel('Latitude')
    plt.xlabel('Day Relative to %s Onset' % onset_nm)
    plt.title(title)
    xmin, xmax = plt.gca().get_xlim()
    if xmax > 60:
        plt.xticks(range(int(xmin), int(xmax) + 1, 30))


# ----------------------------------------------------------------------
def eddy_decomp(var, nt, lon1, lon2, taxis=0):
    """Decompose variable into mean and eddy fields."""
            
    lonname = atm.get_coord(var, 'lon', 'name')
    tstr = '%d-%s rolling' % (nt, var.dims[taxis])
    lonstr = atm.latlon_labels([lon1, lon2], 'lon', deg_symbol=False)
    lonstr = '-'.join(lonstr)
    
    varbar = atm.rolling_mean(var, nt, axis=taxis, center=True)
    varbarzon = atm.subset(varbar, {lonname : (lon1, lon2)})
    varbarzon = varbarzon.mean(dim=lonname)
    
    comp = xray.Dataset()    
    comp['AVG'] = varbarzon
    comp['AVG'].attrs['title'] = 'Time mean (%s), zonal mean (%s)' % (tstr, lonstr)    
    comp['ST'] = varbar - varbarzon
    comp['ST'].attrs['title'] = 'Stationary eddy'
    comp['TR'] = var - varbar
    comp['TR'].attrs['title'] = 'Transient eddy'
    
    return comp
    
