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
def daily_rel2onset(data, d_onset, npre, npost):
    """Return subset of daily data aligned relative to onset day.

    Parameters
    ----------
    data : xray.DataArray
        Daily data.
    d_onset : ndarray
        Array of onset date (day of year) for each year.
    npre, npost : int
        Number of days before and after onset to extract.

    Returns
    -------
    data_out : xray.DataArray
        Subset of N days of daily data for each year, where
        N = npre + npost + 1 and the day dimension is
        dayrel = day - d_onset.
    """

    name, attrs, coords, dimnames = atm.meta(data)
    yearnm = atm.get_coord(data, 'year', 'name')
    daynm = atm.get_coord(data, 'day', 'name')
    years = atm.makelist(atm.get_coord(data, 'year'))

    if isinstance(d_onset, xray.DataArray):
        d_onset = d_onset.values
    else:
        d_onset = atm.makelist(d_onset)

    relnm = daynm + 'rel'

    for y, year in enumerate(years):
        dmin, dmax = d_onset[y] - npre, d_onset[y] + npost
        subset_dict = {yearnm : (year, None), daynm : (dmin, dmax)}
        sub = atm.subset(data, subset_dict)
        sub = sub.rename({daynm : relnm})
        sub[relnm] = sub[relnm] - d_onset[y]
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
        index['ts_daily'] = tseries['MFC']
    elif onset_nm == 'CHP_PCP':
        if data is None:
            tseries = get_mfc_box(None, datafiles, None, years, *chp_opts)
            data = tseries['PCP_ACC']
        index = indices.onset_changepoint(data)
        index['ts_daily'] = tseries['PCP']

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
    keys = ['THETA', 'MSE', 'DSE', 'V*', 'abs_vort', 'EMFD', 'VFLXMSE']
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

    if varnm.isalpha():
        varid, plev = varnm, None
    else:
        plev = int(varnm[-3:])
        varid = varnm[:-3]

    if varnm == 'precip':
        subset_dict = {'day' : (daymin, daymax),
                       'lon' : (40, 120),
                       'lat' : (-60, 60)}
        var = atm.combine_daily_years('PRECTOT', datafiles, years,
                                      yearname=yearnm, subset_dict=subset_dict)
    elif var_type(varnm) == 'calc':
        if plev is not None:
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
        elif varid == 'VFLXMSE':
            Lv = atm.constants.Lv.values
            var = data['VFLXCPT'] + data['VFLXPHI'] + data['VFLXQV'] * Lv
            var.attrs['units'] = data['VFLXCPT'].attrs['units']
            var.attrs['long_name'] = 'Vertically integrated MSE meridional flux'
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
            daynm_in = atm.get_coord(ds[varid], 'day', 'name')
        var = atm.combine_daily_years(varid, datafiles, years, yearname=yearnm,
                                      subset_dict={daynm_in : (daymin, daymax)})
        var = atm.squeeze(var)

    # Convert precip and evap to mm/day
    if varnm in ['precip', 'EVAP']:
        var = atm.precip_convert(var, var.attrs['units'], 'mm/day')

    # Align relative to onset day:
    if var_type(varnm) == 'basic':
        print('Aligning data relative to onset day')
        daynm_in = atm.get_coord(var, 'day', 'name')
        if daynm_in !=  daynm:
            var = var.rename({daynm_in : daynm})
        if len(years) == 1:
            var = atm.expand_dims(var, yearnm, years[0], axis=0)
        var = daily_rel2onset(var, onset, npre, npost)

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
def plot_colorbar(symmetric, orientation='vertical', ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if symmetric:
        atm.colorbar_symm(orientation=orientation, ax=ax, **kwargs)
    else:
        plt.colorbar(orientation=orientation, ax=ax, **kwargs)


# ----------------------------------------------------------------------
def contourf_lat_time(lat, days, plotdata, title='', cmap='RdBu_r', onset_nm='',
                      zero_line=False, ax=None):
    if ax is None:
        ax = plt.gca()
    vals = plotdata.values.T
    vals = np.ma.array(vals, mask=np.isnan(vals))
    ncont = 40
    symmetric = atm.symm_colors(plotdata)
    cint = atm.cinterval(vals, n_pref=ncont, symmetric=symmetric)
    clev = atm.clevels(vals, cint, symmetric=symmetric)
    cf = ax.contourf(days, lat, vals, clev, cmap=cmap)
    plt.colorbar(mappable=cf, ax=ax)
    #plot_colorbar(symmetric, ax=ax, mappable=cf)
    if symmetric and zero_line:
        ax.contour(days, lat, vals, [0], colors='k')
    ax.grid(True)
    ax.set_ylabel('Latitude')
    ax.set_xlabel('Day Relative to %s Onset' % onset_nm)
    ax.set_title(title)
    xmin, xmax = ax.get_xlim()
    if xmax > 60:
        ax.set_xticks(range(int(xmin), int(xmax) + 1, 30))
    plt.draw()


# ----------------------------------------------------------------------
def eddy_decomp(var, nt, lon1, lon2, taxis=0):
    """Decompose variable into mean and eddy fields."""

    lonname = atm.get_coord(var, 'lon', 'name')
    tstr = 'Time mean (%d-%s rolling)' % (nt, var.dims[taxis])
    lonstr = atm.latlon_labels([lon1, lon2], 'lon', deg_symbol=False)
    lonstr = 'zonal mean (' + '-'.join(lonstr) + ')'
    name, attrs, coords, dims = atm.meta(var)

    varbar = atm.rolling_mean(var, nt, axis=taxis, center=True)
    varbarzon = atm.subset(varbar, {lonname : (lon1, lon2)})
    varbarzon = varbarzon.mean(dim=lonname)
    varbarzon.attrs = attrs

    comp = xray.Dataset()
    comp[name + '_AVG'] = varbarzon
    comp[name + '_AVG'].attrs['component'] = tstr + ', ' + lonstr
    comp[name + '_ST'] = varbar - varbarzon
    comp[name + '_ST'].attrs = attrs
    comp[name + '_ST'].attrs['component'] = 'Stationary eddy'
    comp[name + '_TR'] = var - varbar
    comp[name + '_TR'].attrs = attrs
    comp[name + '_TR'].attrs['component'] = 'Transient eddy'

    return comp


# ----------------------------------------------------------------------
def latlon_data(var, latmax=89):
    """Return lat, lon coords in radians and cos(lat)."""

    data = xray.Dataset()

    # Latitude
    latname = atm.get_coord(var, 'lat', 'name')
    latdim = atm.get_coord(var, 'lat', 'dim')
    lat = atm.get_coord(var, 'lat')
    latcoords = {latname: lat.copy()}
    lat[abs(lat) > latmax] = np.nan
    data['LAT'] = xray.DataArray(lat, coords=latcoords)
    latrad = np.radians(lat)
    data['LATRAD'] = xray.DataArray(latrad, coords=latcoords)
    data['COSLAT'] = np.cos(data['LATRAD'])
    data.attrs['latname'] = latname
    data.attrs['latdim'] = latdim

    # Longitude
    try:
        lonname = atm.get_coord(var, 'lon', 'name')
        londim = atm.get_coord(var, 'lon', 'dim')
        lon = atm.get_coord(var, 'lon')
        loncoords = {lonname : lon.copy()}
        data['LON'] = xray.DataArray(lon, coords=loncoords)
        lonrad = np.radians(lon)
        data['LONRAD'] = xray.DataArray(lonrad, coords=loncoords)
        data.attrs['lonname'] = lonname
        data.attrs['londim'] = londim
    except ValueError:
        data.attrs['lonname'] = None
        data.attrs['londim'] = None

    return data


# ----------------------------------------------------------------------
def advection(uflow, vflow, omegaflow, u, dudp):
    """Return x, y and p components of advective terms in momentum budget.
    """

    a = atm.constants.radius_earth
    latlon = latlon_data(u)
    latdim, londim = latlon.attrs['latdim'], latlon.attrs['londim']
    latrad, coslat = latlon['LATRAD'], latlon['COSLAT']
    if londim is not None:
        lonrad = latlon['LONRAD']

    ds = xray.Dataset()
    if londim is not None:
        ds['X'] = atm.gradient(u, lonrad, londim) * uflow / (a*coslat)
    else:
        ds['X'] = 0.0 * u
    ds['Y'] = atm.gradient(u*coslat, latrad, latdim) * vflow / (a*coslat)
    ds['P'] = omegaflow * dudp

    return data


# ----------------------------------------------------------------------
def fluxdiv(u, v, omega, dudp, domegadp):
    """Return x, y and p components of EMFD terms in momentum budget.
    """

    a = atm.constants.radius_earth
    latlon = latlon_data(u)
    latdim, londim = latlon.attrs['latdim'], latlon.attrs['londim']
    latrad, coslat = latlon['LATRAD'], latlon['COSLAT']
    coslat = latlon['COSLAT']
    coslat_sq = coslat ** 2
    if londim is not None:
        lonrad = latlon['LONRAD']

    ds = xray.Dataset()
    if londim is not None:
        ds['X'] = atm.gradient(u * u, lonrad, londim) / (a*coslat)
    else:
        ds['X'] = 0.0 * u
    ds['Y'] = atm.gradient(u * v * coslat_sq, latrad, latdim) / (a*coslat_sq)
    ds['P'] = omega * dudp + u * domegadp

    return ds


# ----------------------------------------------------------------------
def calc_ubudget(datafiles, ndays, lon1, lon2, plev=200):
    """Calculate momentum budget for daily data in one year.

    Keys of datafiles dict must be: U, V, DUDP, H, OMEGA, DOMEGADP, DUDTANA
    """

    # Read data
    data = xray.Dataset()
    for nm in datafiles:
        print('Reading ' + datafiles[nm])
        with xray.open_dataset(datafiles[nm]) as ds:
            if nm in ds.data_vars:
                var = ds[nm]
            else:
                var = ds[nm + '%d' % plev]
            if 'Day' in var.dims:
                var = var.rename({'Day' : 'day'})
            data[nm] = atm.squeeze(var)
    data['PHI'] = atm.constants.g.values * data['H']

    # Put zeros in for any missing variables (e.g. du/dp)
    for nm in ['OMEGA', 'DUDP', 'DOMEGADP', 'DUDTANA']:
        if nm not in data.data_vars:
            data[nm] = 0.0 * data['U']

    # Eddy decomposition
    taxis = 0
    for nm in data.data_vars:
        print('Eddy decomposition for ' + nm)
        comp = eddy_decomp(data[nm], ndays, lon1, lon2, taxis)
        for compnm in comp:
            data[compnm] = comp[compnm]

    # Momentum budget calcs
    # du/dt = sum of terms in ubudget
    ubudget = xray.Dataset()
    readme = 'Momentum budget: ACCEL = sum of all other data variables'
    ubudget.attrs['readme'] = readme
    ubudget.attrs['ndays'] = ndays
    ubudget.attrs['lon1'] = lon1
    ubudget.attrs['lon2'] = lon2

    # Advective terms
    keypairs = [ ('AVG', 'AVG'), ('AVG', 'ST'), ('ST', 'AVG')]
    print('Computing advective terms')
    for pair in keypairs:
        print(pair)
        ukey, flowkey = pair
        u = data['U_' + ukey]
        dudp = data['DUDP_' + ukey]
        uflow = data['U_' + flowkey]
        vflow = data['V_' + flowkey]
        omegaflow = data['OMEGA_' + flowkey]
        adv = advection(uflow, vflow, omegaflow, u, dudp)
        for nm in adv.data_vars:
            key = 'ADV_%s_%s_%s' % (ukey, flowkey, nm)
            ubudget[key] = - adv[nm]
            long_name = 'Advection of %s momentum by %s' % (ukey, flowkey)
            ubudget[key].attrs['long_name'] = long_name

    # EMFD terms
    keys = ['TR', 'ST']
    print('Computing EMFD terms')
    for key in keys:
        print(key)
        u = data['U_' + key]
        v = data['V_' + key]
        omega = data['OMEGA_' + key]
        dudp = data['DUDP_' + key]
        domegadp = data['DOMEGADP_' + key]
        emfd = fluxdiv(u, v, omega, dudp, domegadp)
        for nm in emfd.data_vars:
            ubudget['EMFC_%s_%s' % (key, nm)] = - emfd[nm]

    # Coriolis terms
    latlon = latlon_data(data['V_ST'])
    lat = latlon['LAT']
    f = atm.coriolis(lat)
    ubudget['COR_AVG'] = data['V_AVG'] * f
    ubudget['COR_ST'] = data['V_ST'] * f

    # Pressure gradient terms
    a = atm.constants.radius_earth.values
    coslat = latlon['COSLAT']
    lonrad = latlon['LONRAD']
    londim = atm.get_coord(data['PHI_ST'], 'lon', 'dim')
    ubudget['PGF_ST'] = - atm.gradient(data['PHI_ST'], lonrad, londim) / (a*coslat)

    # Analysis increment for dU/dt
    ubudget['ANA'] = data['DUDTANA']

    # Time mean
    print('Computing rolling time mean')
    for nm in ubudget.data_vars:
        ubudget[nm] = atm.rolling_mean(ubudget[nm], ndays, axis=taxis, center=True)

    # Acceleration
    nseconds = 60 * 60 * 24 * ndays
    delta_u = np.nan * data['U']
    u = data['U'].values
    delta_u.values[ndays//2:-ndays//2] = (u[ndays:] - u[:-ndays]) / nseconds
    ubudget['ACCEL'] = delta_u

    return ubudget, data


# ----------------------------------------------------------------------
def v_components(ubudget, scale=None, eqbuf=5.0):
    """Return mean, eddy-driven, etc components of v for streamfunction.
    """

    comp_dict = {'MMC' : 'ADV_AVG', 'PGF' : 'PGF_ST', 'EDDY_ST' : 'EMFC_ST',
                 'EDDY_TR' : 'EMFC_TR', 'EDDY_CRS' : 'ADV_CRS'}

    if scale is not None:
        ubudget = ubudget * scale
    latname = atm.get_coord(ubudget, 'lat', 'name')
    lat = ubudget[latname]
    f = atm.coriolis(lat)
    f[abs(lat) < eqbuf] = np.nan

    v = xray.Dataset()
    v['TOT'] = ubudget['COR'] / f
    for nm in sorted(comp_dict):
        v[nm] = - ubudget[comp_dict[nm]] / f
    v['EDDY'] = v['EDDY_CRS'] + v['EDDY_TR'] + v['EDDY_ST']
    v['RESID'] = v['TOT'] - v['MMC'] - v['PGF'] - v['EDDY']

    return v
