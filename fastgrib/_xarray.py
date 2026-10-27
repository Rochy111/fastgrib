"""Fields as an xarray Dataset: a variable each parameter, level kind and
kind of forecast, laid out along whichever of reference time, ensemble
member, forecast step and level vary within it (dimensions 'time',
'member', 'step' and the level kind's name: 'isobaricInhPa', ...), as
cfgrib lays GRIB out; CF coordinates and grid mapping.

No field is ever left out: fields on other grids get their own spatial
dimensions (latitude_1, ...), and fields a variable's dimensions cannot
tell apart (the same field twice) are stacked along a '<name>_field'
dimension instead, with those coordinates along it."""
import re
from collections import Counter, OrderedDict
from datetime import timedelta

import numpy as np

from . import _grids

_AXES = ('time', 'member', 'step', None)        # None: the level kind's own name


def dataset(f, recs, earth_winds=False):
    import xarray as xr
    held = None
    if any(getattr(r, 'grid', None) is None for r in recs):
        # a remote file's index entries: their grids are known once fetched,
        # so the fields are read first (each variable's built from them in turn)
        held, entries, recs = {}, recs, []
        for r, v in f.fields(entries):
            held[r.index] = v.copy()
            recs.append(r)
    grids = list(OrderedDict.fromkeys(_grids.layout(r.grid) for r in recs))
    coords, space = OrderedDict(), {}
    for k, g in enumerate(grids):
        space[g] = _space(g, coords, '' if k == 0 else '_%d' % k)
    groups = OrderedDict()
    for r in recs:
        groups.setdefault(_key(r), []).append(r)
    names = _names(list(groups))
    specs = []                                  # (name, dims, attrs, shape, [(rec, slot)], full)
    for key, name in zip(groups, names):
        members = groups[key]
        r0 = members[0]
        dims, crs = space[_grids.layout(r0.grid)]
        attrs = {'units': r0.units, 'long_name': r0.long_name, 'GRIB_name': r0.name,
                 'GRIB_level_kind': _kind(r0)}
        if any(m.uv_relative for m in members):
            attrs['GRIB_uv_relative_to_grid'] = 1
        extra = _no_member(r0.extra)
        if extra:
            attrs['GRIB_extra'] = extra.rstrip(':')
        if crs:
            attrs['grid_mapping'] = crs
        for attr, vals in (('GRIB_fcst', {m.fcst for m in members}),
                           ('GRIB_level', {m.level for m in members}),
                           ('valid_time', {m.valid_time for m in members})):
            if len(vals) == 1 and None not in vals:
                v = vals.pop()
                attrs[attr] = v.isoformat() if hasattr(v, 'isoformat') else v
        at = [_at(m) for m in members]
        varying = [k for k in range(len(_AXES)) if len({a[k] for a in at}) > 1]
        cube = (len(set(at)) == len(at)
                and all(None not in {a[k] for a in at} for k in varying)
                and len({type(a[3]) for a in at}) == 1)          # layers or levels, not both
        if cube:                                # time x member x step x level, as present
            vdims, index = (), []
            for k in varying:
                vals = sorted({a[k] for a in at})
                dname = _dim(coords, _AXES[k] or _kind(r0), vals, r0)
                vdims += (dname,)
                index.append({v: i for i, v in enumerate(vals)})
            shape = tuple(len(ix) for ix in index)
            full = len(members) == int(np.prod(shape))
            place = [(m, int(np.ravel_multi_index(tuple(ix[a[k]] for ix, k in zip(index, varying)),
                                                  shape)) if shape else 0)
                     for m, a in zip(members, at)]
        else:                                   # fields the axes cannot tell apart: in file order
            fdim = name + '_field'
            vdims = (fdim,)
            coords[fdim] = ((fdim,), [m.index for m in members], {'long_name': 'record number'})
            for k, axis in enumerate(_AXES):
                vals = [a[k] for a in at]
                if k in varying and None not in vals:
                    if any(isinstance(v, tuple) for v in vals):          # layers: 'a-b'
                        vals = np.array(['%g-%g' % v if isinstance(v, tuple) else '%g' % v
                                         for v in vals])
                    else:
                        vals = _coord(axis, vals)
                    coords['%s_%s' % (name, axis or 'level')] = ((fdim,), vals, {})
            shape, full = (len(members),), True
            place = [(m, i) for i, m in enumerate(members)]
        specs.append((name, vdims + dims, attrs, shape, place, full))
    data = OrderedDict((s[0], None) for s in specs)
    if held is not None:                        # a remote file's fields, read already
        uses = Counter(r.index for r in recs)   # a field picked twice fills two slots
        for name, vdims, attrs, shape, place, full in specs:
            ny_nx = place[0][0].grid.shape
            arr = _blank(shape, ny_nx, full)
            flat = arr.reshape((-1,) + ny_nx)
            for m, k in place:
                flat[k] = held[m.index]
                uses[m.index] -= 1
                if not uses[m.index]:
                    del held[m.index]               # let each go once placed
            data[name] = xr.Variable(vdims, arr, attrs)
    else:
        _fill(f, specs, data, xr)
    if earth_winds:
        _earth(f, specs, data)
    refs = sorted({r.ref_time for r in recs if r.ref_time is not None})
    if len(refs) == 1 and 'time' not in coords:
        coords['time'] = ((), np.datetime64(refs[0], 'ns'), {'long_name': 'reference time'})
    return xr.Dataset(data, coords={k: xr.Variable(*v) for k, v in coords.items()},
                      attrs={'source': f.name, 'Conventions': 'CF-1.8',
                             'history': 'decoded by fastgrib'})


def _blank(shape, ny_nx, full):
    """A variable's array: to be filled whole, or NaN where no field lands."""
    return (np.empty if full else lambda s, d: np.full(s, np.nan, d))(shape + ny_nx, np.float32)


def _earth(f, specs, data):
    """The vector components given along the grid's axes turned to east
    and north, in place: each u with its v, where both are in the Dataset.
    Variables are marked GRIB_uv_relative_to_grid 0 or 1 as they end."""
    from . import _file, _winds
    where = {}
    for name, vdims, attrs, shape, place, full in specs:
        for m, k in place:
            where.setdefault(m.index, []).append((name, k))
    flat = {n: v.values.reshape((-1,) + v.shape[-2:]) for n, v in data.items()}
    turned = set()
    for name, vdims, attrs, shape, place, full in specs:
        for m, k in place:
            if m.name not in _file._V_NAME or not m.uv_relative:
                continue
            try:
                v = f._v_of(m)
            except KeyError:
                continue
            for (vn, vk) in where.get(v.index, ()):
                if (name, k, vn, vk) in turned:
                    continue
                g = _grids.layout(m.grid)
                flat[name][k], flat[vn][vk] = _winds.to_earth(flat[name][k], flat[vn][vk], g)
                turned.add((name, k, vn, vk))
                data[name].attrs['GRIB_uv_relative_to_grid'] = 0
                data[vn].attrs['GRIB_uv_relative_to_grid'] = 0


def _fill(f, specs, data, xr):
    """The variables of a local file, decoded straight into their memory:
    those every slot of which a field fills are views of one array a grid,
    read at once (no copies, no batch ring); the rest, NaN where nothing
    lands, are filled field by field."""
    by_grid = OrderedDict()
    for s in specs:
        by_grid.setdefault(_grids.layout(s[4][0][0].grid) if s[5] else None, []).append(s)
    for g, group in by_grid.items():
        if g is None:                           # sparse cubes
            slot = {}
            for name, vdims, attrs, shape, place, full in group:
                ny_nx = place[0][0].grid.shape
                arr = _blank(shape, ny_nx, False)
                flat = arr.reshape((-1,) + ny_nx)
                for m, k in place:
                    slot[m.index] = (flat, k)
                data[name] = xr.Variable(vdims, arr, attrs)
            for r, v in f.fields([m for s in group for m, _ in s[4]]):
                flat, k = slot[r.index]
                flat[k] = v
            continue
        big = np.empty((sum(len(s[4]) for s in group),) + g.shape, np.float32)
        order, at = [], 0
        for name, vdims, attrs, shape, place, full in group:
            n = len(place)
            order += [m for m, _ in sorted(place, key=lambda mk: mk[1])]
            data[name] = xr.Variable(vdims, big[at:at + n].reshape(shape + g.shape), attrs)
            at += n
        f._read(order, False, False, big, fresh=True)       # new memory: the CPU's


def _key(r):
    """What makes a variable: the parameter, level kind, grid, kind of
    forecast (an instant, or a statistic over how long) and the rest of
    what wgrib2 says (probability, percentile, ensemble statistic) -- not
    the member, step, level or reference time, which become its axes."""
    return (r.name, _kind(r), _fcst_kind(r.fcst), _no_member(r.extra), _grids.layout(r.grid))


def _kind(r):
    """A record's level kind for laying it out: ecCodes' -- but pressure
    levels all in hPa, whole or not, so a profile stays one variable."""
    k = r.level_kind
    return 'isobaricInhPa' if k == 'isobaricInPa' else k


def _at(r):
    """Where r lies on the axes: (reference time, member, step, level); a
    layer's level is its two sides."""
    step = r.fcst_hours
    lev = r.level_value
    if r.level_type2 == r.level_type and r.level_value2 is not None and lev is not None:
        lev = (lev, r.level_value2)
    return (r.ref_time, r.member, None if step is None else step, lev)


def _fcst_kind(fcst):
    m = re.match(r'^(\d+)-(\d+) (\S+) (.+) fcst$', fcst)
    if m:
        return '%d %s %s' % (int(m.group(2)) - int(m.group(1)), m.group(3), m.group(4))
    if fcst == 'anl' or re.match(r'^\d+ \S+ fcst$', fcst):
        return ''
    return fcst


def _no_member(extra):
    return re.sub(r'(^|:)(MM-)?ENS=[^:]*:', r'\1', extra)


def _coord(axis, vals):
    if axis == 'time':
        return np.array([np.datetime64(v, 'ns') for v in vals])
    if axis == 'step':
        return np.array([np.timedelta64(timedelta(hours=v)) for v in vals]).astype('m8[ns]')
    if axis == 'member':
        return np.array(vals, np.int64)
    return np.array([float('%.7g' % v) for v in vals])      # levels: the decimals written


def _dim(coords, base, vals, r0):
    """The dimension for these values: base's own name when free or holding
    the same values, else base_1, base_2, ... Layers go by their upper side
    where that tells them apart, else as 'a-b', each side a coordinate too."""
    axis = base if base in ('time', 'member', 'step') else None
    sides = None
    if vals and isinstance(vals[0], tuple):
        tops = [v[0] for v in vals]
        if len(set(tops)) == len(tops):
            vals = tops
        else:
            sides, vals = vals, ['%g-%g' % v for v in vals]
    arr = np.array(vals) if sides else _coord(axis, vals)
    k, name = 0, base
    while name in coords and not np.array_equal(np.asarray(coords[name][1]), arr):
        k += 1
        name = '%s_%d' % (base, k)
    if name not in coords:
        attrs = ({'long_name': 'ensemble member'} if axis == 'member' else
                 {'long_name': 'reference time'} if axis == 'time' else
                 {'long_name': 'time since the reference time'} if axis == 'step' else
                 {'GRIB_level_kind': _kind(r0)})
        coords[name] = ((name,), arr, attrs)
        if sides:
            for k, side in enumerate(('first', 'second')):
                coords['%s_%s' % (name, side)] = ((name,), _coord(None, [v[k] for v in sides]),
                                                  {'long_name': "the layer's %s side" % side})
    return name


def _names(keys):
    """A variable name each key: the bare parameter where it is unique, its
    level kind added where not, then what tells the rest apart."""
    out = []
    for key in keys:
        name, kind = key[0], key[1]
        same = [k for k in keys if k[0] == name]
        if len(same) == 1:
            out.append(name)
            continue
        n = '%s_%s' % (name, kind)
        alike = [k for k in same if k[1] == kind]
        if len(alike) > 1:
            parts = [key[i] for i in (2, 3) if len({k[i] for k in alike}) > 1 and key[i]]
            grid = [k[4] for k in alike]
            if len(set(grid)) > 1 and grid.index(key[4]):
                parts.append('grid%d' % list(OrderedDict.fromkeys(grid)).index(key[4]))
            if parts:
                n += '_' + re.sub(r'[^0-9A-Za-z%]+', '_', ' '.join(parts)).strip('_')
        out.append(n)
    seen = {}
    for i, n in enumerate(out):                 # still alike (a name made twice): number them
        if n in seen:
            seen[n] += 1
            out[i] = '%s_%d' % (n, seen[n])
        else:
            seen[n] = 1
    return [n.replace('%', 'pct') for n in out]


def _space(g, coords, sfx):
    """(dims, grid-mapping name or None) of a grid's two dimensions, its
    coordinates added to coords, names ending in sfx."""
    if g.template not in _grids._PLACED:        # coefficients, or points not placed: values
        return ('values' + sfx,), None
    if g.pl or g.template == 130:               # reduced, or listed: its points in a row
        dim = ('values' + sfx,)
        lat, lon = _grids.latlon(g)
        coords['latitude' + sfx] = (dim, lat, {'units': 'degrees_north',
                                               'standard_name': 'latitude'})
        coords['longitude' + sfx] = (dim, lon, {'units': 'degrees_east',
                                                'standard_name': 'longitude'})
        if g.template not in _grids._ROTATED:
            return dim, None
        coords['crs' + sfx] = ((), 0, _crs(g))
        return dim, 'crs' + sfx
    if g.template in _grids._ROTATED and _grids.stagger(g) is None:  # rotated: CF's rotated pole
        dims = ('rlat' + sfx, 'rlon' + sfx)
        rlat, rlon = _grids._axes(g)
        coords['rlat' + sfx] = ((dims[0],), rlat, {'units': 'degrees',
                                                   'standard_name': 'grid_latitude'})
        coords['rlon' + sfx] = ((dims[1],), rlon, {'units': 'degrees',
                                                   'standard_name': 'grid_longitude'})
        lat, lon = _grids.latlon(g)
        coords['latitude' + sfx] = (dims, lat, {'units': 'degrees_north',
                                                'standard_name': 'latitude'})
        coords['longitude' + sfx] = (dims, lon, {'units': 'degrees_east',
                                                 'standard_name': 'longitude'})
        coords['crs' + sfx] = ((), 0, _crs(g))
        return dims, 'crs' + sfx
    ax = _grids.axes(g)
    if ax is not None:
        lat, lon = ax
        coords['latitude' + sfx] = (('latitude' + sfx,), lat,
                                    {'units': 'degrees_north', 'standard_name': 'latitude'})
        coords['longitude' + sfx] = (('longitude' + sfx,), lon % 360.0,
                                     {'units': 'degrees_east', 'standard_name': 'longitude'})
        return ('latitude' + sfx, 'longitude' + sfx), None
    yx = ('y' + sfx, 'x' + sfx)
    lat, lon = _grids.latlon(g)
    coords['latitude' + sfx] = (yx, lat, {'units': 'degrees_north', 'standard_name': 'latitude'})
    coords['longitude' + sfx] = (yx, lon, {'units': 'degrees_east', 'standard_name': 'longitude'})
    xy = _cf_xy(g)
    if xy is not None:
        coords['x' + sfx] = ((yx[1],), xy[0], {'units': 'm',
                                               'standard_name': 'projection_x_coordinate'})
        coords['y' + sfx] = ((yx[0],), xy[1], {'units': 'm',
                                               'standard_name': 'projection_y_coordinate'})
    coords['crs' + sfx] = ((), 0, _crs(g))
    return yx, 'crs' + sfx


def _cf_xy(g):
    """Projection x, y (m) as the CF grid mapping (and proj4()) places
    them, for a spherical earth; None otherwise."""
    a, b = g.earth or (6371229.0, 6371229.0)
    if abs(a - b) > 1e-6:
        return None
    x, y = _grids._xy(g)
    x, y = np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
    if g.template == 30:                        # from the cone's apex to lat_0's parallel
        p = g.params
        south = p[7]
        lt1, lt2, lat0 = (-p[3], -p[4], -p[8]) if south else (p[3], p[4], p[8])
        n, t = _grids._cone(lt1, lt2)
        rf = a * np.cos(lt1 * _grids.D2R) * t(lt1) ** n / n
        rho0 = rf / t(lat0) ** n
        y = y - rho0 if south else y + rho0
    return x, y


def _crs(g):
    """CF grid-mapping attributes of a projected grid."""
    a, b = g.earth or (6371229.0, 6371229.0)
    p = g.params
    att = {'proj4': _grids.proj4(g)}
    if abs(a - b) < 1e-6:
        att['earth_radius'] = a
    else:
        att['semi_major_axis'], att['semi_minor_axis'] = a, b
    if g.template == 30:
        att.update(grid_mapping_name='lambert_conformal_conic', standard_parallel=[p[3], p[4]],
                   longitude_of_central_meridian=p[2], latitude_of_projection_origin=p[8])
    elif g.template == 20:
        att.update(grid_mapping_name='polar_stereographic', standard_parallel=p[2],
                   straight_vertical_longitude_from_pole=p[3],
                   latitude_of_projection_origin=-90.0 if p[6] else 90.0)
    elif g.template == 10:
        att.update(grid_mapping_name='mercator', standard_parallel=p[2],
                   longitude_of_projection_origin=p[1])
    elif g.template in _grids._ROTATED:
        att.update(grid_mapping_name='rotated_latitude_longitude',
                   grid_north_pole_latitude=-p[6],
                   grid_north_pole_longitude=(p[7] + 180.0) % 360.0,
                   north_pole_grid_longitude=-p[8])
    return att
