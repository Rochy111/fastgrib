"""Grids: every point where wgrib2 and ecCodes put it -- projections,
scanning modes, rotated and reduced grids, ellipsoids -- and winds turned
from the grid's axes to east and north."""
import numpy as np
import pytest

import fastgrib

from helpers import grib1, path, reference, same


@pytest.mark.parametrize('name', ['nps', 'sps', 'merc', 'gauss', 'lcc', 'll_ns',
                                  'lcc_wgs84', 'nps_wgs84', 'merc_wgs84'])
def test_latlon_is_wgrib2s(name):
    f = fastgrib.open(path(name))
    lat, lon = f.latlon()
    w = np.load(path(name, '_latlon.npz'))
    wlat, wlon = w['lat'].reshape(lat.shape), w['lon'].reshape(lat.shape) % 360
    assert np.abs(lat - wlat).max() < 1e-4
    assert np.abs((lon - wlon + 180) % 360 - 180).max() < 1e-3   # wgrib2 -csv prints ~6 digits
    assert f.grid().shape == lat.shape


def test_scanning_modes():
    """Columns consecutive (0x20), every other row reversed (0x10), with
    0x80 / 0x40 and bitmaps: the same values and lat/lon as the original;
    fields differing only in 0x20 / 0x10 read as one stack."""
    f, g = fastgrib.open(path('gfs_scan')), fastgrib.open(path('gfs_simple'))
    assert len({r.grid.scan for r in f.records}) == 10 and any(r.bitmap >= 0 for r in f.records)
    for r, w in zip(f.records, g.records):
        assert np.array_equal(f.read(r), g.read(w), equal_nan=True), repr(r)
        assert all(np.array_equal(a, b) for a, b in zip(f.latlon(r), g.latlon(w)))
    alike = [r for r in f.records if r.grid.scan & 0xC0 == 0]          # 0x20, 0x10, 0x30
    assert len({r.grid.scan for r in alike}) == 3
    assert all(np.array_equal(v, g.read(r.index - 1), equal_nan=True)
               for r, v in zip(alike, f.read(alike)))


def test_mixed_scans_with_bitmaps(route):
    """Fields of one layout but other scans, the later one bitmapped, in
    one read (its map keyed by scan as well)."""
    f, ref = reference('gfs_scan')
    picks = [r for r in f.records if r.index in (10, 18)]           # 0xf0, then 0xd0 bitmapped
    assert picks[1].bitmap >= 0 and picks[0].grid.scan != picks[1].grid.scan
    assert all(same(v, ref[r.index - 1]) for r, v in zip(picks, f.read(picks)))


def test_ellipsoid_longitudes_and_probability_units():
    for name in ('lcc_wgs84', 'merc_wgs84', 'nps_wgs84'):
        lon = fastgrib.open(path(name)).latlon()[1]
        assert lon.min() >= 0 and lon.max() < 360, name
    f = fastgrib.open(path('nbm'))
    probs = [r for r in f.records if r.product in (5, 9)]
    assert probs and all(r.units == '%' for r in probs)


@pytest.mark.parametrize('name', ['lcc', 'nps'])
def test_winds_to_earth(name, route):
    """Winds given along a Lambert / polar grid's axes, turned to east and
    north, against wgrib2's own east-north interpolation of the same winds
    (to their packing, 0.1 m/s)."""
    g, e = fastgrib.open(path(name + '_uvgrid')), fastgrib.open(path(name + '_uvearth'))
    assert all(r.uv_relative for r in g.records) and not any(r.uv_relative for r in e.records)
    u, v = g.winds()
    ue, ve = e.winds()
    assert u.shape == (2,) + g.grid().shape
    assert np.nanmax(np.abs(u - ue)) < 0.12 and np.nanmax(np.abs(v - ve)) < 0.12
    us, vs = g.winds(earth=False)                       # as stored
    assert np.array_equal(us, g.read(name='UGRD'), equal_nan=True)
    u850, v850 = g.winds(level=850)
    assert u850.ndim == 2 and np.array_equal(u850, u[0])
    from fastgrib import _winds
    a, b = _winds.to_grid(ue, ve, g.grid())
    assert np.nanmax(np.abs(a - us)) < 0.12 and np.nanmax(np.abs(b - vs)) < 0.12


def test_vector_axes_geometry():
    """Both hemispheres: the grid x axis's bearing measured from latlon()
    along each row is the angle the rotation uses."""
    from fastgrib import _grids, _winds
    grids = [fastgrib.open(path(n)).grid() for n in ('nps', 'sps', 'lcc')]
    p = list(grids[2].params)
    p[0], p[3], p[4], p[7], p[8] = -p[0], -p[3], -p[4], True, -p[8]
    grids.append(grids[2]._replace(params=tuple(p)))                 # a southern Lambert
    for g in grids:
        la, lo = map(np.radians, _grids.latlon(g))
        dlon = (lo[:, 2:] - lo[:, :-2] + np.pi) % (2 * np.pi) - np.pi
        meas = np.arctan2(la[:, 2:] - la[:, :-2], dlon * np.cos(la[:, 1:-1]))
        c, s = _winds.angle(g)
        err = (meas - np.arctan2(s, c)[:, 1:-1] + np.pi) % (2 * np.pi) - np.pi
        assert np.degrees(np.abs(err)).max() < 0.2, g


ROTRED = ['rot_ll', 'rot_ll_ns', 'rot_gg', 'red_gg', 'red_rot_gg']


@pytest.mark.parametrize('name', ROTRED)
def test_rotated_and_reduced_latlon_are_eccodes(name):
    """Each point's latitude and longitude as ecCodes gives it (which does
    not unrotate a regular rotated Gaussian grid: rot_gg's go through our
    unrotation, the one checked on the other rotated grids)."""
    from fastgrib import _cpu, _grids
    f = fastgrib.open(path(name))
    r = f.records[0]
    ec = np.load(path(name, '_ec.npz'))
    elat, elon = ec['lat'], ec['lon']
    if name == 'rot_gg':
        elat, elon = _grids.unrotate(elat, elon, *r.grid.params[6:9])
    at = _cpu._defined(f._map, r)
    lat, lon = (a.ravel() for a in f.latlon())
    assert np.abs(lat[at] - elat).max() < 1e-5
    assert np.abs((lon[at] - elon + 180) % 360 - 180).max() < 1e-5
    assert np.abs(f.read(r).ravel()[at] - ec['values']).max() < 1e-4
    assert lon.min() >= 0 and lon.max() < 360


def test_rotated_and_reduced_shapes_and_xarray():
    pytest.importorskip('xarray')
    red = fastgrib.open(path('red_gg'))
    assert red.read(0).shape == (6114,) == red.latlon()[0].shape
    ds = red.to_xarray()
    v = next(iter(ds.data_vars.values()))
    assert v.dims == ('values',) and ds['latitude'].shape == (6114,)
    rot = fastgrib.open(path('rot_ll')).to_xarray()
    v = next(iter(rot.data_vars.values()))
    assert v.dims == ('rlat', 'rlon') and v.attrs['grid_mapping'] == 'crs'
    assert rot['crs'].attrs['grid_mapping_name'] == 'rotated_latitude_longitude'
    assert rot['latitude'].dims == ('rlat', 'rlon')


def test_rotated_proj4_and_wind_axes():
    """proj4() places the rotated grid's points as our unrotation does; the
    winds' axes there are the rotated rows' bearings."""
    pyproj = pytest.importorskip('pyproj')
    from fastgrib import _grids, _winds
    for name in ('rot_ll', 'rot_ll_ns', 'red_rot_gg'):
        g = fastgrib.open(path(name)).grid()
        rlat, rlon = _grids.rotated(g)
        lat, lon = _grids.latlon(g)
        t = pyproj.Transformer.from_crs(pyproj.CRS(_grids.proj4(g)),
                                        pyproj.CRS('+proj=longlat +a=6371229 +b=6371229'),
                                        always_xy=True)
        x, y = t.transform(rlon, rlat)
        assert np.abs(y - lat).max() < 1e-9 and np.abs((x - lon + 180) % 360 - 180).max() < 1e-9
    g = fastgrib.open(path('rot_ll')).grid()
    la, lo = map(np.radians, _grids.latlon(g))
    dlon = (lo[:, 2:] - lo[:, :-2] + np.pi) % (2 * np.pi) - np.pi
    meas = np.arctan2(la[:, 2:] - la[:, :-2], dlon * np.cos(la[:, 1:-1]))
    c, s = _winds.angle(g)
    err = (meas - np.arctan2(s, c)[:, 1:-1] + np.pi) % (2 * np.pi) - np.pi
    assert np.degrees(np.abs(err)).max() < 0.01


def test_space_view_is_wgrib2s():
    """Space views (3.90): each point where wgrib2 puts it (CGMS's
    navigation), NaN where the satellite looks past the Earth (wgrib2: a
    999 placeholder); ecCodes agrees where its rows run north to south
    (with j scanning positively it still takes them so, where WMO's flag
    and wgrib2 say south to north)."""
    f = fastgrib.open(path('sv'))
    w = np.load(path('sv', '_latlon.npz'))
    z = np.load(path('sv_ec', '.npz'))
    at = 0
    for k, r in enumerate(f.records):
        lat, lon = f.latlon(r)
        n = lat.size
        wlat, wlon = w['lat'][at:at + n].reshape(lat.shape), w['lon'][at:at + n].reshape(lat.shape)
        at += n
        off = wlat > 90
        assert np.array_equal(np.isnan(lat), off), r
        assert np.abs(lat - wlat)[~off].max(initial=0) < 1e-4
        assert np.abs((lon - wlon + 180) % 360 - 180)[~off].max(initial=0) < 1e-3
        if not r.grid.scan & 0x40:              # ecCodes' points, value by value
            llv = z['llv%d' % k]
            seen = ~((llv[:, 0] == 0) & (llv[:, 1] == 0))
            where = np.argsort(f.read(r).ravel())[llv[seen, 2].astype(np.int64)]
            assert np.abs(lat.ravel()[where] - llv[seen, 0]).max(initial=0) < 1e-9
            assert np.abs((lon.ravel()[where] - llv[seen, 1] + 180) % 360 - 180).max(initial=0) < 1e-9
    assert np.isnan(f.latlon(5)[0]).all()      # a sector past the Earth's edge


def test_space_view_regrid_and_ellipsoid():
    """at() on a space view gives each point's own value at its own
    latitude and longitude (the forward navigation inverts the inverse),
    NaN past the Earth; on an ellipsoid, the points are PROJ's geos
    projection's (MSG's y sweep)."""
    pyproj = pytest.importorskip('pyproj')
    from fastgrib import _grids
    f = fastgrib.open(path('sv'))
    for k in range(5):
        lat, lon = f.latlon(k)
        assert np.array_equal(f.at(lat, lon, k, method='nearest'), f.read(k), equal_nan=True)
        assert np.allclose(f.at(lat, lon, k), f.read(k), atol=1e-6, equal_nan=True)
    assert np.isnan(f.at(0.0, 180.0, 0))       # the far side of the Earth
    a, b = 6378169.0, 6356583.8
    g = _grids.Grid(371, 371, 0x00, 90, (0.0, -75.2, 362.2, 362.2, 185.6, 185.6, 0.0, 6.610839,
                                         0, 0), (a, b))
    lat, lon = _grids.space_view(g)
    x, y = np.meshgrid(*_grids._view_steps(g))
    h = 6.610839 * a
    rx = 2 * np.arcsin(1 / 6.610839) / 362.2
    p = pyproj.Proj(proj='geos', h=h - a, lon_0=-75.2, a=a, b=b, sweep='y')
    _same_points(lat, lon, *p(x * rx * (h - a), y * rx * (h - a), inverse=True)[::-1])


def _same_points(lat, lon, plat, plon, tol=1e-8):
    plat = np.where(np.abs(plat) > 1e20, np.nan, plat)
    assert np.array_equal(np.isnan(lat), np.isnan(plat))
    ok = ~np.isnan(lat)
    assert ok.sum() > lat.size // 4
    assert np.abs(lat - plat)[ok].max() < tol
    ok &= np.abs(lat) < 90 - 1e-9                # (a pole's longitude: any)
    assert np.abs((lon - plon + 180) % 360 - 180)[ok].max() < tol


def test_space_view_anywhere():
    """Space views neither wgrib2 nor ecCodes places: a camera off the
    equator, over a pole, its image turned -- as PROJ's tilted
    perspective (no tilt, the plane turned by the orientation) sees the
    Earth -- and orthographic views (Nr missing), as PROJ's orthographic;
    each point at() back to itself."""
    pyproj = pytest.importorskip('pyproj')
    from fastgrib import _grids
    R = 6371229.0
    nr = 6.610839
    for lap, lop, ori in ((35.0, -100.0, 30.0), (-60.0, 20.0, -45.0), (90.0, 0.0, 0.0),
                          (-90.0, 0.0, 120.0), (10.0, 140.0, 180.0)):
        g = _grids.Grid(81, 61, 0x40, 90, (lap, lop, 400.0, 400.0, 40.0, 30.0, ori, nr, 0, 0),
                        (R, R))
        lat, lon = _grids.space_view(g)
        x, y = np.meshgrid(*_grids._view_steps(g))
        ang = 2 * np.arcsin(1 / nr) / 400.0     # the scan angles: the tangent plane's x, y
        tx, ty = x * ang, y * ang
        hh = (nr - 1) * R
        # (over a pole GRIB's reference meridian is 180 E at the North, 0 at the South: the
        # one PROJ hangs below its pole)
        lon0 = 180.0 if lap == 90 else 0.0 if lap == -90 else lop
        p = pyproj.Proj(proj='tpers', lat_0=lap, lon_0=lon0, h=hh, azi=ori, tilt=0, R=R)
        _same_points(lat, lon, *p(hh * np.tan(tx), hh * np.tan(ty) / np.cos(tx),
                                  inverse=True)[::-1])
        c, r = _grids.space_view_steps(g, lat, lon)
        ok = ~np.isnan(lat)
        assert np.abs(c - np.arange(81)[None, :])[ok].max() < 1e-7
        assert np.abs(r - np.arange(61)[:, None])[ok].max() < 1e-7
    for lap, earth in ((0.0, (6378137.0, 6356752.314)), (40.0, (R, R)), (-75.0, (R, R))):
        g = _grids.Grid(90, 70, 0x40, 90, (lap, 10.0, 80.0, 80.0, 44.5, 34.5, 0.0, np.inf, 0, 0),
                        earth)
        lat, lon = _grids.space_view(g)
        x, y = np.meshgrid(*_grids._view_steps(g))
        m = 2 * earth[0] / 80.0
        p = pyproj.Proj(proj='ortho', lat_0=lap, lon_0=10.0, a=earth[0], b=earth[1])
        _same_points(lat, lon, *p(x * m, y * m, inverse=True)[::-1], tol=1e-7)
    # as messages (tools/testdata/make_other_grids.py, by ecCodes, which places none of
    # them): each point the camera's, at() back to itself; GRIB1's as ecCodes places it
    f = fastgrib.open(path('sv_more'))
    assert [r.grid.params[7] for r in f.records] == [6.610839, 6.610839, 6.610839, np.inf]
    for r in f.records:
        lat, lon = f.latlon(r)
        assert np.array_equal(lat, _grids.space_view(r.grid)[0], equal_nan=True)
        assert np.isfinite(lat).sum() > lat.size // 4
        assert np.array_equal(f.at(lat, lon, r, method='nearest'), f.read(r), equal_nan=True)
    g1 = grib1('sv1')
    lat, lon = g1.latlon(0)
    llv = np.load(path('sv1_ec', '.npz'))['llv']
    where = np.argsort(g1.read(0).ravel())[llv[:, 2].astype(np.int64)]
    assert np.abs(lat.ravel()[where] - llv[:, 0]).max() < 1e-9
    assert np.abs((lon.ravel()[where] - llv[:, 1] + 180) % 360 - 180).max() < 1e-9
    # an orthographic image turned is the same Earth, its axes turned
    g0 = _grids.Grid(91, 91, 0x40, 90, (30.0, 0.0, 80.0, 80.0, 45.0, 45.0, 0.0, np.inf, 0, 0), (R, R))
    g9 = g0._replace(params=(30.0, 0.0, 80.0, 80.0, 45.0, 45.0, 90.0, np.inf, 0, 0))
    a0, a9 = _grids.space_view(g0), _grids.space_view(g9)
    inner = slice(17, 74)                       # (on the disc, clear of its limb)
    for k in (0, 1):
        d = np.rot90(a0[k], -1)[inner, inner] - a9[k][inner, inner]
        assert np.abs((d + 180) % 360 - 180).max() < 1e-9


def test_staggered_grids():
    """Staggered grids (WMO's scanning flags 0x08/0x04/0x02/0x01: rows or
    points offset half a step, offset rows a point short): each point
    where wgrib2 puts it, at() back to itself; NCEP's rotated E-grid
    (3.32768) as the same grid in 3.1, its B-grid (3.32769, NAM's native)
    where NCEP's own NLAT/ELON put it (to their packing: 0.01 and 0.1
    degrees); NCEP's templates from other centres not placed."""
    f = fastgrib.open(path('stag'))
    w = np.load(path('stag', '_latlon.npz'))
    at = 0
    for r in f.records:
        lat, lon = f.latlon(r)
        n = lat.size
        assert np.abs(lat.ravel() - w['lat'][at:at + n]).max() < 1e-3, hex(r.grid.scan)
        assert np.abs((lon.ravel() - w['lon'][at:at + n] + 180) % 360 - 180).max() < 1e-3
        at += n
        v = f.read(r)
        assert v.shape == ((21, 31) if not r.grid.scan & 1 else (n,))
        assert np.array_equal(f.at(lat, lon, r, method='nearest'), v)
        assert np.allclose(f.at(lat, lon, r), v, atol=1e-6)
    assert at == w['lat'].size
    e = fastgrib.open(path('egrid'))
    assert e.keys(0)['gridDefinitionTemplateNumber'] == 32768
    la, lo = e.latlon(0)
    la2, lo2 = f.latlon(1)
    assert np.abs(la - la2).max() < 1e-6 and np.abs((lo - lo2 + 180) % 360 - 180).max() < 1e-6
    g1 = grib1('g1_egrid')                      # GRIB1's 203 and 205 (to its millidegrees)
    la1, lo1 = g1.latlon(0)
    assert np.abs(la1 - la).max() < 1e-3 and np.abs((lo1 - lo + 180) % 360 - 180).max() < 1e-3
    from fastgrib import _grids
    rl, ro = np.meshgrid(np.linspace(-4, 4, 21), np.linspace(-7.5, 7.5, 31), indexing='ij')
    la1, lo1 = g1.latlon(1)
    el, eo = _grids.unrotate(rl, ro, -50.0, 260.0)
    assert np.abs(la1 - el).max() < 1e-3 and np.abs((lo1 - eo + 180) % 360 - 180).max() < 1e-3
    nam = fastgrib.open(path('nam_bgrd'))
    ll = np.load(path('nam_bgrd_ll', '.npz'))
    lat, lon = nam.latlon(0)
    assert lat.shape == (835, 954)
    assert np.abs(lat.ravel()[::13] - ll['lat']).max() < 0.0062
    assert np.abs((lon.ravel()[::13] - ll['lon'] + 180) % 360 - 180).max() < 0.051
    b = bytearray(open(path('egrid'), 'rb').read())
    b[16 + 5:16 + 7] = (98).to_bytes(2, 'big')    # (another centre's 3.32768: its own)
    other = fastgrib.open(bytes(b))
    assert other.read(0).shape == (651,)
    with pytest.raises(ValueError):
        other.latlon(0)


def test_irregular_points():
    """Points listed one by one (3.130): each where wgrib2 and ecCodes say
    (ecCodes places none, its keys hold them); at() and regrid() the
    nearest of them by default, bilinear refused (no such thing here)."""
    f = fastgrib.open(path('irr'))
    lat, lon = f.latlon(0)
    v = f.read(0)
    assert v.shape == lat.shape == (500,) and f.keys(0)['gridType'] == 'irregular_latlon'
    w, z = np.load(path('irr', '_latlon.npz')), np.load(path('irr_ec', '.npz'))
    assert np.abs(lat - w['lat']).max() < 1e-3 and np.abs(lat - z['lat']).max() < 1e-9
    assert np.abs((lon - z['lon'] + 180) % 360 - 180).max() < 1e-9
    assert np.array_equal(f.at(lat, lon, 0), v)
    assert np.isfinite(f.regrid((np.arange(-80, 81, 40.0), np.arange(0, 360, 90.0)), 0)).all()
    with pytest.raises(ValueError, match='nearest'):
        f.at(10.0, 10.0, 0, method='bilinear')
    pytest.importorskip('xarray')
    ds = f.to_xarray()
    assert next(iter(ds.data_vars.values())).dims == ('values',) and 'latitude' in ds.coords


def test_grids_not_placed():
    """A grid fastgrib does not place (ICON's unstructured 3.101): its
    values read all the same, a row of them; its keys name its grid file;
    what needs its points says where they are."""
    f = fastgrib.open(path('ug', '.grb2'))
    v = f.read(0)
    assert v.shape == (20480,) and np.allclose(v, 250 + np.arange(20480) * 0.001, atol=1e-3)
    k = f.keys(0)
    assert (k['gridType'], k['numberOfGridUsed'], k['numberOfGridInReference']) == \
        ('unstructured_grid', 26, 1)
    assert k['uuidOfHGrid'] == 'a27b8de618c411e4820ab5b098c6a5c0'
    with pytest.raises(ValueError, match='grid file'):
        f.latlon()
    with pytest.raises(ValueError):
        f.at(10.0, 10.0, 0)
    pytest.importorskip('xarray')
    assert next(iter(f.to_xarray().data_vars.values())).dims == ('values',)
