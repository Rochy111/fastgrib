"""Regridding: as wgrib2's -new_grid where IPOLATES is right, and points
where they are where it is not."""
import numpy as np
import pytest

import fastgrib

from helpers import UNDEF, needs_pyproj, path


REGRID = {'lcc': (25.528, 235.169, 0.348, 39, 63), 'nps': (20.015, 200.27, 0.556, 39, 50),
          'gauss': (-61.903, 232.5, 3.095, 40, 81), 'll_ns': (-65.0, 230.0, 3.25, 40, 78)}


def _axes_of(case):
    lat0, lon0, d, ny, nx = REGRID[case]
    return lat0 + d * np.arange(ny), lon0 + d * np.arange(nx)


def _step(f, r):
    """Half the packing step of r: wgrib2 packs what it interpolates to it."""
    import struct
    from fastgrib import _meta
    b = f._map
    e = _meta._sm(struct.unpack_from('>H', b, r.s5 + 15)[0], 2)
    d = _meta._sm(struct.unpack_from('>H', b, r.s5 + 17)[0], 2)
    return 0.5 * 2.0 ** e * 10.0 ** -d


@pytest.mark.parametrize('case', sorted(REGRID))
@pytest.mark.parametrize('method', ['bilinear', 'nearest'])
def test_regrid_is_wgrib2s(case, method):
    """regrid() onto a lat-lon grid as wgrib2 -new_grid (IPOLATES) does it
    -- to the packing wgrib2 gives its output, the same points defined
    (gaps and edges by IPOLATES' half-the-weight rule), nearest's ties as
    its NINT."""
    f = fastgrib.open(path(case))
    lats, lons = _axes_of(case)
    want = np.fromfile(path('regrid_%s_%s' % (case, 'neighbor' if method == 'nearest'
                                               else method), '.bin'), np.float32)
    want = np.where(want == UNDEF, np.nan, want).reshape(lats.size, lons.size)
    got = f.regrid((lats, lons), 0, method=method)
    assert got.shape == want.shape and got.dtype == np.float32
    assert np.array_equal(np.isnan(got), np.isnan(want))
    assert np.nanmax(np.abs(got - want)) <= _step(f, f.records[0]) * 1.001 + 1e-5


@pytest.mark.parametrize('name', ['gfs', 'lcc', 'nps', 'sps', 'merc', 'gauss', 'red_gg',
                                  'rot_ll', 'rot_ll_ns', 'red_rot_gg', 'lcc_wgs84', 'll_ns'])
def test_regrid_places_points(name):
    """Every grid's points where they are: onto its own grid, a field
    comes back as it is; a field of each point's own latitude and
    longitude, interpolated at the centre of each cell, gives the mean of
    its corners' (the middle of each row's two points, between two rows,
    on reduced grids) -- wgrib2's IPOLATES is 12 degrees out on Mercator
    and 0.4 on polar stereographic at this."""
    needs_pyproj(name)
    from fastgrib import _grids
    f = fastgrib.open(path(name))
    la, lo = f.latlon()
    x = f.read(0)
    assert np.array_equal(f.regrid(f.grid(), 0), x, equal_nan=True)
    assert np.array_equal(f.at(la, lo, 0), x, equal_nan=True)
    east = lo if name == 'merc' else (lo + 180.0) % 360.0 - 180.0   # no seam in the domain
    c = fastgrib.open(f.encode(la, 0, packing='ieee') + f.encode(east, 0, packing='ieee'))
    g = f.grid()
    if g.pl:                                    # reduced: rows' midpoints and between rows
        rows, start, k = _grids._rows(g)[:3]
        own = _grids.rotated(g)[1] if g.template in _grids._ROTATED else lo   # its frame's
        tl, tn = [], []
        for j in range(0, rows.size - 1, 3):
            a = start[j] + k[j] // 3
            tl += [rows[j], (rows[j] + rows[j + 1]) / 2]
            tn += [(own[a] + own[a + 1]) / 2 if own[a + 1] > own[a] else own[a]] * 2
        tl, tn = np.array(tl), np.array(tn)
        if g.template in _grids._ROTATED:       # built in the grid's frame: geographic
            tl, tn = _grids.unrotate(tl, tn, *g.params[6:9])
        got = c.at(tl, tn)
        want_lat = np.array(tl)
        keep = np.abs(tl) < 80
    else:                                       # each cell's centre: the mean of its corners
        def mid(a):
            return (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:]) / 4
        span = np.maximum.reduce([east[:-1, :-1], east[1:, :-1], east[:-1, 1:], east[1:, 1:]]) -             np.minimum.reduce([east[:-1, :-1], east[1:, :-1], east[:-1, 1:], east[1:, 1:]])
        keep = (np.abs(mid(la)) < 80) & (span < 30)           # off the poles and the seam
        v = np.radians(la), np.radians(lo)              # the centre on the sphere
        xyz = [mid(np.cos(v[0]) * np.cos(v[1])), mid(np.cos(v[0]) * np.sin(v[1])),
               mid(np.sin(v[0]))]
        tl = np.degrees(np.arctan2(xyz[2], np.hypot(xyz[0], xyz[1])))[keep]
        tn = np.degrees(np.arctan2(xyz[1], xyz[0]))[keep] % 360.0
        got = c.at(tl, tn)
        want_lat = mid(la)[keep]
        keep = np.ones(tl.size, bool)
    assert np.isfinite(got[:, keep]).all()
    cell = 180.0 / (len(g.pl) if g.pl else max(x.shape))    # a cell's size, about
    assert np.abs(got[0][keep] - want_lat[keep]).max() < 0.02 * cell
    tn_east = tn if name == 'merc' else (tn + 180.0) % 360.0 - 180.0
    keep &= (np.abs(tn_east) < 170) | (name == 'merc')     # (the field jumps at its seam)
    assert np.abs(got[1][keep] - tn_east[keep]).max() < 0.05 * cell / np.cos(np.radians(60))


def test_regrid_winds_and_targets():
    """Winds along the grid's axes come out east and north (a uniform
    west wind stays one, where wgrib2's nearest turns it by the target's
    angle); targets: a Grid, a GribFile, a Record, axes; at() for points;
    picks stack."""
    from fastgrib import _winds
    f = fastgrib.open(path('lcc_uvgrid'))
    u, g = f.select(name='UGRD')[0], f.grid()
    ug, vg = _winds.to_grid(np.full(g.shape, 10.0), np.zeros(g.shape), g)
    c = fastgrib.open(f.encode(ug, u, packing='ieee') + f.encode(vg, f._v_of(u), packing='ieee'))
    assert all(r.uv_relative for r in c.records)
    lats, lons = _axes_of('lcc')
    for method in ('bilinear', 'nearest'):
        uv = c.regrid((lats, lons), method=method)
        assert np.nanmax(np.abs(uv[0] - 10)) < 1e-4 and np.nanmax(np.abs(uv[1])) < 1e-4
        v_only = c.regrid((lats, lons), 1, method=method)    # a v alone: its u found
        assert np.allclose(v_only, uv[1], atol=1e-5, equal_nan=True)
    gfs = fastgrib.open(path('gfs'))
    a = gfs.regrid(fastgrib.open(path('lcc')), name='TMP')
    assert a.shape == (len(gfs.select(name='TMP')),) + g.shape
    assert np.array_equal(gfs.regrid(g, name='TMP'), a, equal_nan=True)
    assert np.array_equal(gfs.regrid(fastgrib.open(path('lcc')).records[0], name='TMP'), a,
                          equal_nan=True)
    one = gfs.at(40.0, 260.0, 0)
    assert np.ndim(one) == 0 and np.isfinite(one)
    assert gfs.at([40.0, 41.0], [260.0, 261.0], name='TMP').shape == (len(a), 2)
    assert np.isnan(gfs.at(-40.0, 10.0, 0))                 # off the grid
    with pytest.raises(ValueError):
        gfs.regrid((lats, lons), 0, method='cubic')


def _redtoreg(v, pl, nx):
    """pygrib's redtoreg (its Cython loop, transcribed), from float64."""
    out = np.empty((len(pl), nx))
    s = 0
    for j, k in enumerate(pl):
        for i in range(nx):
            zxi = i * float(k) / nx
            im = int(zxi)
            zdx = zxi - im
            im, ip = (im + k) % k, (im + 1 + k) % k
            a, b = v[s + im], v[s + ip]
            if np.isnan(a) or np.isnan(b):
                out[j, i] = a if zdx < 0.5 else b
            else:
                out[j, i] = a * (1.0 - zdx) + b * zdx
        s += k
    return out.astype(np.float32)


@pytest.mark.parametrize('name, ext', [('red_gg', '.grib2'), ('red_rot_gg', '.grib2'),
                                       ('g1_ec', '.grb')])
def test_expand_reduced_grids(name, ext):
    """A reduced Gaussian grid on its regular one: pygrib's redtoreg along
    each row (checked against pygrib 2.1.8: the same float32s), through
    read(), fields(), expand() and an out= array; the points those of a
    regular Gaussian grid of the same rows, rotated as the grid is."""
    import warnings
    from fastgrib import _grids
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f = fastgrib.open(path(name, ext))
    r = next(r for r in f.records if r.grid.pl)
    v = f.read(r)
    nx = max(r.grid.pl)
    rows = list(r.grid.pl)[::1 if r.grid.scan & 0x40 else -1]     # south to north, as read()
    e = f.read(r, expand=True)
    assert e.shape == (len(rows), nx) and e.dtype == np.float32
    assert np.array_equal(e, _redtoreg(v.astype(np.float64), rows, nx))
    gap = v.copy()
    gap[rows[0] + 3] = np.nan                   # a missing value: the nearer point's beside it
    assert np.array_equal(fastgrib.expand(gap, r), _redtoreg(gap.astype(np.float64), rows, nx),
                          equal_nan=True)
    assert np.array_equal(fastgrib.expand(v, r, nx=2 * nx)[:, ::2], e)     # (every other: on it)
    out = np.empty((2, len(rows), nx), np.float32)
    assert f.read([r, r], expand=True, out=out) is out and np.array_equal(out[1], e)
    got = [x.copy() for rr, x in f.fields([r, r], expand=True)]
    assert all(np.array_equal(x, e) for x in got)
    lat, lon = f.latlon(r, expand=True)
    assert lat.shape == lon.shape == e.shape
    la1, lo1, la2, lo2, _, n = r.grid.params[:6]
    want = _grids._gaussian(int(n), min(la1, la2), max(la1, la2), len(rows))[::-1]
    if r.grid.template == 41:                   # in the grid's frame: rotated back
        rl, ro = _grids.unrotate(np.repeat(want, nx).reshape(-1, nx)[::-1],
                                 (lo1 + np.arange(nx) * 360.0 / nx)[None, :].repeat(len(rows), 0),
                                 *r.grid.params[6:9])
        assert np.allclose(lat, rl) and np.allclose((lon - ro + 180) % 360 - 180, 0)
    else:
        assert np.allclose(lat[:, 0], np.sort(want)) and np.allclose(np.diff(lon[0]), 360.0 / nx)
    g = fastgrib.open(path('gfs'))              # not reduced: as read() gives it
    assert np.array_equal(g.read(0, expand=True), g.read(0), equal_nan=True)


# (file, box (west, east, south, north), its shape, first and last points as wgrib2
# -small_grib places them)
BOXES = [('gfs', (-100, -80, 30, 45), (16, 21), (30.0, 260.0), (45.0, 280.0)),
         ('gfs_scan', (-100, -80, 30, 45), (16, 21), (30.0, 260.0), (45.0, 280.0)),
         ('gfs_jpeg', (-100, -80, 30, 45), (16, 21), (30.0, 260.0), (45.0, 280.0)),
         ('ecmwf', (-10, 10, -5, 5), (41, 81), (-5.0, 350.0), (5.0, 10.0)),
         ('ecmwf', (170, -170, 40, 50), (41, 81), (40.0, 170.0), (50.0, 190.0)),
         ('gauss', (-30, 40, -20, 20), (22, 38), (-19.999708, 330.0), (19.999708, 39.375)),
         ('lcc', (-100, -90, 35, 42), (10, 3), (35.35748, 260.099653), None),
         ('merc', (200, 220, 0, 20), (15, 13), None, None),
         ('nps', (220, 250, 40, 60), (13, 8), None, None)]


@pytest.mark.parametrize('name, box, shape, first, last', BOXES,
                         ids=['%s:%s' % (b[0], ','.join(map(str, b[1]))) for b in BOXES])
def test_box(name, box, shape, first, last, tmp_path):
    """A box cut from fields (wgrib2's -small_grib): the rows and columns of
    the points in it (round the seam of a global grid: from the box's west
    edge on), a projected grid's rectangle round them; written as GRIB2 on
    the smaller grid, its values read back unchanged and its points where
    they were -- wgrib2's grid (its first and last points, checked against
    wgrib2 3.1.3's -small_grib)."""
    f = fastgrib.open(path(name))
    v = f.read(0, box=box)
    lat, lon = f.latlon(0, box=box)
    assert v.shape == lat.shape == shape
    full_lat, full_lon = f.latlon(0)
    inside = (full_lat >= box[2]) & (full_lat <= box[3]) & \
        (((full_lon - box[0]) % 360) <= (box[1] - box[0]) % 360)
    assert inside.sum() <= v.size and inside.sum() > 0
    if first is not None:
        assert np.allclose((lat[0, 0], lon[0, 0] % 360), first, atol=1e-5)
    if last is not None:
        assert np.allclose((lat[-1, -1], lon[-1, -1] % 360), last, atol=1e-5)
    n = min(5, len(f))
    out = tmp_path / 'box.grib2'
    assert f.write(str(out), f.records[:n], box=box) == n
    g = fastgrib.open(str(out))
    assert [repr(r).split(':', 2)[2] for r in g.records] == \
        [repr(r).split(':', 2)[2] for r in f.records[:n]]
    for k in range(n):
        assert np.array_equal(g.read(k), f.read(k, box=box), equal_nan=True)
    la, lo = g.latlon(0)
    assert np.allclose(la, lat, atol=1e-5) and np.allclose((lo - lon + 180) % 360 - 180, 0,
                                                           atol=1e-5)
    got = [x.copy() for _, x in f.fields(f.records[:2], box=box)]
    assert np.array_equal(got[-1], f.read(len(got) - 1, box=box), equal_nan=True)


def test_box_refused():
    """A box with no point of the grid in it, or on a grid it is not cut
    from, says so; so does a box that is not one."""
    f = fastgrib.open(path('gfs'))
    with pytest.raises(ValueError):
        f.read(0, box=(0, 10, -60, -50))
    with pytest.raises(ValueError):
        f.read(0, box=(0, 10, 50, 40))
    with pytest.raises(TypeError):
        f.read(0, box=(0, 10))
    with pytest.raises(ValueError):
        fastgrib.open(path('red_gg')).read(0, box=(0, 10, 0, 10))


def test_small_grib_command(tmp_path):
    from fastgrib.__main__ import main
    out = tmp_path / 'box.grib2'
    assert main([path('gfs'), '-match', ':TMP:', '-small_grib', '-100:-80', '30:45',
                 str(out)]) == 0
    f, g = fastgrib.open(path('gfs')), fastgrib.open(str(out))
    want = f.find(':TMP:')
    assert len(g.records) == len(want)
    assert np.array_equal(g.read(g.records), f.read(want, box=(-100, -80, 30, 45)),
                          equal_nan=True)
