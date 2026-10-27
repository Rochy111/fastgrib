"""Regridding: as wgrib2's -new_grid where IPOLATES is right, and points
where they are where it is not."""
import numpy as np
import pytest

import fastgrib

from helpers import UNDEF, path


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
