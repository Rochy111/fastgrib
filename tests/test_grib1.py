"""GRIB1 read as the GRIB2 it would be: names, levels, times, grids
(predefined too), simple, second-order and spectral packing."""
import numpy as np
import pytest

import fastgrib

from helpers import grib1, path


G1_INVENTORY = [
    'TMP:2 m above ground:anl:', 'TMP:2 m above ground:anl:', 'TMP:2 m above ground:anl:',
    'TMP:1000 mb:anl:', 'TMP:2 m above ground:anl:', 'TMP:surface:anl:',
    'TMP:2 m above ground:anl:', 'TMP:2 m above ground:anl:', 'TMP:2 m above ground:anl:',
    'TMP:2 m above ground:anl:ENS=+5:', 'tp:surface:0-6 hour acc fcst:',
    'tp:surface:0-6 hour acc fcst:', 'TMP:2 m above ground:0-6 hour max fcst:',
    'TMP:850 mb:anl:', 'TMP:surface:anl:']


NARR_INVENTORY = [
    'PRMSL:mean sea level:anl:', 'CLMR:100 mb:anl:', 'TSOIL:0-0.1 m below ground:anl:',
    'UGRD:10 m above ground:anl:', 'VGRD:10 m above ground:anl:', 'LFTX:500-1000 mb:anl:',
    '4LFTX:180-0 mb above ground:anl:', 'APCP:surface:0-3 hour acc fcst:',
    'LHTFL:surface:0-3 hour ave fcst:']


def test_grib1_inventory():
    """ECMWF's parameters at the level and statistic they imply (2t at 2 m,
    tp accumulated -- also as ECMWF codes it, a step from the start -- mx2t
    a maximum), NCEP's by its own tables, an ensemble member, spherical
    harmonics; offsets the file's own."""
    for name, want in (('g1_ec', G1_INVENTORY), ('g1_narr', NARR_INVENTORY)):
        f = grib1(name)
        assert [repr(r).split(':', 3)[3] for r in f.records] == want
        raw = open(path(name, '.grb'), 'rb').read()
        assert all(raw[r.offset:r.offset + 4] == b'GRIB' and raw[r.offset + 7] == 1
                   for r in f.records)
    f = grib1('g1_ec')
    assert f.damaged == [] and f.records[-1].grid.kind == 'spherical harmonics'
    assert [r.short_name for r in f.records][:2] == ['2t', '2t']
    assert f.records[9].member == 5 and f.records[12].short_name == 'mx2t6'
    assert grib1('g1_narr').records[1].short_name == 'clwmr'


@pytest.mark.parametrize('name', ['g1_ec', 'g1_narr'])
def test_grib1_values_and_points_are_eccodes(name, route):
    """Every value bit-identical to ecCodes' (but constant fields: GRIB's
    formula scales the reference value by 10^-D, which ecCodes and g2clib
    skip -- NARR's 1e-12 kg/kg of cloud water at 100 hPa, not 0.001; wgrib2
    -g2clib 0 agrees), every point where ecCodes puts it (but polar
    stereographic, where ecCodes walks the rows north whatever the scanning
    mode says: checked against the projection below)."""
    import copy
    from fastgrib import _cpu, _grids
    f = grib1(name)
    z = np.load(path(name, '_ec.npz'))
    keys, st = list(z['keys']), int(z['stride'])
    for k, r in enumerate(f.records):
        kk = dict(zip(keys, z['keys%d' % k]))
        if 'lat%d' % k in z.files:
            elat, elon = z['lat%d' % k], z['lon%d' % k]
        full = copy.copy(r)
        full.bitmap = -1
        at = _cpu._defined(f._map, full)
        v, e = f.read(r).ravel()[at], z['v%d' % k]
        if r.short_name == 'clwmr':
            assert np.allclose(v[~np.isnan(v)], 1e-12) and np.allclose(e[~np.isnan(e)], 1e-3)
        else:
            assert np.array_equal(v, e, equal_nan=True), r
        if kk['gridType'] in ('polar_stereographic', 'sh'):     # (sh: coefficients, no points)
            continue
        la, lo = (a.ravel()[at][::st] for a in _grids.latlon(r.grid))
        if kk['gridType'] == 'rotated_gg':                  # ecCodes leaves these rotated
            elat, elon = _grids.unrotate(elat, elon, *r.grid.params[6:9])
        assert np.abs(la - elat).max() < 1e-5, r
        assert np.abs((lo - elon + 180) % 360 - 180).max() < 1e-5, r


def test_grib1_polar_rows_follow_scanning_mode():
    """GRIB1 polar stereographic, rows going south (jScansPositively 0):
    the first point is the top-left one, 60 km steps along x and down y in
    the grid's own projection (wgrib2 places them so too)."""
    pyproj = pytest.importorskip('pyproj')
    from fastgrib import _grids
    f = grib1('g1_ec')
    for r, first in ((f.records[7], (20.0, 230.0)), (f.records[8], (-20.0, 100.0))):
        g = r.grid
        lat, lon = _grids.latlon(g)
        assert np.allclose((lat[-1, 0], lon[-1, 0]), first)          # the file's first point
        x, y = pyproj.Transformer.from_crs('+proj=longlat +R=6367470', _grids.proj4(g),
                                           always_xy=True).transform(lon, lat)
        assert np.allclose(np.diff(x, axis=1), 60000, atol=0.01)
        assert np.allclose(np.diff(y, axis=0), 60000, atol=0.01)


def test_grib1_winds_xarray_and_mixed_files(tmp_path):
    """GRIB1's wind component flag: NARR's says east-north (read as they
    are); set to the grid's axes, turned to east-north keeping their speed.
    xarray of GRIB1; a file of GRIB1 and GRIB2 messages reads both, each at
    its own offset."""
    f = grib1('g1_narr')
    assert not any(r.uv_relative for r in f.records[3:5])
    raw = bytearray(open(path('g1_narr', '.grb'), 'rb').read())
    for r in f.records[3:5]:                    # the GDS's resolution and component flags
        p = r.offset + 8
        g = p + int.from_bytes(raw[p:p + 3], 'big')
        raw[g + 16] |= 0x08
    q = tmp_path / 'narr_uvgrid.grb'
    q.write_bytes(bytes(raw))
    f = fastgrib.open(str(q))
    u, v = f.records[3], f.records[4]
    assert u.uv_relative and v.uv_relative
    gu, gv = f.read(u), f.read(v)
    eu, ev = f.winds()
    assert np.allclose(np.hypot(eu, ev), np.hypot(gu, gv), atol=1e-4, equal_nan=True)
    assert np.nanmax(np.abs(eu - gu)) > 1                              # turned, not copied
    pytest.importorskip('xarray')
    ds = f.to_xarray()
    assert 'prmsl' in ds or 'PRMSL' in ds or any('prmsl' in k.lower() for k in ds.data_vars)
    g2 = open(path('lcc'), 'rb').read()
    g1 = open(path('g1_narr', '.grb'), 'rb').read()
    p = tmp_path / 'mixed.grb'
    p.write_bytes(g2 + g1 + g2)
    m = fastgrib.open(str(p))
    n2 = len(fastgrib.open(path('lcc')).records)
    assert len(m.records) == 2 * n2 + 9 and not m.damaged
    assert m.records[n2].offset == len(g2) and m.records[n2 + 9].offset == len(g2) + len(g1)
    assert np.array_equal(m.read(m.records[n2 + 1]), f.read(f.records[1]), equal_nan=True)
    assert [x.msg for x in m.records] == list(range(1, len(m.records) + 1))


def test_grib1_remote(server):
    """A remote GRIB1 by its wgrib inventory (.inv, as NCEI's NARR has):
    the entries worded as the records, the fetched messages read as the
    local file's."""
    f = fastgrib.open(server + 'g1_narr.grb')
    loc = grib1('g1_narr')
    assert [repr(r) for r in f.records] == [repr(r) for r in loc.records]
    picks = f.select(level='10 m above ground')
    assert [r.name for r in picks] == ['UGRD', 'VGRD']
    for r, v in zip(picks, f.read(picks)):
        assert np.array_equal(v, loc.read(r.index - 1), equal_nan=True)
    assert np.array_equal(f.read(name='CLMR'), loc.read(name='CLMR'), equal_nan=True)


def test_grib1_second_order(route):
    """GRIB1 second-order packing, every form, as ecCodes reads it, to the
    bit: ECMWF's general extended (no, 1st, 2nd, 3rd order spatial
    differencing; boustrophedonic rows; bitmaps; a reduced Gaussian grid)
    and WMO's row by row, constant width, general -- and written out as
    GRIB2 simple packing that holds the same values."""
    import copy
    from fastgrib import _cpu
    f = grib1('g1_2nd')
    z = np.load(path('g1_2nd_ec', '.npz'))
    assert len(f.records) == 11 and not f.damaged
    for k, r in enumerate(f.records):
        full = copy.copy(r)
        full.bitmap = -1
        got = f.read(r).ravel()[_cpu._defined(f._map, full)]
        assert np.array_equal(got, z['v%d' % k], equal_nan=True), (k, str(z['packing%d' % k]))
    g = fastgrib.open(f.message(0))
    assert g.keys(0)['packingType'] == 'grid_simple'
    assert np.array_equal(g.read(0), f.read(0), equal_nan=True)


def test_grib1_predefined_grids():
    """GRIB1 messages naming their grid instead of describing it (no GDS):
    WMO's international grids 21-26 and 61-64 (quarter globes, the pole a
    single value), NCEP's from ON388 Table B -- NARR's grid 221 placed to
    the bit as its own GDS places it. (ecCodes reads their values but not
    their points.)"""
    f = grib1('g1_predef')
    assert not f.damaged and len(f.records) == 13
    want = {0: (1333, (0, 0), (90, 0)), 1: (1333, (0, 180), (90, 180)),
            2: (1333, (-90, 0), (0, 180)), 5: (1297, (-90, 0), (0, 355)),
            9: (4096, (-90, 180), (0, 0))}
    for k, (n, first, last) in want.items():
        r = f.records[k]
        v = f.read(r)
        lat, lon = f.latlon(r)
        assert v.shape == (n,) and np.array_equal(v, np.arange(n, dtype=np.float32))
        assert np.allclose((lat[0], lon[0]), first) and np.allclose((lat[-1], lon[-1]), last)
    g = f.latlon(f.records[0])                          # 37 rows of 37, the pole once
    assert np.count_nonzero(g[0] == 90) == 1 and np.count_nonzero(g[0] == 0) == 37
    narr = grib1('g1_narr')
    for a, b in zip(narr.records[:3], f.records[10:]):
        assert repr(a).split(':', 2)[2] == repr(b).split(':', 2)[2]
        assert np.array_equal(narr.read(a), f.read(b), equal_nan=True)
    assert all(np.array_equal(x, y) for x, y in zip(narr.latlon(0), f.latlon(10)))


def test_grib1_matrix_of_values():
    """GRIB1's matrix of values at each grid point (tools/testdata/
    make_grib1_matrix.py, ECMWF's layout: secondary bitmaps, a primary
    one too): each element a field of its own, its points as present in
    both bitmaps, its values as made -- ecCodes, reading the first, takes
    its secondary bitmaps for values -- worded by its coordinates (given,
    and geometric); one value a point with the matrix flags, as ecCodes."""
    f = grib1('g1_matrix')
    z = np.load(path('g1_matrix_ec', '.npz'))
    assert len(f.records) == 13
    assert repr(f.records[4]).endswith(':matrix 2,2 direction=180 frequency=0.03795:')
    for m in (0, 1):
        present, sec, x = z['present%d' % m], z['sec%d' % m], z['x%d' % m]
        full = np.full((present.size, sec.shape[1]), np.nan)
        have = np.zeros(full.shape, bool)
        have[np.flatnonzero(present)] = sec
        full[have] = x                          # point by point, each point's elements
        for k in range(6):
            want = full[:, k].reshape(4, 6)[::-1]  # (the file's rows north first)
            assert np.array_equal(f.read(6 * m + k), want, equal_nan=True), (m, k)
    assert not np.array_equal(z['v0'][:112], z['x0'])
    assert np.array_equal(f.read(12)[::-1].ravel(), z['v2'])
