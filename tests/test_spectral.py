"""Spherical harmonics: coefficients as ecCodes gives them, and synthesis."""
import numpy as np
import pytest

import fastgrib

from helpers import grib1, path


def test_spectral_coefficients_are_eccodes(route):
    """Spherical-harmonic fields (GRIB2 3.50 with 5.51 complex and 5.50
    simple packing; GRIB1's, complex as GRIBEX writes it -- its subset's
    last wavenumber scaled by the Laplacian -- and simple): every
    coefficient as ecCodes gives it, to the bit."""
    z = np.load(path('sh_ec', '.npz'))
    for name, ext, stem in (('sh', '.grb2', 'sh'), ('g1_sh', '.grb', 'g1_sh')):
        f = grib1(name) if ext == '.grb' else fastgrib.open(path(name, ext))
        assert len(f.records) == 4 and not f.damaged
        for k, r in enumerate(f.records):
            assert r.grid.kind == 'spherical harmonics' and r.grid.params[:3] == (63, 63, 63)
            v = f.read(r)
            assert v.shape == (4160,)
            assert np.array_equal(v, z['v_%s%d' % (stem, k)].astype(np.float32)), (name, k)
    k = fastgrib.open(path('sh', '.grb2')).keys(0)
    assert (k['gridType'], k['J'], k['packingType'], k['TS']) == ('sh', 63, 'spectral_complex', 462)


def test_spectral_synthesis():
    """regrid()/at() synthesise spherical harmonics at any points, as
    pyshtools does (ECMWF's normalisation: the (0, 0) coefficient the
    global mean), GRIB1 and GRIB2 alike; the field is the one it should be
    (24 April, 12 UTC: the Sahara at 311 K, the Pacific at night 295)."""
    z = np.load(path('sh_ec', '.npz'))
    f = fastgrib.open(path('sh', '.grb2'))
    for k in (0, 2):
        got = f.at(z['lat'], z['lon'], k)
        assert np.abs(got - z['synth_sh%d' % k]).max() < 5e-5
    g = f.regrid((np.arange(-90, 90.1, 1.0), np.arange(0, 360, 1.0)), 0)
    w = np.cos(np.radians(np.arange(-90, 90.1, 1.0)))
    assert abs(np.average(g.mean(1), weights=w) - f.read(0)[0]) < 0.01
    assert abs(f.at(23, 10, 0) - 311.0) < 0.1 and abs(f.at(23, 190, 0) - 295.0) < 0.1
    g1 = grib1('g1_sh')
    assert np.allclose(g1.at(z['lat'], z['lon'], 0), f.at(z['lat'], z['lon'], 0), atol=1e-4)
    assert np.isnan(f.at(95.0, 0.0, 0))
    with pytest.raises(ValueError):
        f.latlon()
    assert np.array_equal(fastgrib.open(f.message(1)).read(0), f.read(1))


def test_spectral_encode():
    """encode() packs coefficients as 5.50 or 5.51: a field's own back to
    the bit, the other packing or fewer bits close, the field still the
    field (ecCodes reads them all as fastgrib does: checked when made)."""
    for f in (fastgrib.open(path('sh', '.grb2')), grib1('g1_sh')):
        for r in f.records:
            c = f.read(r)
            assert np.array_equal(fastgrib.open(f.encode(c, r)).read(0), c)
            for packing in ('simple', 'complex'):
                g = fastgrib.open(f.encode(c, r, packing=packing, bits=12))
                assert g.keys(0)['packingType'] == 'spectral_' + packing
                assert np.abs(g.read(0) - c).max() < 1e-5 * np.abs(c).max()
                assert abs(g.at(23, 10, 0) - f.at(23, 10, r)) < 0.3
    with pytest.raises(ValueError):
        f.encode(c[:-2], 0)
    with pytest.raises(ValueError):
        f.encode(c, 0, packing='png')


def test_spectral_rotated_stretched():
    """Rotated, stretched, and stretched and rotated spherical harmonics
    (GRIB2 3.51-3.53, GRIB1 60/70/80; ecCodes reads their coefficients and
    places none): coefficients as ecCodes's; the field synthesised in its
    own frame -- rotated as rotated grids are, stretched by Schmidt's
    transform about its pole (latitude 90 - 2 atan(c tan((90 - lat) / 2))
    there) -- GRIB1 as GRIB2."""
    from fastgrib._regrid import rotate
    z = np.load(path('sh_ec', '.npz'))
    base = fastgrib.open(path('sh', '.grb2'))  # its record 2: the same coefficients, unrotated
    f, g1 = fastgrib.open(path('sh_more', '.grb2')), grib1('g1_sh_more')
    assert [r.grid.kind for r in f.records] == [
        'rotated spherical harmonics', 'stretched spherical harmonics',
        'stretched rotated spherical harmonics']
    for k, r in enumerate(f.records):
        assert np.array_equal(f.read(r), z['v_shm%d' % k].astype(np.float32))
        assert np.array_equal(g1.read(k), z['v_g1_shm%d' % k].astype(np.float32))
        assert g1.records[k].grid.params[5:] == r.grid.params[5:]     # (their samples: modes 2, 1)
    lat, lon = z['lat'], z['lon']

    def stretched(la, c=2.4):
        return 90 - np.degrees(2 * np.arctan(c * np.tan(np.radians(90 - la) / 2)))
    rlat, rlon = rotate(lat, lon, -40.0, 10.0, 25.0)
    want = [base.at(rlat, rlon, 2), base.at(stretched(lat), lon, 2)]
    rlat, rlon = rotate(lat, lon, -46.5, 182.5, 0.0)    # (its stretching pole: the frame's north)
    want.append(base.at(stretched(rlat), rlon, 2))
    for k in range(3):
        assert np.abs(f.at(lat, lon, k) - want[k]).max() < 1e-3, k
        assert np.allclose(g1.at(lat, lon, k), f.at(lat, lon, k), atol=1e-3)
    k = f.keys(2)
    assert (k['gridType'], k['latitudeOfStretchingPoleInDegrees'], k['stretchingFactor'],
            k['latitudeOfSouthernPoleInDegrees']) == ('stretched_rotated_sh', 46.5, 2.4, -46.5)
    c = f.read(0)
    assert np.array_equal(fastgrib.open(f.encode(c, 0)).read(0), c)
    assert np.isfinite(f.regrid((np.arange(-90, 91, 30.0), np.arange(0, 360, 60.0)), 1)).all()
