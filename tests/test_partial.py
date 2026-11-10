"""Partial reads: a simple or IEEE field's values at some points, read where
they lie -- the same float32s a whole read gives -- locally, and remotely
by fetching only the bytes that hold them."""
import functools
import http.server
import os
import threading

import numpy as np
import pytest

import fastgrib
from fastgrib import _partial

from helpers import Ranges, path


def _whole(monkeypatch):
    """Partial reads off: every field decoded whole."""
    monkeypatch.setattr(_partial, 'FRACTION', 10 ** 12)


@pytest.mark.parametrize('name', ['gfs_simple', 'gfs_ieee', 'gfs_ieee64', 'gfs_scan', 'lcc',
                                  'nps', 'gauss', 'red_gg', 'rot_ll', 'gfs_bm254', 'nbm',
                                  'gfs', 'gfs_png', 'gfs_aec', 'gfs_jpeg', 'ecmwf', 'gefs',
                                  'nbm_qmd_pr'])
def test_values_at_points_are_a_whole_reads(name):
    """Any point's value, read by itself -- where it lies (simple, IEEE),
    or decoding up to it (complex, PNG, CCSDS) -- is what decoding the
    whole field gives (every scanning, bitmaps, IEEE of 32 and 64 bits,
    constant fields, missing-value management); NaN where it has none."""
    f = fastgrib.open(path(name))
    rng = np.random.default_rng(len(name))
    seen = 0
    for r in f.records[:40]:
        if not _partial.can(r):
            continue
        full = f.read(r).ravel()
        for top in (full.size // 10 + 1, full.size):      # points early on, and anywhere
            lay = np.concatenate(([0, top - 1], rng.integers(0, top, 200)))
            for prefix in (False, True):
                assert np.array_equal(_partial.values(f._map, r, lay, prefix=prefix), full[lay],
                                      equal_nan=True), (r, top, prefix)
        assert np.array_equal(_partial.values(f._map, r, lay.reshape(-1, 2)),
                              full[lay].reshape(-1, 2), equal_nan=True)
        seen += 1
    assert seen or name in ('lcc', 'nps', 'gauss', 'red_gg', 'rot_ll')


def _repacked(tmp_path, packing):
    """ECMWF's 1440 x 721 field packed simple or IEEE, with its index."""
    out = str(tmp_path / ('ec_%s.grib2' % packing))
    fastgrib.open(path('ecmwf')).write(out, packing=packing)
    fastgrib.open(out).write_index(out + '.idx')
    return out


@pytest.mark.parametrize('packing', ['simple', 'ieee', 'jpeg'])
def test_partial_reads_local(packing, tmp_path, monkeypatch):
    """at(), regrid() onto a few points and read(box=...) read only the
    points they take of a big field, and give what whole reads give."""
    f = fastgrib.open(_repacked(tmp_path, packing))
    lat, lon = np.array([51.5, -33.9, 35.7, 89.9]), np.array([0.0, 18.4, 139.7, 300.0])
    box = (5, 15, 45, 55)
    got = (f.at(lat, lon), f.at(lat, lon, method='nearest'), f.read(0, box=box),
           f.regrid((np.array([40.0, 41.0]), np.array([10.0, 11.0]))))
    _whole(monkeypatch)
    want = (f.at(lat, lon), f.at(lat, lon, method='nearest'), f.read(0, box=box),
            f.regrid((np.array([40.0, 41.0]), np.array([10.0, 11.0]))))
    for a, b in zip(got, want):
        assert np.array_equal(a, b, equal_nan=True)


@pytest.mark.parametrize('packing', ['simple', 'ieee'])
def test_partial_reads_remote(packing, tmp_path):
    """Remotely, only a field's first sections and the byte ranges of the
    values wanted are fetched: a twentieth of the message or less for a few
    points or a small box, the values those of the whole file; a field
    another way packed fetched whole, as before."""
    out = _repacked(tmp_path, packing)
    loc = fastgrib.open(out)
    size = os.path.getsize(out)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                                          functools.partial(Ranges, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = 'http://127.0.0.1:%d/%s' % (srv.server_address[1], os.path.basename(out))
    try:
        lat, lon = np.array([51.5, -33.9, 35.7]), np.array([0.0, 18.4, 139.7])
        Ranges.gets.clear()
        f = fastgrib.open(url)
        assert np.array_equal(f.at(lat, lon), loc.at(lat, lon), equal_nan=True)
        fetched = sum(n for p, n in Ranges.gets if not p.endswith('.idx'))
        assert fetched < size // 20
        assert f.records[0].grid is not None            # what the index said, set right
        Ranges.gets.clear()
        f = fastgrib.open(url)
        box = (5, 15, 45, 55)
        assert np.array_equal(f.read(0, box=box), loc.read(0, box=box), equal_nan=True)
        assert sum(n for p, n in Ranges.gets if not p.endswith('.idx')) < size // 5
        g = fastgrib.open(url)                          # whole reads after: the message whole
        assert np.array_equal(g.read(0), loc.read(0), equal_nan=True)
    finally:
        srv.shutdown()


def test_partial_reads_remote_in_order(server, tmp_path):
    """A complex-packed field remotely: its group headers, then its values
    up to the last group a point needs (a point early in the file's order:
    a part of the message; late: all of it, no byte fetched twice); a PNG
    or CCSDS one fetched on until its stream holds the point, the same; a
    JPEG2000 one its headers, then the code-blocks reaching the point.
    Each the values of the local file. Winds along the
    grid's axes turned to east and north remotely as locally (an index
    does not say which they are: each field's head tells, before reading)."""
    packings = ('complex', 'ccsds', 'png', 'jpeg')
    for name, packing in (('ecmwf', p) for p in packings):
        out = str(tmp_path / ('%s_%s.grib2' % (name, packing)))
        fastgrib.open(path(name)).write(out, packing=packing)
        fastgrib.open(out).write_index(out + '.idx')
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                                          functools.partial(Ranges, directory=str(tmp_path)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = 'http://127.0.0.1:%d/' % srv.server_address[1]
    try:
        for packing in packings:
            name = 'ecmwf_%s.grib2' % packing
            loc = fastgrib.open(str(tmp_path / name))
            size = os.path.getsize(str(tmp_path / name))
            for lat, part in ((85.0, True), (30.0, True), (-85.0, False)):     # (rows run south)
                Ranges.gets.clear()
                f = fastgrib.open(base + name)
                assert np.array_equal(f.at(lat, 10.0), loc.at(lat, 10.0), equal_nan=True)
                got = sum(n for p, n in Ranges.gets if not p.endswith('.idx'))
                assert got < size * 0.75 if part else got <= size + 16, (packing, lat, got, size)
                if lat == 85.0:
                    assert got < size // 4, (packing, got, size)
    finally:
        srv.shutdown()
    for name in ('lcc_uvgrid', 'gfs_uv'):
        loc = fastgrib.open(path(name))
        lat, lon = np.array([35.0, 40.0]), np.array([255.0, 265.0])
        u = [r for r in loc.records if r.name == 'UGRD'][:1]
        if not u:
            continue
        d = tmp_path / ('w_' + name)
        d.mkdir()
        (d / 'f.grib2').write_bytes(open(path(name), 'rb').read())
        loc.write_index(str(d / 'f.grib2.idx'))
        srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
                                              functools.partial(Ranges, directory=str(d)))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            f = fastgrib.open('http://127.0.0.1:%d/f.grib2' % srv.server_address[1])
            assert np.array_equal(f.at(lat, lon), loc.at(lat, lon), equal_nan=True), name
        finally:
            srv.shutdown()
