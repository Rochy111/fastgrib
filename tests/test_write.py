"""Writing GRIB2: subsets, metadata set, values encoded every packing."""
import os

import numpy as np
import pytest

import fastgrib

from helpers import DATA, grib1, path, reference, same


def test_write_subsets(tmp_path):
    """write(): each field picked as its own GRIB2 message -- cut from a
    message of several, a reused bitmap given, GRIB1 turned into GRIB2 --
    read back as the original; picks and appending."""
    for name, ext in (('gfs_uv', '.grib2'), ('gfs_bm254', '.grib2'), ('gefs', '.grib2'),
                      ('red_gg', '.grib2'), ('gfs_scan', '.grib2'), ('g1_narr', '.grb')):
        f = grib1(name) if ext == '.grb' else fastgrib.open(path(name, ext))
        out = str(tmp_path / (name + '.grib2'))
        assert f.write(out) == len(f.records)
        g = fastgrib.open(out)
        assert [repr(r).split(':', 2)[2] for r in g.records] == \
            [repr(r).split(':', 2)[2] for r in f.records]
        assert all(r.sub == 0 for r in g.records) and not g.damaged
        raw = open(out, 'rb').read()
        from fastgrib import _encode
        assert all(raw[_encode.sections(raw, r)[6][0] + 5] != 254 for r in g.records)
        for a, b in zip(f.records, g.records):
            assert np.array_equal(f.read(a), g.read(b), equal_nan=True), a
    f = fastgrib.open(path('gfs'))
    out = str(tmp_path / 'picks.grib2')
    n = f.write(out, name='TMP')
    assert n == len(f.select(name='TMP')) and f.write(out, append=True, name='HGT') > 0
    g = fastgrib.open(out)
    assert [r.name for r in g.records] == ['TMP'] * n + ['HGT'] * (len(g.records) - n)


ENCODE_FILES = ['gfs', 'gfs_scan', 'red_gg', 'rot_ll_ns', 'nps', 'lcc', 'gfs_ieee']


@pytest.mark.parametrize('name', ENCODE_FILES)
def test_encode_round_trip(name, route, tmp_path):
    """A field's own values encoded again -- as its packing, simple,
    complex (second-order spatial differencing), IEEE, PNG, JPEG2000 and
    CCSDS -- read back to the bit, whatever the grid and scanning mode."""
    f = fastgrib.open(path(name))
    picks = f.records[::max(1, len(f.records) // 12)]
    for packing in (None, 'simple', 'complex', 'ieee', 'png', 'jpeg', 'ccsds'):
        msgs, kept = [], []
        for r in picks:
            try:
                msgs.append(f.encode(f.read(r), r, packing=packing))
                kept.append(r)
            except ValueError as e:             # JPEG2000 asked of a field deeper than 16 bits
                assert packing == 'jpeg' and 'JPEG2000 of' in str(e)
        if not msgs:
            continue
        out = tmp_path / ('enc_%s.grib2' % packing)       # (a file mapped stays: one each)
        out.write_bytes(b''.join(msgs))
        g = fastgrib.open(str(out))
        for a, b in zip(kept, g.records):
            x, y = f.read(a), g.read(b)
            if name == 'gfs_ieee' and packing not in (None, 'ieee'):    # floats: 24 bits
                span = np.nanmax(x) - np.nanmin(x)
                assert np.nanmax(np.abs(x - y)) <= span * 2.0 ** -24 + 1e-6 * np.nanmax(abs(x))
            else:
                assert np.array_equal(x, y, equal_nan=True), (packing, a)
            assert repr(a).split(':', 2)[2] == repr(b).split(':', 2)[2]


def test_encode_new_values(tmp_path):
    """New values (gaps included): to the field's own precision (half its
    2^E 10^-D step), to `decimals`, to `bits`, IEEE exactly as float32;
    complex packing smaller than simple."""
    import struct
    from fastgrib import _meta
    f = fastgrib.open(path('gfs'))
    r = f.select(name='TMP')[0]
    v = f.read(r).astype(np.float64)
    v = v + np.linspace(0, 3.3, v.size).reshape(v.shape)
    v[::7, ::5] = np.nan
    b = f._map
    e = _meta._sm(struct.unpack_from('>H', b, r.s5 + 15)[0], 2)
    d = _meta._sm(struct.unpack_from('>H', b, r.s5 + 17)[0], 2)
    step = 2.0 ** e * 10.0 ** -d

    def back(m):
        return fastgrib.open(m).read(0)

    for packing in ('simple', 'complex'):
        y = back(f.encode(v, r, packing=packing))
        assert np.array_equal(np.isnan(y), np.isnan(v))
        assert np.nanmax(np.abs(y - v)) <= step / 2 * 1.001 + 1e-4
    assert np.nanmax(np.abs(back(f.encode(v, r, decimals=1)) - v)) <= 0.05 + 1e-4
    y = back(f.encode(v, r, bits=10))
    assert np.nanmax(np.abs(y - v)) <= (np.nanmax(v) - np.nanmin(v)) / 1023 + 1e-3
    assert np.array_equal(back(f.encode(v, r, packing='ieee')), v.astype(np.float32),
                          equal_nan=True)
    assert len(f.encode(v, r, packing='complex')) < len(f.encode(v, r, packing='simple'))
    with pytest.raises(ValueError):
        f.encode(v[:3], r)


def test_message_metadata(tmp_path):
    """message(): the field's name, level, reference time and forecast
    time set (statistics as 4.8), its packed values untouched."""
    from datetime import datetime
    f = fastgrib.open(path('gfs'))
    r = f.select(name='TMP')[0]
    ms = [f.message(r, name='DPT', level='700 mb', fcst='0-6 hour max fcst'),
          f.message(r, date=datetime(2030, 1, 2, 12), fcst='9 hour fcst'),
          f.message(r, level=(103, 2.0)), f.message(r, param=(0, 1, 1), fcst='anl')]
    (tmp_path / 'm.grib2').write_bytes(b''.join(ms))
    g = fastgrib.open(str(tmp_path / 'm.grib2'))
    assert [repr(x).split(':', 2)[2] for x in g.records] == [
        'd=2026102300:DPT:700 mb:0-6 hour max fcst:', 'd=2030010212:TMP:0.04 mb:9 hour fcst:',
        'd=2026102300:TMP:2 m above ground:6 hour fcst:', 'd=2026102300:RH:0.04 mb:anl:']
    assert g.records[0].product == 8 and g.records[1].valid_time == datetime(2030, 1, 2, 21)
    assert all(np.array_equal(g.read(x), f.read(r), equal_nan=True) for x in g.records)
    with pytest.raises(ValueError):
        f.message(r, name='NOT-A-NAME')
    with pytest.raises(TypeError):
        f.message(r, colour='blue')


def test_written_is_wgrib2s(route):
    """What fastgrib writes (tools/testdata/make_written.py: new values packed three
    ways and to asked precisions, metadata set) as wgrib2 reads it, and
    made again byte for byte."""
    import importlib.util
    loc, ref = reference('written')
    want = [l.strip().split(':', 2)[2].rstrip(':') for l in open(path('written', '.inv'))]
    assert [repr(r).split(':', 2)[2].rstrip(':') for r in loc.records] == want
    for r, v in loc.fields():                   # (three grids: read a field at a time)
        assert same(v, ref[r.index - 1]), r
    spec = importlib.util.spec_from_file_location(
        'make_written', os.path.join(os.path.dirname(__file__), '..', 'tools', 'testdata', 'make_written.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert b''.join(mod.messages(DATA)) == open(path('written'), 'rb').read()


def test_write_remote(server, tmp_path):
    """A remote file's picks written: only they are fetched, as the local
    file's own messages; GRIB1 too, as GRIB2."""
    f = fastgrib.open(server + 'gfs_uv.grib2')
    loc = fastgrib.open(path('gfs_uv'))
    out = str(tmp_path / 'r.grib2')
    assert f.write(out, name='UGRD') == len(loc.select(name='UGRD'))
    g = fastgrib.open(out)
    for a, b in zip(loc.select(name='UGRD'), g.records):
        assert np.array_equal(loc.read(a), g.read(b), equal_nan=True)
    axes = (np.arange(22, 48, 0.7), np.arange(252, 288, 0.7))      # regridded as fetched
    assert np.array_equal(f.regrid(axes, name=['PRMSL', 'VGRD']), loc.regrid(axes, name=['PRMSL', 'VGRD']),
                          equal_nan=True)
    f = fastgrib.open(server + 'g1_narr.grb')
    out = str(tmp_path / 'r1.grib2')
    f.write(out)
    g, loc = fastgrib.open(out), grib1('g1_narr')
    assert [repr(r).split(':', 2)[2] for r in g.records] == \
        [repr(r).split(':', 2)[2] for r in loc.records]
    msg = f.encode(loc.read(1), 1)
    assert fastgrib.open(msg).read(0).shape == loc.grid().shape
