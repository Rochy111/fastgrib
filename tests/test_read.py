"""Reading GRIB2 as wgrib2 does: inventories, values to the bit (CPU and
GPU), names and level kinds as ecCodes gives them, picks, batches, routing."""
import glob
import hashlib
import os
import struct
import threading
from datetime import timedelta

import numpy as np
import pytest

import fastgrib
from fastgrib import _cpu

from helpers import DATA, FILES, as_wgrib2, gpu_ok, path, reference, same


@pytest.mark.parametrize('name', FILES)
def test_inventory_is_wgrib2s(name):
    """wgrib2 -s's lines -- but for a parameter its tables lack ('var
    discipline=...'), which fastgrib names by ecCodes', and a level type
    this wgrib2 (3.1.3) has as 'reserved', which newer wgrib2 words."""
    f = fastgrib.open(path(name))
    want = [l.rstrip(':') for l in open(path(name, '.inv')).read().splitlines() if l.strip()]
    assert len(f.records) == len(want)
    for r, w in zip(f.records, want):
        got, w = repr(r).rstrip(':').split(':'), w.split(':')
        if w[3].startswith('var discipline=') and got[3] != w[3]:
            assert not got[3].startswith('var'), repr(r)
            got[3] = w[3]
        if w[4] == 'reserved':
            got[4] = w[4]
        if r.product == 44:                     # wgrib2 reads WMO's 2-octet forecast
            got[5] = w[5]                       # time of 4.44 as 4 octets; ecCodes doesn't
        assert got == w


ECC = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, '*.ecc')))


@pytest.mark.parametrize('name', ECC)
def test_short_names_and_level_kinds_are_eccodes(name):
    """short_name and level_kind as ecCodes' shortName and typeOfLevel."""
    f = fastgrib.open(path(name))
    want = [l.split() for l in open(path(name, '.ecc')).read().splitlines() if l.strip()]
    assert [[r.short_name or 'unknown', r.level_kind] for r in f.records] == want


def test_ecmwf_local_parameters():
    """ECMWF's own parameter numbers, which wgrib2 cannot name: ecCodes'
    names; its soil and most-unstable-parcel levels worded and kinded."""
    f = fastgrib.open(path('ecmwf_local'))
    tp = f.select(name='tp')[0]
    assert (tp.long_name, tp.units, tp.fcst) == ('Total precipitation', 'm',
                                                  '0-6 hour acc@(fcst,dt=450 sec),missing=0')
    assert {r.name for r in f.records} >= {'tp', 'lcc', 'tcc', 'ro'}
    sot = f.select(short_name='sot')
    assert sot and all(r.level_kind == 'soilLayer' for r in sot)
    assert 'soil level 1 - soil level 2' in {r.level for r in sot}
    cape = f.select(short_name='mucape')[0]
    assert cape.level_kind == 'mostUnstableParcel' and cape.level.startswith('Departure level')
    assert f.select(short_name='mx2t3')[0].fcst.startswith('3-6 hour max')


def test_parameters_and_times():
    f = fastgrib.open(path('gfs'))
    r = f.select(name='TMP', level_type='isobaricInhPa')[-1]
    assert f.select(name='TMP', level=r.level_value) == [r]
    assert (r.long_name, r.units, r.level_kind) == ('Temperature', 'K', 'isobaricInhPa')
    assert r.level == '%g mb' % r.level_value
    assert r.fcst_hours == 6.0 and r.valid_time > r.ref_time
    assert not [x for x in f.records if x.name.startswith('var')]       # all named


def _remade(f, i, tmpl, s4):
    """f's field i, alone, with section 4 `s4` made product template
    4.tmpl (its length and number set)."""
    s = f.sections(i)
    s4 = bytearray(s4)
    s4[0:4], s4[7:9] = struct.pack('>I', len(s4)), struct.pack('>H', tmpl)
    body = b''.join(s[k] for k in (1, 3)) + bytes(s4) + b''.join(s[k] for k in (5, 6, 7))
    m = f.message(i)
    return m[:8] + struct.pack('>Q', 16 + len(body) + 4) + body + b'7777'


# 4.32's one spectral band (series, satellite, instrument, central wave number);
# 4.33 adds an ensemble member (type, perturbation, size of the ensemble)
_BAND = struct.pack('>BHHHBI', 1, 333, 72, 37, 0, 2310)


_SAT = 'sat=METEOSAT 13 Unknown 692.52 GHz '     # wgrib2 3.1.3's words for _BAND


@pytest.mark.parametrize('tmpl, unit, ft, tail, fcst, step, extra', [
    (32, 1, 6, _BAND, '6 hour fcst', timedelta(hours=6), _SAT),
    (33, 1, 6, _BAND + bytes([3, 4, 10]), '6 hour fcst', timedelta(hours=6), 'ENS=+4:' + _SAT),
    (1000, 0, 90, b'', '90 min fcst', timedelta(minutes=90), ''),
    (32, 1, 0, _BAND, 'anl', timedelta(0), _SAT)], ids=['4.32', '4.33', '4.1000', '4.32-anl'])
def test_times_without_a_level(tmpl, unit, ft, tail, fcst, step, extra):
    """Simulated satellite imagery (4.32, 4.33) and a cross section
    (4.1000) give 4.0's times without its level: valid time and wording
    as wgrib2 3.1.3's lines for these messages ('no_level', the member,
    the satellite band), the level left alone."""
    g = fastgrib.open(path('gfs'))
    s4 = bytearray(g.sections(0)[4][:22]) + tail        # 4.0's first 22 octets: the time
    s4[17], s4[18:22] = unit, struct.pack('>I', ft)
    f = fastgrib.open(_remade(g, 0, tmpl, s4))
    r = f.records[0]
    assert r.product == tmpl and r.ref_time == g.records[0].ref_time
    assert (r.fcst, r.valid_time, r.fcst_hours) == (fcst, r.ref_time + step,
                                                   step.total_seconds() / 3600)
    assert repr(r).split(':', 4)[4] == 'no_level:%s:%s' % (fcst, extra)
    assert (r.level_type, r.level_value, r.level_type2, r.level_value2) == (255, None, 255, None)
    assert (r.member, r.ensemble) == ((4, 'perturbed') if tmpl == 33 else (None, None))
    k = f.keys(r)
    assert (k['forecastTime'], k['indicatorOfUnitOfTimeRange'], k['stepRange']) == (
        ft, unit, str(ft))
    assert 'typeOfFirstFixedSurface' not in k
    if tmpl != 1000:                            # the band, as ecCodes names its keys
        assert {n: k[n] for n in ('NB', 'satelliteSeries', 'satelliteNumber', 'instrumentType',
                                  'scaleFactorOfCentralWaveNumber',
                                  'scaledValueOfCentralWaveNumber')} == {
            'NB': 1, 'satelliteSeries': 333, 'satelliteNumber': 72, 'instrumentType': 37,
            'scaleFactorOfCentralWaveNumber': 0, 'scaledValueOfCentralWaveNumber': 2310}
    assert np.array_equal(f.read(r), g.read(g.records[0]), equal_nan=True)


def test_satellite_bands_listed():
    """Several spectral bands: each band key a list (ecCodes' octets 24 on,
    11 a band; a missing wave number None), and 4.33's member after them."""
    g = fastgrib.open(path('gfs'))
    two = struct.pack('>BHHHBI', 2, 333, 72, 37, 0, 2310) + \
        struct.pack('>HHHBI', 334, 17, 207, 255, 0xFFFFFFFF)
    f = fastgrib.open(_remade(g, 0, 33, bytearray(g.sections(0)[4][:22]) + two +
                              bytes([3, 4, 10])))
    k = f.keys(0)
    assert (k['NB'], k['satelliteSeries'], k['instrumentType'],
            k['scaledValueOfCentralWaveNumber']) == (2, [333, 334], [37, 207], [2310, None])
    assert (k['perturbationNumber'], f.records[0].member) == (4, 4)
    assert repr(f.records[0]).split(':', 4)[4] == (    # wgrib2 3.1.3's line
        'no_level:6 hour fcst:ENS=+4:sat=Multi-sat 2 bands: -inf to 2.31e+03 m-1 ')


@pytest.mark.parametrize('tmpl, at, band, words', [
    (32, 22, (333, 72, 207, 1, 9259), 'sat=METEOSAT 13 SEVIRI 277.58 GHz '),
    (32, 22, (333, 72, 207, 0, 92590), 'sat=METEOSAT 13 SEVIRI 10.80 um '),
    (32, 22, (333, 72, 207 | 2 << 13, 0, 92590), 'sat=METEOSAT 13 SEVIRI 10.80 um H pol.'),
    (31, 13, (333, 72, 207, 0, 92590), 'sat=METEOSAT 13 SEVIRI 10.80 um ')])
def test_satellite_band_words(tmpl, at, band, words):
    """A satellite product's band as wgrib2 words it (Spectral_bands.c,
    BUFR tables 0 01 007 and 0 02 019): a wavelength in um or a frequency
    in GHz, a polarization; checked against wgrib2 3.1.3 (4.31's too)."""
    g = fastgrib.open(path('gfs'))
    s4 = bytearray(g.sections(0)[4][:at]) + struct.pack('>BHHHBI', 1, *band)
    r = fastgrib.open(_remade(g, 0, tmpl, s4)).records[0]
    assert repr(r).rsplit(':', 1)[1] == words and r.level == 'no_level'


def test_simulated_satellite_statistic():
    """4.34, a member's statistic of simulated satellite data: its time
    ranges and end after the spectral bands and the member -- as the 4.8
    field they were taken from has them."""
    g = fastgrib.open(path('ecmwf_local'))
    base = g.records[2]
    assert base.product == 8
    s4 = g.sections(2)[4]
    f = fastgrib.open(_remade(g, 2, 34, s4[:22] + _BAND + bytes([3, 4, 10]) + s4[34:]))
    r = f.records[0]
    assert (r.fcst, r.valid_time, r.member) == (base.fcst, base.valid_time, 4)
    assert repr(r).split(':', 4)[4] == 'no_level:%s:ENS=+4:%s' % (base.fcst, _SAT)
    want = g.keys(base)
    assert {k: f.keys(r)[k] for k in ('stepType', 'stepRange', 'lengthOfTimeRange')} == {
        k: want[k] for k in ('stepType', 'stepRange', 'lengthOfTimeRange')}


# templates that are 4.0, 4.1, 4.8 or 4.11 with octets inserted: (template,
# file, field of the template it extends, where the octets go, the octets --
# counts where the template has them)
_INSERTED = [
    (51, 'ecmwf_local', 0, 34, bytes([2]) + bytes(24)),                 # 2 categories
    (91, 'ecmwf_local', 2, 34, bytes([2]) + bytes(24)),
    (53, 'ecmwf_local', 0, 11, bytes([5, 2, 0, 1, 0, 2, 0, 1])),        # 2 partitions
    (54, 'gefs', 0, 11, bytes([5, 2, 0, 1, 0, 2, 0, 1])),
    (55, 'ecmwf_local', 0, 11, bytes([1, 2, 3, 4, 5, 6])),              # a tile
    (56, 'ecmwf_local', 0, 11, bytes([1, 2, 3, 4, 5, 6])),
    (59, 'gefs', 0, 11, bytes([1, 2, 3, 4, 5, 6])),
    (62, 'ecmwf_local', 2, 11, bytes([1, 2, 3, 4, 5, 6])),
    (63, 'gefs', 8, 11, bytes([1, 2, 3, 4, 5, 6])),
    (57, 'ecmwf_local', 0, 11, bytes(8) + bytes([2]) + bytes(10)),      # 2 parameters
    (58, 'gefs', 0, 11, bytes(8) + bytes([2]) + bytes(10)),
    (67, 'ecmwf_local', 2, 11, bytes(8) + bytes([255])),                # (none)
    (68, 'gefs', 8, 11, bytes(8) + bytes([1]) + bytes(5)),
    (70, 'ecmwf_local', 0, 11, bytes([0, 1, 0, 98, 2])),                # post-processing
    (71, 'gefs', 0, 11, bytes([0, 1, 0, 98, 2])),
    (72, 'ecmwf_local', 2, 11, bytes([0, 1, 0, 98, 2])),
    (73, 'gefs', 8, 11, bytes([0, 1, 0, 98, 2]))]


@pytest.mark.parametrize('tmpl, name, i, at, octets', _INSERTED,
                         ids=['4.%d' % t[0] for t in _INSERTED])
def test_templates_read_as_the_one_they_extend(tmpl, name, i, at, octets):
    """Categorical, partitioned, tiled, distribution-function and
    post-processed templates: level, times and member read as the
    template they extend, as wgrib2 does (which words no member of 4.54)."""
    g = fastgrib.open(path(name))
    base = g.records[i]
    s4 = g.sections(i)[4]
    f = fastgrib.open(_remade(g, i, tmpl, s4[:at] + octets + s4[at:]))
    r = f.records[0]
    assert r.product == tmpl
    assert repr(r).split(':', 3)[3] == ('%s:%s:%s:' % (base.name, base.level, base.fcst)
                                        if tmpl == 54 else repr(base).split(':', 3)[3])
    assert (r.level_type, r.level_value, r.valid_time, r.member) == (
        base.level_type, base.level_value, base.valid_time, base.member)
    k, want = f.keys(r), g.keys(base)
    for key in ('typeOfFirstFixedSurface', 'forecastTime', 'stepRange', 'perturbationNumber'):
        assert k.get(key) == want.get(key), key
    assert np.array_equal(f.read(r), g.read(base), equal_nan=True)


@pytest.mark.parametrize('name', FILES)
def test_values_are_wgrib2s(name, route):
    f, ref = reference(name)
    vals = [f.read(r) for r in f.records]
    if ref is None:
        digest = hashlib.sha256(b''.join(as_wgrib2(v).tobytes() for v in vals)).hexdigest()
        assert digest == open(path(name, '.sha256')).read().strip()
        return
    for r, v, w in zip(f.records, vals, ref):
        assert same(v, w), repr(r)


@pytest.mark.parametrize('name', ['gfs', 'nbm', 'gfs_png', 'gfs_aec', 'gfs_uv'])
def test_fields_and_batches(name, route):
    f, ref = reference(name)
    got = {r.index: v.copy() for r, v in f.fields()}
    assert len(got) == len(f)
    for r, w in zip(f.records, ref):
        assert same(got[r.index], w), repr(r)
    by_grid = {}
    for r, w in zip(f.records, ref):
        by_grid.setdefault(r.grid, []).append((r, w))
    for recs in by_grid.values():
        stack = f.read([r for r, _ in recs])
        for v, (r, w) in zip(stack if stack.ndim == 3 else [stack], recs):
            assert same(v, w), repr(r)


def test_split_records_and_small_budgets():
    """A record split across cores, and decode_iter with tiny blocks."""
    f, ref = reference('nbm')
    buf = np.fromfile(path('nbm'), np.uint8)
    cols = np.arange(f.records[0].grid.ni * f.records[0].grid.nj)
    old = _cpu._MIN_RUN
    _cpu._MIN_RUN = 64                          # small fields split too
    try:
        for r, w in zip(f.records, ref):
            v = _cpu.decode(buf, [r], cols, undef=np.float32(np.nan))[0]
            assert same(v.reshape(w.shape), w), repr(r)
    finally:
        _cpu._MIN_RUN = old
    for i, block in _cpu.decode_iter(buf, f.records, cols, budget=3 * cols.size * 4,
                                     undef=np.float32(np.nan)):
        for j, v in enumerate(block):
            assert same(v.reshape(ref[i + j].shape), ref[i + j])


@pytest.mark.skipif(not gpu_ok(), reason='no usable GPU')
@pytest.mark.parametrize('name', ['gfs', 'nbm', 'gfs_jpeg'])
def test_on_the_card(name):
    f, ref = reference(name)
    for r, w in zip(f.records, ref):
        assert same(f.read(r, gpu=True).get(), w), repr(r)


@pytest.mark.skipif(not gpu_ok(), reason='no usable GPU')
def test_on_the_card_north_up(monkeypatch):
    """Complex-packed fields whose rows run north first (as NCEP's GFS
    files have them), no bitmap: put in file order and turned round on the
    card, no column map built (a 0.018 deg global SST's took 2 s, rebuilt
    every read). The GEFS sample complex packed, its scanning mode set to
    0: the same values, rows the other way round."""
    f = fastgrib.open(path('gefs'))
    v = f.read()
    msgs = []
    for i, r in enumerate(f.records):
        m = bytearray(f.encode(v[i], r, packing='complex'))
        p = bytes(m).find(fastgrib.open(bytes(m)).sections(0)[3])
        assert m[p + 71] == 0x40                 # octet 72 of 3.0: scanning mode
        m[p + 71] = 0
        msgs.append(bytes(m))
    g = fastgrib.open(b''.join(msgs))
    assert all(g.keys(k)['dataRepresentationTemplateNumber'] == 3 for k in range(len(msgs)))

    def no_map(*a):
        raise AssertionError('a column map was built')
    monkeypatch.setattr(_cpu, '_defined', no_map)
    want = v[:, ::-1]
    assert same(g.read(gpu=True).get(), want)
    got = np.empty_like(v)
    g.read(gpu=True, out=got)
    assert same(got, want)


def test_picks():
    f = fastgrib.open(path('gfs'))
    assert f.read(f.records[3]).shape == f.grid().shape
    assert f.read([0, 1]).shape == (2,) + f.grid().shape
    one = f.find(':PRMSL:')
    assert len(one) == 1 and f.read(':PRMSL:').ndim == 2
    assert f.select(name=['TMP', 'RH'], level_type='isobaricInhPa')
    assert f.select(level_value=lambda v: v is not None and v > 900)
    with pytest.raises(KeyError, match='It holds'):
        f.read('NOSUCHFIELD')
    with pytest.raises(TypeError):
        f.select(colour='red')


def test_ensembles():
    """GEFS members (4.1, 4.11) and statistics (4.2, 4.12), as wgrib2 words
    them, picked by member."""
    f = fastgrib.open(path('gefs'))
    assert {r.product for r in f.records} == {1, 2, 11, 12}
    kinds = {(r.member, r.ensemble) for r in f.records}
    assert kinds == {(1, 'perturbed'), (0, 'control'), (None, 'ens mean'),
                     (None, 'ens std dev')}
    tmax = f.select(name='TMAX', member=1)
    assert tmax and tmax[0].fcst == '0-6 hour max fcst' and 'ENS=+1' in tmax[0].extra
    assert tmax[0].valid_time - tmax[0].ref_time == timedelta(hours=6)
    assert all('ENS=low-res ctl' in r.extra for r in f.select(ensemble='control'))


def test_cli(tmp_path, capsys):
    from fastgrib.__main__ import main
    assert main([path('gfs')]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [l.rstrip(':') for l in lines] == \
        [l.rstrip(':') for l in open(path('gfs', '.inv')).read().splitlines()]
    out = tmp_path / 'v.bin'
    assert main([path('gfs'), '-bin', str(out)]) == 0
    assert open(out, 'rb').read() == open(path('gfs', '.bin'), 'rb').read()
    assert main([path('gfs'), '-select', 'name=TMP', 'level_type=isobaricInhPa', '-stats']) == 0
    assert 'mean=' in capsys.readouterr().out


def test_png_pipeline(monkeypatch):
    """A big PNG field read alone (inflate, unfilter and conversion
    overlapped): forced onto the small PNG file, a few rows a step --
    wgrib2's values, and a cut-short stream an error."""
    from fastgrib import _images
    monkeypatch.setattr(_images, 'PNG_PIPE', 1)
    monkeypatch.setattr(_images, 'PNG_BLOCK', 200)
    f, ref = reference('gfs_png')
    for r in f.records:
        assert same(f.read(r), ref[r.index - 1]), repr(r)
    r = next(x for x in f.records if x.packing == 41 and x.s7end - x.s7 > 200)
    b = bytearray(open(path('gfs_png'), 'rb').read()[r.offset:])
    b = bytes(b[:struct.unpack('>Q', b[8:16])[0]])
    damaged = bytearray(b)
    q = r.s7 - r.offset + 5 + 60                 # inside the PNG's data
    damaged[q:q + 40] = bytes(40)
    g = fastgrib.open(bytes(damaged))
    with pytest.raises(ValueError):
        g.read(g.records[0])


def test_select_index_is_a_scan():
    """select() through its index picks what testing every field picks."""
    from fastgrib._file import _matches
    f = fastgrib.open(path('gfs'))
    cases = [dict(name='TMP'), dict(name=['UGRD', 'VGRD'], level_type='isobaricInhPa'),
             dict(level=500), dict(level_value=lambda v: v is not None and v > 900),
             dict(level_kind='heightAboveGround', fcst='6 hour fcst'), dict(short_name='2t'),
             dict(level_value=0.4), dict(name='NOSUCH'), dict(level_type=103),
             dict(fcst_hours=6.0, name=('TMP', 'RH'))]
    for c in cases:
        got = f.select(**c)
        tests = [('level_value' if k == 'level' and isinstance(v, (int, float)) else
                  'level_kind' if k == 'level_type' and isinstance(v, str) else k, v)
                 for k, v in c.items()]
        want = [r for r in f.records if all(_matches(getattr(r, k), v) for k, v in tests)]
        assert got == want, c
    assert f.select() == f.records


def test_auto_routing(monkeypatch):
    """'auto' as measured: fresh results the CPU's; big fields into a
    reused array the GPU's; fields() the GPU's only for a long stream once
    it is up; small fields always the CPU's."""
    import types
    from fastgrib import _file
    monkeypatch.setattr(_file, 'DEVICE', 'auto')
    monkeypatch.setattr(_file, '_try_gpu', lambda: True)
    grid = types.SimpleNamespace(ni=1440, nj=721)            # 1.04M points
    big = [types.SimpleNamespace(grid=grid, packing=3) for _ in range(200)]
    small = [types.SimpleNamespace(grid=types.SimpleNamespace(ni=720, nj=361), packing=3)] * 2000
    out = np.empty((1, 1), np.float32)
    for up in (None, True):
        monkeypatch.setattr(_file, '_GPU_STATE', up)
        assert not _file._want_gpu(big, 'fresh')
        assert _file._want_gpu(big, out)
        assert not _file._want_gpu(small, out)
        assert _file._want_gpu(big * 6, 'stream') == bool(up)
        assert not _file._want_gpu(big[:5], 'stream')
    monkeypatch.setattr(_file, '_GPU_STATE', False)                # no usable GPU
    assert not _file._want_gpu(big, out)
    monkeypatch.setattr(_file, 'DEVICE', 'gpu')
    assert _file._want_gpu(big, 'fresh')


def test_select_from_threads():
    import sys
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for _ in range(20):
            f = fastgrib.open(path('gfs'))
            want = [r for r in f.records if r.name == 'TMP']
            bar = threading.Barrier(8)
            res = []

            def go():
                bar.wait()
                res.append(f.select(name='TMP'))
            ts = [threading.Thread(target=go) for _ in range(8)]
            [t.start() for t in ts]
            [t.join() for t in ts]
            assert len(res) == 8 and all(r == want for r in res)
    finally:
        sys.setswitchinterval(old)


def test_damaged_png_header_and_empty_grid():
    """A PNG header claiming a vast image, a grid of no points: errors,
    no allocation, no IndexError."""
    f = fastgrib.open(path('gfs_png'))
    r = next(x for x in f.records if x.packing == 41)
    b = bytearray(open(path('gfs_png'), 'rb').read()[r.offset:])
    b = b[:struct.unpack('>Q', b[8:16])[0]]
    q = b.index(b'IHDR', r.s7 - r.offset)
    b[q + 8:q + 12] = struct.pack('>I', 0x7FFFFFF0)                 # height
    with pytest.raises(ValueError):
        fastgrib.open(bytes(b)).read(0)
    g = fastgrib.open(path('gfs_simple'))
    r = g.records[0]
    m = bytearray(open(path('gfs_simple'), 'rb').read()[:g.records[1].offset])
    p = 16
    while m[p + 4] != 3:
        p += struct.unpack('>I', m[p:p + 4])[0]
    m[p + 6:p + 10] = bytes(4)                                       # 0 points
    m[p + 30:p + 34] = bytes(4)                                      # Ni = 0
    with pytest.warns(UserWarning):
        h = fastgrib.open(bytes(m))
    assert len(h) == 0 and 'no points' in h.damaged[0]


def test_extended_product_templates():
    """4.40-4.49 (chemistry, aerosols) and 4.60/4.61 (reforecasts) read as
    the templates they extend: level, times, member as ecCodes reads them
    (4.44's 2-octet forecast time too), wgrib2's wording of the rest."""
    f = fastgrib.open(path('tmpl_more'))
    assert [r.product for r in f.records] == [41, 41, 43, 43, 44, 46, 49, 49, 60, 61]
    assert all(r.level_value == 500 and r.fcst_hours == 6 for r in f.records)
    assert [r.member for r in f.records] == [3, 3, 3, 3, None, None, 3, 3, 3, 3]
    assert f.records[4].fcst == '6 hour fcst'
    a = fastgrib.open(path('gefs_aero'))
    assert all('aerosol=' in r.extra for r in a.records)
    assert all('aerosol_wavelength' in r.extra for r in a.records if r.name.endswith('AOTK'))
    lines = open(path('gefs_aero', '.inv')).read().splitlines()
    assert [repr(r) for r in a.records] == lines            # wgrib2's, to the last colon
    assert {r.level_kind for r in a.records} == {'atmosphere', 'surface'}


@pytest.mark.parametrize('name', ['gfs', 'gefs', 'nbm', 'gfs_uv', 'ecmwf_local', 'lcc'])
def test_inventory_lines_are_wgrib2s(name):
    """Every inventory line as wgrib2 -s writes it, to the last colon: one
    after the forecast time, none after the items following it (ENS=+1,
    prob ...)."""
    lines = [l for l in open(path(name, '.inv')).read().splitlines() if l.strip()]
    got = [repr(r) for r in fastgrib.open(path(name)).records]
    if name == 'ecmwf_local':                   # (fastgrib names what wgrib2 3.1.3's tables
        got, lines = zip(*[(g, w) for g, w in zip(got, lines)    # do not: parameters, levels)
                           if 'var discipline' not in w and ':reserved:' not in w])
    assert list(got) == list(lines)


def test_compressed_files_contains_and_pandas(tmp_path):
    """A gzip, bzip2 or xz file (or bytes) opens as the GRIB inside it;
    `pattern in f` asks find(); to_pandas() lists the fields, and its rows
    pick fields back."""
    import bz2
    import gzip
    import lzma
    raw = open(path('gfs'), 'rb').read()
    f = fastgrib.open(path('gfs'))
    for mod, ext in ((gzip, 'gz'), (bz2, 'bz2'), (lzma, 'xz')):
        p = tmp_path / ('gfs.grib2.' + ext)
        p.write_bytes(mod.compress(raw))
        g = fastgrib.open(str(p))
        assert [repr(r) for r in g.records] == [repr(r) for r in f.records]
        assert np.array_equal(g.read(7), f.read(7), equal_nan=True)
    assert len(fastgrib.open(gzip.compress(raw))) == len(f)
    assert ':TMP:' in f and ':NO SUCH:' not in f and f.records[3] in f
    pd = pytest.importorskip('pandas')
    df = f.to_pandas()
    assert isinstance(df, pd.DataFrame) and len(df) == len(f)
    assert list(df['name']) == [r.name for r in f.records]
    hot = df[(df['name'] == 'TMP') & (df['level_kind'] == 'isobaricInhPa')]
    assert np.array_equal(f.read(hot), f.read(name='TMP', level_kind='isobaricInhPa'),
                          equal_nan=True)


def test_keys_units_and_names_are_eccodes():
    """keys()'s name and units are ecCodes' (its concept's), as its
    shortName is: 'm s**-1', 'Pressure reduced to MSL'; a probability's
    percent; the record keeps wgrib2's words."""
    f = fastgrib.open(path('gfs'))
    k = f.keys(0)
    assert (k['shortName'], k['name'], k['units']) == ('prmsl', 'Pressure reduced to MSL', 'Pa')
    v = f.find(':VGRD:')[0]
    assert f.keys(v)['units'] == 'm s**-1' and v.units == 'm s-1'
    assert fastgrib.open(path('nbm')).keys(0)['units'] == '%'


def test_keys_found_in_real_files():
    """Keys as ecCodes gives them where a sweep of operational files found
    them otherwise: NCEP's rotated E-grid's size and corners (3.32768, as
    3.0's octets); a statistic over no time's stepRange its one step ('0',
    not '0-0'); no typeOfLevel or level for a template without one."""
    e = fastgrib.open(path('egrid'))
    k = e.keys(0)
    assert (k['gridDefinitionTemplateNumber'], k['Ni'], k['Nj']) == (32768, 31, 21)
    assert 'gridType' not in k or k['gridType'] is None
    g = fastgrib.open(path('gfs'))
    r = g.find(':APCP:')[0]
    m = bytearray(g.message(r))
    p = 16
    while m[p + 4] != 4:
        p += int.from_bytes(m[p:p + 4], 'big')
    m[p + 18:p + 22] = bytes(4)                 # forecast time 0, the range's length 0
    m[p + 49:p + 53] = bytes(4)
    z = fastgrib.open(bytes(m))
    assert z.records[0].fcst == '0-0 day acc fcst'      # (wgrib2's words)
    assert z.keys(0)['stepRange'] == '0'
    s4 = bytearray(g.sections(0)[4][:22]) + _BAND
    k = fastgrib.open(_remade(g, 0, 32, s4)).keys(0)
    assert 'typeOfLevel' not in k and 'level' not in k and k['stepRange'] == '6'
