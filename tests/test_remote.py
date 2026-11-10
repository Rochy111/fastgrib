"""Remote files: .idx and ECMWF's .index, byte ranges, the fetch cache."""
import http.server
import os
import threading

import numpy as np
import pytest

import fastgrib

from helpers import Ranges, grib1, path, reference, same


@pytest.mark.parametrize('name', ['gfs', 'nbm', 'gfs_uv', 'gefs'])
def test_remote(server, name):
    f = fastgrib.open(server + name + '.grib2')
    loc, ref = reference(name)
    assert [repr(r).rstrip(':') for r in f.records] == [repr(r).rstrip(':') for r in loc.records]
    picks = f.records[::3]
    stack = f.read(picks)
    for v, r in zip(stack, picks):
        assert same(v, ref[r.index - 1]), repr(r)
    got = {r.index: v.copy() for r, v in f.fields(f.records[1::4])}
    for k, v in got.items():
        assert same(v, ref[k - 1])
    assert f.latlon()[0].shape == f.grid().shape
    assert [(e.member, e.ensemble) for e in f.records] ==         [(r.member, r.ensemble) for r in loc.records]


def test_remote_ecmwf_index(tmp_path):
    """An ECMWF-style .index (JSON lines, ecCodes' names): each entry named,
    levelled and kinded as its record will be, before anything is fetched."""
    import functools
    import json
    f = fastgrib.open(path('ecmwf_local'))
    shorts = [l.split()[0] for l in open(path('ecmwf_local', '.ecc')).read().splitlines()]
    (tmp_path / 'ec.grib2').write_bytes(open(path('ecmwf_local'), 'rb').read())
    ends = [r.offset for r in f.records[1:]] + [os.path.getsize(path('ecmwf_local'))]
    with open(tmp_path / 'ec.index', 'w') as out:
        for r, sn, end in zip(f.records, shorts, ends):
            d = {'date': '20261023', 'time': '0000', 'step': '6', 'param': sn,
                 'levtype': 'sol' if r.level_kind == 'soilLayer' else 'sfc',
                 '_offset': r.offset, '_length': end - r.offset}
            if d['levtype'] == 'sol':
                d['levelist'] = '%d' % r.level_value2
            out.write(json.dumps(d) + '\n')
    handler = functools.partial(Ranges, directory=str(tmp_path))
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        g = fastgrib.open('http://127.0.0.1:%d/ec.grib2' % srv.server_address[1])
        keys = ('name', 'short_name', 'level', 'level_kind', 'level_type', 'level_value')
        assert [[getattr(e, k) for k in keys] for e in g.records] == \
            [[getattr(r, k) for k in keys] for r in f.records]
        assert len(g.select(short_name='sot')) == len(f.select(short_name='sot'))
        loc, ref = reference('ecmwf_local')
        for e in g.records[::2]:
            assert same(g.read(e), ref[e.index - 1]), repr(e)
    finally:
        srv.shutdown()


def _tail(r):
    """An inventory line but for its number (a part of a file numbers from 1)."""
    return repr(r).split(':', 1)[1]


@pytest.mark.parametrize('name, ext', [('gfs', '.grib2'), ('g1_narr', '.grb')])
def test_remote_without_index_fetches_only_what_is_wanted(server, monkeypatch, name, ext):
    """No index: max_records and byte_range fetch the messages wanted, a
    window at a time in parallel ranges (not a request a message, nor the
    whole file), each placed and read as in the file; without either, the
    whole file in parallel ranges."""
    from fastgrib import _remote
    monkeypatch.setattr(_remote, 'WALK_BYTES', 4096)     # windows smaller than a message
    monkeypatch.setattr(_remote, 'RUN_BYTES', 8192)      # ... each in several ranges
    url = server + 'noidx_' + name + ext
    loc = fastgrib.open(path(name, ext)) if ext == '.grib2' else grib1(name)
    size = os.path.getsize(path(name, ext))
    Ranges.gets.clear()
    f = fastgrib.open(url, max_records=4)
    assert [repr(r) for r in f.records] == [repr(r) for r in loc.records[:4]]
    assert sum(n for _, n in Ranges.gets) < size
    assert len(Ranges.gets) < 40
    for r, s in zip(f.records, loc.records):
        assert np.array_equal(f.read(r), loc.read(s), equal_nan=True)
    a, b = loc.records[3].offset, loc.records[7].offset
    want = [r for r in loc.records if a <= r.offset < b]
    Ranges.gets.clear()
    f = fastgrib.open(url, byte_range=(a - 1, b))         # (from inside the message before)
    assert [_tail(r) for r in f.records] == [_tail(r) for r in want]
    f = fastgrib.open(url, byte_range=(a, b))
    assert [_tail(r) for r in f.records] == [_tail(r) for r in want]
    assert np.array_equal(f.read(f.records), loc.read(want), equal_nan=True)
    assert f.message(f.records[0]) == loc.message(want[0])
    assert f.keys(f.records[-1]) == loc.keys(want[-1])
    Ranges.gets.clear()
    f = fastgrib.open(url)
    assert [repr(r) for r in f.records] == [repr(r) for r in loc.records]
    assert len(Ranges.gets) > 1 and sum(n for _, n in Ranges.gets) == size


def test_index_written_and_given(server, tmp_path):
    """write_index(): wgrib2 -s lines, which open(url, index=...) reads (a
    path or a URL) for a file without one beside it -- its last message
    fetched to its own end, not the file's; an index of part of a file
    serves that part. The command line's -grib writes the fields picked."""
    from fastgrib.__main__ import main
    loc = fastgrib.open(path('gfs'))
    text = loc.write_index()
    assert text.splitlines() == [repr(r) for r in loc.records]
    idx = tmp_path / 'gfs.idx'
    loc.write_index(str(idx))
    url = server + 'noidx_gfs.grib2'
    for index in (str(idx), server + 'gfs.grib2.idx'):
        f = fastgrib.open(url, index=index)
        assert [repr(e) for e in f.records] == [repr(r) for r in loc.records]
        assert np.array_equal(f.read(f.records[-1]), loc.read(loc.records[-1]), equal_nan=True)
    part = tmp_path / 'part.idx'
    part.write_text(''.join(text.splitlines(True)[5:9]))
    Ranges.gets.clear()
    f = fastgrib.open(url, index=str(part))
    last = f.records[-1]
    assert np.array_equal(f.read(last), loc.read(loc.records[8]), equal_nan=True)
    assert sum(n for _, n in Ranges.gets) < os.path.getsize(path('gfs')) // 4
    with pytest.raises(ValueError):
        fastgrib.open(path('gfs'), index=str(idx))
    out = tmp_path / 'tmp.grib2'
    assert main([url, '-index', str(idx), '-match', ':TMP:', '-grib', str(out)]) == 0
    g = fastgrib.open(str(out))
    want = loc.find(':TMP:')
    assert [_tail(r).split(':', 1)[1] for r in g.records] == \
        [_tail(r).split(':', 1)[1] for r in want]
    assert np.array_equal(g.read(g.records), loc.read(want), equal_nan=True)


def test_remote_disk_cache(server, tmp_path):
    """cache=: messages fetched once are kept on disk, a file each, and a
    later open (another process's alike) reads them from there -- only the
    index fetched again; the last message (no end in the index) too; a
    damaged copy on disk fetched again."""
    url = server + 'gfs.grib2'
    loc = fastgrib.open(path('gfs'))
    picks = [0, 5, len(loc) - 1]
    f = fastgrib.open(url, cache=str(tmp_path))
    first = [f.read(i) for i in picks]
    assert sum(len(fs) for _, _, fs in os.walk(tmp_path)) >= len(picks)
    Ranges.gets.clear()
    g = fastgrib.open(url, cache=str(tmp_path))
    again = [g.read(i) for i in picks]
    assert [p for p, _ in Ranges.gets] == ['/gfs.grib2.idx']      # nothing but the index
    for a, b, i in zip(first, again, picks):
        assert np.array_equal(a, b, equal_nan=True) and np.array_equal(b, loc.read(i),
                                                                      equal_nan=True)
    folder = next(d for d, _, fs in os.walk(tmp_path) if fs)
    for name in os.listdir(folder):             # cut short on disk: fetched anew
        with open(os.path.join(folder, name), 'r+b') as fh:
            fh.truncate(10)
    h = fastgrib.open(url, cache=str(tmp_path))
    assert np.array_equal(h.read(5), loc.read(5), equal_nan=True)


def test_remote_limits_with_an_index(server):
    """With an index, max_records and byte_range pick among its entries."""
    loc = fastgrib.open(path('gfs'))
    f = fastgrib.open(server + 'gfs.grib2', max_records=5)
    assert [repr(e) for e in f.records] == [repr(r) for r in loc.records[:5]]
    a, b = loc.records[10].offset, loc.records[20].offset
    f = fastgrib.open(server + 'gfs.grib2', byte_range=(a, b), max_records=6)
    assert [repr(e) for e in f.records] == [repr(r) for r in loc.records[10:16]]
    g = fastgrib.open(path('gfs'), byte_range=(a, None))
    assert [repr(r) for r in g.records] == [repr(r) for r in loc.records[10:]]
    with pytest.raises(ValueError):
        fastgrib.open(path('gfs'), max_records=0)
    with pytest.raises(ValueError):
        fastgrib.open(path('gfs'), byte_range=(b, a))


def test_fetch_cache_counts_what_it_holds(monkeypatch):
    import time as _t
    from fastgrib import _remote
    monkeypatch.setattr(_remote, 'fetch', lambda url, a, b: (_t.sleep(0.01), b'x' * (b - a + 1))[1])
    fe = _remote.Fetcher('http://x')
    spans = [(i * 1000, i * 1000 + 999) for i in range(10)]
    for _ in range(3):
        ts = [threading.Thread(target=fe.messages, args=(spans,)) for _ in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    assert fe.held == sum(len(v) for v in fe.cache.values()) == 10000


def test_index_names_local_parameters():
    """A .idx line wgrib2 could not name ('var discipline=3 center=7 ...', as
    NOAA's HAFS indexes have for simulated brightness temperatures) gives the
    record the name the message itself gets once read; another centre's
    local number stays unnamed."""
    from fastgrib import _remote
    text = ('1:0:d=2024100900:var discipline=3 center=7 local_table=1 parmcat=192 '
            'parm=58:top of atmosphere:12 hour fcst:\n'
            '2:100:d=2024100900:var discipline=3 center=98 local_table=1 parmcat=192 '
            'parm=58:top of atmosphere:12 hour fcst:\n')
    ncep, other = _remote._noaa(text)
    assert (ncep.name, ncep.long_name) == (
        'SBTAGR13', 'Simulated Brightness Temperature for nadir ABI GOES-R, Band-13')
    assert ncep.line.startswith('1:0:d=2024100900:var discipline=3 center=7')   # as published
    assert other.name == 'var3_192_58'


def test_fetch_backs_off_and_retries(monkeypatch):
    """A throttled request (S3's 503 Slow Down) is tried again after a growing,
    jittered wait, and a Retry-After header is honoured; a 404 is not retried."""
    import urllib.error
    from fastgrib import _remote
    _remote._net()
    calls, sleeps = [], []
    monkeypatch.setattr(_remote.time, 'sleep', sleeps.append)
    monkeypatch.setattr(_remote, '_proxied', lambda: False)

    def flaky(url, head, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 503, 'Slow Down', {'Retry-After': '2'}, None)
        if len(calls) == 2:
            raise urllib.error.HTTPError(url, 503, 'Slow Down', {}, None)
        return b'GRIB'

    monkeypatch.setattr(_remote, '_get', flaky)
    assert _remote.fetch('https://example.invalid/f') == b'GRIB'
    assert len(calls) == 3 and sleeps[0] == 2.0
    assert _remote.BACKOFF * 2 * 0.5 <= sleeps[1] <= _remote.BACKOFF * 2 * 1.5

    def missing(url, head, timeout):
        calls.append(url)
        raise urllib.error.HTTPError(url, 404, 'Not Found', {}, None)

    calls.clear()
    monkeypatch.setattr(_remote, '_get', missing)
    with pytest.raises(urllib.error.HTTPError):
        _remote.fetch('https://example.invalid/g')
    assert len(calls) == 1
