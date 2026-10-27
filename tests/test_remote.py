"""Remote files: .idx and ECMWF's .index, byte ranges, the fetch cache."""
import http.server
import os
import threading

import pytest

import fastgrib

from helpers import Ranges, path, reference, same


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
