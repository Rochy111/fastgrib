"""A file past 4 GiB: fields beyond the 32-bit offsets read as anywhere --
scanned, numbered, located, decoded on the CPU and the GPU, fetched by
byte range. It writes a 4.3 GB file, so it runs only when asked:

    FASTGRIB_BIG_TEST=1 pytest tests/test_big.py
"""
import os

import numpy as np
import pytest

import fastgrib
from helpers import gpu_ok, path, reference, same

pytestmark = pytest.mark.skipif(not os.environ.get('FASTGRIB_BIG_TEST'),
                                reason='writes a 4.3 GB file: set FASTGRIB_BIG_TEST=1')


@pytest.fixture(scope='module')
def big(tmp_path_factory):
    """gfs.grib2, 4.25 GiB of zeros, gfs.grib2 and nbm.grib2 again."""
    p = tmp_path_factory.mktemp('big') / 'big.grib2'
    gfs, nbm = open(path('gfs'), 'rb').read(), open(path('nbm'), 'rb').read()
    gap = (4 << 30) + (1 << 28)
    with open(p, 'wb') as f:
        f.write(gfs)
        zeros = bytes(64 << 20)
        for _ in range(gap // len(zeros)):
            f.write(zeros)
        f.write(gfs)
        f.write(nbm)
    yield str(p), len(gfs), gap
    os.remove(p)


def test_past_4_gib(big):
    p, n, gap = big
    f = fastgrib.open(p)
    g, ref = reference('gfs')
    nb, nref = reference('nbm')
    assert len(f) == 2 * len(g) + len(nb) and not f.damaged
    late = f.records[len(g):]
    assert min(r.offset for r in late) == n + gap > 1 << 32
    for r, w in zip(late[:len(g)], g.records):
        assert repr(r).split(':', 2)[2] == repr(w).split(':', 2)[2]
        assert r.offset == w.offset + n + gap
    want = ref + ref + nref
    parts = (late[:len(g)], late[len(g):])                  # GFS's grid, then NBM's
    for part in parts:
        assert all(same(v, want[r.index - 1]) for r, v in zip(part, f.read(part)))
    for r, v in f.fields(late[-25:]):
        assert same(v, want[r.index - 1])
    if gpu_ok():
        for part in parts:
            vals = f.read(part, gpu=True).get()
            assert all(same(v, want[r.index - 1]) for r, v in zip(part, vals))
    f.close()


def test_past_4_gib_remote(big, tmp_path):
    """Byte ranges past 4 GiB, from an index of the file."""
    import functools
    import http.server
    import threading
    from helpers import Ranges
    p, n, gap = big
    f = fastgrib.open(p)
    with open(p + '.idx', 'w') as idx:
        idx.write(''.join(repr(r) + '\n' for r in f.records))
    handler = functools.partial(Ranges, directory=os.path.dirname(p))
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = 'http://127.0.0.1:%d/%s' % (srv.server_address[1], os.path.basename(p))
        g = fastgrib.open(url)
        late = g.records[-12:]
        assert late[0].offset > 1 << 32
        assert all(np.array_equal(a, b, equal_nan=True)
                   for a, b in zip(g.read(late), f.read(f.records[-12:])))
    finally:
        srv.shutdown()
        f.close()
        os.remove(p + '.idx')
