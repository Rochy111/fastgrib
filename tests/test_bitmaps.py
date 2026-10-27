"""Bitmaps given before (254): within a message, across messages, fetched."""
import http.server
import os
import threading

import pytest

import fastgrib

from helpers import Ranges, path, reference, same


def test_bitmap_from_an_earlier_message_is_refused():
    """A message that reuses an earlier message's bitmap, read alone."""
    f = fastgrib.open(path('gfs'))
    r = next(x for x in f.records if x.bitmap >= 0)
    b = bytearray(open(path('gfs'), 'rb').read())
    s6 = r.s6
    b[s6 + 5] = 254                            # 'the bitmap given before'
    end = next(x.offset for x in f.records if x.offset > r.offset)
    with pytest.warns(UserWarning, match='damaged'):
        f = fastgrib.open(bytes(b[r.offset:end]))
    assert len(f) == 0 and 'earlier message' in f.damaged[0]


def _split(msg):
    """A message of two fields as two messages, the second's bitmap section
    still saying 254: its bitmap then is the first message's."""
    import struct
    secs, p = [], 16
    while p < len(msg) - 4:
        ln = struct.unpack('>I', msg[p:p + 4])[0]
        secs.append(msg[p:p + ln])
        p += ln
    nums = [s[4] for s in secs]
    k = nums.index(4, nums.index(7))            # the second field's product section
    out = []
    for body in (b''.join(secs[:k]), b''.join(s for s in secs if s[4] in (1, 2, 3)) +
                 b''.join(secs[k:])):
        out.append(msg[:8] + struct.pack('>Q', 16 + len(body) + 4) + body + b'7777')
    return out


@pytest.fixture(scope='module')
def bm254_split(tmp_path_factory):
    """gfs_bm254 with its two-field messages split: second messages whose
    bitmap is the message before's (beyond the standard, which keeps 254
    within a message; wgrib2 refuses them) -- read as the joined file."""
    import struct
    b = open(path('gfs_bm254'), 'rb').read()
    msgs, pos = [], 0
    while (pos := b.find(b'GRIB', pos)) >= 0:
        n = struct.unpack('>Q', b[pos + 8:pos + 16])[0]
        msgs.append(b[pos:pos + n])
        pos += n
    out = []
    for m in msgs:
        out += _split(m) if m.count(b'\x00\x00\x00\x06\x06\xfe') else [m]
    d = tmp_path_factory.mktemp('bm254')
    p = d / 'split.grib2'
    p.write_bytes(b''.join(out))
    f = fastgrib.open(str(p))
    (d / 'split.grib2.idx').write_text(''.join(repr(r) + '\n' for r in f.records))
    return str(d)


def test_bitmap_reused_across_messages(bm254_split):
    f = fastgrib.open(os.path.join(bm254_split, 'split.grib2'))
    loc, ref = reference('gfs_bm254')
    assert len(f) == len(loc)
    reused = 0
    for r, w in zip(f.records, ref):
        assert same(f.read(r), w), repr(r)
        reused += r.bitmap >= 0 and r.bitmap < r.offset
    assert reused == 6


def test_bitmap_reused_remote(bm254_split):
    """Remote picks of the fields whose bitmap an earlier message defines:
    that bitmap fetched too -- and fields reusing one within their own
    message, from wgrib2's file."""
    import functools
    handler = functools.partial(Ranges, directory=bm254_split)
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = 'http://127.0.0.1:%d/split.grib2' % srv.server_address[1]
        f = fastgrib.open(url)
        loc, ref = reference('gfs_bm254')
        local = fastgrib.open(os.path.join(bm254_split, 'split.grib2'))
        picks = [e for e, r in zip(f.records, local.records) if r.bitmap >= 0 and r.bitmap < r.offset]
        assert len(picks) == 6
        for e in picks:
            assert same(f.read(e), ref[e.index - 1]), repr(e)
        stack = f.read(picks)
        assert all(same(v, ref[e.index - 1]) for v, e in zip(stack, picks))
    finally:
        srv.shutdown()


def test_bitmap_reused_within_a_message_remote(server):
    f = fastgrib.open(server + 'gfs_bm254.grib2')
    loc, ref = reference('gfs_bm254')
    subs = [e for e in f.records if e.sub == 2]
    assert len(subs) == 6
    for e in subs:
        assert same(f.read(e), ref[e.index - 1]), repr(e)
