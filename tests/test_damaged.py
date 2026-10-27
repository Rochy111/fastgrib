"""Damaged files: every intact field still read (and equal to wgrib2's),
the damaged ones reported, never a crash or a read past the data -- on the
CPU and on the GPU."""
import struct
import warnings

import numpy as np
import pytest

import fastgrib
from helpers import gpu_ok, path, reference, same


def messages(name):
    b = open(path(name), 'rb').read()
    out, pos = [], 0
    while True:
        pos = b.find(b'GRIB', pos)
        if pos < 0:
            return out
        n = struct.unpack('>Q', b[pos + 8:pos + 16])[0]
        out.append(b[pos:pos + n])
        pos += n


def opened(data):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f = fastgrib.open(data)
    return f, w


def intact(f, ref_by_line):
    """Every field f read equals wgrib2's value of the same field."""
    for r in f.records:
        line = repr(r).split(':', 2)[2]
        assert same(f.read(r), ref_by_line[line]), repr(r)


@pytest.fixture(scope='module')
def gfs():
    f, ref = reference('gfs')
    by_line = {repr(r).split(':', 2)[2]: v for r, v in zip(f.records, ref)}
    return messages('gfs'), by_line


@pytest.mark.parametrize('cut', [0.13, 0.5, 0.61, 0.97])
def test_cut_short(gfs, cut):
    msgs, ref = gfs
    whole = b''.join(msgs)
    f, w = opened(whole[:int(len(whole) * cut)])
    complete = sum(1 for k in range(len(msgs)) if sum(map(len, msgs[:k + 1])) <= len(whole) * cut)
    assert len(f) == complete
    assert len(f.damaged) == (0 if complete == len(msgs) else 1) and w
    intact(f, ref)


def test_garbage_and_bad_messages(gfs):
    msgs, ref = gfs
    rng = np.random.default_rng(5)
    junk = lambda n: rng.integers(0, 256, n, dtype=np.uint8).tobytes().replace(b'GRIB', b'GRIX')  # noqa: E731
    bad_len = bytearray(msgs[3])
    bad_len[8:16] = struct.pack('>Q', 1 << 40)                 # a length past anything
    no_end = bytearray(msgs[5])
    no_end[-4:] = b'7778'
    grib1 = b'GRIB' + (40).to_bytes(3, 'big') + b'\x01' + bytes(28) + b'7777'
    parts = [junk(100), msgs[0], junk(3000), msgs[1], msgs[2], bytes(bad_len), msgs[4],
             bytes(no_end), grib1, msgs[6], junk(17)]
    f, w = opened(b''.join(parts))
    kept = {repr(r).split(':', 2)[2] for r in f.records}
    want = [msgs[k] for k in (0, 1, 2, 4, 6)]
    assert len(f) == len(want)
    assert len(f.damaged) == 3 and any('GRIB1' in d for d in f.damaged)
    intact(f, ref)
    assert kept


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
def test_fuzzed_data_never_crashes(device):
    """Bytes flipped in the packing and data sections of single messages:
    each decodes to something or raises ValueError -- no crash, no read
    out of bounds (the GPU's context stays healthy after)."""
    if device == 'gpu' and not gpu_ok():
        pytest.skip('no usable GPU')
    fastgrib.set_device(device)
    rng = np.random.default_rng(11)
    try:
        for name in ('gfs', 'nbm'):
            for m in messages(name)[::3]:
                f0 = fastgrib.open(m)
                r = f0.records[0]
                lo = r.s5                                    # packing, bitmap and data sections
                for _ in range(6):
                    b = bytearray(m)
                    for _ in range(rng.integers(1, 6)):
                        b[rng.integers(lo, len(b) - 4)] = rng.integers(0, 256)
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        f = fastgrib.open(bytes(b))
                    for rec in f.records:
                        try:
                            v = f.read(rec, gpu=device == 'gpu')
                            if device == 'gpu':
                                v = v.get()
                            assert v.shape == rec.grid.shape
                        except ValueError:
                            pass
        # and the decoder is still right afterwards
        f, ref = reference('nbm')
        for rr, w in zip(f.records, ref):
            v = f.read(rr, gpu=device == 'gpu')
            assert same(v.get() if device == 'gpu' else v, w)
    finally:
        fastgrib.set_device('auto')
