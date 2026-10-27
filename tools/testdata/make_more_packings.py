"""Test data in packings no tool at hand writes, from messages that are:

    python tools/testdata/make_more_packings.py IEEE32.grib2 SIMPLE.grib2 OUT_DIR

gfs_ieee64.grib2  the 32-bit IEEE messages (5.4, precision 1) as 64-bit
                  ones (precision 2): the same values, widened
gfs_rle.grib2     the simple-packed messages without a bitmap as run-length
                  (5.200, JMA's): their integers binned into 12 levels, a
                  stretch of each made undefined (level code 0), 8-bit codes

wgrib2 reads 5.200; not 64-bit 5.4 -- that file's inventory (written here,
from IEEE32.inv) and values (the 32-bit file's) come from the 32-bit one.
"""
import os
import struct
import sys

import numpy as np


def messages(path):
    b = open(path, 'rb').read()
    out, pos = [], 0
    while (pos := b.find(b'GRIB', pos)) >= 0:
        n = struct.unpack('>Q', b[pos + 8:pos + 16])[0]
        out.append(b[pos:pos + n])
        pos += n
    return out


def sections(m):
    out, p = [], 16
    while p < len(m) - 4:
        ln = struct.unpack('>I', m[p:p + 4])[0]
        out.append(bytearray(m[p:p + ln]))
        p += ln
    return out


def message(m, secs):
    body = b''.join(bytes(s) for s in secs)
    return m[:8] + struct.pack('>Q', 16 + len(body) + 4) + body + b'7777'


def ieee64(m):
    secs = sections(m)
    s5 = next(s for s in secs if s[4] == 5)
    s7 = next(s for s in secs if s[4] == 7)
    assert struct.unpack('>H', s5[9:11])[0] == 4 and s5[11] == 1
    s5[11] = 2
    vals = np.frombuffer(bytes(s7[5:]), '>f4').astype('>f8')
    s7[5:] = vals.tobytes()
    s7[0:4] = struct.pack('>I', len(s7))
    return message(m, secs)


def _encode(codes, mv, nbits):
    """Level codes as wgrib2 reads run-length: each code, then its repeat
    count less one in base 2^nbits - 1 - mv digits (least first), each
    digit + mv + 1."""
    span = (1 << nbits) - 1 - mv
    out = []
    i = 0
    while i < len(codes):
        j = i
        while j < len(codes) and codes[j] == codes[i]:
            j += 1
        out.append(int(codes[i]))
        rest = j - i - 1
        while rest:
            out.append(mv + 1 + rest % span)
            rest //= span
        i = j
    return out


def rle(m, nlev=12, nbits=8):
    secs = sections(m)
    s5 = next(s for s in secs if s[4] == 5)
    s6 = next(s for s in secs if s[4] == 6)
    s7 = next(s for s in secs if s[4] == 7)
    if struct.unpack('>H', s5[9:11])[0] != 0 or s6[5] != 255 or s5[19] == 0:
        return None
    n, nb = struct.unpack('>I', s5[5:9])[0], s5[19]
    bits = np.unpackbits(np.frombuffer(bytes(s7[5:]), np.uint8))[:n * nb].reshape(n, nb)
    ints = bits.dot(1 << np.arange(nb - 1, -1, -1, dtype=np.int64))
    edges = np.quantile(ints, np.linspace(0, 1, nlev + 1)[1:-1])
    codes = np.searchsorted(edges, ints, side='right') + 1             # 1..nlev
    codes[n // 3:n // 3 + n // 10] = 0                                  # undefined
    centre = [int(np.median(ints[codes == k])) if (codes == k).any() else 0
              for k in range(1, nlev + 1)]
    lv = [max(-32767, min(32767, c - 16000)) for c in centre]          # signed, some < 0
    stream = _encode(codes, nlev, nbits)
    packed = np.packbits(np.array([[(c >> (nbits - 1 - k)) & 1 for k in range(nbits)]
                                   for c in stream], np.uint8).ravel())
    sm = lambda v: (0x8000 | -v) if v < 0 else v                        # noqa: E731
    new5 = bytearray(struct.pack('>IBIH', 0, 5, n, 200))
    new5 += bytes([nbits]) + struct.pack('>HH', nlev, nlev) + bytes([0])
    new5 += b''.join(struct.pack('>H', sm(v)) for v in lv)
    new5[0:4] = struct.pack('>I', len(new5))
    new7 = bytearray(struct.pack('>IB', 5 + len(packed), 7)) + packed.tobytes()
    return message(m, [s if s[4] not in (5, 7) else (new5 if s[4] == 5 else new7)
                       for s in secs])


def main(ieee, simple, out):
    wide = [ieee64(m) for m in messages(ieee)]
    with open(os.path.join(out, 'gfs_ieee64.grib2'), 'wb') as f:
        f.write(b''.join(wide))
    # its inventory: the 32-bit file's (wgrib2 -s stops at 64-bit data), at
    # the new offsets
    inv = os.path.splitext(ieee)[0] + '.inv'
    if os.path.exists(inv):
        at, lines = 0, []
        for m, line in zip(wide, open(inv).read().splitlines()):
            num, _, rest = line.split(':', 2)
            lines.append('%s:%d:%s' % (num, at, rest))
            at += len(m)
        open(os.path.join(out, 'gfs_ieee64.inv'), 'w').write(''.join(x + '\n' for x in lines))
    got = [x for x in (rle(m) for m in messages(simple)) if x]
    with open(os.path.join(out, 'gfs_rle.grib2'), 'wb') as f:
        f.write(b''.join(got))
    print('gfs_ieee64.grib2, gfs_rle.grib2 (%d messages)' % len(got))


if __name__ == '__main__':
    main(*sys.argv[1:4])
