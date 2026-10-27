"""Test data: simple-packed lat-lon messages rewritten in other scanning
modes -- columns consecutive (0x20), every other row reversed (0x10), and
their mixes with 0x80 / 0x40 -- the same fields, their points (values and
bitmap bits) put in the new order by wgrib2's own rule (Scan.c's ij2p).

    python tools/testdata/make_scan_grib.py IN.grib2 OUT.grib2
"""
import struct
import sys

import numpy as np

MODES = (0x20, 0x60, 0x10, 0x50, 0x30, 0x70, 0x90, 0xD0, 0xA0, 0xF0)


def ij2p(i, j, scan, nx, ny):
    """wgrib2's ij2p: the file index of we:sn point (i, j)."""
    j = j if scan & 64 else ny - 1 - j
    i = np.where(bool(scan & 16) & (j % 2 == 1), nx - 1 - i, i)
    i = nx - 1 - i if scan & 128 else i
    return j + i * ny if scan & 32 else i + nx * j


def sections(m):
    out, p = [], 16
    while p < len(m) - 4:
        ln = struct.unpack('>I', m[p:p + 4])[0]
        out.append(bytearray(m[p:p + ln]))
        p += ln
    return out


def rescan(m, new):
    secs = sections(m)
    s3 = next(s for s in secs if s[4] == 3)
    s5 = next(s for s in secs if s[4] == 5)
    s6 = next(s for s in secs if s[4] == 6)
    s7 = next(s for s in secs if s[4] == 7)
    assert struct.unpack('>H', s3[12:14])[0] == 0, 'lat-lon grids only'
    assert struct.unpack('>H', s5[9:11])[0] == 0, 'simple packing only'
    nx, ny = struct.unpack('>II', s3[30:38])
    old = s3[71]
    n, nbits = nx * ny, s5[19]
    jj, ii = np.divmod(np.arange(n), nx)                    # we:sn points
    src, dst = ij2p(ii, jj, old, nx, ny), ij2p(ii, jj, new, nx, ny)
    perm = np.empty(n, np.int64)
    perm[dst] = src                                        # new file index -> old
    if s6[5] == 0:
        bits = np.unpackbits(np.frombuffer(bytes(s6[6:]), np.uint8), count=n).astype(bool)
    else:
        bits = np.ones(n, bool)
    nv = int(bits.sum())
    if nbits:
        raw = np.unpackbits(np.frombuffer(bytes(s7[5:]), np.uint8))[:nv * nbits]
        vals = raw.reshape(nv, nbits)
        rank = np.cumsum(bits) - 1                         # old file index -> value number
        newbits = bits[perm]
        vals = vals[rank[perm[newbits]]]
        s7[5:] = np.packbits(vals.reshape(-1)).tobytes()
        s7[0:4] = struct.pack('>I', len(s7))
    else:
        newbits = bits[perm]
    if s6[5] == 0:
        s6[6:] = np.packbits(newbits).tobytes()
    s3[71] = new
    if (old ^ new) & 0x80:                      # the first point is now at the other side
        s3[50:54], s3[59:63] = s3[59:63], s3[50:54]
    if (old ^ new) & 0x40:
        s3[46:50], s3[55:59] = s3[55:59], s3[46:50]
    body = b''.join(bytes(s) for s in secs)
    return m[:8] + struct.pack('>Q', 16 + len(body) + 4) + body + b'7777'


def main(src, out):
    b = open(src, 'rb').read()
    msgs, pos = [], 0
    while (pos := b.find(b'GRIB', pos)) >= 0:
        n = struct.unpack('>Q', b[pos + 8:pos + 16])[0]
        msgs.append(b[pos:pos + n])
        pos += n
    res = [rescan(m, MODES[k % len(MODES)]) for k, m in enumerate(msgs)]
    open(out, 'wb').write(b''.join(res))
    print('%d messages in scanning modes %s' % (len(res), ', '.join('%#x' % m for m in MODES)))


if __name__ == '__main__':
    main(*sys.argv[1:3])
