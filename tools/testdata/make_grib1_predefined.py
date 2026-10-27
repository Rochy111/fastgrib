"""Test data for GRIB1 predefined grids (no GDS: the grid only by its
number, PDS octet 7): tests/data/g1_predef.grb --

- WMO's international grids 21-26 and 61-64, each value its point's
  index in the message (simple packing, 16 bits), for the tests to place;
- g1_narr.grb's first three messages with their GDS taken out (NCEP's
  grid 221), for the tests to compare with the same messages described.

    python tools/testdata/make_grib1_predefined.py DATA_DIR
"""
import os
import sys

COUNTS = {21: 1333, 22: 1333, 23: 1333, 24: 1333, 25: 1297, 26: 1297, 61: 4096, 62: 4096,
          63: 4096, 64: 4096}


def messages(narr):
    out, pos = [], 0
    while True:
        pos = narr.find(b'GRIB', pos)
        if pos < 0:
            return out
        n = int.from_bytes(narr[pos + 4:pos + 7], 'big')
        out.append(bytes(narr[pos:pos + n]))
        pos += n


def strip_gds(m):
    m = bytearray(m)
    plen = int.from_bytes(m[8:11], 'big')
    g = 8 + plen
    glen = int.from_bytes(m[g:g + 3], 'big')
    m[15] &= 0x7F                               # PDS octet 8: no GDS
    m = m[:g] + m[g + glen:]
    m[4:7] = len(m).to_bytes(3, 'big')
    return bytes(m)


def wmo(pds, grid, n):
    p = bytearray(pds)
    p[6], p[7] = grid, p[7] & 0x3F              # the grid; no GDS, no bitmap
    p[26:28] = b'\0\0'                          # D = 0
    data = b''.join(i.to_bytes(2, 'big') for i in range(n))
    bds = bytes([0, 0, 0, 0]) + b'\0\0' + b'\0\0\0\0' + bytes([16]) + data
    if len(bds) % 2:
        bds += b'\0'
    bds = len(bds).to_bytes(3, 'big') + bds[3:]
    m = b'GRIB' + b'\0\0\0' + bytes([1]) + bytes(p) + bds + b'7777'
    return m[:4] + len(m).to_bytes(3, 'big') + m[7:]


def main(data):
    narr = messages(open(os.path.join(data, 'g1_narr.grb'), 'rb').read())
    pds = narr[0][8:8 + int.from_bytes(narr[0][8:11], 'big')]
    out = [wmo(pds, g, n) for g, n in COUNTS.items()] + [strip_gds(m) for m in narr[:3]]
    with open(os.path.join(data, 'g1_predef.grb'), 'wb') as f:
        f.write(b''.join(out))
    print(len(out), 'messages')


if __name__ == '__main__':
    main(sys.argv[1])
