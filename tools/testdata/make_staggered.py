"""Test data for staggered grids: tests/data/stag.grib2, small rotated
lat-lon grids (3.1) staggered every way WMO's scanning flags allow -- odd
or even rows offset, every point offset along j, offset rows a point
short, either direction -- each value its point's index (wgrib2 -csv
gives each point's place: make_test_data.sh); and egrid.grib2, NCEP's
rotated E-grid (3.32768), the same grid as stag.grib2's second message.

    python3 tools/testdata/make_staggered.py OUT_DIR
"""
import os
import struct
import sys

import numpy as np

NX, NY = 31, 21
DI, DJ = 0.5, 0.4                               # rotated degrees
CLAT, CLON = 40.0, 260.0                        # where the rotated equator meets its prime meridian
SCANS = (0x48, 0x44, 0x49, 0x45, 0x4A, 0x4B, 0xC8, 0x08, 0x88)


def sm(v, n):
    return (abs(v) | (1 << (8 * n - 1) if v < 0 else 0)).to_bytes(n, 'big')


def points(scan):
    """How many values: offset rows a point short when flag 0x01 says."""
    rows = NY - ((scan >> 1) & scan & 1)
    f = np.arange(rows)
    return int((NX - (scan & 1) * np.where(f % 2 == 0, scan >> 3 & 1, scan >> 2 & 1)).sum())


def unrotate(rlat, rlon):
    la, lo = np.radians(rlat), np.radians(rlon)
    v = np.array([np.cos(lo) * np.cos(la), np.sin(lo) * np.cos(la), np.sin(la)])
    t, o = np.radians(-CLAT), np.radians(-CLON)       # (the southern pole at CLAT - 90, CLON)
    st, ct, so, co = np.sin(t), np.cos(t), np.sin(o), np.cos(o)
    m = np.array([[ct * co, -ct * so, -st], [so, co, 0.0], [st * co, -st * so, ct]])
    g = np.linalg.solve(m, v)
    return float(np.degrees(np.arcsin(g[2]))), float(np.degrees(np.arctan2(g[1], g[0])) % 360)


def message(scan, s3body, n):
    s1 = struct.pack('>IBHHBBBHBBBBBBB', 21, 1, 7, 0, 2, 1, 1, 2026, 10, 26, 0, 0, 0, 0, 1)
    earth = bytes([6]) + b'\xff' * 15
    s3 = bytes([0]) + struct.pack('>I', n) + bytes([0, 0]) + s3body(earth)
    s3 = struct.pack('>IB', 5 + len(s3), 3) + s3
    s4 = struct.pack('>IBHH', 34, 4, 0, 0) + bytes([0, 0, 2, 0, 96]) + struct.pack('>HB', 0, 0) + \
        bytes([1]) + struct.pack('>I', 0) + bytes([1, 0]) + struct.pack('>I', 0) + \
        bytes([255, 255]) + struct.pack('>I', 0)
    s5 = struct.pack('>IBIHfhhBB', 21, 5, n, 0, 0.0, 0, 0, 16, 0)
    s6 = struct.pack('>IBB', 6, 6, 255)
    s7 = struct.pack('>IB', 5 + 2 * n, 7) + np.arange(n, dtype='>u2').tobytes()
    body = s1 + s3 + s4 + s5 + s6 + s7
    return b'GRIB' + bytes([0, 0, 0, 2]) + struct.pack('>Q', 16 + len(body) + 4) + body + b'7777'


def rotated(scan):
    """3.1: first and last points in the rotated frame (La2, Lo2 those of
    a row not offset, as wgrib2 checks them)."""
    rlat1, rlon1 = -(NY - 1) / 2 * DJ, -(NX - 1) / 2 * DI
    la1, la2 = (rlat1, -rlat1) if scan & 0x40 else (-rlat1, rlat1)
    lo1, lo2 = (-rlon1, rlon1) if scan & 0x80 else (rlon1, -rlon1)

    def body(earth):
        return struct.pack('>H', 1) + earth + struct.pack('>IIII', NX, NY, 0, 0xFFFFFFFF) + \
            sm(round(la1 * 1e6), 4) + struct.pack('>I', round(lo1 * 1e6) % 360000000) + \
            bytes([0x38]) + sm(round(la2 * 1e6), 4) + \
            struct.pack('>I', round(lo2 * 1e6) % 360000000) + \
            struct.pack('>II', round(DI * 1e6), round(DJ * 1e6)) + bytes([scan]) + \
            sm(round((CLAT - 90) * 1e6), 4) + struct.pack('>I', round(CLON * 1e6)) + \
            struct.pack('>f', 0.0)
    return message(scan, body, points(scan))


def egrid():
    """NCEP's 3.32768 of stag.grib2's 0x44 grid: its first point and
    centre geographic, Di, Dj; no offset flags (the E-grid's own)."""
    la1, lo1 = unrotate(-(NY - 1) / 2 * DJ, -(NX - 1) / 2 * DI)

    def body(earth):
        return struct.pack('>H', 32768) + earth + struct.pack('>IIII', NX, NY, 0, 0xFFFFFFFF) + \
            sm(round(la1 * 1e6), 4) + struct.pack('>I', round(lo1 * 1e6)) + bytes([0x38]) + \
            sm(round(CLAT * 1e6), 4) + struct.pack('>I', round(CLON * 1e6)) + \
            struct.pack('>II', round(DI * 1e6), round(DJ * 1e6)) + bytes([0x40])
    return message(0x40, body, NX * NY)


def sm1(v):
    """GRIB1's 3-octet millidegrees, the sign in the first bit."""
    v = round(v * 1000)
    return (abs(v) | (0x800000 if v < 0 else 0)).to_bytes(3, 'big')


def grib1(pds, kind, last=None):
    """A GRIB1 message on NCEP's grid 203 (E-grid) or 205 (B-grid) of the
    domain above, each value its point's index."""
    la1, lo1 = unrotate(-(NY - 1) / 2 * DJ, -(NX - 1) / 2 * DI)
    gds = bytes([0, 255, kind]) + struct.pack('>HH', NX, NY) + sm1(la1) + sm1(lo1) + \
        bytes([0x88]) + sm1(CLAT) + sm1(CLON) + struct.pack('>HH', round(DI * 1000),
                                                             round(DJ * 1000)) + bytes([0x40])
    gds += sm1(last[0]) + sm1(last[1]) if last else b'\0' * 4
    gds = (len(gds) + 3).to_bytes(3, 'big') + gds
    p = bytearray(pds)
    p[6], p[7] = 255, 0x80                      # a GDS of its own, no bitmap
    p[26:28] = b'\0\0'                          # D = 0
    n = NX * NY
    bds = bytes([0, 0, 0, 0]) + b'\0\0' + b'\0\0\0\0' + bytes([16]) + \
        b''.join(i.to_bytes(2, 'big') for i in range(n))
    if len(bds) % 2:
        bds += b'\0'
    bds = len(bds).to_bytes(3, 'big') + bds[3:]
    m = b'GRIB' + b'\0\0\0' + bytes([1]) + bytes(p) + gds + bds + b'7777'
    return m[:4] + len(m).to_bytes(3, 'big') + m[7:]


if __name__ == '__main__':
    out = sys.argv[1]
    open(os.path.join(out, 'stag.grib2'), 'wb').write(b''.join(rotated(s) for s in SCANS))
    open(os.path.join(out, 'egrid.grib2'), 'wb').write(egrid())
    narr = open(os.path.join(out, 'g1_narr.grb'), 'rb').read()
    pds = narr[8:8 + int.from_bytes(narr[8:11], 'big')]
    last = unrotate((NY - 1) / 2 * DJ, (NX - 1) / 2 * DI)
    open(os.path.join(out, 'g1_egrid.grb'), 'wb').write(grib1(pds, 203) + grib1(pds, 205, last))
    print('stag.grib2 (%d grids), egrid.grib2, g1_egrid.grb (203, 205)' % len(SCANS))
