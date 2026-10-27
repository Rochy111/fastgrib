"""Test data for GRIB1's matrix of values at each grid point:
tests/data/g1_matrix.grb, hand-made to ECMWF's layout (ecCodes reads
them; it writes none) -- a 2 x 3 matrix (directions given, frequencies
geometric) at each point of a 6 x 4 grid with secondary bitmaps, the
same with a primary bitmap too -- and ecCodes' own 'grid_simple_matrix'
of one value a point; with each message's truth (g1_matrix_ec.npz:
present points, secondary bitmaps, the values) and what ecCodes reads
(its 'values': the first's coded values as they are -- it expands none --
and the last's).

    python3 tools/testdata/make_grib1_matrix.py OUT_DIR

Needs ecCodes' Python bindings (pip install eccodes eccodeslib).
"""
import math
import os
import sys

import numpy as np
import eccodes

NI, NJ, NR, NC = 6, 4, 2, 3


def ibm(v):
    """An IBM single-precision float's 4 octets."""
    if v == 0:
        return b'\0\0\0\0'
    s = 0x80 if v < 0 else 0
    v = abs(v)
    e = math.ceil(math.log(v, 16)) if v > 0 else 0
    m = round(v / 16.0 ** (e - 6))
    if m >= 1 << 24:
        m >>= 4
        e += 1
    return bytes([s | (e + 64)]) + m.to_bytes(3, 'big')


def lat3(v):
    v = round(v * 1000)
    return (abs(v) | (0x800000 if v < 0 else 0)).to_bytes(3, 'big')


def message(points_missing, seed):
    rng = np.random.default_rng(seed)
    npts = NI * NJ
    present = np.ones(npts, bool)
    present[list(points_missing)] = False
    p = int(present.sum())
    sec = rng.random((p, NR * NC)) > 0.25          # each point's elements present
    x = rng.integers(0, 4096, int(sec.sum()))     # their 12-bit values, point by point
    pds = bytes([0, 0, 28, 2, 7, 96, 255, 0x80 | (0x40 if points_missing else 0), 140, 1, 0,
                 0, 26, 10, 25, 0, 0, 1, 0, 0, 0, 0, 0, 0, 21, 0, 0, 0])
    gds = bytes([0, 0, 32, 0, 255, 0]) + NI.to_bytes(2, 'big') + NJ.to_bytes(2, 'big') + \
        lat3(3.0) + lat3(0.0) + bytes([0x80]) + lat3(0.0) + lat3(5.0) + \
        (1000).to_bytes(2, 'big') + (1000).to_bytes(2, 'big') + bytes([0]) + b'\0' * 4
    bms = b''
    if points_missing:
        bits = np.packbits(present).tobytes()
        bms = (6 + len(bits)).to_bytes(3, 'big') + bytes([len(bits) * 8 - npts]) + b'\0\0' + bits
    coefs = ibm(0.0) + ibm(180.0) + ibm(0.0345) + ibm(1.1)
    secb = np.packbits(sec.ravel()).tobytes()
    data = np.zeros((x.size * 12 + 7) // 8, np.uint8)
    bits = np.unpackbits(x.astype('>u2').view(np.uint8)).reshape(-1, 16)[:, 4:].ravel()
    data[:] = np.packbits(bits)[:data.size]
    body = bytes([0, 0, 0, 0x10]) + b'\0\0' + ibm(0.0) + bytes([12]) + p.to_bytes(2, 'big') + \
        bytes([0x60]) + NR.to_bytes(2, 'big') + NC.to_bytes(2, 'big') + \
        bytes([0, 2, 11, 2, 1, 2]) + coefs + secb + data.tobytes()
    if len(body) % 2:
        body += b'\0'
    bds = len(body).to_bytes(3, 'big') + body[3:]
    m = b'GRIB' + b'\0\0\0' + bytes([1]) + pds + gds + bms + bds + b'7777'
    return m[:4] + len(m).to_bytes(3, 'big') + m[7:], present, sec, x


def single():
    """ecCodes' grid_simple_matrix of one value a point (its matrix description
    there all the same)."""
    h = eccodes.codes_grib_new_from_samples('GRIB1')
    for k, v in (('Ni', NI), ('Nj', NJ), ('iDirectionIncrementInDegrees', 1.0),
                 ('jDirectionIncrementInDegrees', 1.0), ('latitudeOfFirstGridPointInDegrees', 3.0),
                 ('longitudeOfFirstGridPointInDegrees', 0.0),
                 ('latitudeOfLastGridPointInDegrees', 0.0),
                 ('longitudeOfLastGridPointInDegrees', 5.0)):
        eccodes.codes_set(h, k, v)
    eccodes.codes_set_values(h, np.arange(NI * NJ, dtype=float) * 1.5)
    eccodes.codes_set(h, 'packingType', 'grid_simple_matrix')
    eccodes.codes_set(h, 'NR', NR)
    eccodes.codes_set(h, 'NC', NC)
    eccodes.codes_set_values(h, np.arange(NI * NJ, dtype=float) * 1.5)
    m = eccodes.codes_get_message(h)
    eccodes.codes_release(h)
    return m


if __name__ == '__main__':
    out = sys.argv[1]
    made = [message((), 1), message((2, 9, 14, 23), 2)]
    msgs = [m[0] for m in made] + [single()]
    path = os.path.join(out, 'g1_matrix.grb')
    open(path, 'wb').write(b''.join(msgs))
    ref = {}
    for k, (_, present, sec, x) in enumerate(made):
        ref['present%d' % k], ref['sec%d' % k], ref['x%d' % k] = present, sec, x
    with open(path, 'rb') as f:
        for k in range(len(msgs)):
            h = eccodes.codes_grib_new_from_file(f)
            eccodes.codes_set(h, 'missingValue', 9999.0)
            ref['v%d' % k] = eccodes.codes_get_values(h)
            eccodes.codes_release(h)
    np.savez_compressed(os.path.join(out, 'g1_matrix_ec.npz'), **ref)
    print({k: v.shape for k, v in ref.items()})
