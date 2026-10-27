"""Test data for GRIB1 second-order packing: tests/data/g1_2nd.grb, with
ecCodes' reading of it (g1_2nd_ec.npz: each message's values, float32,
in the message's order, NaN for none) --

- ecCodes' own encodings, ECMWF's 'general extended' second-order packing:
  without spatial differencing and with 1st, 2nd and 3rd order,
  boustrophedonic rows, on a lat-lon and a reduced Gaussian grid, with and
  without a bitmap;
- the older forms ecCodes reads but does not write -- row by row, constant
  width with a secondary bitmap, general (widths and secondary bitmap) --
  packed here from the same fields' simple packing, for ecCodes to read.

    python3 tools/testdata/make_grib1_second_order.py OUT_DIR

Needs ecCodes' Python bindings (pip install eccodes eccodeslib).
"""
import os
import struct
import sys

import numpy as np
import eccodes

GENERAL = [('regular_ll_sfc_grib1', 'grid_second_order', False),
           ('regular_ll_sfc_grib1', 'grid_second_order_no_SPD', False),
           ('regular_ll_sfc_grib1', 'grid_second_order_SPD1', True),
           ('regular_ll_sfc_grib1', 'grid_second_order_SPD3', False),
           ('regular_ll_sfc_grib1', 'grid_second_order_boustrophedonic', False),
           ('regular_ll_sfc_grib1', 'grid_second_order', True),
           ('reduced_gg_pl_32_grib1', 'grid_second_order', False),
           ('reduced_gg_pl_32_grib1', 'grid_second_order_SPD1', True)]
OLDER = ['row_by_row', 'constant_width', 'general']


def field(lat, lon, rough):
    v = 280 + 25 * np.cos(np.radians(lat)) + 6 * np.sin(np.radians(2 * lon)) + 0.01 * lon
    return v + rough * np.sin(lat * 7.3 + lon * 3.1)          # some groups wider than others


def simple(sample, gaps, rough=0.0, bits=16):
    h = eccodes.codes_grib_new_from_samples(sample)
    lat = eccodes.codes_get_array(h, 'latitudes')
    lon = eccodes.codes_get_array(h, 'longitudes')
    v = field(lat, lon, rough)
    if gaps:
        eccodes.codes_set(h, 'bitmapPresent', 1)
        eccodes.codes_set(h, 'missingValue', 9999.0)
        v = np.where((lat > 20) & (lat < 40) & (lon > 30) & (lon < 90), 9999.0, v)
    eccodes.codes_set_values(h, v)
    eccodes.codes_set(h, 'bitsPerValue', bits)          # (after values: Gaussian samples)
    eccodes.codes_set_values(h, v)
    return h


def _bits(vals, width):
    out = []
    acc = n = 0
    for v in vals:
        acc, n = (acc << width) | int(v), n + width
        while n >= 8:
            n -= 8
            out.append((acc >> n) & 0xFF)
        acc &= (1 << n) - 1
    if n:
        out.append((acc << (8 - n)) & 0xFF)
    return bytes(out)


def older(kind):
    """A message of the simple-packed field, its BDS packed `kind` of
    second order (the WMO GRIB1 forms): groups of points (rows, or runs cut
    every 7th point), a reference each (first-order values), the rest in
    the group's width."""
    h = simple('regular_ll_sfc_grib1', False, rough=0.3)
    m = eccodes.codes_get_message(h)
    ni, nj = eccodes.codes_get(h, 'Ni'), eccodes.codes_get(h, 'Nj')
    eccodes.codes_release(h)
    p = 8
    p += int.from_bytes(m[p:p + 3], 'big')              # PDS
    p += int.from_bytes(m[p:p + 3], 'big')              # GDS
    b = p                                               # BDS
    nbits = m[b + 10]
    n = ni * nj
    acc = int.from_bytes(m[b + 11:b + 11 + (n * nbits + 7) // 8], 'big')
    total = 8 * ((n * nbits + 7) // 8)
    x = [(acc >> (total - (i + 1) * nbits)) & ((1 << nbits) - 1) for i in range(n)]
    if kind == 'row_by_row':
        starts = list(range(0, n, ni))
    else:
        starts = sorted(set(range(0, n, 7)) | set(range(0, n, 11)))
    bounds = starts + [n]
    refs, widths, second = [], [], []
    for a, z in zip(bounds, bounds[1:]):
        g = x[a:z]
        r = min(g)
        refs.append(r)
        widths.append(max(v - r for v in g).bit_length())
        second.append([v - r for v in g])
    if kind == 'constant_width':
        widths = [max(widths)] * len(widths)
    head = bytearray(m[b:b + 11])                       # length, flags, E, R, first-order width
    head[3] = (head[3] & 0x0F) | 0x40                   # complex (second-order) grid point
    head[10] = max(refs).bit_length() or 1
    flag = {'row_by_row': 0x10, 'constant_width': 0x20, 'general': 0x30}[kind]
    p1 = len(refs)
    wid = bytes([widths[0]]) if kind == 'constant_width' else bytes(widths)
    sec = b''
    if kind != 'row_by_row':
        marks = np.zeros(n, np.uint8)
        marks[starts] = 1
        sec = np.packbits(marks).tobytes()
    first = _bits(refs, head[10])
    secv = b''.join([])                                 # all groups' values in one bit run
    allv, allw = [], []
    for s, w in zip(second, widths):
        allv += s
        allw += [w] * len(s)
    acc = nb = 0
    out = []
    for v, w in zip(allv, allw):
        acc, nb = (acc << w) | v, nb + w
        while nb >= 8:
            nb -= 8
            out.append((acc >> nb) & 0xFF)
        acc &= (1 << nb) - 1
    if nb:
        out.append((acc << (8 - nb)) & 0xFF)
    secv = bytes(out)
    n1 = 21 + len(wid) + len(sec) + 1                   # octet numbers, from 1
    n2 = n1 + len(first)
    body = bytes(head[:11]) + struct.pack('>HB', n1, flag) + struct.pack('>HHHB', n2, p1, n, 0) + \
        wid + sec + first + secv
    if len(body) % 2:
        body += b'\0'
    body = struct.pack('>I', len(body))[1:] + body[3:]
    msg = m[:b] + body + b'7777'
    total = len(msg)
    return msg[:4] + struct.pack('>I', total)[1:] + msg[7:]


def main(out):
    msgs = []
    for sample, packing, gaps in GENERAL:
        h = simple(sample, gaps, rough=0.4)
        eccodes.codes_set(h, 'packingType', packing)
        msgs.append(eccodes.codes_get_message(h))
        eccodes.codes_release(h)
    for kind in OLDER:
        msgs.append(older(kind))
    path = os.path.join(out, 'g1_2nd.grb')
    with open(path, 'wb') as f:
        f.write(b''.join(msgs))
    ref = {}
    with open(path, 'rb') as f:
        k = 0
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            eccodes.codes_set(h, 'missingValue', 9.999e20)
            v = eccodes.codes_get_values(h)
            ref['v%d' % k] = np.where(v == 9.999e20, np.nan, v).astype(np.float32)
            ref['packing%d' % k] = np.array(eccodes.codes_get(h, 'packingType'))
            eccodes.codes_release(h)
            k += 1
    np.savez_compressed(os.path.join(out, 'g1_2nd_ec.npz'), **ref)
    print(k, 'messages:', [str(ref['packing%d' % i]) for i in range(k)])


if __name__ == '__main__':
    main(sys.argv[1])
