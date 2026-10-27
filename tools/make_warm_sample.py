"""Make fastgrib/_warm.grib2, the small file fastgrib.warm() decodes: one
field of each packing the decoders handle (the first found in tests/data),
a spectral field, and a GRIB1 message of second-order packing -- every
kernel compiled, in a few kilobytes.

    python tools/make_warm_sample.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastgrib                                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'tests', 'data')
OUT = os.path.join(ROOT, 'fastgrib', '_warm.grib2')
# packing templates wanted, and where to look first
WANT = (0, 3, 4, 40, 41, 42, 200)             # (5.2: the same kernels as 5.3)
FILES = ('gfs', 'nbm', 'gfs_simple', 'gfs_ieee', 'gfs_jpeg', 'gfs_png', 'gfs_aec', 'gfs_rle',
         'gfs_bm254', 'gefs', 'ecmwf', 'lcc', 'written', 'nam_bgrd')


def main():
    warnings.simplefilter('ignore')
    parts, got = [], {}
    for name in FILES:
        f = fastgrib.open(os.path.join(DATA, name + '.grib2'))
        for k, r in enumerate(f.records):
            key = (r.packing, r.bitmap >= 0)
            if r.packing in WANT and key not in got:
                got[key] = name
                parts.append(f.message(k))
    for name, ext in (('sh', '.grb2'), ('g1_2nd', '.grb')):    # spectral; GRIB1 second-order
        with open(os.path.join(DATA, name + ext), 'rb') as fh:
            data = fh.read()
        f = fastgrib.open(data)
        if name == 'sh':
            parts.append(f.message(0))
            got[('5.50/51', False)] = name
        else:                                   # GRIB1: the message itself, as it is
            n = int.from_bytes(data[4:7], 'big')
            parts.append(data[:n])
            got[('grib1 2nd-order', False)] = name
    missing = [p for p in WANT if not any(k[0] == p for k in got)]
    with open(OUT, 'wb') as fh:
        fh.write(b''.join(parts))
    for (packing, bm), name in sorted(got.items(), key=str):
        print('  %-18s bitmap %-5s from %s' % (packing, bm, name))
    print('%s: %d messages, %d bytes%s' % (OUT, len(parts), os.path.getsize(OUT),
                                          '; no sample of 5.%s' % missing if missing else ''))


if __name__ == '__main__':
    main()
