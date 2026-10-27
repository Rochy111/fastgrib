"""Test data for writing: tests/data/written.grib2, made by fastgrib --
GFS fields given new values (with gaps) packed simple, complex, IEEE, PNG,
JPEG2000 (to 30 bits) and CCSDS,
to their own precision and to asked ones, on lat-lon, Lambert and
scanning-mode grids, and fields with their metadata set -- for wgrib2 to
read (written.inv, written.bin: make_test_data.sh) and the tests to
compare with, and to make again byte for byte.

    python tools/testdata/make_written.py TESTS_DATA_DIR
"""
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import fastgrib                                 # noqa: E402


def values(f, r, k):
    """r's values moved and given gaps: new values, deterministic."""
    v = f.read(r).astype(np.float64)
    j, i = np.indices(v.shape)
    v = v + 0.37 * np.sin(0.3 * i + 0.11 * k) * np.cos(0.2 * j) * (np.nanstd(v) + 1)
    v[(i + 2 * j + k) % 17 == 0] = np.nan
    return v


def messages(data):
    out = []
    gfs = fastgrib.open(os.path.join(data, 'gfs.grib2'))
    for k, r in enumerate(gfs.records[::12]):
        v = values(gfs, r, k)
        out.append(gfs.encode(v, r, packing='complex'))
        out.append(gfs.encode(v, r, packing='simple', decimals=1))
        out.append(gfs.encode(v, r, packing='ieee'))
        out.append(gfs.encode(v, r, packing='complex', bits=12))
    for name in ('lcc', 'gfs_scan'):
        f = fastgrib.open(os.path.join(data, name + '.grib2'))
        for k, r in enumerate(f.records[:4]):
            out.append(f.encode(values(f, r, k), r, packing='complex'))
    for k, r in enumerate(gfs.records[::30]):    # the image packings, gaps and no gaps
        v = values(gfs, r, k)
        for packing in ('png', 'jpeg', 'ccsds'):
            out.append(gfs.encode(v, r, packing=packing))
            out.append(gfs.encode(gfs.read(r), r, packing=packing, bits=10 + k))
    for bits in (24, 30):                       # JPEG2000 past 25 bits a value: OpenJPEG's limit
        out.append(gfs.encode(values(gfs, gfs.records[5], bits), gfs.records[5],
                              packing='jpeg', bits=bits))
    r = gfs.select(name='TMP')[0]
    out.append(gfs.message(r, name='DPT', level='700 mb', fcst='0-6 hour max fcst'))
    out.append(gfs.message(r, date=datetime(2030, 1, 2, 12), fcst='9 hour fcst'))
    out.append(gfs.message(r, level=(103, 2.0)))
    return out


def main(data):
    with open(os.path.join(data, 'written.grib2'), 'wb') as f:
        for m in messages(data):
            f.write(m)
    print('written.grib2:', os.path.getsize(os.path.join(data, 'written.grib2')), 'bytes')


if __name__ == '__main__':
    main(sys.argv[1])
