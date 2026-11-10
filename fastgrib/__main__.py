"""fastgrib on the command line, wgrib2's way.

    fastgrib FILE|URL                        the inventory (wgrib2 -s)
    fastgrib FILE -match ':TMP:' -v          matching fields, with units and times
    fastgrib FILE -select name=TMP level=500 -stats
    fastgrib FILE -match ':HGT:500 mb:' -o z500.nc      NetCDF (xarray)
    fastgrib FILE -bin out.bin               float32 we:sn, as wgrib2 -bin -no_header -order we:sn
    fastgrib FILE -csv out.csv               lon,lat,value a point
    fastgrib FILE -grid                      the grids the fields lie on
    fastgrib FILE -match ':TMP:' -grib t.grib2           the fields picked, as GRIB2 (wgrib2 -grib)
    fastgrib FILE -small_grib -100:-80 30:45 box.grib2   cut to a box (wgrib2 -small_grib)
    fastgrib FILE -grib out.grib2 -packing ccsds         packed again, on every core
    fastgrib URL -max_records 20 > URL.idx   the first 20 fields of a remote file without an
                                             index, fetched by themselves; their index saved
    fastgrib URL -index my.idx -match ':HGT:'            a remote file through an index given
    fastgrib -warm                           compile every decoder now (installing already does)
"""
import argparse
import sys
import time

import numpy as np

from . import open as gopen, __version__
from ._cpu import UNDEF


def _value(v):
    for kind in (int, float):
        try:
            return kind(v)
        except ValueError:
            pass
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(prog='fastgrib', description=__doc__.split('\n\n')[0],
                                 epilog=__doc__.split('\n\n', 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', nargs='?', help='a GRIB2 file, or an http(s):// or s3:// URL')
    ap.add_argument('-match', action='append', metavar='RE',
                    help='fields whose inventory line matches (all given must)')
    ap.add_argument('-not', dest='not_', action='append', metavar='RE',
                    help='leave out fields whose line matches')
    ap.add_argument('-select', nargs='+', metavar='KEY=VALUE',
                    help='pygrib-style picks: name=TMP level=500 level_type=heightAboveGround')
    ap.add_argument('-v', action='store_true', help='verbose: units, long name, valid time, grid')
    ap.add_argument('-stats', action='store_true', help='min, max, mean and undefined points')
    ap.add_argument('-grid', action='store_true', help='describe the grids')
    ap.add_argument('-o', metavar='OUT.nc', help='write the fields picked to NetCDF')
    ap.add_argument('-bin', metavar='OUT', help='write float32 we:sn, undefined 9.999e20')
    ap.add_argument('-csv', metavar='OUT', help='write lon,lat,value lines (with -match: a field)')
    ap.add_argument('-grib', metavar='OUT', help='write the fields picked as GRIB2, each message '
                    'standing alone (a remote file: only those fetched)')
    ap.add_argument('-packing', metavar='TYPE',
                    choices=('simple', 'complex', 'ccsds', 'jpeg', 'png', 'ieee'),
                    help='with -grib or -small_grib: pack the fields so (simple, complex, '
                    'ccsds, jpeg, png, ieee; on every core: wgrib2 -set_grib_type)')
    ap.add_argument('-small_grib', action='store_true',
                    help='-small_grib W:E S:N OUT: write the fields picked cut to a box of '
                    'longitudes W to E and latitudes S to N, as GRIB2 (wgrib2 -small_grib)')
    ap.add_argument('-max_records', type=int, metavar='N', help='only the first N fields')
    ap.add_argument('-byte_range', metavar='START:STOP',
                    help='only the messages starting in these bytes (STOP empty: the end)')
    ap.add_argument('-index', metavar='IDX', help="a remote file's index, a path or URL")
    ap.add_argument('-gpu', action='store_true', help='decode on the GPU')
    ap.add_argument('-time', action='store_true', help='report how long it took')
    ap.add_argument('-warm', action='store_true',
                    help='compile every decoder now and wait for it (installing already does, '
                         'and fastgrib does it again by itself when they stop matching)')
    ap.add_argument('-version', action='version', version='fastgrib ' + __version__)
    argv = list(sys.argv[1:] if argv is None else argv)
    small = None
    if '-small_grib' in argv:                   # its values taken first: '-100:-80' is no option
        k = argv.index('-small_grib')
        small = argv[k + 1:k + 4]
        if len(small) < 3:
            ap.error('-small_grib W:E S:N OUT')
        del argv[k:k + 4]
    a = ap.parse_args(argv)
    a.small_grib = small
    if a.warm:
        from . import warm
        print('fastgrib: decoders compiled in %.1f s' % warm())
        if a.src is None:
            return 0
    elif a.src is None:
        ap.error('a file or URL to read (or -warm)')

    t0 = time.perf_counter()
    rng = None
    if a.byte_range:
        lo, _, hi = a.byte_range.partition(':')
        rng = (int(lo or 0), int(hi) if hi else None)
    f = gopen(a.src, max_records=a.max_records, byte_range=rng, index=a.index)
    recs = list(f.records)
    if a.select:
        crit = dict(kv.split('=', 1) for kv in a.select)
        recs = f.select(**{k: _value(v) for k, v in crit.items()})
    import re
    for rx in a.match or ():
        recs = [r for r in recs if re.search(rx, repr(r))]
    for rx in a.not_ or ():
        recs = [r for r in recs if not re.search(rx, repr(r))]
    out = sys.stdout
    if a.grid:
        seen = {}
        for r in recs:
            g = f.grid(r)
            seen.setdefault(g, []).append(r.index)
        for g, idx in seen.items():
            out.write('%r  %s  earth %s  fields %d-%d (%d)\n'
                      % (g, ', '.join('%g' % x if isinstance(x, float) else str(x) for x in g.params),
                         g.earth, idx[0], idx[-1], len(idx)))
    want_values = a.stats or a.bin or a.csv or a.o or a.grib or a.small_grib
    if a.grib:
        f.write(a.grib, recs, packing=a.packing)
    if a.small_grib:
        (w, e), (s, n) = (tuple(float(x) for x in v.split(':')) for v in a.small_grib[:2])
        f.write(a.small_grib[2], recs, box=(w, e, s, n), packing=a.packing)
    if not want_values and not a.grid:
        for r in recs:
            line = repr(r)
            if a.v:
                line += ('' if line.endswith(':') else ':') + 'units=%s:%s:vt=%s:' % (
                    r.units, r.long_name,
                    r.valid_time.strftime('%Y%m%d%H%M') if r.valid_time else '?')
            out.write(line + '\n')
    if a.o:
        f.to_xarray(recs).to_netcdf(a.o)
    if a.stats or a.bin or a.csv:
        fb = open(a.bin, 'wb') if a.bin else None
        fc = open(a.csv, 'w') if a.csv else None
        for r, v in f.fields(recs, gpu=a.gpu):
            v = v.get() if hasattr(v, 'get') else v
            if a.stats:
                ok = ~np.isnan(v)
                out.write('%r%s\n' % (r, 'ndata=%d:undef=%d:min=%g:max=%g:mean=%g:' % (
                    v.size, v.size - int(ok.sum()), np.nanmin(v), np.nanmax(v), np.nanmean(v))
                    if ok.any() else 'ndata=%d:undef=%d:' % (v.size, v.size)))
            if fb:
                np.where(np.isnan(v), UNDEF, v).astype('<f4').tofile(fb)
            if fc:
                lat, lon = f.latlon(r)
                for la, lo, x in zip(lat.ravel(), lon.ravel(), v.ravel()):
                    fc.write('%s,%s,%s\n' % (lo, la, 'nan' if x != x else repr(float(x))))
        for h in (fb, fc):
            if h:
                h.close()
    if a.time:
        sys.stderr.write('%d fields in %.3f s\n' % (len(recs), time.perf_counter() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
