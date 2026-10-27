"""fastgrib: GRIB2 (and GRIB1) files, fast and simply -- read, regridded,
written back.

    import fastgrib

    f = fastgrib.open('gfs.t00z.pgrb2.0p25.f006')     # a path, URL or bytes
    print(f)                                          # every field, as wgrib2 -s
    t2m = f.read('TMP:2 m above ground')              # (ny, nx) float32, NaN = no value
    t500 = f.select(name='TMP', level_value=500)      # pygrib-style picks
    lat, lon = f.latlon()
    for rec, field in f.fields():                     # a whole file, memory flat
        ...
    u, v = f.winds(level=850)                         # east and north components
    z = f.regrid((lats, lons), ':HGT:500 mb:')        # onto lat-lon axes, a Grid; at() points
    f.write('subset.grib2', name='TMP')               # GRIB2 messages; encode(), message()
    f.keys(0)                                         # a field's keys, as ecCodes names them

Values are wgrib2's to the bit. CPU (numba) or GPU (cupy) is picked for
you. See README.md for everything else.
"""
from ._cpu import UNDEF
from ._file import GribFile, open, release, set_device, set_threads, warm    # noqa: A004
from ._grids import Grid
from ._meta import Record

__version__ = '0.1.0'
__all__ = ['open', 'set_device', 'set_threads', 'release', 'warm', 'GribFile', 'Record',
           'Grid', 'UNDEF']
