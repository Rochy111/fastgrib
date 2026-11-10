"""fastgrib as an xarray engine: xr.open_dataset('gfs.grib2',
engine='fastgrib') and open_mfdataset(...), as cfgrib's -- every variable
read when indexed, only the fields indexed (a remote file's: those fetched
then read). xarray finds it by the entry point fastgrib's install
registers; engine=FastgribBackend works without that."""
import os

from xarray.backends import BackendArray, BackendEntrypoint

from ._xarray import _Fields


class Fields(_Fields, BackendArray):
    """A variable's fields, read when indexed (_xarray._Fields), as xarray's
    BackendArray -- picklable, for dask: the file reopened where unpickled."""

# cfgrib's filter_by_keys, as ecCodes names the keys -> fastgrib's Record attributes
_KEYS = {'shortName': 'short_name', 'typeOfLevel': 'level_kind', 'level': 'level_value',
         'name': 'long_name', 'units': 'units', 'number': 'member'}
_EXTENSIONS = ('.grib2', '.grb2', '.grib', '.grb', '.grib1', '.grb1', '.gb2', '.gr2')


def _picks(f, filter_by_keys, match):
    """The records picked: by filter_by_keys (ecCodes' names, or fastgrib's
    attributes) and a pattern of the inventory lines."""
    crit = {_KEYS.get(k, k): v for k, v in (filter_by_keys or {}).items()}
    recs = f.select(**crit) if crit else list(f.records)
    if match is not None:
        hit = {id(r) for r in f.find(match)}
        recs = [r for r in recs if id(r) in hit]
    if not recs:
        raise KeyError('no field of %s matches %s' % (f.name, dict(filter_by_keys or {},
                                                                    **({'match': match}
                                                                       if match else {}))))
    return recs


class FastgribBackend(BackendEntrypoint):
    """xarray's engine 'fastgrib': GRIB2 and GRIB1, local or remote."""
    description = 'GRIB2 and GRIB1 files and URLs through fastgrib (values as wgrib2\'s)'
    url = 'https://github.com/Rochy111/fastgrib'
    open_dataset_parameters = ('filename_or_obj', 'drop_variables', 'filter_by_keys', 'match',
                               'earth_winds', 'max_records', 'byte_range', 'index', 'cache')

    def open_dataset(self, filename_or_obj, *, drop_variables=None, filter_by_keys=None,
                     match=None, earth_winds=False, max_records=None, byte_range=None,
                     index=None, cache=None):
        from . import open as gopen
        from ._xarray import dataset
        src = os.fspath(filename_or_obj) if isinstance(filename_or_obj, os.PathLike) else \
            filename_or_obj
        f = gopen(src, max_records=max_records, byte_range=byte_range, index=index, cache=cache)
        try:
            ds = dataset(f, _picks(f, filter_by_keys, match), earth_winds, lazy=True)
        except Exception:
            f.close()
            raise
        if drop_variables:
            ds = ds.drop_vars([v for v in drop_variables if v in ds.variables])
        ds.set_close(f.close)
        return ds

    def guess_can_open(self, filename_or_obj):
        try:
            name = os.fspath(filename_or_obj)
        except TypeError:
            return False
        return isinstance(name, str) and name.lower().endswith(_EXTENSIONS)
