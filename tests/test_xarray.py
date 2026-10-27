"""xarray Datasets: every field kept, laid out by what tells them apart."""
import numpy as np
import pytest

import fastgrib

from helpers import path


def test_xarray():
    xr = pytest.importorskip('xarray')
    f = fastgrib.open(path('gfs'))
    ds = f.to_xarray(':TMP:[0-9.]+ mb:')
    t = ds['TMP']
    assert t.dims[0] == 'isobaricInhPa' and t.attrs['units'] == 'K'
    lev = float(t['isobaricInhPa'][-1])
    want = f.read(name='TMP', level=lev)
    assert np.array_equal(t.sel(isobaricInhPa=lev).values, want, equal_nan=True)
    assert isinstance(ds, xr.Dataset)
    g = fastgrib.open(path('lcc')).to_xarray()
    assert g['crs'].attrs['grid_mapping_name'] == 'lambert_conformal_conic'


def _every_field(f, ds):
    """Each of f's fields is in ds, exactly once, at its place."""
    seen = 0
    for r in f.records:
        hits = [v for v in ds.data_vars.values() if v.attrs['GRIB_name'] == r.name]
        want = f.read(r)
        for v in hits:
            for ix in np.ndindex(v.shape[:-2]):
                if np.array_equal(v.values[ix], want, equal_nan=True):
                    seen += 1
                    break
            else:
                continue
            break
    assert seen == len(f.records)
    assert sum(int(np.prod(v.shape[:-2])) for v in ds.data_vars.values()) >= len(f.records)


@pytest.mark.parametrize('name', ['gfs', 'nbm', 'gefs', 'gfs_uv', 'gfs_bm254', 'ecmwf_local'])
def test_xarray_keeps_every_field(name):
    pytest.importorskip('xarray')
    f = fastgrib.open(path(name))
    _every_field(f, f.to_xarray())


def test_xarray_layers():
    """Soil layers along one axis by their upper sides, the sides kept."""
    pytest.importorskip('xarray')
    f = fastgrib.open(path('ecmwf_local'))
    sot = f.to_xarray(short_name='sot')
    v = next(iter(sot.data_vars.values()))
    assert v.dims[0] == 'soilLayer' and list(sot['soilLayer'].values) == [0, 1, 2, 3]
    want = f.read(short_name='sot', level_value=2)
    assert np.array_equal(v.sel(soilLayer=2).values, want, equal_nan=True)


def test_xarray_members_and_grids(tmp_path):
    pytest.importorskip('xarray')
    f = fastgrib.open(path('gefs'))
    ds = f.to_xarray()
    v = ds['VGRD_isobaricInhPa']
    assert v.dims[:2] == ('member', 'isobaricInhPa') and list(v['member'].values) == [0, 1]
    want = f.read(name='VGRD', level=850, member=1)
    assert np.array_equal(v.sel(member=1, isobaricInhPa=850).values, want, equal_nan=True)
    # fields on several grids: each grid its own dimensions, nothing left out
    mixed = tmp_path / 'mixed.grib2'
    mixed.write_bytes(b''.join(open(path(n), 'rb').read() for n in ('gfs_simple', 'lcc', 'nps')))
    m = fastgrib.open(str(mixed))
    ds = m.to_xarray()
    _every_field(m, ds)
    assert {'latitude', 'y_1', 'y_2', 'crs_1', 'crs_2'} <= set(ds.coords) | set(ds.dims)


def test_xarray_duplicates_kept(tmp_path):
    """The same fields twice (a file written twice over): every copy kept,
    along a '_field' dimension, where cfgrib would refuse or drop them."""
    pytest.importorskip('xarray')
    twice = tmp_path / 'twice.grib2'
    twice.write_bytes(open(path('gefs'), 'rb').read() * 2)
    f = fastgrib.open(str(twice))
    ds = f.to_xarray()
    _every_field(f, ds)
    assert any(d.endswith('_field') for d in ds.dims)


def test_winds_in_xarray():
    pytest.importorskip('xarray')
    g, e = fastgrib.open(path('lcc_uvgrid')), fastgrib.open(path('lcc_uvearth'))
    ds = g.to_xarray(earth_winds=True)
    ref = e.to_xarray()
    for k in ref.data_vars:
        assert np.nanmax(np.abs(ds[k].values - ref[k].values)) < 0.12, k
        assert ds[k].attrs['GRIB_uv_relative_to_grid'] == 0
    assert g.to_xarray()['UGRD_isobaricInhPa'].attrs['GRIB_uv_relative_to_grid'] == 1
    with pytest.raises(KeyError):
        fastgrib.open(path('gfs')).winds(':TMP:')


def test_xarray_remote(server):
    pytest.importorskip('xarray')
    f = fastgrib.open(server + 'gefs.grib2')
    loc = fastgrib.open(path('gefs'))
    a, b = f.to_xarray(), loc.to_xarray()
    assert set(a.data_vars) == set(b.data_vars)
    for k in b.data_vars:
        assert np.array_equal(a[k].values, b[k].values, equal_nan=True), k


def test_xarray_remote_same_field_twice(server):
    pytest.importorskip('xarray')
    picks = ([':TMP:', ':2 m above ground:'])
    a = fastgrib.open(server + 'gfs.grib2').to_xarray(picks)
    b = fastgrib.open(path('gfs')).to_xarray(picks)
    assert set(a.data_vars) == set(b.data_vars)
    for k in b.data_vars:
        assert np.array_equal(a[k].values, b[k].values, equal_nan=True), k
