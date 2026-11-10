# xarray

```python
ds = f.to_xarray()                        # every field
ds = f.to_xarray(':(TMP|HGT):[0-9]+ mb:') # or the ones you pick
```

This needs `xarray` (install the `xarray` or `all` extra).

## As an xarray engine

Installing fastgrib registers it with xarray, so it can stand in for cfgrib:

```python
import xarray as xr

ds = xr.open_dataset('gfs.t00z.pgrb2.0p25.f006', engine='fastgrib')
t = xr.open_dataset(path, engine='fastgrib',
                    filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName': 't'})
ds = xr.open_mfdataset(paths, engine='fastgrib', combine='nested', concat_dim='step')
```

Opening reads only the inventory. A field is decoded when its values are
asked for, and only the fields that are indexed get decoded: `ds['TMP'][3]`
decodes one field. `filter_by_keys` takes cfgrib's ecCodes names
(`shortName`, `typeOfLevel`, `level`, `number`) or fastgrib's record
attributes. `match=` picks by a pattern in the inventory line, and
`max_records`, `byte_range` and `index` work as they do for `fastgrib.open`.
The Dataset is the same as `to_xarray()` makes. `f.to_xarray(lazy=True)`
gives the lazy one from an open file.

A lazy Dataset pickles small: its file is opened again wherever it is
unpickled, not copied. That means `chunks={}` (dask), `open_mfdataset(...,
parallel=True)` and dask's distributed workers all work. A `GribFile`
pickles the same way, by its path or URL, so it can be passed to a
process pool.

## Layout

You get one variable per parameter, level kind and kind of forecast. Each
is laid out along whichever of these vary within it:

- `time`, the reference time;
- `member`, the ensemble member;
- `step`, the forecast step;
- a level dimension named after the level kind (`isobaricInhPa`,
  `heightAboveGround` and so on), with layers described by both their
  bounds.

Coordinates and attributes follow CF conventions: 1-D `latitude` and
`longitude` on lat-lon grids; 2-D ones, projection x/y and a CF `crs` grid
mapping on projected grids; units, long names and times.

## Fields outside the common layout

Fields on a different grid from the rest get their own spatial dimensions.
Fields that no coordinate tells apart, such as the same field stored twice,
go along a `<name>_field` dimension instead of overwriting each other.

## Winds

`to_xarray(earth_winds=True)` turns grid-relative winds (HRRR, NAM and RAP's
Lambert grids, rotated grids) to east and north before building the
Dataset. See [Regridding and winds](regridding.md#winds).

## NetCDF from the command line

```bash
fastgrib file.grib2 -match ':HGT:[0-9]+ mb:' -o heights.nc
```
