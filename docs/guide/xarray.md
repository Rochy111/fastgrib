# xarray

```python
ds = f.to_xarray()                        # every field
ds = f.to_xarray(':(TMP|HGT):[0-9]+ mb:') # or the ones you pick
```

This needs `xarray` (install the `xarray` or `all` extra).

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

## Nothing is dropped

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
