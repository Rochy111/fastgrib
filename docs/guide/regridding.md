# Regridding and winds

## Onto a lat-lon grid

```python
lats = np.arange(20, 55, 0.25)
lons = np.arange(230, 300, 0.25)
z = f.regrid((lats, lons), ':HGT:500 mb:')          # (len(lats), len(lons))
```

The source can be any grid fastgrib reads: Lambert, polar stereographic,
rotated, reduced Gaussian, even spherical harmonics. Longitudes may run
either side of 0.

## Onto another file's grid

```python
t = gfs.regrid(hrrr.grid(), name='TMP')             # GFS temperature on HRRR's grid
```

Pass a `Grid`, a `GribFile` or a `Record` and the result lines up point for
point with that file's fields.

## At points

```python
p = f.at(40.0, 255.0, 'PRMSL')                      # one point
p = f.at(station_lats, station_lons, 'PRMSL')       # any shapes that broadcast
```

## Methods

Bilinear by default, or `method='nearest'`. Gaps and edges are treated as
wgrib2's `-new_grid` (IPOLATES) treats them: a missing or off-grid corner is
left out of the weights, and the point is defined as long as half the
weight is. Where IPOLATES places points consistently, the results match
wgrib2's to its output precision.

Points are placed with each grid's own forward projection. Regridding a
field of each point's own latitude gives the target's latitudes to within
0.001°, where IPOLATES can be 0.4° out on a polar stereographic grid and
12° in longitude on a Mercator grid that crosses the date line.

The corners and weights are worked out once per pair of grids and reused
for every field. NBM's 3.7 million-point Lambert grid onto 580,000 points
takes 57 ms for the first field and 4 ms for each one after.

## Winds

Many regional grids (HRRR, NAM and RAP's Lambert grids, rotated grids) store
wind components along the grid's own axes rather than east and north.

```python
u, v = f.winds(level=850)                 # east and north, whatever the grid
u, v = f.winds(level=850, earth=False)    # as stored
```

`winds` pairs each `UGRD` with its `VGRD` and turns them to east and north,
matching wgrib2's `-new_grid_winds earth` to within its 0.1 m/s.

`regrid` and `at` do the same for wind components: they come out east and
north, turned at their own points, so a uniform wind stays uniform.
