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

### Only the points needed

When `at()`, `regrid()` onto a few points or `read(box=...)` needs only a
small part of a field, fastgrib reads only what that part takes. The
values are always the same float32s a whole read gives.

How much it saves depends on the packing:

- **Simple and IEEE** (ECMWF's archives, ERA5, fields written with
  `packing='simple'`) store each value at a place its index gives, so
  those values are read where they lie and nothing else is decoded.
- **Complex packing** (NOAA's GFS, HRRR, NAM, RAP, NBM), **CCSDS** (ECMWF's
  open data) and **PNG** (MRMS) can only be decoded in order: with spatial
  differencing each value builds on the ones before it. These are decoded
  up to the last point needed and no further, with the fields side by side
  on every core. A point early in the file's order costs little, and one
  near its end costs about a whole read (fastgrib then decodes the field
  whole, so it is never slower). Where that is depends on the grid's
  scanning: HRRR's rows run south to north, GFS's north to south.
- **JPEG2000** codes its image in blocks of wavelet coefficients, 64 × 64
  at each level. fastgrib parses the packets, decodes only the blocks whose
  coefficients reach the points needed (16 of an HRRR field's 577 for one
  point), and runs the inverse wavelet over a window round each group of
  points, not the whole image. One point of an HRRR field takes 4 ms
  against 68 for the whole field on a core, and one point from 24 fields
  takes 64 ms against 228. Ten points scattered over the field take a
  third of a whole read; a hundred need nearly every block and cost what a
  whole read does, never more.

On GFS's 3072 × 1536 flux file packed simple (113 fields, 41 of them
bitmapped):

| | Points alone | Whole fields |
|---|---:|---:|
| `at()`, one point, every field | 25 ms | 300 ms |
| `at()`, 50 points | 29 ms | 281 ms |
| `read(box=...)`, 10° × 10° | 47 ms | 390 ms |

One point from every field of the complex-packed originals:

| | Decoded up to the point | Whole fields |
|---|---:|---:|
| HRRR (173 fields), southern Texas | 27 ms | 124 ms |
| HRRR, Kansas | 54 ms | 122 ms |
| HRRR, North Dakota | 102 ms | 113 ms |
| GFS flux (113 fields), 75°N | 43 ms | 342 ms |
| GFS flux, 5°N | 87 ms | 337 ms |
| GFS flux, 70°S | 190 ms | 337 ms |

From a remote file the gain is in bytes. fastgrib fetches each message's
first sections (8 KB, plus its bitmap if it has one), then only the byte
ranges holding the values wanted. Ranges close together are joined, and
the rest are fetched side by side. Three points from a simple-packed 0.25°
global field took 13 KB of its 1.2 MB message in 4 requests, and a 10° box
73 KB in 2. For a complex-packed field fastgrib fetches its group headers,
then its values up to the last group needed. One point from 30 of HRRR's
fields served over HTTP fetched 36% of their bytes for southern Texas, 65%
for Kansas, and all of them (no byte twice) for North Dakota. In a CCSDS
or PNG stream, where a value ends is only known by decoding up to it. So
fastgrib fetches a guess from the share of the values wanted, decodes, and
fetches half as much again until the stream holds the point (or 70% of the
message is in hand, when it fetches the rest). The first guess nearly
always holds. A point near the start of a 0.25° global field fetched 7% of
its message in three requests. A JPEG2000 field's packets each begin with
a header saying how long its body is, so fastgrib fetches its headers one
after another (a few hundred bytes each, the bodies between skipped),
then only the bodies of the code-blocks the points need. One point from
24 HRRR fields fetched 9% of their bytes, from fastgrib's own JPEG2000 and
from OpenJPEG's (ecCodes') alike. That is a few round trips a field, one
after another, and the fields side by side.

Winds along the grid's axes are turned to east and north only at the grid
points the interpolation takes. The values are the same as turning whole
fields, without re-reading each pair. One point from all 173 fields of an
HRRR file takes 0.12 s, against 0.36 s before 0.3.3.

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
