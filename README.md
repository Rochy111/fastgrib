# fastgrib

[![tests](https://github.com/Rochy111/fastgrib/actions/workflows/tests.yml/badge.svg)](https://github.com/Rochy111/fastgrib/actions/workflows/tests.yml)

GRIB2 (and GRIB1) for Python, fast: open a file (or a URL), pick fields,
get numpy arrays -- and write them back, regrid them, read any key. Values
are **bit-identical to wgrib2**; decoding is several times faster on the
CPU (numba) and faster still on an NVIDIA GPU (cupy).

```python
import fastgrib

f = fastgrib.open('gfs.t00z.pgrb2.0p25.f006')     # a path, an http(s):// or s3:// URL, or bytes
print(f)                                          # every field, as wgrib2 -s lists them
t2m = f.read('TMP:2 m above ground')              # (ny, nx) float32, NaN where there is no value
t500 = f.read(name='TMP', level=500)              # pygrib-style keywords
tp = f.read(short_name='tp')                      # or ecCodes' (pygrib's, cfgrib's) names
lat, lon = f.latlon()                             # same shape, rows south to north
for rec, field in f.fields(':UGRD:'):             # many fields, memory stays flat
    print(rec.name, rec.level, rec.valid_time, field.mean())
ds = f.to_xarray()                                # an xarray Dataset of every field
u, v = f.winds(level=850)                         # east and north, whatever the grid's axes
z = f.regrid((lats, lons), ':HGT:500 mb:')        # onto a lat-lon grid, another file's, points
f.write('subset.grib2', name='TMP')               # fields picked, as their own GRIB2 messages
f.keys(0)['bitsPerValue']                         # any key, as ecCodes names it
```

## Install

```
pip install git+https://github.com/Rochy111/fastgrib              # numpy and numba only
pip install "fastgrib[all] @ git+https://github.com/Rochy111/fastgrib"   # + GPU (cupy), xarray/NetCDF, pyproj
```

From a clone, `pip install -e .` (or `-e ".[all]"`). Python 3.9+, numpy 1.x
or 2.x; tested on Linux, Windows and macOS. The GPU path needs an NVIDIA GPU
with CUDA 12 and `cupy` (no CI runner has one: it is tested by hand, on an
RTX 5070 Ti); without one everything runs on the CPU.

numba compiles fastgrib's decoders for your machine the first time each is
used and keeps them: after installing (or upgrading), the first field of
each kind of packing takes several seconds more. `fastgrib -warm` (or
`fastgrib.warm()`) compiles them all at once instead (~20-30 s) -- worth
running in an install script, a container image's build or a job's set-up,
after which every first read is as quick as any other.

## Reading

**Picking fields.** By a regular expression searched in each field's
inventory line, exactly as `wgrib2 -match` does
(`'12:0:d=2026012000:TMP:2 m above ground:6 hour fcst:'`); by keywords as
pygrib's `select` (`name`, `short_name`, `level`, `level_value`,
`level_kind`, `level_type`, `fcst`, `fcst_hours`, `valid_time`, `member`,
`ensemble`, ...: a value, a list of values, or a function); by `Record` or
position; or a list of those. One field reads as `(ny, nx)`, several as
`(n, ny, nx)`.

```python
f.find(':HGT:')                                   # Records whose line matches
f.select(name=['UGRD', 'VGRD'], level_type='isobaricInhPa', level=lambda v: v >= 200)
f.read(f.records[10])
```

**Records** carry `name`, `long_name`, `units` (NCEP's parameter tables,
then ecCodes' for what NCEP's lack -- ECMWF's local `tp`, `lcc`, `ro`,
...), `short_name` (ecCodes' shortName: `2t`, `tp`, `mx2t3`, ...),
`level` (wgrib2's wording), `level_type`/`level_value`/`level_kind`
(ecCodes' typeOfLevel: `'isobaricInhPa'`, `'heightAboveGroundLayer'`,
`'soilLayer'`, ...), `ref_time`/`valid_time` (datetimes), `fcst_hours`,
`member`/`ensemble` (an ensemble member's number -- 0 the control -- or
the statistic: `'ens mean'`, `'ens std dev'`), `grid` and more;
`rec.as_dict()` for all of it.

**Grids.** `f.grid()` gives `.shape`, `.kind`, `.params`, `.earth`;
`f.latlon()` every point's latitude and longitude (longitudes in
[0, 360), as wgrib2 prints them). Every grid comes out west to east and
south to north (wgrib2's `-order we:sn`), however the file scans it.

**Remote files.** `fastgrib.open('https://.../file.grib2')` (or `s3://`)
reads the index published beside it -- NOAA's `.idx`, ECMWF's `.index` --
and fetches only the fields you read, by HTTP byte ranges, in parallel;
`print(f)`, `find` and `select` work before any download, with the same
names, levels and level kinds the fields have once fetched. Without an
index the file is downloaded whole. Connections are kept open between
requests (a new one costs ~0.1 s of set-up), except through a proxy.

```python
f = fastgrib.open('https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20261023/00/atmos/gfs.t00z.pgrb2.1p00.f006')
z500 = f.read(':HGT:500 mb:')                     # ~0.05 s: one field fetched (connections kept)
everything = f.read()                             # 743 fields, 45 MB: ~1.2 s
e = fastgrib.open('https://data.ecmwf.int/forecasts/20261023/00z/ifs/0p25/oper/20261023000000-6h-oper-fc.grib2')
t2 = e.read(short_name='2t')                      # ECMWF open data (CCSDS-packed)
```

**xarray.** `f.to_xarray(pick)` gives a Dataset: a variable per parameter,
level kind and kind of forecast, laid out along whichever of reference
time (`time`), ensemble member (`member`), forecast step (`step`) and
level (a dimension named for the kind; layers by their sides) vary within
it; CF coordinates (1-D latitude/longitude on lat-lon grids; 2-D ones
plus projection x/y and a CF `crs` grid mapping on projected grids),
units, long names and times. No field is ever dropped: fields on other
grids get their own spatial dimensions, and fields no coordinate tells
apart (the same field twice) go along a `<name>_field` dimension. Needs
`xarray`.

**GPU.** `read(..., gpu=True)` returns cupy arrays that stay on the card
-- the fastest path by far when the next step runs on the GPU too. For
numpy results the choice is automatic and goes by measurement: a fresh
result is the CPU's (the GPU would page-lock new memory: 2-3x slower),
big fields read into an array you reuse (`read(..., out=array)`) the
GPU's (GFS 0.25°: 0.14 ms a field against 0.21), `fields()` the CPU's
unless the GPU is already up and the stream long (all of NBM's 769
fields: 0.37 ms a field against 0.55). Force a device with
`fastgrib.set_device('cpu' | 'gpu' | 'auto')` or the environment
variable `FASTGRIB_DEVICE` -- `'gpu'` keeps the CPU nearly idle while
decoding (NBM: a twelfth of the CPU time).

**Threads.** The CPU decodes on a thread per logical CPU;
`fastgrib.set_threads(n)` or the environment variable `FASTGRIB_THREADS`
uses n (`set_threads(None)`: back to all). Fewer leave the machine to
other work, and where every core reading is limited by memory's
bandwidth -- big simply or complex packed fields through `fields()` --
they cost little time: GFS 0.25° on 8 of 24 threads, 4% slower for a
fifth less CPU (IFS 0.25°: as fast, a quarter less).

**GRIB1.** A GRIB1 file (or one of both editions) opens like any other:
each message reads as the GRIB2 it would be -- NCEP's parameters by its
ON388 tables (NARR's MSLET, 4LFTX ...), ECMWF's by ecCodes' with the level
and statistic they imply (`2t` at 2 m, `tp` accumulated from the start,
`mx2t` a maximum), levels and time ranges as GRIB2 says them, ensembles
(NCEP's PDS extension, ECMWF's local definition 1) as 4.1/4.11. Lat-lon,
Gaussian (thinned too), rotated, Mercator, Lambert and polar
stereographic grids, spherical harmonics (rotated and stretched too),
space views, NCEP's rotated E- and B-grids (203, 205); grids named but not described
(no GDS) -- WMO's international ones (21-26, 61-64: quarter globes, the
pole one value) and NCEP's ON388 Table B (67 grids: 211, 212, 218, 221,
2, 3, 4 ...), which ecCodes and pygrib cannot place; simple packing, every
form of second-order packing (ECMWF's general extended, with spatial
differencing and boustrophedonic rows; WMO's row by row, constant width,
general), bitmaps, and matrices of values at each point (wave spectra:
each element a field of its own, worded by its direction and frequency --
ecCodes misreads them, taking the secondary bitmaps for values, and
writes none). Messages are turned when
read, a batch at a time: opening NARR's 56 MB analysis takes 0.03 s and
7 MB, reading all 423 fields 0.04 s (ecCodes 0.35 s, pygrib 0.46 s). A
remote GRIB1 reads by its wgrib inventory (`.inv` beside it, as NCEI's
NARR has). Values are ecCodes' to the bit but for constant fields with a
decimal scale, where fastgrib applies GRIB's formula (NARR's cloud water
at 100 hPa: 1e-12 kg/kg) and ecCodes and g2clib do not (0.001; wgrib2
`-g2clib 0` agrees with fastgrib).

**Winds.** `f.winds(...)` gives `(u, v)` of the vector fields picked
(every UGRD by default, each with its VGRD), turned east and north where
the grid gives them along its own axes (HRRR's, NAM's, RAP's Lambert
winds, rotated grids): wgrib2's `-new_grid_winds earth`, to its 0.1 m/s.
`earth=False` gives them as stored.

**Spherical harmonics** (GRIB2 3.50 with 5.50/5.51 packing; GRIB1's,
GRIBEX's subset quirk undone as ecCodes does): `read()` gives the
coefficients as ecCodes and pygrib do, each (n, m) a real and an imaginary
part, m outermost -- to the bit -- and `regrid()` / `at()` synthesise the
field at any points, which neither does: ECMWF's normalisation (the (0, 0)
coefficient the global mean, no Condon-Shortley phase), as pyshtools gives
it to 1e-5 K. A T63 field onto a global 0.5-degree grid: 12 ms. Rotated
and stretched ones (3.51-3.53, GRIB1's 60/70/80: ARPEGE's kind) are
synthesised in their own frame -- rotated as rotated grids are, stretched
by Schmidt's transform about the stretching pole. `encode()`
packs coefficients back, `'simple'` (5.50) or `'complex'` (5.51, the
field's own subset and Laplacian power kept), a field's own to the bit.

**Satellite images** (space view, 3.90: Meteosat's, Himawari's, as
EUMETSAT and JMA put them in GRIB; GRIB1's too) are placed by CGMS's
navigation on the field's own ellipsoid: wgrib2's points, PROJ's geos to
1e-10 degrees, NaN past the Earth's edge (wgrib2 prints a sentinel,
ecCodes (0, 0)); a full MSG disk, 13.8M points, in 0.04 s. Views neither
wgrib2 nor ecCodes places -- a camera off the equator or over a pole, the
image turned, orthographic views (Nr missing) -- are placed as PROJ's
tilted perspective and orthographic projections place them (to 1e-8
degrees). `regrid()` and `at()` work from them as from any grid.

**Staggered and listed grids.** Rows offset half a step (WMO's scanning
flags 0x08/0x04, 0x02 along j, 0x01 offset rows a point short) are placed
as wgrib2 places them; NCEP's rotated E-grid (3.32768) and B-grid
(3.32769: NAM's native output, which wgrib2 cannot place) as rotated
grids about their centre -- NAM's points where NCEP's own NLAT/ELON fields
put them, to those fields' packing. Points listed one by one (3.130) read
as flat `values` with each point's latitude and longitude, `at()` and
`regrid()` their nearest. Grids whose points are not in the message
(ICON's unstructured 3.101, curvilinear 3.204) read as flat `values`,
`keys()` naming the grid file (`uuidOfHGrid`, `numberOfGridUsed`).

## Regridding

```python
z = f.regrid((np.arange(20, 55, 0.25), np.arange(230, 300, 0.25)), ':HGT:500 mb:')
t = gfs.regrid(hrrr.grid(), name='TMP')           # onto another file's grid: they line up
p = f.at(40.0, 255.0, 'PRMSL')                    # at points (any shapes that broadcast)
```

From any grid fastgrib reads (rotated, reduced, spectral) onto a lat-lon
grid's axes, a Grid (a GribFile or Record: its grid) or points; bilinear,
or `method='nearest'` (the default on points listed one by one). Gaps and edges as IPOLATES (wgrib2's `-new_grid`)
treats them -- a missing or off-grid corner weighed out, the point defined
while half the weight is -- and nearest's ties as its NINT, so where
IPOLATES places points consistently the results are wgrib2's to its output
packing (Lambert, polar stereographic, lat-lon, Gaussian). Points are
placed by each grid's own forward projection: a field of every point's
own latitude, regridded, gives the target's to 0.001 degrees, where
IPOLATES is 0.4 degrees out on polar stereographic and 12 in longitude on
a Mercator grid across the date line. Winds along the grid's axes come out
east and north, turned at their own points (a uniform wind stays uniform;
wgrib2's nearest turns it by the target's angle). A plan of corners and
weights is made once per grid and applied to each field: NBM's 3.7M-point
Lambert grid onto 580k points, 57 ms the first field, 4 ms each after.

## Writing

```python
f.write('tmp.grib2', name='TMP')                  # a subset: each field its own message
f.write('out.grib2', append=True, level='500 mb') # GRIB1 files and remote ones too (as GRIB2)
m = f.message(rec, name='DPT', level='700 mb', fcst='0-6 hour max fcst', date=datetime(2030, 1, 2))
m = f.encode(values, like=rec)                    # new values as a field like rec: bytes
m = f.encode(values, rec, packing='complex', decimals=1)
fastgrib.open(m).read(0)                          # messages are bytes: open them, write them
```

`write` copies each field picked as its own GRIB2 message: a message of
one field as it is, one of several cut to the field's sections, a bitmap
given before (254) made explicit, a GRIB1 field turned into GRIB2 (so
`fastgrib.open(g1).write(out)` converts a GRIB1 file). `message` sets the
name (or `param=(discipline, category, number)`), level (wgrib2's wording
or `(type, value[, type2, value2])`), reference time and forecast time
(statistics as 4.8/4.11) with the packed values untouched. `encode` packs
new values (`read()`'s layout, NaN for none: a bitmap) in the field's
grid and product: `packing='simple'`, `'complex'` (second-order spatial
differencing, groups merged where cheaper: 0.8x NCEP's own bytes on NBM,
1.07x on GFS), `'ieee'` (lossless), `'png'`, `'jpeg'` (JPEG2000, lossless,
to 31 bits) or `'ccsds'` (CCSDS 121.0 as ECMWF's open data are, 0.77x
GFS's bytes) -- each fastgrib's own encoder -- by default as the field's;
spherical harmonics' `'simple'` or `'complex'`; to the
field's own precision (its 2^E 10^-D) unless `decimals` or `bits` say --
a field's own values encode back to the bit. Checked: everything written
reads in wgrib2 and ecCodes as in fastgrib (`tests/data/written.grib2`
and wgrib2's reading of it are kept, and made again byte for byte).
NBM's 3.7M points pack in 32-70 ms (complex), 20 ms (IEEE), 34 ms
(CCSDS), 64 ms (JPEG2000), 100-400 ms (PNG: zlib). Every packing at every
depth from 1 to 30 bits, gaps or none, reads in wgrib2 and ecCodes (libaec,
OpenJPEG, libpng) as in fastgrib -- JPEG2000 past 25 bits too, which
ecCodes itself writes corrupt (OpenJPEG's encoder overflows there).

**JPEG2000** is fastgrib's own codec (numba, no library): any Part-1
codestream decodes as OpenJPEG decodes it -- every code-block style alone
and together, the five progression orders, tiles, precincts, quality
layers, the 9/7 wavelet (to its float rounding), regions of interest --
checked sample for sample against OpenJPEG on each; and correctly past
OpenJPEG's limits (it crashes encoding 24-bit terminated passes and
loses samples past 25 bits; Pillow's reading, past 16: fastgrib's
earlier reader, through Pillow, read 17-25-bit fields wrong). Damaged
codestreams raise `ValueError` (fuzzed).

## Keys

`f.keys(pick)` gives one field's keys as ecCodes (pygrib, cfgrib) names
them, read from its octets -- `discipline`, `parameterCategory`,
`typeOfFirstFixedSurface`, `scaledValueOfFirstFixedSurface`, `Ni`, `Nj`,
`latitudeOfFirstGridPointInDegrees`, `LoVInDegrees`, `forecastTime`,
`typeOfStatisticalProcessing`, `lengthOfTimeRange`, `perturbationNumber`,
`packingType`, `referenceValue`, `bitsPerValue`, `numberOfMissing`,
`shortName`, `typeOfLevel`, `level`, `stepRange`, `stepType`, ... -- equal
to ecCodes' on every key of every test field (17,000 checked; negative PV
levels excepted, which ecCodes reads unsigned). `f.sections(pick)` gives
the raw sections for any octet not named.

## Damaged and unusual files

- **Damaged files** -- cut short, bytes lost, garbage between messages, a
  wrong length: every intact field is still read, each damaged message is
  skipped and described in `f.damaged` (and one warning), and corrupt
  packed data raises `ValueError` -- never a crash or a read past the data,
  on the CPU or the GPU (fuzzed: tens of thousands of corrupted messages).
- **Several fields in one message** (wgrib2's `-submsg_uv`, and others):
  each a field of its own, numbered `5.1`, `5.2` as wgrib2 numbers them.
- **Bitmaps given before** (indicator 254): within a message, and from an
  earlier message -- also when only that field is fetched from a remote
  file (the bitmap's message is fetched with it).
- **Files past 4 GB**, **any scanning order** (rows or columns first,
  every other row reversed, either direction) and **many threads at once**
  (one file shared or each its own; GPU work serialised per card).

## Command line

```
fastgrib FILE|URL                                 # the inventory, as wgrib2 -s
fastgrib FILE -match ':TMP:' -not ':0\.' -v       # with units, long names, valid times
fastgrib FILE -select name=TMP level=500 -stats   # min, max, mean, undefined points
fastgrib FILE -grid                               # the grids the fields lie on
fastgrib FILE -match ':HGT:[0-9]+ mb:' -o z.nc    # NetCDF (xarray)
fastgrib FILE -bin out.bin                        # float32 we:sn, as wgrib2 -order we:sn -no_header -bin
fastgrib FILE -csv out.csv                        # lon,lat,value a point
fastgrib -warm                                    # compile every decoder now (once, after installing)
```

(`python -m fastgrib ...` without installing.) Listing, matching and
selecting run no kernel, so numba is not even imported: an inventory of
GFS 0.25° takes 0.27 s from start to finish, most of it Python and numpy
starting. numba loads with the first field decoded.

## What it reads

| | |
|---|---|
| Grids (3.x) | 0 lat-lon, 1 rotated lat-lon, 10 Mercator, 20 polar stereographic, 30 Lambert conformal, 40 Gaussian, 41 rotated Gaussian -- regular, reduced (thinned) or staggered; 50-53 spherical harmonics (rotated, stretched); 90 space view (anywhere, turned, orthographic); 130 points listed; NCEP's 32768 and 32769 (rotated E- and B-grids); every scanning mode (0x01-0xF0); spherical earths (ellipsoidal ones through pyproj); any other (101 unstructured, 204 curvilinear ...) as its values, not placed |
| Products (4.x) | 4.0 to 4.15: analyses and forecasts, ensemble members (1, 11) and statistics (2, 12), clusters (3, 4, 13, 14), probabilities (5, 9), percentiles (6, 10), statistics over time (8-14) and space (15); chemicals and aerosols 4.40-4.49 (WMO's C-14 names, sizes, wavelengths); 4.60/4.61 re-forecasts -- worded as wgrib2 words them (its whole level table and time-range rules) |
| Packing (5.x) | 0 simple, 2 and 3 complex (spatial differencing, missing values), 4 IEEE (32 and 64 bits), 40 JPEG2000 (any Part-1 codestream, to 31 bits), 41 PNG, 42 CCSDS/AEC (ECMWF's open data), 50 and 51 spherical harmonics (simple, complex), 200 run-length (JMA) |
| Bitmaps | present, absent, or given before (254) |
| Parameters | NCEP's tables (1757 parameters, WMO numbers and NCEP's local ones), then ecCodes' concepts (WMO, ECMWF's and NCEP's local, legacy; chemicals and aerosols by their constituent, size and wavelength -- `aod`, `pm2p5` -- with ecCodes' chemical-split concepts last, as ecCodes before 2.49 named them) |
| GRIB1 | grids 0, 1, 3, 4, 5, 10, 14 (thinned too), 50, 60, 70, 80 (spherical harmonics), 90 (space view), 203, 205 (NCEP's E- and B-grids), predefined (WMO's, NCEP's Table B); simple, second-order and spectral (simple, complex) packing; matrices of values; bitmaps; NCEP's tables 2 and 128-141, ECMWF's (3954 parameters) |

Not placed, by design: grids whose points live elsewhere (3.101's grid
file, 3.204's coordinate fields) and GRIB1's 201/202 E-grids, which carry
no rotation centre (NCEP retired them). Not read: JPEG2000 with progression
order changes or packed packet headers (POC, PPM/PPT markers: no GRIB
writer uses them) and GRIB1 second-order packing of a matrix of values
(nothing writes it; ecCodes neither). Anything not covered raises a clear
error instead of guessing.

## Accuracy

The same integer unpacking and floating-point arithmetic as wgrib2 --
`(R + X·2^E)·10^-D` in double with wgrib2's integer powers, rounded once
to float32 -- so every value matches wgrib2's to the bit, and NaN sits
exactly where wgrib2 writes 9.999e20. Checked against wgrib2 3.1.3 on:
every field of GFS 1° and 0.5° (1486 fields, 110 with undefined points),
NBM CONUS (Lambert; probabilities, percentiles, ensemble statistics),
GEFS members and statistics, JPEG2000 / CCSDS / PNG re-encodings, MRMS
(PNG), ECMWF open data (CCSDS, 191 fields), fields regridded onto every
supported projection (latitudes and longitudes too), every scanning mode.

Inventory lines equal wgrib2's, and the `.idx` files NOAA publishes, line
for line: GFS, GEFS, NBM, HRRR, RAP, NAM and RTMA files checked whole
(wgrib2 3.1.3 lacks a few parameter names and level types; there
fastgrib gives ecCodes' names and newer wgrib2's level wording).
`short_name` and `level_kind` equal ecCodes' `shortName` and
`typeOfLevel` on every field tried (1276 fields: GFS, GEFS, NBM, ECMWF).
Rotated and reduced grids' points equal ecCodes' (to 1e-5 degrees), their
values wgrib2's; IEEE and run-length values wgrib2's; GRIB1 values
ecCodes' (NARR's 423 fields, ecCodes' GRIB1 samples) and wgrib2's of the
same messages as GRIB2. `tests/` repeats these checks on small files
(wgrib2's and ecCodes' own output stored beside each); `pytest` runs them
without wgrib2, ecCodes or a network.

## Performance

Decoding every field of a file, ms a field (i7-13700K 8P+8E, RTX 5070 Ti,
Windows 11; wgrib2 3.1.3 with OpenMP on all 24 threads, in WSL2, writing
to /dev/null):

| | wgrib2 | fastgrib CPU | fastgrib GPU → numpy | fastgrib GPU, on the card |
|---|---|---|---|---|
| NBM CONUS, 769 fields of 3.7M points | 4.4 | 0.54 | 0.37 | 0.10 |
| GFS 0.25°, 743 fields of 1.04M points | 1.1 | 0.27 | 0.26 | 0.17 |
| GFS 0.5°, 743 fields of 260k points | 0.36 | 0.076 | 0.10 | 0.055 |
| CCSDS (GFS 0.5° re-encoded), 86 fields | | 0.36 | | |
| JPEG2000 (GFS 0.5° re-encoded), 86 fields | 31 | 3.2 (its own codec; OpenJPEG: 1.6 through imagecodecs, 21 through Pillow) | | |
| MRMS reflectivity (PNG), 24.5M points | 130 | 75 | | |

(The table above predates the decoder changes of October 2026; on the
files below, CCSDS and JPEG2000 decode about twice as fast as then, on
one core and on all of them.)

Against wgrib2 3.1.3 and pygrib 2.1.6 on operational files, all in WSL2
on the same machine, ms a field (wgrib2 `-bin /dev/null`, pygrib
`.values` of every message; one core, then every core -- wgrib2's
OpenMP is slower on 24 threads than on one for all of these):

| | wgrib2, 1 thread | pygrib | fastgrib, 1 core | fastgrib, every core | one field read alone (pygrib / fastgrib) |
|---|---|---|---|---|---|
| GFS 0.25°, 743 fields, complex packing | 2.67 | 7.32 | 1.25 | 0.28 | 11.2 / 1.1 |
| HAFS, 749 fields of 2.3M points, complex packing, bitmaps | 5.02 | 23.8 | 2.93 | 0.47 | 28.8 / 2.5 |
| ECMWF IFS 0.25°, 78 fields, simple packing (12 bits) | 1.43 | 3.55 | 0.87 | 0.25 | 2.9 / 1.3 |
| ECMWF open data, 157 fields, CCSDS | 10.1 | 11.9 | 4.28 | 0.52 | 11.8 / 4.8 |
| RAP 13 km, 355 fields, JPEG2000 | 8.76 | 10.1 | 4.75 | 0.98 | 11.9 / 2.1 |

Opening and listing GFS 0.25°: 10 ms (wgrib2 `-s` 141 ms, pygrib 1.28 s).

One NBM field read alone: 2.0 ms on the CPU (1.4 into a reused array),
0.7 ms on the GPU into a reused array. The GPU → numpy rate is the PCIe
copy (~50 GB/s), not the decoding; the CPU at full load is RAM-bound.
`python tools/bench.py FILE` measures your machine.

How: GIL-free numba kernels on a thread per core, branch-free bit reads,
records too big for one core split across all of them in one pass (runs
decoded from a zero state, then corrected -- the spatial differencing is
linear -- each thread taking the next run as it comes, so threads that
start late or run on slower cores just take fewer); on the GPU, one CUDA
graph per batch: a scan over chunks for the differencing, a per-field
float table, a warp-uniform decode loop, uploads and downloads overlapped
on their own streams. Memory stays bounded: `fields()` decodes into a
fixed ring of buffers, each decoding thread keeps at most 32 MB of
JPEG2000 scratch arrays (fresh ones each field were page faults, which no
two threads take at once), and page-locked memory is at most the file
being read on the GPU (2 GB and an eighth of RAM at most) and two arrays
you reuse with `out=`; `f.close()` lets the file's go,
`fastgrib.release()` everything.

## Lower level

`fastgrib._cpu` (`scan`, `defined`, `decode`, `decode_iter`) and
`fastgrib._gpu` (`decode`, `decode_iter`, `release`) decode onto any set
of grid points into arrays you give, with wgrib2's 9.999e20 (or any value)
for missing points. They are what the front end is built on; their
interfaces may change.

## Development

[CONTRIBUTING.md](CONTRIBUTING.md) has the setup, the rules (values stay
wgrib2's to the bit), how to measure, and how kernels are written;
[docs/internals.md](docs/internals.md) maps the code; [CHANGELOG.md](CHANGELOG.md)
lists what changed.

```
pytest                                            # the checks against wgrib2's and ecCodes' stored output
FASTGRIB_BIG_TEST=1 pytest tests/test_big.py      # a 4.3 GB file (written and removed)
python tools/bench.py FILE                        # timings on your machine
tools/testdata/make_test_data.sh GFS NBM ECMWF_MSG ECMWF_FC GEFS_DIR NAM_BGRD tests/data   # rebuild the test data
python tools/tables/make_params.py PAGES_DIR             # tables/params.py from NCEP's tables
python tools/tables/make_eccodes.py DEFINITIONS_DIR      # tables/eccodes.py from ecCodes' definitions
python tools/tables/make_levels.py LEVEL_C DEFINITIONS_DIR   # tables/levels.py from wgrib2's Level.c and ecCodes
python tools/tables/make_params1.py TABLE2_HTML          # tables/params1.py from NCEP's ON388 Table 2 page
python tools/tables/make_params1_ecmwf.py DEFINITIONS_DIR    # tables/params1_ecmwf.py from ecCodes' GRIB1 definitions
python tools/tables/make_chem.py C14.csv                 # tables/chem.py from WMO's Common Code Table C-14
python tools/tables/make_ncep_grids.py TABLEB_HTML       # tables/ncep_grids.py from NCEP's ON388 Table B
python tools/make_warm_sample.py                         # fastgrib/_warm.grib2, warm()'s sample, from tests/data
```

The package: `_file` (the front end) on `_cpu` / `_gpu` (decoding),
`_grids`, `_regrid`, `_winds`, `_spectral`, `_images` (PNG, CCSDS, and
`_jpeg2000`, the JPEG2000 codec), `_grib1` (with `_grib1_2nd`,
`_grib1_matrix`), `_encode`, `_keys`, `_meta`, `_remote`, `_xarray`, with
the byte and bit readers they share in
`_octets`, `_jit` (numba's `njit` and `intrinsic`, numba imported with the
first kernel run) and the generated tables in `fastgrib/tables/` (loaded
when first used). `tools/tables/` makes those tables, `tools/testdata/` the
test data; `tests/` has a module per topic, `helpers.py` their shared
paths and comparisons.

Parameter names, descriptions and units come from NCEP's GRIB2
documentation (a US government work, public domain) and ecCodes'
definitions (Apache License 2.0); level wording from wgrib2's Level.c
(public domain). wgrib2 is the reference the values are checked against;
no other wgrib2 code is included. See [NOTICE](NOTICE).

## License

MIT -- see [LICENSE](LICENSE); the ecCodes-derived tables are under the
Apache License 2.0 ([LICENSE-ecCodes](LICENSE-ecCodes)), WMO's code table
C-14 under the MIT License ([LICENSE-WMO-CCT](LICENSE-WMO-CCT)); see
[NOTICE](NOTICE).
