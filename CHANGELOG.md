# Changelog

All notable changes. Values decoded by fastgrib stay bit-identical to wgrib2
in every release; a change that alters them is a bug.

## 0.3.3 -- 2026-11-10

### Added
- `open(url, max_records=N)` and `byte_range=(start, stop)`: a remote file
  with no index fetches only the messages wanted, a window at a time (8 MB,
  then twice as much), each window as parallel range requests, never a
  request per message. The first 20 fields of a 540 MB GFS file took 1.2 s
  from S3. Records keep their offsets in the whole file. With an index, or
  on a local file, the options pick among its fields.
- `GribFile.write_index(path)` writes the inventory as wgrib2 -s lines
  (NOAA's .idx), and `open(url, index=path_or_url)` reads a remote file
  through an index given (wgrib2 -s, ECMWF's .index or wgrib's .inv): a file
  without one need only be scanned once. The last message an index lists is
  fetched to its own length, not to the end of the file.
- `xr.open_dataset(path, engine='fastgrib')` (and `open_mfdataset`), in
  place of cfgrib: opening reads only the inventory, and only the fields
  indexed are decoded. `filter_by_keys` takes cfgrib's ecCodes names.
  `to_xarray(lazy=True)` gives the same from an open file.
- `read(..., expand=True)`, `fields(expand=True)`, `latlon(expand=True)` and
  `fastgrib.expand(values, grid)`: reduced Gaussian (and other reduced)
  grids on their regular grid, each row interpolated as pygrib's `redtoreg`
  does. The float32s are identical on an O1280 field, in 34 ms on one core
  and 3.8 ms on all of them (`redtoreg`: 124 ms).
- `keys()` of simulated satellite imagery (4.32 to 4.34) gives its spectral
  bands as ecCodes names them: `NB`, `satelliteSeries`, `satelliteNumber`,
  `instrumentType`, `scaleFactorOfCentralWaveNumber`,
  `scaledValueOfCentralWaveNumber` (a list each for several bands).
- `keys()` gives `paramId`, ecCodes 2.50's (0 where it has none), from the
  same concept as `shortName`: equal to ecCodes' on every GRIB2 test field
  whose shortName is. tools/tables/memfs_definitions.py writes out the
  definition files a conda-forge ecCodes for Windows holds in memory, to
  build the tables from.
- NCEP's local parameter 0.0.205, which RRFS writes and NCEP's pages do
  not list yet, is named as wgrib2 names it: ELMELT, Effective Layer (EL)
  Temperature, in C.
- `write(path, packing='complex'|'ccsds'|'jpeg'|..., decimals=, bits=)`:
  a file (or the fields picked, or a box) packed again, as wgrib2's
  -set_grib_type, on every core: fields are packed side by side while the
  next decodes, the bytes as packing each alone. HRRR's 173 fields: complex
  0.87 s (wgrib2 131.6 s), CCSDS 0.47 s (8.7 s), JPEG2000 4.8 s (27.2 s),
  every value read back unchanged. Command line: `-packing TYPE` with
  `-grib` or `-small_grib`.
- `open(url, cache=folder or True)`: each remote message fetched is kept on
  disk, one file per message, and any later read takes it from there, in
  this process or another. Only the index is fetched again. A copy cut
  short is fetched again.
- `f.to_pandas()`: the inventory as a DataFrame (name, short_name, level,
  level_kind, fcst, fcst_hours, ref_time, valid_time, member, ...), whose
  rows any pick takes back: `f.read(df.query('level_value >= 500'))`.
- Files compressed whole with gzip, bzip2 or xz open as the GRIB inside
  them (as some archives ship them), and so do such bytes.
- `':TMP:2 m above ground:' in f`: whether any field matches.
- Command line: `-grib OUT` writes the fields picked as GRIB2 (wgrib2's
  -grib); `-max_records`, `-byte_range` and `-index` as above.
- `write(path, box=(west, east, south, north))`: each field cut to a box
  and written on the smaller grid, as wgrib2's -small_grib, with wgrib2's
  grid sections. Fields are packed again as they were, so their values read
  back unchanged. Lat-lon and Gaussian grids, across the 0° seam or the
  antimeridian too; Mercator, polar stereographic and Lambert grids (the
  rectangle round the box). An HRRR surface file's 173 fields took 0.87 s,
  against wgrib2's 6.2 s (simple packing) and 27.7 s (complex packing, as
  fastgrib writes it). `read(box=...)`, `fields(box=...)` and
  `latlon(box=...)` cut the same box out of arrays; on the command line,
  `-small_grib W:E S:N OUT`.

### Faster (every value unchanged)
- Packing: a field packs in ~22 ms rather than 38 (HRRR, complex). Its scaling, differencing and the like are each one loop
  that vectorises, and its working arrays are kept between fields rather
  than made anew (a fresh field-sized array's pages are faulted in one at a
  time, whatever the threads). `write(packing=...)` decodes each field on
  the thread that packs it, so no thread waits on one decoding for all.
  HRRR's whole file: complex 0.87 s, CCSDS 0.47 s, JPEG2000 4.8 s.
- `write(packing=...)` packing many fields at once was held up by memory's
  bandwidth, not the cores: 4 threads were as fast as 24 (7.4 ms a field).
  Its packers now make their integers a cache-sized chunk at a time and
  difference them from there, never storing them; pack a field with no
  gaps from the values as given rather than a float64 copy; and move the
  groups left together after each pass of the group search, so the next
  walks along memory. HRRR's whole file, 16 threads: 4.9 ms a field. The
  bytes are the same (981 messages compared). `encode()` of one field,
  with memory to itself, keeps the single pass.
- Arrays a kernel returns (run from the kernel image) are no longer copied
  out: numpy takes the kernel's memory, freed with the array.
- Bitmapped fields on grids already in read()'s order (RRFS, most models)
  build their bitmap's map in one kernel rather than numpy's passes:
  RRFS's 70 bitmapped fields, each its own bitmap, 1.71 ms a field read
  alone (2.77 before), 0.74 through `fields()` (1.79).
- GRIB1 fields (ERA5) take partial reads too: a simply packed field made
  its GRIB2 message (one copy, nothing decoded) and only its points read.
  ERA5, three points from each field: 2.2 ms rather than 6.4.
- Partial reads: `at()`, `regrid()` onto a few points and `read(box=...)`
  of a simple-packed or IEEE field read only the values they need, where
  they lie, when that is a small part of the field. GFS's 3072 x 1536 flux
  file packed simple: a point from all 113 fields 25 ms rather than 300, a
  10 deg box 47 ms rather than 390. A remote field is fetched in part too:
  its first sections (8 KB, and its bitmap), then only the byte ranges of
  the values wanted, joined where close and fetched side by side. Three
  points of a 0.25 deg global field took 13 KB of its 1.2 MB in 4
  requests, a 10 deg box 73 KB in 2.
- Complex packing, CCSDS and PNG, which decode only in order, are decoded
  up to the last point wanted and no further, side by side on every core,
  or whole where that is no slower. One point from every field: HRRR 27 ms
  (southern Texas) to 102 ms (North Dakota) rather than 113-124; GFS's
  flux file 43 ms (75 N) to 190 ms (70 S) rather than ~340. Remotely, a
  complex-packed field's group headers, then its values up to the last
  group wanted (HRRR, southern Texas: 36% of the bytes).
- Remote CCSDS and PNG fields are fetched in part too, though where a
  value ends in their streams is known only by decoding up to it: a guess
  from the share wanted, then half as much again until the stream holds
  the point. A point near the start of a 0.25 deg global field: 7-8% of
  its message, in three requests.
- JPEG2000 fields take partial reads: the codestream's packets parsed, then
  only the code-blocks whose wavelet coefficients reach the points wanted
  decoded (16 of an HRRR field's 577 for a point), and the inverse wavelet
  run over a window round each group of points. One point of an HRRR
  field: 4 ms rather than 68; one from 24 fields, 64 ms rather than 228;
  ten scattered points, a third of a whole read, and never more than one.
  `_jpeg2000.decode(..., want=)` is checked sample for sample against
  whole decodes, on OpenJPEG's codestreams of every coding (9/7, tiles off
  the origin, precincts, layers) and on fastgrib's own.
- Remote JPEG2000 fields are fetched in part: the codestream's headers,
  then each packet's header (each lies after the bodies before it: walked
  one by one, the bodies skipped), then only the bodies of the code-blocks
  a read decodes. One point from 24 HRRR fields: 9% of their bytes, from
  fastgrib's JPEG2000 and OpenJPEG's (ecCodes') alike.
- Gaussian latitudes are worked out once per N, not at every `regrid()`
  or `at()` (O1280: 25 ms each time before).
- `at()` and `regrid()` of winds along the grid's axes (HRRR, NAM, RAP)
  turn each u and v pair only at the grid points the interpolation takes,
  and read each pair once. One point from all 173 fields of an HRRR file:
  0.12 s rather than 0.36.

### Fixed
- JPEG2000 encoding of a field under a bitmap (one row of its values)
  asked for 1.6 GB on HRRR: each code-block was given a full block's room,
  in 8-byte integers, and such a row is 30,000 blocks of 64 x 1. Each block
  now has room for its own size, in bytes; the output is the same.
- `at()` and `regrid()` of a remote file turn winds given along the grid's
  axes (HRRR's, NAM's, RAP's) to east and north, as on a local file: an
  index does not say which they are, so they were left as they came. Each
  field's first sections are read first now (8 KB, side by side, kept for
  the read after).
- JPEG2000 encoding of an image wider than 32,768 samples (any field with
  a bitmap and more points than that, which packs as one row of its
  values; a grid wider than that) wrote one packet where the image has
  several precincts, so wgrib2, ecCodes and fastgrib could not read it
  back. Each precinct now gets its own packet (HRRR's HGT at the lifted
  condensation level: 1.9M values, read back exactly by all three).
- Inventory lines are wgrib2's to the last character, checked line by line
  against wgrib2 3.1.3 on ~40 operational files (GFS, GEFS, HRRR, RAP, NAM,
  RRFS, HAFS, HWRF, AROME, MSC's GDPS, RDPS and HRDPS, ECMWF):
  - a line ending in items after the forecast time (`ENS=+1`, `prob ...`,
    `31% level`) has no colon after the last, as wgrib2 writes it (GEFS,
    NBM); `find()` patterns written against wgrib2's lines match the same;
  - simulated and observed satellite imagery (4.31 to 4.34: HWRF's and
    HAFS's brightness temperatures, AROME's) says `no_level`, and gives
    wgrib2's `sat=` item: the satellite, the instrument and the band
    (`sat=DMSP 17 SSMIS 19.35 GHz V pol.`, `sat=GOES 15 IMAGER 10.68 um `).
    Each band of HWRF's simulated imagery is then a variable of its own in
    xarray;
  - any other template without a level (4.20, 4.30, 4.1000 ...) says
    `no_level`, as wgrib2, not `product template 4.N`.
- `keys()`' `name` and `units` are ecCodes' (`m s**-1`, 'Pressure reduced
  to MSL'), from the same concept as its `shortName`; the record keeps
  wgrib2's words (`m s-1`). Compared with ecCodes on 8,600 operational
  messages, `keys()` also now gives NCEP's rotated E- and B-grids' size and
  corners (`Ni`, `Nj`, ...; RAP's native grid `gridType` 'ncep_32769'), a
  statistic over no time's `stepRange` as its one step ('0', not '0-0'),
  and no `typeOfLevel` or `level` for a template without a level.
- An accumulated flux is in J m-2, not W m-2: ECMWF's `ttr`, `ssrd`, `str`
  and the rest, GRIB1 and GRIB2, as ecCodes has them (GRIB2's parameter
  table gives the flux's units; an accumulation of it over time is an
  energy). Averages, as NCEP writes its fluxes, keep W m-2.
- A GRIB1 statistic over no time is worded in its own unit ('0-0 hour acc
  fcst'), not '0-0 day acc fcst'. That is how wgrib2 words a GRIB2 one, and
  GRIB2 fields still read that way, as in wgrib2.

### Changed
- Complex packing (`encode()`, `write(packing='complex')`) finds its groups
  as wgrib2's tightest packer does (neighbours merged where that saves
  bits, wider ranges allowed in turn, group lengths given more bits while
  that saves, boundary values moved to the narrower side), from blocks of
  8: 9% smaller than 0.3.2's, so its bytes differ. HRRR's surface file
  packed again: 128.5 MiB (NCEP's own 143, wgrib2's -set_grib_type c3
  125.6) in 0.87 s (wgrib2 131.6 s). The values read back unchanged, as
  before.
- A `GribFile` pickles as its path or URL (opened again where unpickled),
  not its bytes: 143 MB for an HRRR file before, under 1 KB now. Lazy
  xarray Datasets pickle too, so dask's `chunks=`, `open_mfdataset(...,
  parallel=True)` and distributed workers work.
- A remote file without an index is downloaded as several range requests
  at a time when the server takes them, not one request (one connection to
  S3 often slows to 10 MB/s).
- Each GitHub release carries a `SHA256SUMS` of its wheel and sdist, and
  its notes start with the line that installs it.

## 0.3.2 -- 2026-11-08

### Faster (every field's values unchanged)
- Reading one complex-packed field at a time takes 15-33% less time
  (GFS's 3072 x 1536 flux file 6.0 ms rather than 7.3, HRRR 1.1 rather
  than 1.6): the scan that cuts a field into runs for all cores, which ran
  on one core before any of them started, reads the group widths and
  lengths in order rather than each from its own load, 3 times as fast.
- Where fastgrib runs on numba rather than its kernel image (another CPU,
  not x86-64, an edited install), the bit reader inside every complex-packed
  group, simple-packed value and GRIB1 second-order group is 7 times as
  fast: a call in its end-of-buffer branch, never taken in practice, made
  numba's code slow on every read. Single reads there take 40-58% less
  time and `fields()` 10-30% less.

### Fixed
- Simulated satellite imagery (product templates 4.32 to 4.34, as in HWRF
  and AROME's brightness temperatures) and cross sections (4.1000) have
  their valid time and forecast time ('6 hour fcst', as wgrib2 says) rather
  than `valid_time = None` and 'anl'. They have 4.0's times but no level,
  and times were read only with 4.0's level. 4.33 and 4.34's ensemble
  member is read too (`ENS=+4`, as wgrib2 words it), and `keys()` gives
  their `forecastTime`, `stepRange` and the rest.
- Templates that are 4.0, 4.1, 4.8 or 4.11 with octets inserted now read as
  those, with their level, times and member (as wgrib2 reads them), rather
  than as 'product template 4.N', 'anl': categorical forecasts (4.51,
  4.91), partitioned parameters (4.53, 4.54), tiles (4.55, 4.56, 4.59,
  4.62, 4.63), distribution functions (4.57, 4.58, 4.67,
  4.68) and post-processed products (4.70 to 4.73).

## 0.3.1 -- 2026-11-02

### Faster (every field's values unchanged)
- A new process's first field takes 4 ms rather than 760 (a JPEG2000 one 12
  rather than 850; a 31 MB GFS file's first 10 rather than 800), and its first
  values come 155 ms after it starts rather than 915. Nearly all of that time
  went to starting numba, not to the decoders: `warm()`, and so `pip install`,
  now also links the compiled decoders into one file for your CPU, which
  later processes run from without importing numba. Reading the rest of a
  big file is a little quicker too (the 31 MB file: 55 ms rather than 71),
  numba's own memory copies on Windows being slower than the C library's.
- Wherever that file cannot serve (another CPU, an edited fastgrib, a decoder
  it does not hold, a damaged file, not x86-64), fastgrib uses numba as
  before, and rebuilds it in the background after the first such field.
  `FASTGRIB_IMAGE=0` turns it off; `FASTGRIB_CACHE_DIR` says where it is kept
  (else beside numba's cache in the package, or in your user cache).

## 0.3.0 -- 2026-10-31

### Changed
- `pip install` compiles the decoders for your CPU, so the first field is
  already at full speed and `fastgrib -warm` is no longer needed. A new
  install's first field takes 0.7 s rather than 7.6, and its first read of
  every packing and grid 1.8 s rather than 44. The install takes about a
  minute longer, and building needs numba.
- When the compiled decoders don't match the machine, for example after a
  numba upgrade, in an editable install or on a different CPU, the first
  field read compiles them again in a low-priority background process.
  `FASTGRIB_AUTOWARM=0` turns this off, and `FASTGRIB_BUILD_KERNELS=0` skips
  compiling during the install.
- `fastgrib.warm()` compiles every CPU kernel, not only the decoders of
  fields read: latitudes and longitudes on every projection, regridding, wind
  rotation, spherical harmonics' synthesis, every encoder, JPEG2000's 9-7
  wavelet and big PNG fields' pipeline each used to compile on first use.
  It takes about 50 s on an empty cache, rather than 30, and under a second
  on a full one.

## 0.2.1 -- 2026-10-29

### Faster (every field's values unchanged)
- Complex packing under missing-value management (template 5.3 with
  octet 23 set to 1 or 2, as on every NBM field) decodes in loops of its
  own: inside the general decoder these values took ~20 ns each. NBM QMD
  percentiles, one thread: 74.5 -> 8.8 ms a field (APCP), 80 -> 17.5 ms
  (WIND); on 24 threads, where runs after the first had the same slow
  loop, 13.2 -> 6.8 ms and 14.7 -> 8.9 ms.
- GPU reads of fields whose rows run north first, with no bitmap (GFS,
  most global lat-lon grids), decode in file order and have their rows
  turned round on the card, as the CPU already did: no column map is
  built on the host. SPoRT's 0.018 deg global SST (177.9M points) 2050 ->
  259 ms; 57 GFS 0.25 deg TMP fields in one read 54 -> 7.4 ms (their five
  maps outnumbered the four kept, so all were rebuilt every call).

### Fixed
- Remote files: a parameter wgrib2 cannot name in an index line (`var
  discipline=3 center=7 ...`, HAFS's simulated brightness temperatures) is
  named as its message is once read (`SBTAGR13`), not `var3_192_58`.
- Remote reads wait between retries (0.5 s doubling, jittered, or the
  server's Retry-After) and try four times: S3's 503 Slow Down outlasted
  three instant retries.
- Worker processes forked after their parent had decoded (`multiprocessing`
  on Linux and macOS) waited for ever on the first field split across
  threads: they inherited the parent's thread pool without its threads.
  A forked child now makes its own, and its own HTTP connections.

### Changed
- The version comes from git (setuptools-scm): every commit after a tag has
  a higher one (0.2.2.dev3+g...), so `pip install -U
  git+https://github.com/Rochy111/fastgrib` upgrades an install. pip used to
  find 0.2.0 already installed and keep the old code.

### Added
- `tools/time_files.py FILE ...`: time decoding every message of files on
  one thread and on all, with the machine's details.
- README: what each new process's first decode costs (~0.5 s), and warming
  a pool's workers with `initializer=fastgrib.warm`.

## 0.2.0 -- 2026-10-27

### Faster (every field's values unchanged)
- JPEG2000: tier-1 decoding rewritten around one flag word per stripe column
  with a branch-free MQ decoder, the inverse wavelet run along rows, and
  scratch arrays reused between fields -- about twice as fast on one core and
  on all of them (RAP: 11.3 -> 5.4 ms a field on one core).
- CCSDS (ECMWF open data): decoded in one pass, block by block, with the
  inverse prediction free of unpredictable branches -- about twice as fast
  (9.3 -> 4.9 ms a field on one core).
- Simple packing: unpacked a chunk at a time with vectorised conversion
  (IFS: 1.6 -> 1.1 ms a field on one core).
- Complex packing: a dedicated loop for fields without missing values, and
  bitmaps counted with a population count (HAFS on every core: 0.77 -> 0.50
  ms a field).
- numba helpers no longer use `inline='always'`, which under numba 0.63
  made complex packing ten times slower (GFS on Linux: 14.7 -> 1.25 ms a
  field on one core).
- Single big reads split across cores take their runs from a shared counter
  instead of a Python task each; numba's `prange` replaced by fastgrib's own
  threads everywhere (a JPEG2000 field read alone on Linux: 71 -> 11 ms of
  CPU).
- `winds()` keeps each grid's rotation angles (RAP: 11 -> 4.8 ms a pair).
- Opening a file parses each grid once rather than once a message
  (~25% faster `open()`).
- Gaussian grids' latitudes by Newton's method, as ecCodes finds them,
  rather than numpy's eigenvalue-based `leggauss` (LAPACK): as accurate, and
  30x faster on O1280 (1.5 s -> 0.05 s for the first `latlon()`).

### Added
- `fastgrib.warm()` and `fastgrib -warm`: compile every decoder at once
  (after installing, in an image build), not with the first field of each
  kind of packing read -- which otherwise took 2-9 s apiece (complex
  packing and JPEG2000 ~8 s each), ~20 s in all.
- `fastgrib.set_threads(n)` and the environment variable `FASTGRIB_THREADS`.
- `fastgrib.release()` also frees the CPU threads' JPEG2000 scratch arrays.

### Changed
- numba is imported with the first field decoded, not with fastgrib:
  `fastgrib FILE` (an inventory) 0.6 -> 0.27 s; Python's HTTP modules load
  only for remote files.
- `GribFile.keys()` returns plain Python ints where some values were numpy
  integers.

### Fixed
- Works under numpy 2: keys (`dataDate`, `dataTime` ...), spectral
  unpacking, GRIB1 second-order packing and CCSDS parameters no longer raise
  or wrap round under numpy 2's integer promotion (NEP 50).
- Reading an ellipsoidal-earth grid without pyproj gives the intended
  "needs pyproj" error instead of a `TypeError`.
- Tests on ellipsoidal-earth grids skip when pyproj is not installed.

## 0.1.0

The first version: GRIB2 and GRIB1 decoding bit-identical to wgrib2 on the
CPU (numba) and GPU (cupy), remote reads by byte range, regridding, wind
rotation, writing and encoding, ecCodes-named keys, xarray datasets, and
fastgrib's own JPEG2000, PNG and CCSDS codecs.
