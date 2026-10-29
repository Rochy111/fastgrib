# Changelog

All notable changes. Values decoded by fastgrib stay bit-identical to wgrib2
in every release; a change that alters them is a bug.

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
