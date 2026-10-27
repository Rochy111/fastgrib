# Changelog

All notable changes. Values decoded by fastgrib stay bit-identical to wgrib2
in every release; a change that alters them is a bug.

## Unreleased

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
