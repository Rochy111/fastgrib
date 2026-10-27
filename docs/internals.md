# How fastgrib works

A map of the code for anyone changing it. The README says what fastgrib
does; this says where and how.

## Modules

| Module | What it holds |
|---|---|
| `_file` | The front end: `GribFile` (`open`, `find`, `select`, `read`, `fields`, `regrid`, `winds`, `write`, ...), device choice, `set_device`, `set_threads`, `release` |
| `_meta` | Scanning a GRIB2 file into `Record`s and wording them as wgrib2 does (levels, forecast times, products) |
| `_grids` | Section 3 into a `Grid` (a named tuple), and every grid's points (`latlon`) |
| `_cpu` | The CPU decoder: placing records (`_place`), decoding them (`decode`, `decode_iter`), complex and simple packing kernels, the thread pool |
| `_images` | Packings that arrive as a coded stream: PNG (zlib), CCSDS, IEEE, run-length, spectral coefficients; JPEG2000 through `_jpeg2000` |
| `_jpeg2000` | fastgrib's own JPEG2000 codec (decoder and lossless encoder) |
| `_gpu` | The cupy decoder for complex packing |
| `_grib1`, `_grib1_2nd`, `_grib1_matrix` | GRIB1, read by turning each message into the GRIB2 message it would be |
| `_encode`, `_keys` | Writing messages and fields; ecCodes-named keys |
| `_regrid`, `_winds`, `_spectral` | Interpolation plans, wind rotation, spherical-harmonic synthesis |
| `_remote` | HTTP(S)/S3 reads through the published `.idx` / `.index` |
| `_xarray` | `to_xarray()` |
| `_octets` | Byte and bit readers shared by everything, numba intrinsics (`clz64`, `take`, `ones`) |
| `_jit` | `njit` / `intrinsic` stand-ins that import numba on the first kernel call |
| `tables/` | Generated parameter, level and grid tables (loaded when first used) |

## Opening a file

`open(path)` memory-maps the file (nothing is read in) and `_meta.scan`
walks its messages in Python: each section's octets through a `memoryview`,
a `Record` per field with its name, level, times and the byte offsets of its
sections 5, 6 and 7. A message's grid (section 3) is parsed once and shared
by every message with the same section (`_meta._grid_of`). GRIB1 messages
are noted during the scan and turned into GRIB2 records by `_grib1`.

No numba is involved: listing, `find` and `select` never import it.

## Decoding

`read()` and `fields()` end in `_cpu.decode` / `_cpu.decode_iter`:

1. **`_place`** turns the records into a parameter table `P` (one int64 row
   a record: packing template, bit widths, offsets ...), a float table `F`
   (reference value, binary and decimal scale), and column maps `D` that say
   where each defined point's value goes. North-up grids without a bitmap
   need no map: they decode in file order and have their rows turned round
   afterwards (`_turn_rows`), which is cheaper than going through a map.
2. **Kernels** decode each record: `_simple` (5.0), `_span` (5.2 / 5.3,
   complex packing with spatial differencing), and the coded streams in
   `_images`. All use wgrib2's arithmetic -- `(R + X 2^E) 10^-D` in double,
   rounded once to float32 -- which is what makes the values bit-identical.
3. **Threads.** Records go to fastgrib's pool (`_cpu._pool`, a thread per
   logical CPU unless `set_threads` says otherwise), biggest first.
   `decode_iter` keeps a ring of result blocks so the threads never run dry
   at a block's end. A call of only a few big complex-packed records splits
   each into runs (`_split`): every run decodes at once from a zero state,
   and because the spatial differencing is linear, each run's true start is
   added afterwards (`_chain_run`, `_chain_end`, `_fix`). The runs are taken
   off a shared counter (`_runs_work`), so a thread that starts late or sits
   on a slow core just takes fewer.

`_cpu.share(work, n)` is the general form of that: the caller plus up to
`n - 1` pool threads run the same numba kernel, each taking items off an
atomic counter (`_octets.take`). JPEG2000's code-blocks, the space-view
navigation and spherical-harmonic synthesis use it.

## The JPEG2000 codec

`_jpeg2000.decode` parses the codestream in Python, decodes packet headers
(tier-2) in numba, gathers each code-block's segments, decodes code-blocks
(tier-1), then runs the inverse wavelet. Tier-1 keeps one 64-bit flag word
per stripe column (OpenJPEG's layout: a 3x6 window of significance, sign,
refinement and visited bits), so contexts and idle-column tests are register
operations; the MQ arithmetic decoder is branch-free in the decision and
renormalises with a leading-zero count. The wavelet runs its vertical
lifting a row at a time. Per-thread scratch arrays (`_scratch`) are reused
from field to field. Every change is checked sample for sample against the
codestreams in `tests/data/j2k.npz`, which OpenJPEG decodes identically.

## CCSDS

`_images._aec` decodes block by block into a block-sized buffer, undoes the
prediction and converts to floats while the block is in L1 -- one pass, no
field-sized intermediate. Split-sample codes are read with a leading-zero
count from one 8-byte load.

## Memory kept between calls

| What | Where | Bound |
|---|---|---|
| Column maps per (grid, bitmap) | `_cpu._MAPS` | 4 maps, 64 MB |
| Grid points | `_grids._LL` | 4 grids, 256 MB |
| Wind rotation angles | `_winds._ANGLES` | 64 MB |
| JPEG2000 scratch | `_jpeg2000._SCRATCH` | 32 MB a thread |
| GPU buffers and page locks | `_gpu` | the file and two reused arrays |

`fastgrib.release()` frees the JPEG2000 scratch and everything the GPU keeps.

## numba

Kernels are declared with `fastgrib._jit.njit`. Until a kernel is first
called these are stand-ins; the first call imports numba, compiles (or loads
from cache) every kernel known so far and replaces the stand-ins in every
fastgrib module, so kernels calling kernels see real dispatchers. numba
caches compiled code per machine in `fastgrib/__pycache__`; the first use
after an install or upgrade compiles (seconds per kind of packing).

See [CONTRIBUTING.md](../CONTRIBUTING.md) for the rules kernels follow.
