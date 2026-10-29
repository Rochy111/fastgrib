# Contributing to fastgrib

Thanks for helping. fastgrib has one non-negotiable rule and a few hard-won
ones; this page lists them, then how to set up, test, measure and submit.

**The rule: decoded values stay bit-identical to wgrib2.** Every change to a
decoder must leave every field's float32 bits exactly as they were (and as
wgrib2 writes them). A faster decoder that is off by one ulp is a bug.

## Set up

```
git clone https://github.com/Rochy111/fastgrib
cd fastgrib
pip install -e ".[xarray,proj,test]"      # add 'gpu' (cupy) if you have an NVIDIA GPU
pytest
```

Python 3.9+, numpy 1.22+ (1.x and 2.x), numba 0.57+. The first `pytest`
run compiles every numba kernel (a few minutes); later runs load them from
numba's cache in `fastgrib/__pycache__`.

## Tests

`pytest` checks fastgrib against wgrib2's and ecCodes' own output, stored
beside small files in `tests/data/` -- it needs neither tool, nor a network.
Tests that need optional pieces skip without them: the GPU tests without a
working cupy and an NVIDIA GPU, the ellipsoidal-earth grids without pyproj. `FASTGRIB_BIG_TEST=1 pytest tests/test_big.py` writes,
reads and removes a 4.3 GB file. CI runs the suite on Linux, Windows and
macOS (Apple Silicon), Python 3.9-3.13, numpy 1 and 2 -- but no CI runner has
an NVIDIA GPU, so run the GPU tests on a machine with one before a release.

The test files are small, so for a change to a decoder also check real files
before and after: hash every field's bytes with the old code and the new,
and compare.

```python
import hashlib, fastgrib
f = fastgrib.open('gfs.t00z.pgrb2.0p25.f006')
digest = [hashlib.sha1(v.tobytes()).hexdigest() for _, v in f.fields()]
# ... and single reads too: f.read(r) splits big fields across cores differently
```

Operational files cover what the test files cannot: GFS (complex packing),
HRRR/NAM/HAFS (complex packing with bitmaps), ECMWF open data (CCSDS), IFS
(simple packing), RAP/NAM products (JPEG2000), MRMS (PNG).

## Measuring performance

`python tools/bench.py FILE` times `fields()` and single reads on your
machine. To compare two versions, alternate them (A, B, A, B ...) and look
at the spread, not one run. Things that will mislead you:

- **Hybrid CPUs on Windows** (Intel 12th gen and later): a script started
  from a terminal counts as background work, and Windows parks its main
  thread on an efficiency core -- single-core timings swing 30-50%. Call
  `fastgrib._cpu._fast_thread()` on the timing thread. (fastgrib's own
  decoding threads already do.)
- **CPU time on Windows** (`time.process_time()`) ticks every 15.6 ms: it
  means nothing over runs shorter than a second or so.
- **`PYTHONDONTWRITEBYTECODE`** (set in some tool environments) makes every
  import recompile fastgrib from source -- import and cold-start timings
  come out ~0.1-0.2 s too slow.
- **numba's cache** can keep serving an edited kernel's old code: delete
  `fastgrib/__pycache__/*.nbi` and `*.nbc` before timing an edit.
- **Fresh memory**: a new result array is faulted in by the OS page by page
  (on Windows ~0.45 ms per 4 MB, on any number of threads). Time
  `read(..., out=array)` to see the decoder alone.

## Writing kernels

fastgrib's decoders are numba `@njit(cache=True, nogil=True)` functions run
on fastgrib's own thread pool. What has been measured to matter:

- **Import `njit`, `intrinsic` from `fastgrib._jit`, not from numba.** They
  defer importing numba until the first kernel runs, so listing a file never
  pays for it.
- **No `inline='always'`.** Plain `@njit` helpers are inlined by LLVM anyway;
  numba's own IR inlining of helpers that take arrays left reference counting
  in hot loops (ten times slower under numba 0.63).
- **Unsigned indices in hot loops** (`np.uint64`): numba adds a
  negative-index fix-up to every signed index. Mind numba's typing: uint64
  combined with int64 gives float64, so make constants unsigned too.
- **Loops, not slice assignment**, to copy between arrays: numba's
  `a[i:j] = b[k:l]` goes through a temporary.
- **Unpredictable branches cost more than arithmetic.** A branch on data that
  goes either way at random (a parity, a decoded bit) mispredicts half the
  time; make such choices selects or bit arithmetic, and keep branches that
  are rare or follow a pattern.
- **No `parallel=True` / `prange`.** Spread work with `fastgrib._cpu.share`:
  the caller and some pool threads take items off a shared counter
  (`fastgrib._octets.take`) until none is left. numba's parallel threads spin
  after each loop (on Linux a single JPEG2000 read took 70 ms of CPU for 3).
- **No fresh large arrays per field** in code that runs on every core: page
  faults do not run in parallel. Reuse scratch memory (see `_jpeg2000._scratch`)
  or write into the caller's arrays.

## Python code and numpy 2

numpy 2 keeps arithmetic on `numpy.uint8` scalars 8-bit (NEP 50):
`b[p] * 100` wraps round and `b[p] * 100 + 20260000` raises. Python code that
reads octets from a numpy buffer (the mapped file) must read them as Python
ints: wrap the buffer with `fastgrib._octets.octets(buf)` (a memoryview, no
copy) and keep the numpy array for numba. To find any site that behaves
differently under numpy 2, run the suite on numpy 1.26 in its migration mode:

```python
# conftest-style: numpy 1.26 only
import numpy as np
np._set_promotion_state('weak_and_warn')   # warns (or raises) wherever numpy 2 would differ
```

## Tables and test data

The generated tables in `fastgrib/tables/` and the test data in
`tests/data/` are made by the scripts in `tools/` -- the README's
Development section lists the commands. Don't edit generated tables by hand.

## Releasing

The version comes from git (setuptools-scm): a tag `v0.2.1` is version
0.2.1, and the commits after it 0.2.2.dev1+g..., 0.2.2.dev2+g..., so
`pip install -U git+https://github.com/Rochy111/fastgrib` always upgrades.
It is written to `fastgrib/_version.py` when installed; never edit it.

1. Give `CHANGELOG.md`'s Unreleased section the version and the date
   (`## 0.2.1 -- 2026-10-29`), and run the GPU tests on a GPU machine.
2. Optionally rehearse: Actions > release > Run workflow, publishing to
   `testpypi` (or `nowhere`: build the package and test the installed wheel).
3. Tag and push: `git tag v0.2.1 && git push origin v0.2.1`. The release
   workflow builds the sdist and wheel, checks their version is the tag's,
   installs the wheel in a folder without the source and runs the whole test
   suite against it, and makes the GitHub release -- that version's CHANGELOG
   section as its notes, the wheel and sdist attached. Users can then
   `pip install git+https://github.com/Rochy111/fastgrib@v0.2.1`.
4. PyPI, when wanted: Actions > release > Run workflow, on the tag, publishing
   to `pypi` (`gh workflow run release.yml --ref v0.2.1 -f target=pypi`). A
   version published to PyPI can never be uploaded again, even if deleted.

Publishing uses PyPI's trusted publishing, so no token is stored. Set up
once, on pypi.org (and test.pypi.org) > Account > Publishing > add a
(pending) publisher: project `fastgrib`, owner `Rochy111`, repository
`fastgrib`, workflow `release.yml`, environment `pypi` (`testpypi` on
TestPyPI). PyPI's is set up; TestPyPI's is not.

## Submitting

Open an issue first for anything large. A pull request should say what it
changes and why; for performance work, the before and after numbers (with
the machine) and how you checked the values are unchanged. Keep comments in
the code's style: short, saying why, with the measurement that justified a
choice.
