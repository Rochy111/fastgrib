"""warm(): every CPU kernel compiled once a machine. Nobody has to call it:
`pip install` does (setup.py), and when what it compiled is not what is
needed -- a numba upgrade, an editable install, a new CPU under an old
image -- the first kernel to run starts it in the background (keep_up()).
fastgrib.warm() and `fastgrib -warm` remain, for a process pool's workers
and for doing it now.

numba compiles a kernel the first time it runs with given argument types
and keeps the machine code in its cache, so a new machine's first field
of each packing takes seconds, and so does its first latlon() of each
projection, regrid, encode ... warm() does one of each on _warm.grib2, a
small file of every packing and grid (tools/make_warm_sample.py), so that
nothing is left to compile later. _rest() names what a small file cannot
reach. test_warm.py finds any kernel it misses.

warm() leaves a file beside the cache saying it finished, named for what
the cache's entries are good for (this Python, numba and CPU: _machine())
and holding a hash of the sources. That file missing or differing, or the
cache not valid here, is how keep_up() knows, with a few file reads as
numba comes up, that a warm() is due: one at a time however many processes
start (a lock file beside it), at low priority, and outliving the program
that started it. A field read meanwhile compiles what it needs, as ever.
FASTGRIB_AUTOWARM=0 turns that off.
"""
import glob
import hashlib
import os
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, '_warm.grib2')
LOCK_AGE = 1800                 # s: a lock this old is a warm() that died
_warming = False                # warm() here: nothing for keep_up() to start


def _decoders(f):
    """Every field of f as single reads, through fields() and split across
    threads."""
    from ._cpu import decode, decode_iter
    from ._file import NAN
    for r in f.records:
        buf, real = f._bytes([r], False)
        cols = f._grid_cols(real[0].grid)
        decode(buf, real, cols, undef=NAN)                         # a field read alone
        for _ in decode_iter(buf, real, cols, undef=NAN):          # fields()
            pass
        if r.packing in (2, 3):                 # a big field's runs, on a small one
            decode(buf, real, cols, undef=NAN, _min_run=16)


def _places(f):
    """Every grid's points, regridding from every kind of grid (spherical
    harmonics' synthesis among them), and winds along a grid's axes turned."""
    from . import _grids, _winds
    lats, lons = np.array([30.0, 40.0]), np.array([250.0, 260.0])
    for r in {_grids.layout(r.grid): r for r in f.records}.values():
        if r.grid.template not in _grids._SPECTRAL:
            f.latlon(r)
        for method in (None, 'nearest'):
            f.regrid((lats, lons), r, method=method)
    u, v = f.winds(name='UGRD')
    g = _grids.layout(f.find(':UGRD:')[0].grid)
    _winds.to_grid(u, v, g)
    _winds.to_earth(u.astype(np.float64), v.astype(np.float64), g)


def _packers(f):
    """A field packed every way, from float32 and from float64, and
    spherical harmonics' coefficients."""
    k = next(k for k, r in enumerate(f.records) if r.packing == 0 and r.bitmap < 0)
    v = f.read(k)
    for packing in ('simple', 'complex', 'ieee', 'png', 'jpeg', 'ccsds'):
        f.encode(v, k, packing=packing)
    f.encode(v.astype(np.float64), k, packing='simple')
    k = next(k for k, r in enumerate(f.records) if r.packing == 51)
    f.encode(f.read(k), k, packing='simple')


def _rest():
    """What no small field reaches, called directly: a big PNG's pipeline
    (_images.PNG_PIPE), JPEG2000's 9-7 wavelet (lossy: no GRIB2 packer here
    writes it), GRIB1's boustrophedonic rows."""
    from . import _cpu, _grib1_2nd, _images, _jpeg2000
    p = np.zeros(_cpu._END, np.int64)
    p[_cpu._DEST] = -1
    _images._convert(np.zeros(1, np.uint8), 0, 1, 1, True, p, np.ones(4), np.empty(0, np.int32),
                     np.empty(1, np.float32))
    z = np.zeros(4)
    _jpeg2000._interleave(z, z, z, z, z, 0, 0, 2, 2)
    _jpeg2000._idwt97(z, 0, 0, 2, 2)
    _grib1_2nd._unturn(np.zeros(2, np.int64), np.ones(2, np.int64))


def warm():
    """Compile every CPU kernel now: ~50 s once, on a desktop, which numba
    then keeps for every later process (~1 s after that, which is all it
    takes a process pool's worker to have them loaded). Installing does
    this, and the first field read does it in the background if it has
    to; call it to wait for that. `fastgrib -warm` on the command line.
    Returns the seconds taken."""
    from ._file import GribFile
    global _warming
    _warming = True
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f = GribFile(SAMPLE)
        _decoders(f)
        _places(f)
        _packers(f)
        _rest()
    f.close()
    try:
        with open(_done_file(), 'w') as fh:
            fh.write(_sources())
    except (OSError, AttributeError):           # a read-only cache, or another numba's insides
        pass
    return time.perf_counter() - t


def _kernel():
    """A kernel, for numba's view of the cache."""
    from . import _cpu
    return _cpu._decode_one


def _machine():
    """What numba's cache entries are for, as its index keys them."""
    from numba.core.registry import cpu_target
    return cpu_target.target_context.codegen().magic_tuple()


def _sources():
    """A hash of the package's sources (the stamps numba keeps are file by file)."""
    h = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(HERE, '*.py'))):
        with open(path, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()


def _done_file():
    """warm()'s file beside numba's cache, wherever numba keeps it, for this
    Python, numba and CPU."""
    import numba
    key = repr((sys.implementation.cache_tag, numba.__version__, _machine()))
    return os.path.join(_kernel()._cache._cache_path,
                        'warmed.' + hashlib.sha256(key.encode()).hexdigest()[:16])


def stale():
    """Are this install's kernels not all compiled, for this machine? (numba
    is asked too: before 0.68 its stamps are the sources' times, which pip's
    copying changes, so that a cache built in the wheel is good for nothing
    there and warm()'s file says otherwise.)"""
    try:
        with open(_done_file()) as f:
            if f.read() != _sources():
                return True
    except OSError:
        return True
    return not _kernel()._cache._cache_file._load_index()


def keep_up():
    """warm() started in the background if it is due (numba loaded).
    Quietly nothing if it cannot be: kernels still compile as they run."""
    if _warming or os.environ.get('FASTGRIB_AUTOWARM', '1') == '0' or getattr(sys, 'frozen', False):
        return
    try:
        if stale():
            lock = _done_file() + '.lock'
            if os.path.exists(lock):
                if time.time() - os.path.getmtime(lock) < LOCK_AGE:
                    return
                os.remove(lock)
            _start(lock)
    except Exception:                                   # noqa: BLE001 -- a head start only
        pass


def _start(lock):
    env = dict(os.environ, FASTGRIB_AUTOWARM='0')
    if sys.platform == 'win32':
        how = dict(creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS
                   | subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        how = dict(start_new_session=True)
    subprocess.Popen([sys.executable, '-c', 'from fastgrib import _warm; _warm.background(%r)' % lock],
                     env=env, cwd=os.path.dirname(HERE), stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **how)


def background(lock):
    """keep_up()'s process: warm(), unless another holds the lock. One that
    fails leaves it, so that it is tried again in LOCK_AGE rather than by
    every program from now on."""
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except OSError:
        return
    if hasattr(os, 'nice'):
        os.nice(10)
    warm()
    os.remove(lock)
