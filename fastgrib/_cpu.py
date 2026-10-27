"""The CPU decoder: GRIB2 values, wgrib2's to the bit, straight into the
caller's columns (fastgrib's front end, fastgrib.open, is built on it).

    recs = scan(buf)                          # every field's metadata
    cols = defined(buf, recs[i])              # the grid points one covers
    out = decode(buf, picked, cols)           # (len(picked), len(cols))
    for i, block in decode_iter(buf, recs, cols):   # a file, batched

The same integer unpacking and float arithmetic as wgrib2, (R + X 2^E)
10^-D in double, rounded once to float32, and `undef` (9.999e20 unless
given) where a record has no value. Data templates 5.0 (simple), 5.2 and
5.3 (complex, spatial differencing, missing-value management) decode
here, 5.40/5.41/5.42 (JPEG2000, PNG, CCSDS) through _images; bitmaps
present, absent or carried over. Anything else raises rather than
guessing.

Records decode on a thread per logical CPU, GIL-free numba kernels taking
the next record as they finish one, biggest first. A call of a few
records splits each across the cores instead, every run at once: the
first from the seeds, the others as if their spatial differencing started
from zero, in exact integers; each then takes in its true start (the
differencing is linear) as soon as the runs before it have ended -- one
pass over the packed bits.
"""
import os
import struct
import sys
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ._jit import intrinsic, njit

UNDEF = np.float32(9.999e20)          # wgrib2's undefined value
BATCH_BYTES = 64 << 20                # decode_iter's batch of results, aimed at
BUDGET_BYTES = 256 << 20              # and at most (less when RAM is short)
RING_BLOCKS = 16                      # decode_iter: blocks its buffer is cut into

from . import _grids
from ._octets import be64 as _be64, bits as _bits, ones as _ones, rd as _rd   # noqa: F401
from ._octets import octets as _octets, take as _take
from ._grids import Grid, _int_power, _sm        # noqa: F401 (re-exported)
from ._meta import NCEP, Record, scan            # noqa: F401 (re-exported)

_check_grid = _grids.check


def latlon(rec):
    """(lat, lon) axes of a lat-lon or Gaussian record's grid, south-first
    and west-first -- the order decode() lays columns out in (wgrib2's
    -order we:sn); latlon2d() covers every grid."""
    ax = _grids.axes(_check_grid(rec))
    if ax is None:
        raise ValueError('%r: a projected grid has no lat-lon axes; use latlon2d()' % rec)
    return ax


def latlon2d(rec):
    """(lat, lon) of every point of a record's grid, (nj, ni) float64 in
    decode()'s we:sn order, so lat.ravel()[cols] goes with decode()'s
    columns. Longitudes in [0, 360), as wgrib2 prints them."""
    return _grids.latlon(_check_grid(rec))


def defined(buf, rec):
    """Flat we:sn grid indices (int64, ascending) of the points `rec` has
    values for -- its bitmap, or the whole grid without one."""
    return np.sort(_defined(_u8(buf), rec))


def _defined(arr, rec):
    """The we:sn index of each point `rec` defines, in file order."""
    ni, nj, scanm = _check_grid(rec)[:3]
    n = ni * nj
    if rec.bitmap < 0:
        f = np.arange(n, dtype=np.int64)
    else:
        bits = np.unpackbits(arr[rec.bitmap:rec.bitmap + (n + 7) // 8],
                             count=n)
        f = np.flatnonzero(bits)
    if scanm & 0xF0 == 0x40:                    # already west-east, south-up
        return f
    pl = rec.grid.pl
    if pl:                                      # reduced: rows of their own lengths
        if scanm & 0x30:
            raise ValueError('%r: a reduced grid scanned by columns or alternate rows'
                             % rec)
        pl = np.asarray(pl, np.int64)
        start = np.concatenate(([0], np.cumsum(pl)[:-1]))         # each file row's
        j = np.searchsorted(start, f, side='right') - 1
        i = f - start[j]
        if scanm & 0x80:
            i = pl[j] - 1 - i
        if not scanm & 0x40:                    # rows south first: the last row's first
            out = np.concatenate(([0], np.cumsum(pl[::-1])[:-1]))[::-1]
            return out[j] + i
        return start[j] + i
    # wgrib2's ij2p, inverted: the file's rows (columns with 0x20) in order,
    # 0x80 turning each, 0x10 every other one back (by the file row's
    # parity), 0x40 south-up
    if scanm & 0x20:
        i, j = np.divmod(f, nj)
    else:
        j, i = np.divmod(f, ni)
    if scanm & 0x80:
        i = ni - 1 - i
    if scanm & 0x10:
        i = np.where(j & 1, ni - 1 - i, i)
    if not scanm & 0x40:
        j = nj - 1 - j
    return j * ni + i


def _u8(buf):
    """buf as the kernels take it: a read-only uint8 array. Read-only
    always, a map's or bytes' or an array of ours: numba compiles a kernel
    again for each kind (a writable copy cost _decode_one a second 5 s
    compile on first use)."""
    a = buf if isinstance(buf, np.ndarray) else np.frombuffer(buf, np.uint8)
    if a.flags.writeable:
        a = a.view()
        a.flags.writeable = False
    return a


# per-record decode parameters (int64 row of P)
(_DRT, _NVAL, _NBITS, _NG, _REFW, _BITSW, _REFL, _INCL, _LASTL, _BITSL,
 _ORDER, _EXTRA, _MGMT, _DATA, _DEST, _ROW, _FILL, _END) = range(18)


def decode(buf, recs, cols, out=None, undef=UNDEF, _min_run=None):
    """Decode `recs` onto the grid points `cols` (ascending flat we:sn
    indices without repeats, e.g. defined() of one of them): returns
    (len(recs), len(cols)) float32, row i for recs[i], `undef` (wgrib2's
    9.999e20 unless given, e.g. NaN) where a record has no value. Records
    must share one grid. `out` may be given to decode into; decode_iter()
    batches a long list."""
    arr = _u8(buf)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    if out is None:
        out = np.empty((len(recs), cols.size), dtype=np.float32)
    if not recs:
        return out
    P, F, D = _place(arr, recs, cols, undef)
    status = np.zeros(len(recs), dtype=np.int64)
    runs = _split(arr, P, out, undef, _min_run or _MIN_RUN)

    def dest(i):
        o = P[i, _DEST]
        return D[o:o + P[i, _NVAL]] if o >= 0 else D[:0]

    def one(i):
        status[i] = _whole(arr, P[i], F[i], D, recs[i], out[P[i, _ROW]], alone)

    def rest(i):                        # after run 0, serially (its seeds not all in it)
        cuts, st = runs[i]
        _span(arr, P[i], F[i], dest(i), out[P[i, _ROW]], False, 0, cuts[1, 0],
              P[i, _NG], st[0, _SK], st[0, _SBIT], st[0, _SLAST],
              st[0, _SPEN], st[0, _SCNT], st[-1])

    # Biggest first, a record per task: whichever core is free takes the
    # next, so long and short records, fast and slow cores even out. The
    # records split into runs go to tasks of their own, each taking runs
    # as it comes (_runs_work), queued first.
    by_size = sorted((i for i in range(len(recs)) if i not in runs),
                     key=lambda i: recs[i].s7 - recs[i].s7end)
    alone = len(by_size) == 1 and not runs
    work = []
    if runs:
        ids = np.array(sorted(runs), dtype=np.int64)
        nrun = [len(runs[i][1]) for i in ids]
        cuts3 = np.zeros((len(ids), max(nrun) + 1, 3), dtype=np.int64)
        st3 = np.zeros((len(ids), max(nrun), _NST), dtype=np.int64)
        flags = np.zeros((len(ids), max(nrun)), dtype=np.int64)
        for r, i in enumerate(ids):
            cuts3[r, :nrun[r] + 1] = runs[i][0]
            runs[i] = (cuts3[r, :nrun[r] + 1], st3[r, :nrun[r]])
        trec = np.repeat(np.arange(len(ids), dtype=np.int64), nrun)
        trun = np.concatenate([np.arange(m, dtype=np.int64) for m in nrun])
        args = (arr, P, F, D, out, ids, cuts3, st3, flags, trec, trun,
                np.zeros(2, dtype=np.int64), _REL_LIMIT, _SPINS)

        def runs_task(_):
            t = _runs_work(*args, -1)
            while t >= 0:               # waited a while: the core back to the OS a moment
                time.sleep(0)
                t = _runs_work(*args, t)
        # a task per two logical CPUs: the runs are taken before more
        # would start, and two on one core's hyperthreads only slow each
        # other (GFS 0.25 deg, a field: 0.73 ms and 3.7 ms of CPU with a
        # task per logical CPU and 2^15 spins, 0.72 and 2.9 so)
        work = [runs_task] * min(trec.size, max(2, threads() // 2))
    _pool_map(lambda t: one(t) if isinstance(t, int) else t(None), work + by_size)
    later = [i for i, (cuts, st) in runs.items()
             if P[i, _ORDER] > 0 and not st[0, _SERR] and st[0, _SCNT] < P[i, _ORDER]]
    for i in later:
        runs[i][1][1:, _SERR] = 0
    _pool_map(rest, later)
    for i, (cuts, st) in runs.items():
        status[i] = _runs_status(P[i], cuts, st)
        if P[i, _DEST] == _FLIP and not status[i]:      # decoded in file order
            _turn_rows(out[P[i, _ROW]], recs[i].grid.ni)
    bad = np.flatnonzero(status)
    if bad.size:
        raise ValueError('%r: the packed data do not add up (code %d)'
                         % (recs[bad[0]], status[bad[0]]))
    return out


def batch_size(ncols, nrecs, budget=None, target=BATCH_BYTES):
    """Records a batch of `ncols`-point fields: about `target` bytes of
    result -- a few dozen MB stay friendlier to caches and page locking
    than hundreds (GFS 0.5 deg: 0.096 ms a field at 256 MB, 0.092 at 64;
    on the GPU 0.122 and 0.098) -- but no fewer records than cores, which
    big fields need (NBM: 1.88 ms a field 2 at a time, 0.76 at 17), and
    no more than `budget` bytes (default_budget()). At least 1."""
    row = 4 * max(ncols, 1)
    budget = default_budget() if budget is None else budget
    n = max(target // row, threads())
    return int(max(1, min(nrecs, n, budget // row)))


def default_budget():
    """BUDGET_BYTES, or a sixteenth of the RAM free now when that is less
    (16 MB at the least)."""
    avail = memory()[1]
    return BUDGET_BYTES if not avail else int(min(BUDGET_BYTES, max(16 << 20, avail // 16)))


def memory():
    """(physical, available) RAM in bytes, 0 for what the OS will not say."""
    try:
        if sys.platform == 'win32':
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (k, ctypes.c_ulonglong) for k in ('phys', 'avail_phys', 'page', 'avail_page',
                                                      'virt', 'avail_virt', 'avail_ext')]
            m = MemoryStatus()
            m.length = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return int(m.phys), int(m.avail_phys)
        phys = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        avail = 0
        try:                                    # (macOS and the BSDs have no /proc:
            with open('/proc/meminfo') as f:    # what is free goes unsaid, not the total)
                for line in f:
                    if line.startswith('MemAvailable:'):
                        avail = int(line.split()[1]) << 10
        except OSError:
            pass
        return phys, avail
    except Exception:
        return 0, 0


def decode_iter(buf, recs, cols, budget=None, undef=UNDEF):
    """decode() over any number of records in one reused buffer of
    batch_size() rows, so memory stays flat: yields (i, block), block's
    rows for recs[i:i + len(block)], valid until the next is asked for.
    The buffer is a ring of up to RING_BLOCKS blocks, every record a task
    of its own on decode()'s threads: while you work on one block the
    others are decoding, so the threads do not run dry at each block's end
    (NBM, 17 rows: 0.63 ms a field as decode() calls of 17, ~0.4 so)."""
    arr = _u8(buf)
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    if not recs:
        return
    rows = batch_size(cols.size, len(recs), budget)
    nb = max(1, min(RING_BLOCKS, rows, len(recs)))
    per = rows // nb
    res = np.empty((nb * per, cols.size), dtype=np.float32)
    pool = _pool()
    queue = deque()
    for t, i in enumerate(range(0, len(recs), per)):
        if len(queue) == max(nb - 1, 1):    # the oldest out before its slot's next
            yield _block_done(*queue.popleft())
        part = recs[i:i + per]
        block = res[(t % nb) * per:(t % nb) * per + len(part)]
        P, F, D = _place(arr, part, cols, undef)
        queue.append((i, part, block,
                      [(j, pool.submit(_whole, arr, P[j], F[j], D, part[j],
                                       block[j])) for j in range(len(part))]))
    while queue:
        yield _block_done(*queue.popleft())


def _block_done(i, part, block, futs):
    """(i, block) once its records are in, or the first one's error."""
    for j, fut in futs:
        code = fut.result()
        if code:
            raise ValueError('%r: the packed data do not add up (code %d)'
                             % (part[j], code))
    return i, block


def _whole(arr, p, f, D, rec, row, alone=False):
    """One record decoded on one core into row: _decode_one's status. A
    north-up record placed by _FLIP decodes in file order, then has its
    rows turned round. alone: the call's only record (a JPEG2000 one may
    take every core)."""
    o = p[_DEST]
    dest = D[o:o + p[_NVAL]] if o >= 0 else D[:0]
    if p[_DRT] >= 40 or p[_DRT] == 4:           # JPEG2000, PNG, CCSDS, IEEE, spectral, RLE
        from . import _images
        code = _images.decode(arr, p, f, dest, row, alone)
    else:
        code = _decode_one(arr, p, f, dest, row)
    if o == _FLIP and not code:
        _turn_rows(row, rec.grid.ni)
    return code


# Column maps onto the whole grid, by (grid, bitmap): a north-up grid's
# row flip costs more to build than a small field takes to decode (GFS
# 0.5 deg: 5 ms against 0.2), so each is built once. Bounded, oldest out.
_MAPS = OrderedDict()
_MAPS_KEEP, _MAPS_BYTES = 4, 64 << 20
_MAPS_LOCK = threading.Lock()


_FLIP = -2       # _DEST of a record decoded in file order, then its rows turned round


def _place(arr, recs, cols, undef=UNDEF, flip=True):
    """Each record's decode parameters and where its values go: (P, F, D),
    P[i] with _DEST (offset of its column map in D, -1 when its defined
    points are `cols` in order) and _FILL (columns it does not reach start
    out undefined). F[i] = (R, 2^E, 10^-D, the value for undefined points).
    Records must share one grid's layout (_grids.layout). _gpu uses it too."""
    g0 = _check_grid(recs[0])
    lay = _grids.layout(g0)
    n = g0[0] * g0[1]
    # ascending and without repeats, so this is every point of the grid
    whole = cols.size == n and cols[0] == 0 and cols[-1] == n - 1
    pos = None                                  # we:sn index -> column
    maps, chunks, total = {}, [], 0
    P = np.zeros((len(recs), 18), dtype=np.int64)
    F = np.zeros((len(recs), 4), dtype=np.float64)
    F[:, 3] = undef
    for i, r in enumerate(recs):
        g = _check_grid(r)
        if g is not g0 and _grids.layout(g) != lay:
            raise ValueError('%r: a different grid from %r' % (r, recs[0]))
        bits = None if r.bitmap < 0 else arr[r.bitmap:r.bitmap + (n + 7) // 8].tobytes()
        key = r.bitmap if bits is None else bits
        if g.scan != g0.scan:                   # its points in another order: a map of its own
            key = (g.scan, key)
        hit = maps.get(key)
        if hit is None:
            if whole and flip and r.bitmap < 0 and g[2] & 0xF0 == 0 and not g.pl:
                # north-up rows, every point: no map -- decoded in file
                # order and turned round after (GFS 0.5 deg: 1.57 + 0.10 ns
                # a value against 1.80 through a map; MRMS's 24.5M-point
                # map was 98 MB, rebuilt every call)
                d, fill = _FLIP, False
            elif whole:
                d = _whole_map(arr, r, g, key)
                fill = d is not None and d.size < n
            else:
                d = _defined(arr, r)
                if d.size == cols.size and np.array_equal(d, cols):
                    d = None
                else:
                    if pos is None:
                        pos = np.full(n, -1, dtype=np.int32)
                        pos[cols] = np.arange(cols.size, dtype=np.int32)
                    d = pos[d]
                    fill = int(np.count_nonzero(d >= 0)) < cols.size
            # the points it defines: as many values as those, or damaged
            ndef = n if bits is None else _ones(np.frombuffer(bits, np.uint8), n)
            if d is None:
                hit = (-1, False, ndef)         # values land in order
            elif isinstance(d, int):
                hit = (d, False, ndef)
            else:
                chunks.append(d)
                # columns this record does not reach start out undefined
                hit = (total, fill, ndef)
                total += d.size
            maps[key] = hit
        P[i, _DEST], P[i, _FILL] = hit[:2]
        P[i, _ROW] = i
        _params(arr, r, P[i], F[i])
        if P[i, _NVAL] != hit[2]:               # a wrong count would place values out of bounds
            raise ValueError('%r: damaged -- %d values for %d defined points'
                             % (r, P[i, _NVAL], hit[2]))
    if len(chunks) == 1:
        D = chunks[0]                           # a kept map, as it is
    else:
        D = (np.concatenate(chunks) if chunks
             else np.zeros(1, dtype=np.int32))
    return P, F, D


def _whole_map(arr, r, g, key):
    """r's column map onto the whole grid (int32, kept), None when its
    values already land in order."""
    if r.bitmap < 0 and g[2] & 0xF0 == 0x40:
        return None
    k = (g, key)
    with _MAPS_LOCK:
        d = _MAPS.get(k)
        if d is not None:
            _MAPS.move_to_end(k)
            return d
    d = _defined(arr, r).astype(np.int32)
    if d.nbytes <= _MAPS_BYTES // _MAPS_KEEP:
        with _MAPS_LOCK:
            _MAPS[k] = d
            while len(_MAPS) > _MAPS_KEEP:
                _MAPS.popitem(last=False)
    return d


_MIN_RUN = 1 << 16                      # values in a run, at least
_REL_LIMIT = 1 << 31                    # a zero-state run's integers, below
_SPINS = 1 << 11                        # spin-waits before yielding the core (2^15
                                        # spun ~1 ms away: a third more CPU a read)
_SPLIT_DIV = 6                          # split when records x this <= cores,
_SPLIT_DIV_BIG = 4                      # or this, fields of _BIG values or more
_BIG = 1 << 20


def _split(arr, P, out, undef=UNDEF, min_run=_MIN_RUN):
    """The records decode() splits into runs of groups, decoded on several
    cores at once -- complex packing, in a call of a few records -- as
    {record: (cuts, run end states)}. A run per logical CPU.
    Runs cost more than whole records in all (a cut scan, a fix-up), so
    they pay only while whole records would leave cores idle: on 24
    threads up to 4 records a call, or 6 of a million values or more
    (NBM, 6 a call: 1.25 ms a field split, 1.43 not; GFS 0.5 deg, 260k
    values: 0.21 split, 0.19 not)."""
    ncpu = threads()
    runs = {}
    big = int(P[:, _NVAL].min()) >= _BIG
    if len(P) * (_SPLIT_DIV_BIG if big else _SPLIT_DIV) > ncpu:
        return runs
    want = -(-ncpu // len(P))
    for i in range(len(P)):
        nch = min(want, int(P[i, _NVAL]) // min_run)
        if P[i, _DRT] == 0 or P[i, _NG] == 0 or nch < 2:
            continue
        cuts = np.zeros((nch + 1, 3), dtype=np.int64)
        nch = _cuts(arr, P[i], nch, cuts)
        if nch < 2:                     # 0: lengths overrun; _decode_one says
            continue
        if P[i, _FILL]:
            out[P[i, _ROW]] = undef
        runs[i] = (cuts[:nch + 1], np.zeros((nch, _NST), dtype=np.int64))
    return runs


def _runs_status(p, cuts, st):
    """_decode_one's status for a record decoded in runs: each run ended
    where the next begins, and the last at the record's end."""
    err = st[:, _SERR][st[:, _SERR] != 0]
    if err.size:
        return int(err[0])
    if p[_ORDER] > 0 and st[0, _SCNT] < p[_ORDER]:    # run 0, then the rest
        ends = [(st[0], cuts[1]), (st[-1], cuts[-1])]
    else:
        ends = [(st[c], cuts[c + 1]) for c in range(len(st))]
    for s, cut in ends:
        if s[_SK] != cut[1] or s[_SBIT] != cut[2]:
            return 3
    if cuts[-1, 1] != p[_NVAL]:
        return 3
    if (cuts[-1, 2] + 7) // 8 > p[_END]:
        return 4
    return 0


_POOL = None
_POOL_LOCK = threading.Lock()


def _threads_from_env():
    v = os.environ.get('FASTGRIB_THREADS', '').strip()
    if not v:
        return 0
    try:
        return max(1, int(v))
    except ValueError:
        import warnings
        warnings.warn('FASTGRIB_THREADS=%r is not a number of threads: ignored' % v)
        return 0


_THREADS = _threads_from_env()


def threads():
    """How many threads decoding uses: set_threads()'s, else the
    environment variable FASTGRIB_THREADS, else one per logical CPU."""
    return _THREADS or os.cpu_count() or 1


def set_threads(n=None):
    """Decode on n threads (None: one per logical CPU) -- fewer leave the
    rest of the machine to other work, and where decoding is limited by
    memory's bandwidth cost little time (GFS 0.25 deg on 8 of 24: 4% slower,
    a fifth less CPU). Calls already running finish on the threads they
    have."""
    global _THREADS, _POOL
    if n is not None and (int(n) != n or n < 1):
        raise ValueError('set_threads: a whole number of threads, 1 or more (or None)')
    with _POOL_LOCK:
        _THREADS = 0 if n is None else int(n)
        old, _POOL = _POOL, None
    if old is not None:
        old.shutdown(wait=False)


def _pool_map(fn, items):
    """fn(item) for every item on decode()'s own threads, one per logical
    CPU and kept between calls, each taking the next item when it finishes
    one. Only kernels run on them, so a decode() waiting on them never
    holds one up."""
    if len(items) <= 1:
        for it in items:
            fn(it)
        return
    for _ in _pool().map(fn, items):
        pass


def share(work, n):
    """work() on the calling thread and on up to n - 1 of decode()'s
    threads at once -- for a nogil kernel whose callers take its items off
    a shared counter (_octets.take) until none is left, so a helper that
    starts late just takes fewer. The caller works too and, once it is
    done, calls off the helpers not yet started (there is nothing left for
    them): it never waits on a busy pool. (Numba's prange did this with
    threads that spin after each loop: a JPEG2000 field read alone took
    15-90 ms of CPU for 3 ms.)"""
    n = max(1, min(n, threads()))
    futs = [_pool().submit(work) for _ in range(n - 1)]
    try:
        work()
    finally:
        for f in futs:
            if not f.cancel():
                f.result()


def _pool():
    """decode()'s threads, made on first use."""
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(threads(), initializer=_fast_thread)
    return _POOL


def _fast_thread():
    """Declare this thread's work as the user's (high QoS), for Windows on
    a hybrid CPU: a process with no focused window of its own -- a script
    from a terminal -- counts as background work there, and its busy
    threads go to the efficiency cores (NBM decode_iter: 7.7 ms a field
    against 0.6). For decode()'s own threads only, not the caller's
    process; nothing elsewhere."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        from ctypes import wintypes

        class State(ctypes.Structure):
            _fields_ = [('Version', wintypes.ULONG), ('ControlMask', wintypes.ULONG),
                        ('StateMask', wintypes.ULONG)]
        k = ctypes.windll.kernel32
        k.GetCurrentThread.restype = wintypes.HANDLE
        st = State(1, 1, 0)      # execution speed: managed, not throttled
        k.SetThreadInformation(wintypes.HANDLE(k.GetCurrentThread()), 3,  # ThreadPowerThrottling
                               ctypes.byref(st), ctypes.sizeof(st))
    except Exception:
        pass


def _params(arr, r, p, f):
    b = _octets(arr)                            # (Python ints, numpy 1 or 2)
    s5 = r.s5
    drt = int(b[s5 + 9]) << 8 | int(b[s5 + 10])
    if drt not in (0, 2, 3, 4, 40, 41, 42, 50, 51, 200):
        raise ValueError('%r: data template 5.%d is not supported'
                         % (r, drt))
    p[_DRT] = drt
    p[_NVAL] = struct.unpack_from('>I', b, s5 + 5)[0]
    p[_DATA] = r.s7 + 5
    p[_END] = r.s7end
    if drt == 4:        # IEEE: 32 or 64 bits a value (wgrib2: 32 only)
        prec = b[s5 + 11]
        if prec not in (1, 2):
            raise ValueError('%r: IEEE precision %d not supported' % (r, prec))
        p[_NBITS] = 32 if prec == 1 else 64
        f[0], f[1], f[2] = 0.0, 1.0, 1.0
        return
    if drt in (50, 51):  # spherical harmonics: the coefficients, made by _spectral
        g = r.grid
        if g is None or g.template not in _grids._SPECTRAL:
            raise ValueError('%r: spectral packing of a grid of points' % r)
        p[_REFL], p[_NG], p[_REFW], p[_BITSW] = r.s5, g.params[0], g.params[1], g.params[2]
        f[0], f[1], f[2] = 0.0, 1.0, 1.0
        return
    if drt == 200:      # run-length (JMA): bits a value, top value, levels, their scale
        if r.bitmap >= 0:
            raise ValueError('%r: run-length packing with a bitmap is not supported' % r)
        p[_NBITS] = b[s5 + 11]
        p[_MGMT] = struct.unpack_from('>H', b, s5 + 12)[0]           # MV
        p[_EXTRA] = struct.unpack_from('>H', b, s5 + 14)[0]          # MVL: levels given
        dec = b[s5 + 16]
        dec = -(dec - 128) if dec > 127 else dec
        f[0], f[1], f[2] = 0.0, 1.0, _int_power(10.0, -dec)
        p[_REFL] = s5 + 17                                           # the levels' table
        if (p[_NBITS] == 0 or p[_NBITS] > 32 or p[_MGMT] > p[_EXTRA]
                or s5 + 17 + 2 * p[_EXTRA] > r.s6):
            raise ValueError('%r: damaged run-length packing parameters' % r)
        return
    ref0 = struct.unpack_from('>f', b, s5 + 11)[0]   # float -> double
    e = _sm(struct.unpack_from('>H', b, s5 + 15)[0], 2)
    d = _sm(struct.unpack_from('>H', b, s5 + 17)[0], 2)
    f[0], f[1], f[2] = ref0, _int_power(2.0, e), _int_power(10.0, -d)
    p[_NBITS] = b[s5 + 19]
    if p[_NBITS] > 32:
        raise ValueError('%r: %d bits a value is not supported' % (r, p[_NBITS]))
    if drt == 42:       # CCSDS: its flags, block size and reference sample interval
        p[_MGMT], p[_REFW], p[_BITSW] = b[s5 + 21], b[s5 + 22], int(b[s5 + 23]) << 8 | b[s5 + 24]
    if drt in (2, 3):
        if b[s5 + 21] != 1:
            raise ValueError('%r: group splitting method %d not supported'
                             % (r, b[s5 + 21]))
        p[_MGMT] = b[s5 + 22]
        p[_NG] = struct.unpack_from('>I', b, s5 + 31)[0]
        p[_REFW], p[_BITSW] = b[s5 + 35], b[s5 + 36]
        p[_REFL] = struct.unpack_from('>I', b, s5 + 37)[0]
        p[_INCL] = b[s5 + 41]
        p[_LASTL] = struct.unpack_from('>I', b, s5 + 42)[0]
        p[_BITSL] = b[s5 + 46]
        if drt == 3:
            p[_ORDER], p[_EXTRA] = b[s5 + 47], b[s5 + 48]
            if p[_ORDER] not in (1, 2):
                raise ValueError('%r: spatial differencing order %d'
                                 % (r, p[_ORDER]))
        if _layout(arr, p)[3] > p[_END]:
            raise ValueError('%r: damaged -- its group headers run past its data' % r)


@njit(cache=True, nogil=True)
def _groups_ok(buf, p):
    """Whether a complex-packed record's group lengths add up to its
    values and their bits fit in its data -- what decode() finds as it
    goes, checked ahead for the GPU (whose reads stay in the record's bytes
    whatever they say)."""
    ng, n = p[_NG], p[_NVAL]
    _, bw, bl, q = _layout(buf, p)[:4]
    refw, bitsw = p[_REFW], p[_BITSW]
    refl, incl, bitsl, lastl = p[_REFL], p[_INCL], p[_BITSL], p[_LASTL]
    k = bits = 0
    for g in range(ng):
        gw = refw + _bits(buf, bw + g * bitsw, bitsw)
        gl = lastl if g == ng - 1 else refl + _bits(buf, bl + g * bitsl, bitsl) * incl
        k += gl
        bits += gw * gl
    return k == n and q * 8 + bits <= p[_END] * 8


@njit(cache=True, nogil=True)
def _put(row, dest, usedest, k, y):
    """Defined point k's value into its column. Unsigned indices: numba
    then adds no negative-index fix-up to each store."""
    u = np.uint64(k)
    if usedest:
        c = dest[u]
        if c >= 0:
            row[np.uint64(c)] = y
    else:
        row[u] = y


@njit(cache=True, nogil=True)
def _turn_rows(row, ni):
    """The rows of ni values in reverse order, in place."""
    nj = row.size // ni
    for j in range(nj // 2):
        a = row[j * ni:(j + 1) * ni]
        b = row[(nj - 1 - j) * ni:(nj - j) * ni]
        for i in range(ni):
            a[i], b[i] = b[i], a[i]


@njit(cache=True, nogil=True)
def _simple(buf, p, f, dest, row):
    """Simple packing (5.0, and PNG's and CCSDS's samples): nbits a value.
    A function of its own: inside _decode_one, beside all of complex
    packing, LLVM made this loop 15x slower (19 ns a value against 1.3)."""
    n, nbits = p[_NVAL], p[_NBITS]
    usedest = p[_DEST] >= 0
    r0, f2, f10 = f[0], f[1], f[2]
    bit = p[_DATA] * 8
    if bit + n * nbits > p[_END] * 8:           # fewer bits than values say: corrupt
        return 1
    # wgrib2 widens the packed integer through a float (exact to 24 bits;
    # it rounds a 25-bit one, and so does this). A chunk at a time: its
    # integers (int32 when they widen: AVX2 converts those packed), then
    # its floats in a loop of their own, which vectorises; the values whose
    # 8-byte load stays in buf (all but the last few) read with no bounds
    # test (IFS, 12 bits: 1.33 -> 0.72 ns a value, its rows turned included)
    widen = nbits <= 25
    nfast = min(n, max(0, -(-((buf.size - 7) * 8 - bit) // nbits)))
    sh = np.uint64(64 - nbits)
    t32 = np.empty(_CHUNK if widen else 0, np.int32)
    t64 = np.empty(0 if widen else _CHUNK, np.int64)
    k = 0
    while k < n:
        c = min(_CHUNK, n - k)
        b0 = bit + k * nbits
        cf = min(c, max(0, nfast - k))         # values of the chunk read unchecked
        if widen:
            for t in range(cf):
                bb = np.uint64(b0 + t * nbits)
                t32[t] = np.int32((_be64(buf, bb >> np.uint64(3))
                                   << (bb & np.uint64(7))) >> sh)
            for t in range(cf, c):
                t32[t] = _bits(buf, b0 + t * nbits, nbits)
        else:
            for t in range(cf):
                bb = np.uint64(b0 + t * nbits)
                t64[t] = np.int64((_be64(buf, bb >> np.uint64(3))
                                   << (bb & np.uint64(7))) >> sh)
            for t in range(cf, c):
                t64[t] = _bits(buf, b0 + t * nbits, nbits)
        if usedest:
            for t in range(c):
                x = np.float64(np.float32(t32[t])) if widen else np.float64(t64[t])
                _put(row, dest, True, k + t, np.float32((r0 + f2 * x) * f10))
        elif widen:
            o = row[k:k + c]
            for t in range(c):
                o[t] = np.float32((r0 + f2 * np.float64(np.float32(t32[t]))) * f10)
        else:
            o = row[k:k + c]
            for t in range(c):
                o[t] = np.float32((r0 + f2 * np.float64(t64[t])) * f10)
        k += c
    return 0 if (bit + n * nbits + 7) // 8 <= p[_END] else 1


_CHUNK = 2048                           # _simple's values a chunk


@njit(cache=True, nogil=True)
def _decode_one(buf, p, f, dest, row):
    """One record, in wgrib2's order of operations; 0 when the packing
    added up."""
    n = p[_NVAL]
    usedest = p[_DEST] >= 0
    undefv = np.float32(f[3])                   # undefined points' value
    if p[_FILL]:
        row[:] = undefv
    r0, f10 = f[0], f[2]
    nbits = p[_NBITS]
    if (p[_DRT] == 0 and nbits == 0) or (p[_DRT] != 0 and p[_NG] == 0):
        y = np.float32(r0 * f10)                # a constant field
        for k in range(n):
            _put(row, dest, usedest, k, y)
        return 0
    if p[_DRT] == 0:
        return _simple(buf, p, f, dest, row)

    st = np.zeros(_NST, dtype=np.int64)
    _span(buf, p, f, dest, row, False, 0, 0, p[_NG], 0,
          _layout(buf, p)[3] * 8, 0, 0, 0, st)
    if st[_SERR]:
        return st[_SERR]
    if st[_SK] != n:
        return 3
    if (st[_SBIT] + 7) // 8 > p[_END]:
        return 4
    return 0


# where a run of groups ends (int64 row of a state array): its next
# defined point, the bit after its packed values, the spatial differencing
# state there (last two values, values so far), the run's sums (_sums), a
# nonzero code if its packing did not add up, and what _chain_end has left
# to do (_chain_run's result)
_SK, _SBIT, _SLAST, _SPEN, _SCNT, _SM, _SS1, _SS2, _SERR, _SFIX, _NST = range(11)


@njit(cache=True, nogil=True)
def _layout(buf, p):
    """Where a complex-packed record's parts lie: (bit of the group
    references, of the group widths, of the group lengths, byte of the
    packed values, first seed, second seed, minimum difference)."""
    ng = p[_NG]
    eo = p[_EXTRA]
    q = p[_DATA]
    e0 = e1 = 0
    minv = 0
    if eo > 0:
        for k in range(eo):
            e0 = (e0 << 8) | np.int64(buf[q + k])
        q += eo
        if p[_ORDER] == 2:
            for k in range(eo):
                e1 = (e1 << 8) | np.int64(buf[q + k])
            q += eo
        sgn = buf[q] & 0x80
        minv = np.int64(buf[q] & 0x7F)
        for k in range(1, eo):
            minv = (minv << 8) | np.int64(buf[q + k])
        if sgn:
            minv = -minv
        q += eo
    bref = q * 8
    bw = bref + ((p[_NBITS] * ng + 7) // 8) * 8
    bl = bw + ((p[_BITSW] * ng + 7) // 8) * 8
    q = (bl + ((p[_BITSL] * ng + 7) // 8) * 8) // 8
    return bref, bw, bl, q, e0, e1, minv


@njit(cache=True, nogil=True)
def _seek(buf, bit):
    """The streaming bit reader's state (next byte, bits held, how many)
    for reading on from bit `bit`."""
    q = bit >> 3
    s = bit & 7
    if s == 0:
        return q, 0, 0
    return q + 1, np.int64(buf[q]) & ((1 << (8 - s)) - 1), 8 - s


@njit(cache=True, nogil=True)
def _zero_bits(buf, bit, n):
    """Whether the n bits from bit `bit` are all zero: up to 56 in one
    read (most groups' bits), else byte by byte."""
    if n <= 0:
        return True
    if n <= 56 and (bit >> 3) + 8 <= buf.size:
        return _bits(buf, bit, n) == 0
    b0, b1 = bit >> 3, (bit + n - 1) >> 3
    head = 0xFF >> (bit & 7)
    tail = (0xFF << (7 - ((bit + n - 1) & 7))) & 0xFF
    if b0 == b1:
        return buf[b0] & head & tail == 0
    if buf[b0] & head or buf[b1] & tail:
        return False
    for b in range(b0 + 1, b1):
        if buf[b]:
            return False
    return True


# 1.0, 2.0, ...: _ramp's positions as doubles from a table, since AVX2 has
# no packed int64 -> double conversion and with one LLVM leaves the loop
# scalar (measured 0.56 against 0.18 ns a value)
_RAMP = np.arange(1.0, 4097.0)


@njit(cache=True, nogil=True)
def _ramp(row, dest, usedest, k, gl, last, pen, base, r0, f2, f10):
    """gl values of a group whose differences are all `base` (second-order
    spatial differencing, both seeds in), from (last, pen); returns the
    new (last, pen). The values are last + j d + base j (j + 1) / 2 for
    j = 1..gl, d = last - pen: worked out in doubles while every term and
    partial sum stays below 2^53, so each is exactly the double of the
    integer the recurrence gives, and the loop has no carried dependence
    (it vectorises); otherwise -- or for a short group, where setting that
    up costs more than it saves (~18 ns: 30 values 26 ns against 19) --
    the recurrence itself."""
    if gl < _RAMP_MIN:
        return _ramp_steps(row, dest, usedest, k, gl, last, pen, base,
                           r0, f2, f10)
    d = last - pen
    fl, fd, fb = np.float64(last), np.float64(d), np.float64(base)
    fg = np.float64(gl)
    if (not usedest and fg < 67108864.0 and abs(fl) + fg * abs(fd)  # 2^26
            + abs(fb) * fg * (fg + 1.0) * 0.5 < 4503599627370496.0):  # 2^52
        out = row[k:k + gl]
        if gl <= _RAMP.size:                    # the usual: one block
            for t in range(gl):
                x = _RAMP[t]
                v = (fl + x * fd) + fb * (x * (x + 1.0) * 0.5)
                out[t] = np.float32((r0 + v * f2) * f10)
            t = gl * (gl + 1) // 2
            return (last + gl * d + base * t,
                    last + (gl - 1) * d + base * (t - gl))
        j0 = 0
        while j0 < gl:
            c = min(gl - j0, _RAMP.size)
            o = out[j0:j0 + c]
            fo = np.float64(j0)
            for t in range(c):
                x = _RAMP[t] + fo
                v = (fl + x * fd) + fb * (x * (x + 1.0) * 0.5)
                o[t] = np.float32((r0 + v * f2) * f10)
            j0 += c
        t = gl * (gl + 1) // 2
        return (last + gl * d + base * t,
                last + (gl - 1) * d + base * (t - gl))
    return _ramp_steps(row, dest, usedest, k, gl, last, pen, base,
                       r0, f2, f10)


_RAMP_MIN = 48                          # shorter ramps go step by step


@njit(cache=True, nogil=True)
def _ramp_steps(row, dest, usedest, k, gl, last, pen, base, r0, f2, f10):
    """_ramp by the recurrence, value by value."""
    for j in range(gl):
        v = base + 2 * last - pen               # integers: exact
        pen = last
        last = v
        _put(row, dest, usedest, k + j,
             np.float32((r0 + np.float64(v) * f2) * f10))
    return last, pen


@njit(cache=True, nogil=True)
def _puti(irow, dest, usedest, k, v):
    """Defined point k's zero-state integer into its column (int32)."""
    u = np.uint64(k)
    if usedest:
        c = dest[u]
        if c >= 0:
            irow[np.uint64(c)] = v
    else:
        irow[u] = v


@njit(cache=True, nogil=True)
def _ramp_rel(irow, dest, usedest, k, gl, last, pen, base, lim):
    """_ramp's integers for a zero-state run, as int32; returns (last,
    pen, whether they all stayed below lim)."""
    d = last - pen
    fl, fd, fb = np.float64(last), np.float64(d), np.float64(base)
    fg = np.float64(gl)
    t = gl * (gl + 1) // 2
    if abs(fl) + fg * abs(fd) + abs(fb) * fg * (fg + 1.0) * 0.5 >= lim:
        return last, pen, False         # perhaps not; _sums settles it
    if not usedest:
        out = irow[k:k + gl]
        j0 = 0
        while j0 < gl:
            c = min(gl - j0, _RAMP.size)
            o = out[j0:j0 + c]
            fo = np.float64(j0)
            for u in range(c):
                x = _RAMP[u] + fo
                o[u] = np.int32((fl + x * fd) + fb * (x * (x + 1.0) * 0.5))
            j0 += c
    else:
        v, w = last, pen
        for j in range(gl):
            v, w = base + 2 * v - w, v
            _puti(irow, dest, usedest, k + j, v)
    return (last + gl * d + base * t,
            last + (gl - 1) * d + base * (t - gl), True)


@njit(cache=True, nogil=True)
def _fix(row, dest, usedest, k0, k1, order, last, pen, f):
    """A zero-state run's defined points [k0, k1) to their values: the
    j-th integer v' becomes v' + last + (j + 1)(last - pen) (first order:
    v' + last), in place, then wgrib2's float. In doubles while those stay
    exact (below 2^52), a block at a time through a buffer so the loop
    vectorises (the row is read as int32 and written as float32)."""
    r0, f2, f10 = f[0], f[1], f[2]
    irow = row.view(np.int32)
    d = last - pen if order == 2 else 0
    n = k1 - k0
    if (not usedest and abs(np.float64(last)) + (n + 1.0) * abs(np.float64(d))
            + 2147483648.0 < 4503599627370496.0):
        fl, fd = np.float64(last), np.float64(d)
        tmp = np.empty(_RAMP.size, dtype=np.int32)
        j0 = 0
        while j0 < n:
            c = min(n - j0, _RAMP.size)
            tmp[:c] = irow[k0 + j0:k0 + j0 + c]
            o = row[k0 + j0:k0 + j0 + c]
            fo = np.float64(j0)
            for u in range(c):
                v = (np.float64(tmp[u]) + fl) + (_RAMP[u] + fo) * fd
                o[u] = np.float32((r0 + v * f2) * f10)
            j0 += c
        return
    for j in range(n):
        k = k0 + j
        c = dest[k] if usedest else k
        if c >= 0:
            v = np.int64(irow[c]) + last + (j + 1) * d
            row[c] = np.float32((r0 + np.float64(v) * f2) * f10)


@njit(cache=True, nogil=True)
def _group(buf, p, bref, bw, bl, g):
    """Group g's (reference, width, length)."""
    nbits, bitsw, bitsl = p[_NBITS], p[_BITSW], p[_BITSL]
    gref = _bits(buf, bref + g * nbits, nbits)
    gw = p[_REFW] + _bits(buf, bw + g * bitsw, bitsw)
    if g == p[_NG] - 1:
        gl = p[_LASTL]
    else:
        gl = p[_REFL] + _bits(buf, bl + g * bitsl, bitsl) * p[_INCL]
    return gref, gw, gl


@njit(cache=True, nogil=True)
def _cuts(buf, p, nch, cuts):
    """Splits a complex-packed record into up to `nch` runs of whole
    groups holding about equally many values: cuts[c] = (first group,
    first defined point, bit of the first packed value) of run c, the
    row after the last run the end. Returns the number of runs, 0 when
    the group lengths overrun the record. (Equally many groups, tallied in
    parallel, came out slower: group lengths cluster by region, so the
    runs' work did not even out -- NBM 1.96 ms a field against 1.51 --
    and weighing the work by values, packed bits and groups gained
    nothing over values alone.)"""
    n, ng = p[_NVAL], p[_NG]
    _, bw, bl, q = _layout(buf, p)[:4]
    refw, bitsw = p[_REFW], p[_BITSW]
    refl, incl, bitsl = p[_REFL], p[_INCL], p[_BITSL]
    k, bit, c = 0, q * 8, 0
    cuts[0, 0], cuts[0, 1], cuts[0, 2] = 0, k, bit
    lastl = p[_LASTL]
    nxt = n // nch if nch > 1 else n + 1        # where the next run starts
    for g in range(ng):
        if k >= nxt:
            c += 1
            cuts[c, 0], cuts[c, 1], cuts[c, 2] = g, k, bit
            nxt = (c + 1) * n // nch if c + 1 < nch else n + 1
        gw = refw + _bits(buf, bw + g * bitsw, bitsw)
        gl = lastl if g == ng - 1 else refl + _bits(buf, bl + g * bitsl, bitsl) * incl
        if k + gl > n:
            return 0
        k += gl
        bit += gw * gl
    c += 1
    cuts[c, 0], cuts[c, 1], cuts[c, 2] = ng, k, bit
    return c


@njit(cache=True, nogil=True)
def _sums(buf, p, ga, gb, bit, st):
    """Groups [ga, gb) of a spatially differenced record, summed as if
    both seeds were in: m values with differences w (raw + reference +
    minimum), st[_SS1] = sum of w and st[_SS2] = sum of its running sums.
    A run starting from (last, pen) then ends at last + m (last - pen) +
    S2, with last - pen grown by S1 -- in integers, so exactly."""
    bref, bw, bl, _, _, _, minv = _layout(buf, p)
    mgmt = p[_MGMT]
    allone = (1 << p[_NBITS]) - 1
    q, acc, nacc = _seek(buf, bit)
    m = s1 = s2 = 0
    endbit = p[_END] * 8
    for g in range(ga, gb):
        gref, gw, gl = _group(buf, p, bref, bw, bl, g)
        if q * 8 - nacc + gl * gw > endbit:     # corrupt: the run's end check reports it
            break
        zeros = gw == 1 and _zero_bits(buf, q * 8 - nacc, gl)
        if zeros:
            # one difference written as 1-bit zeros (see _span)
            q, acc, nacc = _seek(buf, q * 8 - nacc + gl)
            if mgmt == 2:
                continue                        # missing values
        if gw == 0 or zeros:
            if gw == 0 and mgmt > 0 and (
                    gref == allone or (mgmt == 2 and gref == allone - 1)):
                continue                        # missing values
            w = gref + minv                     # closed form for gl of w
            s2 += gl * s1 + w * (gl * (gl + 1) // 2)
            s1 += gl * w
            m += gl
            continue
        mw = (1 << gw) - 1
        m1 = mw if mgmt > 0 else -1
        m2 = mw - 1 if mgmt == 2 else -1
        base = gref + minv
        for j in range(gl):
            while nacc < gw:
                acc = (acc << 8) | np.int64(buf[q])
                q += 1
                nacc += 8
            nacc -= gw
            raw = (acc >> nacc) & mw
            acc &= (1 << nacc) - 1
            if raw != m1 and raw != m2:
                s1 += raw + base
                s2 += s1
                m += 1
    st[_SM], st[_SS1], st[_SS2] = m, s1, s2


@njit(cache=True, nogil=True)
def _span(buf, p, f, dest, row, rel, lim, ga, gb, k, bit, last, pen, cnt,
          st):
    """Groups [ga, gb) of a complex-packed record, from defined point k
    and packed-value bit `bit`, with the spatial differencing's state
    (last, pen, cnt) there; leaves where it ended in st. rel: the run's
    integers (seeds in, no missing values) go into the row as int32 for
    _fix, st[_SERR] -1 if one is missing or reaches `lim`. The reader's
    state is the bit itself; only the rare byte-stream loops (fields past
    56 bits, the buffer's last bytes, the seeds) stream."""
    irow = row.view(np.int32)
    n = p[_NVAL]
    usedest = p[_DEST] >= 0
    r0, f2, f10 = f[0], f[1], f[2]
    undefv = np.float32(f[3])                   # undefined points' value
    nbits = p[_NBITS]
    order = p[_ORDER]
    mgmt = p[_MGMT]
    bref, bw, bl, _, e0, e1, minv = _layout(buf, p)
    allone = (1 << nbits) - 1
    b = bit
    endbit = p[_END] * 8
    for g in range(ga, gb):
        gref, gw, gl = _group(buf, p, bref, bw, bl, g)
        if k + gl > n:
            st[_SERR] = 2
            return
        if b + gl * gw > endbit:                # corrupt widths or lengths: no reading on
            st[_SERR] = 4
            return
        mw = (1 << gw) - 1
        gmiss = gw == 0 and mgmt > 0 and (
            gref == allone or (mgmt == 2 and gref == allone - 1))
        if order == 2 and cnt > 1:
            # The common case (NCEP's: second-order differences, both
            # seeds in), as tight loops by kind of group. The same integer
            # and float operations as the general loop below, in the same
            # order; cnt only matters until it passes 1, so it stops here.
            base = gref + minv
            if gw == 1 and _zero_bits(buf, b, gl):
                # Every value 0: a group of one difference, which packers
                # that never write width 0 (NBM's) send as 1-bit zeros --
                # most of their values. Missing instead under secondary
                # missing values, where 0 is that value at 1 bit.
                b += gl
                gmiss = mgmt == 2
                gw = 0
            if gmiss:                           # a group of missing values
                if rel:
                    st[_SERR] = -1
                    return
                for j in range(gl):
                    _put(row, dest, usedest, k, undefv)
                    k += 1
                continue
            if gw == 0:                         # every difference the same
                if rel:
                    last, pen, ok = _ramp_rel(irow, dest, usedest, k, gl,
                                              last, pen, base, lim)
                    if not ok:
                        st[_SERR] = -1
                        return
                else:
                    last, pen = _ramp(row, dest, usedest, k, gl, last, pen,
                                      base, r0, f2, f10)
                k += gl
                continue
            # raw values meaning missing; -1 never comes out of the stream
            m1 = mw if mgmt > 0 else -1
            m2 = mw - 1 if mgmt == 2 else -1
            if gw <= 56 and ((b + gl * gw) >> 3) + 8 <= buf.size:
                # Each value one big-endian 8-byte load at its own bit and
                # two shifts: no refill branches, and no value's position
                # waits on the one before (2.4 -> 1.8 ns a value on GFS)
                sh = np.uint64(64 - gw)
                if rel:
                    for j in range(gl):
                        raw = np.int64((_be64(buf, b >> 3) << np.uint64(b & 7)) >> sh)
                        b += gw
                        v = raw + base + 2 * last - pen
                        if raw == m1 or raw == m2 or v >= lim or v < -lim:
                            st[_SERR] = -1
                            return
                        pen = last
                        last = v
                        _puti(irow, dest, usedest, k, v)
                        k += 1
                elif m1 < 0 and not usedest:
                    # no missing values, values in order (most fields): no
                    # test or map lookup a value (GFS: 1.61 -> 1.49 ns a value)
                    o = row[k:k + gl]
                    for j in range(gl):
                        raw = np.int64((_be64(buf, b >> 3) << np.uint64(b & 7)) >> sh)
                        b += gw
                        v = raw + base + 2 * last - pen
                        pen = last
                        last = v
                        o[j] = np.float32((r0 + np.float64(v) * f2) * f10)
                    k += gl
                else:
                    for j in range(gl):
                        raw = np.int64((_be64(buf, b >> 3) << np.uint64(b & 7)) >> sh)
                        b += gw
                        if raw == m1 or raw == m2:
                            y = undefv
                        else:
                            v = raw + base + 2 * last - pen
                            pen = last
                            last = v
                            y = np.float32((r0 + np.float64(v) * f2) * f10)
                        _put(row, dest, usedest, k, y)
                        k += 1
                continue
            q, acc, nacc = _seek(buf, b)
            for j in range(gl):
                while nacc < gw:
                    acc = (acc << 8) | np.int64(buf[q])
                    q += 1
                    nacc += 8
                nacc -= gw
                raw = (acc >> nacc) & mw
                acc &= (1 << nacc) - 1
                if rel:
                    v = raw + base + 2 * last - pen
                    if raw == m1 or raw == m2 or v >= lim or v < -lim:
                        st[_SERR] = -1
                        return
                    pen = last
                    last = v
                    _puti(irow, dest, usedest, k, v)
                    k += 1
                    continue
                if raw == m1 or raw == m2:
                    y = undefv
                else:
                    v = raw + base + 2 * last - pen
                    pen = last
                    last = v
                    y = np.float32((r0 + np.float64(v) * f2) * f10)
                _put(row, dest, usedest, k, y)
                k += 1
            b = q * 8 - nacc
            continue
        if order == 2 and not gmiss and (gw == 0 or (
                gw == 1 and mgmt != 2 and _zero_bits(buf, b, gl))):
            # The seeds inside a group of one difference -- a whole field
            # of one (NBM's all-zero precipitation, 3.7M values) can be a
            # single group: the seeds one by one, then the rest a ramp.
            if gw == 1:
                b += gl
            j = 0
            while j < gl and cnt < 2:
                v = e0 if cnt == 0 else e1
                pen = last
                last = v
                cnt += 1
                _put(row, dest, usedest, k,
                     np.float32((r0 + np.float64(v) * f2) * f10))
                k += 1
                j += 1
            if j < gl:
                last, pen = _ramp(row, dest, usedest, k, gl - j, last, pen,
                                  gref + minv, r0, f2, f10)
                k += gl - j
            continue
        q, acc, nacc = _seek(buf, b)
        for j in range(gl):
            raw = 0
            miss = gmiss
            if gw > 0:
                while nacc < gw:
                    acc = (acc << 8) | np.int64(buf[q])
                    q += 1
                    nacc += 8
                nacc -= gw
                raw = (acc >> nacc) & mw
                acc &= (1 << nacc) - 1
                if mgmt > 0 and (raw == mw or (mgmt == 2 and raw == mw - 1)):
                    miss = True
            if miss and rel:
                st[_SERR] = -1
                return
            if miss:
                y = undefv
            else:
                v = raw + gref
                if order == 1:
                    if cnt > 0:
                        v = v + last + minv
                    else:
                        v = e0
                    last = v
                elif order == 2:
                    if cnt > 1:
                        v = v + minv + last + last - pen
                    elif cnt == 1:
                        v = e1
                    else:
                        v = e0
                    pen = last
                    last = v
                cnt += 1
                if rel:
                    if v >= lim or v < -lim:
                        st[_SERR] = -1
                        return
                    _puti(irow, dest, usedest, k, v)
                    k += 1
                    continue
                y = np.float32((r0 + np.float64(v) * f2) * f10)
            _put(row, dest, usedest, k, y)
            k += 1
        b = q * 8 - nacc
    st[_SK], st[_SBIT] = k, b
    st[_SLAST], st[_SPEN], st[_SCNT] = last, pen, cnt


# A run's flag: 0 running, 1 ended as it should (its sums, run 0 its end,
# in its state row), 2 not. Stored with release and read with acquire
# ordering, so a run that sees 1 sees the sums; read in a loop with the
# CPU's spin-wait hint between, so a waiting hyperthread leaves its core
# to the other.
# (Windows' platform.machine() runs a process: its environment says it)
_X86 = (os.environ.get('PROCESSOR_ARCHITECTURE', '') if sys.platform == 'win32'
        else os.uname().machine).lower() in ('amd64', 'x86_64', 'i386', 'i686', 'x86')


@intrinsic
def _flag_get(typingctx, flags, i):
    from numba import types
    from numba.core import cgutils
    def codegen(context, builder, sig, args):
        ary = context.make_array(sig.args[0])(context, builder, args[0])
        ptr = cgutils.get_item_pointer(context, builder, sig.args[0], ary,
                                       [args[1]])
        return builder.load_atomic(ptr, 'acquire', 8)
    return types.int64(flags, i), codegen


@intrinsic
def _flag_set(typingctx, flags, i, v):
    from numba import types
    from numba.core import cgutils
    def codegen(context, builder, sig, args):
        ary = context.make_array(sig.args[0])(context, builder, args[0])
        ptr = cgutils.get_item_pointer(context, builder, sig.args[0], ary,
                                       [args[1]])
        builder.store_atomic(args[2], ptr, 'release', 8)
        return context.get_dummy_value()
    return types.void(flags, i, v), codegen


@intrinsic
def _spin_hint(typingctx):
    from llvmlite import ir
    from numba import types
    from numba.core import cgutils
    def codegen(context, builder, sig, args):
        if _X86:                        # elsewhere the atomic load is enough
            fn = cgutils.get_or_insert_function(
                builder.module, ir.FunctionType(ir.VoidType(), []),
                'llvm.x86.sse2.pause')
            builder.call(fn, [])
        return context.get_dummy_value()
    return types.void(), codegen


@njit(cache=True, nogil=True)
def _chain_run(buf, p, f, dest, row, cuts, st, flags, c, lim):
    """Run c of a record split for decode(), spatially differenced. Run 0
    decodes from the seeds; the others at the same time as if the
    differencing started from zero, keeping those exact integers in the
    row (int32). The differencing is linear, so a run that starts from
    (last, pen) holds v' + last + (j + 1)(last - pen) at its j-th value,
    and the end it reached from zero gives its sums (S2, S2 - S1 after m
    values). A run that meets missing values or leaves int32 sums its
    groups instead (_sums). Returns what _chain_end has left to do: 1 put
    the start into the values, 0 decode from it, -1 nothing."""
    order = p[_ORDER]
    s = st[c]
    if c == 0:
        _span(buf, p, f, dest, row, False, 0, cuts[0, 0], cuts[1, 0],
              cuts[0, 1], cuts[0, 2], 0, 0, 0, s)
        s[_SFIX] = -1
        _flag_set(flags, 0, 1 if s[_SERR] == 0 and s[_SCNT] >= order else 2)
        return -1
    _span(buf, p, f, dest, row, True, lim, cuts[c, 0], cuts[c + 1, 0],
          cuts[c, 1], cuts[c, 2], 0, 0, 2, s)
    fix = 1 if s[_SERR] == 0 else 0
    if s[_SERR] < 0:                    # missing values, or past int32
        s[_SERR] = 0
        _sums(buf, p, cuts[c, 0], cuts[c + 1, 0], cuts[c, 2], s)
    elif fix:
        s[_SM] = s[_SK] - cuts[c, 1]
        s[_SS1] = s[_SLAST] - s[_SPEN]
        s[_SS2] = s[_SLAST]
    s[_SFIX] = -1 if s[_SERR] else fix          # (published with the flag)
    _flag_set(flags, c, 1 if s[_SERR] == 0 else 2)
    return s[_SFIX]


@njit(cache=True, nogil=True)
def _dest_of(D, p):
    o = p[_DEST]
    return D[o:o + p[_NVAL]] if o >= 0 else D[:0]


@njit(cache=True, nogil=True)
def _runs_work(buf, P, F, D, out, ids, cuts, st, flags, trec, trun, ctr, lim,
               spins, held):
    """decode()'s split records' runs (task t: record ids[trec[t]], run
    trun[t]; cuts, st and flags stacked a record each), taken by whichever
    of decode()'s threads call this, as they come: each takes a run's
    first half (_chain_run) off ctr[0] until none is left, then a second
    half (_chain_end) off ctr[1]. A thread that starts late, or runs on a
    slower core, just takes fewer -- no run waits for a particular thread
    to start, as one task a run on the pool did (each began ~30 us after
    the last: a Python call apiece, one at a time under the GIL). Every
    first half is taken before any second, and each second half waits
    only on first halves already taken, so they all end. Returns -1 when
    nothing is left; else, after `spins` waits, the task it is still
    waiting on, to call again with as `held` once the core has been given
    up a moment."""
    T = trec.size
    if held < 0:
        while True:
            t = _take(ctr, 0)
            if t >= T:
                break
            r, c = trec[t], trun[t]
            p = P[ids[r]]
            cu, s = cuts[r], st[r]
            if p[_ORDER] > 0:
                _chain_run(buf, p, F[ids[r]], _dest_of(D, p), out[p[_ROW]], cu, s,
                           flags[r], c, lim)
            else:                               # no differencing: runs stand alone
                _span(buf, p, F[ids[r]], _dest_of(D, p), out[p[_ROW]], False, 0,
                      cu[c, 0], cu[c + 1, 0], cu[c, 1], cu[c, 2], 0, 0, 2, s[c])
                s[c, _SFIX] = -1
                _flag_set(flags[r], c, 1)
    t = held
    while True:
        if t < 0:
            t = _take(ctr, 1)
            if t >= T:
                return -1
        r, c = trec[t], trun[t]
        fl = flags[r]
        n = spins
        while _flag_get(fl, c) == 0:            # its first half still going
            if n == 0:
                return t
            n -= 1
            _spin_hint()
        p = P[ids[r]]
        fix = st[r][c, _SFIX]
        if fix >= 0 and _chain_end(buf, p, F[ids[r]], _dest_of(D, p), out[p[_ROW]],
                                   cuts[r], st[r], fl, c, fix, spins):
            return t
        t = -1


@njit(cache=True, nogil=True)
def _chain_end(buf, p, f, dest, row, cuts, st, flags, c, fix, spins):
    """The rest of _chain_run's run c once every run before it has ended:
    its start from their sums, then into its values (_fix, while they are
    still in this core's cache) or decoded from it. Returns 1 when, after
    `spins` waits, a run before it is still going."""
    for j in range(c):
        v = _flag_get(flags, j)
        while v == 0:
            if spins == 0:
                return 1
            spins -= 1
            _spin_hint()
            v = _flag_get(flags, j)
        if v != 1:
            return 0                    # decode() sorts it out
    order = p[_ORDER]
    last, pen = st[0, _SLAST], st[0, _SPEN]
    for j in range(1, c):
        if order == 1:
            last = last + st[j, _SS1]
        else:
            d = last - pen
            last = last + st[j, _SM] * d + st[j, _SS2]
            pen = last - (d + st[j, _SS1])
    if fix:
        _fix(row, dest, p[_DEST] >= 0, cuts[c, 1], cuts[c + 1, 1], order,
             last, pen, f)
    else:
        _span(buf, p, f, dest, row, False, 0, cuts[c, 0], cuts[c + 1, 0],
              cuts[c, 1], cuts[c, 2], last, pen, 2, st[c])
    return 0


def run_parallel(fn, items):
    """fn(item) for every item, on a thread per logical CPU, each taking
    the next item when it finishes one. For numba nogil kernels: they run
    concurrently, and a numba parallel kernel's start-up (~0.2 s to load
    and start in each new process) is not paid."""
    items = list(items)
    n = min(len(items), threads())
    if n <= 1:
        for it in items:
            fn(it)
        return
    with ThreadPoolExecutor(n) as ex:
        for _ in ex.map(fn, items):
            pass
