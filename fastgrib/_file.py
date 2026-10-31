"""The front end: open a GRIB2 file (a path, a URL or bytes), pick fields,
get arrays.

Fields are picked by a pattern searched in each one's inventory line
("12:0:d=2026012000:TMP:2 m above ground:132 hour fcst:", as wgrib2
-match does), by keywords as pygrib's select (name='TMP', level_value=500),
by record or position, or a list of those. A pick of one field reads as
(ny, nx), of several as (n, ny, nx), rows south to north and columns west
to east as latlon() gives the grid; NaN where a field has no value.

Where it runs is decided for you: the CPU for small reads, the GPU for big
ones or once it is running (its set-up costs a second or two, more than a
field takes on the CPU); gpu=True keeps the fields on the card as cupy
arrays. Local files are memory-mapped, not read in; a GPU job loads one
that fits in RAM comfortably, which lets the card pull it straight from
page-locked memory. Remote files (http, https, s3) are read through the
index published beside them: only the messages picked are fetched.
"""
import builtins
import copy
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import _cpu as G
from . import _encode
from . import _grids
from . import _remote
from . import _winds
from .tables.params import PARAMS

GPU_MIN_VALUES = 1 << 27      # GPU set up for a numpy job this big (~35 NBM fields), or running
GPU_MIN_FIELD = 1 << 19       # ... of fields this big at least (GFS 0.25 deg yes, 0.5 deg no)
GPU_MIN_STREAM = 1 << 30      # a fields() stream this big goes to a GPU already up (~290 NBM fields)
DEVICE = os.environ.get('FASTGRIB_DEVICE', 'auto').lower()      # 'auto', 'cpu' or 'gpu'
REMOTE_BATCH = 128 << 20      # fields(): bytes fetched a batch, the next fetched meanwhile
NAN = np.float32(np.nan)
_KEYS = ('index', 'name', 'short_name', 'long_name', 'units', 'level', 'level_type',
         'level_value',
         'level_type2', 'level_value2', 'level_kind', 'fcst', 'fcst_hours', 'ref_time',
         'valid_time', 'member', 'ensemble', 'extra', 'date')


def set_device(device):
    """Where numpy reads decode: 'auto' (the default: the GPU for big
    fields read into a reused out= array or through fields(), the CPU for
    the rest -- whichever is faster), 'cpu' or 'gpu'. Also the environment
    variable FASTGRIB_DEVICE."""
    global DEVICE
    if device not in ('auto', 'cpu', 'gpu'):
        raise ValueError("device: 'auto', 'cpu' or 'gpu'")
    DEVICE = device


def set_threads(n=None):
    """Decode on n threads (None, the default: one per logical CPU); also
    the environment variable FASTGRIB_THREADS. Fewer leave the rest of the
    machine to other work; where reading is limited by memory's bandwidth
    (big simply or complex packed fields, every core reading) they cost
    little time."""
    G.set_threads(n)


def release():
    """Give back everything kept between calls: the GPU's buffers and page
    locks (on every device; the next GPU read sets it up again) and the
    CPU threads' JPEG2000 scratch arrays."""
    gg = sys.modules.get(__package__ + '._gpu')
    if gg is not None:
        gg.release()
    jp = sys.modules.get(__package__ + '._jpeg2000')
    if jp is not None:
        jp.release()


def open(src):                # noqa: A001 -- like builtins.open, for a GRIB
    """A GRIB2 file -- a path, an http(s):// or s3:// URL, or bytes -- as a
    GribFile."""
    return GribFile(src)


class GribFile:
    """A GRIB2 file's fields: .records (its inventory, in file order),
    find(), select(), read(), fields(), latlon(), to_xarray(); .damaged
    lists messages skipped as damaged. Usable as a context manager."""

    def __init__(self, src):
        self._fetch = None
        self._subs = None             # _bytes(): {message offset: its records}
        self._orders = {}             # encode(): {grid: its points' layout index, file order}
        self._ram = None
        self._cols = {}
        self._lines = None            # find()'s inventory lines, made on first use
        self._index = {}              # select()'s: attribute -> {value: [positions]}
        self._closed = False
        self.damaged = []             # messages skipped as damaged (or GRIB1), described
        if _remote.is_url(src):
            self.name = src
            url = _remote.https(src)
            entries = _remote.inventory(url)
            if entries is not None:
                self._fetch = _remote.Fetcher(url)
                self._map = None
                self.records = entries
                return
            self._map = np.frombuffer(_remote.fetch(url), np.uint8)   # no index: all of it
        elif isinstance(src, (str, os.PathLike)):
            self.name = os.fspath(src)
            self._map = np.memmap(self.name, np.uint8, 'r')
        else:
            self.name = '<buffer>'
            self._map = G._u8(src)
        g1 = []
        self.records = G.scan(self._map, self.damaged, grib1=g1)
        if g1:                        # GRIB1 messages: read as the GRIB2 they would be,
            from . import _grib1      # scanned from their sections, each turned when read
            self.records, plans = _grib1.records(self._map, g1, self.records, self.damaged)
            self._fetch = _grib1.Local(self._map, plans)
        if self.damaged:
            import warnings
            warnings.warn('%s: %d damaged or unread messages skipped (see .damaged); the '
                          'first: %s' % (os.path.basename(self.name), len(self.damaged),
                                         self.damaged[0]), stacklevel=3)

    # -- the inventory

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, i):
        return self.records[i]

    def __repr__(self):
        base = os.path.basename(self.name.rstrip('/'))
        if getattr(self._fetch, 'remote', True) and self._fetch is not None:
            return '<GribFile %r: %d fields (remote, fetched as read)>' % (base, len(self))
        grids = sorted({r.grid for r in self.records if r.grid is not None})
        shapes = ', '.join('%d x %d %s' % (g.ni, g.nj, g.kind) for g in grids)
        return '<GribFile %r: %d fields on %s>' % (base, len(self), shapes or 'no grid')

    def __str__(self):
        return '\n'.join(repr(r) for r in self.records)

    def find(self, pattern):
        """The fields whose inventory line matches `pattern` (a regular
        expression, searched as wgrib2 -match does)."""
        rx = re.compile(pattern)
        if self._lines is None:                 # formatted once, not every search
            self._lines = [repr(r) for r in self.records]
        return [r for r, line in zip(self.records, self._lines) if rx.search(line)]

    def select(self, **criteria):
        """The fields whose attributes match, as pygrib's select: each
        keyword a Record attribute (name, level, level_value, level_kind,
        level_type, fcst, fcst_hours, valid_time, ...) and its value a
        value, a list of values (any of them) or a function returning
        whether. level=500 (a number) means level_value; level_type takes
        a code or a kind's name ('isobaricInhPa')."""
        tests = []
        for k, v in criteria.items():
            if k == 'level' and isinstance(v, (int, float)):
                k = 'level_value'
            if k == 'level_type' and isinstance(v, str):
                k = 'level_kind'
            if k not in _KEYS:
                raise TypeError('select(): no field attribute %r (there are %s)'
                                % (k, ', '.join(_KEYS)))
            tests.append((k, v))
        # each attribute's distinct values tested, not every field: an
        # index of them, made on first use (GFS, 743 fields: ~350 us a
        # select by scanning, a few us after)
        hit = None
        for k, v in tests:
            idx = self._index.get(k)
            if idx is None:                     # built whole, then published: threads
                idx = {}                        # may share the file
                for i, r in enumerate(self.records):
                    idx.setdefault(getattr(r, k), []).append(i)
                self._index[k] = idx
            got = set()
            for x, where in idx.items():
                if _matches(x, v):
                    got.update(where)
            hit = got if hit is None else hit & got
        return [self.records[i] for i in sorted(hit)] if tests else list(self.records)

    # -- the values

    def read(self, which=None, gpu=False, out=None, **criteria):
        """The fields `which` picks (or select(**criteria) does; all of them
        by default): (ny, nx) float32 for one, (n, ny, nx) for several, NaN
        where a field has no value; cupy arrays on the card with gpu=True.
        out: an array of that shape to decode into instead (numpy, or cupy
        on the card) -- reused, it saves a big field's fresh pages (NBM:
        1.5 ms a field against 3.3). For a whole file, fields() keeps
        memory flat."""
        recs, one = self._pick(which, criteria)
        return self._read(recs, one, gpu, out, fresh=out is None)

    def _read(self, recs, one, gpu, out, fresh):
        """read()'s work; fresh: out (if any) is new memory, not reused."""
        if out is not None and not isinstance(out, np.ndarray):
            gpu = True                          # a cupy array: on the card
        use_gpu = gpu or _want_gpu(recs, 'fresh' if fresh else out)
        buf, recs = self._bytes(recs, use_gpu)
        grid = _grid_of(recs)
        cols = self._grid_cols(grid)
        flat = None if out is None else out.reshape(len(recs), cols.size)
        if gpu:
            res = _gpu().decode(buf, recs, cols, flat, undef=NAN)
        elif use_gpu:
            res = np.empty((len(recs), cols.size), np.float32) if flat is None else flat
            gg = _gpu()
            gg.decode(buf, recs, cols, res, undef=NAN)
            if out is None:
                gg.unlock(res)                  # it is the caller's now
        else:
            res = G.decode(buf, recs, cols, flat, undef=NAN)
        res = res.reshape((len(recs),) + grid.shape)
        return res[0] if one else res

    def fields(self, which=None, gpu=False, **criteria):
        """(record, field) for each field picked (all by default), in file
        order, as read() gives them -- decoded a batch at a time into
        reused memory, so a whole file takes no more than one batch (a
        remote one fetched a batch ahead). A field stays as it is until the
        next is asked for: copy() one to keep it."""
        recs, _ = self._pick(which, criteria)
        use_gpu = gpu or _want_gpu(recs, 'stream')
        if self._fetch is None:
            yield from self._decode_runs(self._buffer(use_gpu), recs, use_gpu, gpu)
            return
        batches = _batches(recs, getattr(self._fetch, 'batch', REMOTE_BATCH))
        with ThreadPoolExecutor(1) as ex:
            nxt = ex.submit(self._bytes, batches[0], use_gpu)
            for k in range(len(batches)):
                buf, real = nxt.result()
                if k + 1 < len(batches):
                    nxt = ex.submit(self._bytes, batches[k + 1], use_gpu)
                yield from self._decode_runs(buf, real, use_gpu, gpu)

    def latlon(self, which=None):
        """(lat, lon) of the grid of the fields picked (the first field's by
        default), each (ny, nx) like read()'s fields."""
        return G.latlon2d(_grid_rec(self._records(which)))

    def grid(self, which=None):
        """The Grid of the fields picked (the first field's by default):
        .shape, .kind, .params, .earth."""
        return _grid_rec(self._records(which)).grid

    def winds(self, which=None, earth=True, gpu=False, **criteria):
        """(u, v) of the vector fields picked, as read() gives them: picked
        by their u components (every UGRD unless which or keywords say --
        level=850, name='UGUST', ':UGRD:10 m above ground:'), each with its
        v (the same level, time, member and grid). earth: east and north
        components, turned from the grid's axes where the file gives them
        along those (HRRR's, NAM's, RAP's Lambert winds; wgrib2's
        -new_grid_winds earth)."""
        if which is None and not ({'name', 'short_name'} & set(criteria)):
            criteria = dict(criteria, name=list(_V_NAME))
        us, one = self._pick(which, criteria)
        us = [u for u in us if u.name in _V_NAME]
        if not us:
            raise KeyError('no u component picked (%s)' % ', '.join(sorted(_V_NAME)))
        vs = [self._v_of(u) for u in us]
        u = self._read(us, False, gpu, None, True)
        v = self._read(vs, False, gpu, None, True)
        if earth:
            rel = [i for i, r in enumerate(us) if r.uv_relative]
            if rel:
                g = _grids.layout(us[0].grid)
                ue, ve = _winds.to_earth(_host(u[rel]), _host(v[rel]), g)
                if gpu:
                    import cupy as cp
                    ue, ve = cp.asarray(ue), cp.asarray(ve)
                u[rel], v[rel] = ue, ve
        return (u[0], v[0]) if one else (u, v)

    def _v_of(self, u):
        """u's v component: the same field but for its name."""
        return self._other(u, _V_NAME[u.name], 1)

    def _u_of(self, v):
        """v's u component."""
        return self._other(v, _U_NAME[v.name], -1)

    def _other(self, a, name, ahead):
        key = lambda r: (r.level_type, r.level_value, r.level_type2, r.level_value2,  # noqa: E731
                         r.fcst, r.extra, r.member, r.ref_time)
        want = (name,) + key(a)
        i = self.records.index(a)
        near = (self.records[i + 1:i + 3] + self.records[max(0, i - 2):i] if ahead > 0 else
                self.records[max(0, i - 2):i][::-1] + self.records[i + 1:i + 3])  # beside it
        for r in near + self.records:
            if (r.name,) + key(r) == want and (r.grid is None or a.grid is None or
                                               _grids.layout(r.grid) == _grids.layout(a.grid)):
                return r
        raise KeyError('%r: no %s beside it' % (a, name))

    def to_xarray(self, which=None, earth_winds=False, **criteria):
        """The fields picked (all by default) as an xarray Dataset: a
        variable each name and level kind, stacked along their levels,
        with latitude and longitude, units and the times (needs xarray).
        earth_winds: vector components given along the grid's axes turned
        to east and north (each u with its v, both picked)."""
        from ._xarray import dataset
        recs, _ = self._pick(which, criteria)
        return dataset(self, recs, earth_winds)

    # -- regridding

    def regrid(self, to, which=None, method=None, **criteria):
        """The fields picked (all by default) interpolated onto `to`: a
        Grid (another file's .grid(): its fields and these then line up),
        a GribFile or Record (its grid), or (lats, lons), the 1-D axes of
        a lat-lon grid (any spacing, longitudes either way of 0). Bilinear,
        as wgrib2's -new_grid (IPOLATES: a gap's corners weighed out, at
        least half the weight defined), or method='nearest' (by default on
        points listed one by one, 3.130: their nearest). Winds along
        the grid's axes come out east and north. Laid out as read() lays a
        field of `to` out; several fields stacked."""
        lat, lon, shape = _target(to)
        return self._interp(lat, lon, shape, which, method, criteria)

    def at(self, lat, lon, which=None, method=None, **criteria):
        """The fields picked at points: lat and lon numbers or arrays (any
        shapes that broadcast), interpolated as regrid() does -- a point's
        value, or an array of the points' shape; several fields stacked."""
        lat, lon = np.broadcast_arrays(np.asarray(lat, np.float64), np.asarray(lon, np.float64))
        return self._interp(lat, lon, lat.shape, which, method, criteria)

    def _interp(self, lat, lon, shape, which, method, criteria):
        from . import _regrid
        recs, one = self._pick(which, criteria)
        out = np.empty((len(recs), int(np.prod(shape))), np.float32)
        plans = {}

        def put(k, r, values):
            g = _grids.layout(r.grid)
            if g is None:
                raise ValueError('%r: its grid is not one fastgrib reads' % r)
            if g.template in _grids._SPECTRAL:  # spherical harmonics: summed at the points
                from . import _spectral
                out[k] = _spectral.synthesise(_host(values), *g.params[:3],
                                              *_grids.spectral_points(g, lat, lon)).ravel()
                return
            if g not in plans:
                plans[g] = _regrid.plan(g, lat, lon, method)
            _regrid.apply(*plans[g], _host(values), out[k])
        turned = {k for k, r in enumerate(recs) if r.uv_relative and self._pair(r) is not None}
        rest = [r for k, r in enumerate(recs) if k not in turned]
        where = {}                              # (fields() may give remote records anew)
        for k, r in enumerate(recs):
            if k not in turned:
                where.setdefault((r.offset, r.sub), []).append(k)
        for r, v in self.fields(rest) if rest else ():
            put(where[(r.offset, r.sub)].pop(0), r, v)
        for k in turned:                        # along the grid's axes: east and north first
            r = recs[k]
            u, is_v = self._pair(r)
            ue, ve = self.winds(u)
            put(k, r, ve if is_v else ue)
        res = out.reshape((len(recs),) + tuple(shape))
        return res[0] if one else res

    def _pair(self, r):
        """(r's u component's record, whether r is the v) of a vector
        component, None for anything else (or a u without its v)."""
        try:
            if r.name in _V_NAME:
                self._v_of(r)
                return r, False
            if r.name in _U_NAME:
                return self._u_of(r), True
        except KeyError:
            pass
        return None

    # -- keys

    def keys(self, which):
        """One field's keys, as ecCodes (pygrib, cfgrib) names them, read
        from its message: {'discipline': 0, 'parameterCategory': 0,
        'typeOfFirstFixedSurface': 103, 'Ni': 1440, 'bitsPerValue': 12,
        'shortName': '2t', 'stepRange': '0-6', ...}. A GRIB1 field's are
        those of the GRIB2 it reads as (editionNumber 1)."""
        from . import _keys
        buf, r = self._one(which)
        g1 = self._fetch is not None and not getattr(self._fetch, 'remote', True) and \
            self._map[r.offset + 7] == 1
        return _keys.keys(buf, r, 1 if g1 else 2)

    def sections(self, which):
        """One field's GRIB2 sections, {number: bytes} -- each whole, its
        length and number first, for any octet keys() does not name (a
        message of several fields: the sections in force for this one)."""
        buf, r = self._one(which)
        return {k: bytes(buf[p:p + ln]) for k, (p, ln) in _encode.sections(buf, r).items()}

    # -- writing

    def message(self, which, **changes):
        """One field's GRIB2 message (bytes), standing alone: cut from a
        message of several, its bitmap given if it reuses one, a GRIB1
        field as GRIB2 -- its packed values untouched. Keywords set its
        metadata: name='TMP' (or param=(discipline, category, number)),
        level='850 mb' (as wgrib2 words it, or (type, value[, type2,
        value2]) as Records give them), date=datetime (the reference time),
        fcst='6 hour fcst' / '0-6 hour acc fcst' / 'anl'."""
        buf, r = self._one(which)
        return _encode.message(buf, r, **changes)

    def encode(self, values, like, packing=None, decimals=None, bits=None, **changes):
        """A GRIB2 message (bytes) of new values (laid out as read() gives
        them, NaN for none) as a field like `like` (one pick): its grid,
        product and metadata, set by keywords as message()'s. packing:
        'simple', 'complex' (second-order spatial differencing), 'ieee'
        (lossless), 'png', 'jpeg' or 'ccsds'; spherical harmonics' 'simple'
        (5.50) or 'complex' (5.51); by default as `like`'s. Precision:
        `like`'s unless decimals (values to 10^-decimals) or bits (bits a
        value) say."""
        buf, r = self._one(like)
        g = r.grid
        if g is None:
            raise ValueError('%r: its grid is not one fastgrib reads' % r)
        if g not in self._orders:               # the layout index of each file point, kept
            full = copy.copy(r)
            full.bitmap = -1
            self._orders[g] = None if g.scan & 0xF0 == 0x40 else G._defined(buf, full)
        order = self._orders[g]
        return _encode.encode(buf, r, _host(values), order, g.ni * g.nj, packing, decimals,
                              bits, **changes)

    def write(self, path, which=None, append=False, **criteria):
        """The fields picked (all by default), each as its own GRIB2
        message (see message()), into a file -- a subset of this one, from
        a remote file only what is picked, a GRIB1 file as GRIB2. Returns
        how many were written."""
        recs, _ = self._pick(which, criteria)
        with builtins.open(path, 'ab' if append else 'wb') as out:
            if self._fetch is None:
                for r in recs:
                    out.write(_encode.message(self._map, r))
            else:
                for batch in _batches(recs, getattr(self._fetch, 'batch', REMOTE_BATCH)):
                    buf, real = self._bytes(batch, False)
                    for r in real:
                        out.write(_encode.message(buf, r))
        return len(recs)

    def _one(self, which):
        """(bytes, Record) of one field picked, its message's bytes in them."""
        recs, single = self._pick(which, {})
        if not single:
            raise ValueError('%d fields picked: one is needed' % len(recs))
        if self._fetch is None:
            return self._map, recs[0]
        buf, real = self._bytes(recs[:1], False)
        return buf, real[0]

    # -- the file

    def close(self):
        """Let go of the file (its memory map, any copy in RAM and its page
        lock, fetched messages)."""
        gg = sys.modules.get(__package__ + '._gpu')
        if gg is not None and self._ram is not None:
            gg.unlock(self._ram)
        self._map = self._ram = self._fetch = None
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- inside

    def _records(self, which):
        """Decodable Records for a pick (remote ones fetched)."""
        recs, _ = self._pick(which if which is not None else 0, {})
        return self._bytes(recs[:1] if which is None else recs, False)[1]

    def _pick(self, which, criteria):
        """(records, whether the pick named a single field)."""
        if self._closed:
            raise ValueError('%s is closed' % self.name)
        if criteria:
            if which is not None:
                raise TypeError('pick by which or by keywords, not both')
            recs = self.select(**criteria)
            if not recs:
                raise KeyError(self._no_match(', '.join('%s=%r' % kv for kv in criteria.items())))
            return recs, len(recs) == 1
        if which is None:
            return list(self.records), len(self.records) == 1
        if isinstance(which, (str, G.Record, _remote.Entry, int, np.integer)):
            items, single = [which], not isinstance(which, str)
        else:
            items, single = list(which), False
        recs = []
        for w in items:
            if isinstance(w, str):
                hit = self.find(w)
                if not hit:
                    raise KeyError(self._no_match(w))
                recs += hit
            elif isinstance(w, (G.Record, _remote.Entry)):
                recs.append(w)
            else:
                recs.append(self.records[int(w)])
        if not recs:
            raise KeyError('nothing picked')
        return recs, single or len(recs) == 1

    def _no_match(self, pattern):
        """What a pick that found nothing says: what is in the file, and
        the nearest inventory lines."""
        import difflib
        from collections import Counter
        names = Counter(r.name for r in self.records)
        lines = sorted({'%s:%s' % (r.name, r.level) for r in self.records})
        near = difflib.get_close_matches(pattern.strip(':'), lines, n=5, cutoff=0.4)
        msg = 'no field of %s matches %s. It holds: %s.' % (
            os.path.basename(self.name), pattern,
            ', '.join('%s (%d)' % kv for kv in names.most_common(12))
            + (', ...' if len(names) > 12 else ''))
        if near:
            msg += ' Nearest: ' + '; '.join(near)
        return msg

    def _decode_runs(self, buf, recs, use_gpu, gpu):
        for run in _runs(recs):
            grid = run[0].grid
            cols = self._grid_cols(grid)
            it = (_gpu().decode_iter(buf, run, cols, host=not gpu, undef=NAN) if use_gpu
                  else G.decode_iter(buf, run, cols, undef=NAN))
            for i, block in it:
                for j in range(len(block)):
                    yield run[i + j], block[j].reshape(grid.shape)

    def _grid_cols(self, grid):
        """Every point of a grid, as the decoders' columns (kept)."""
        c = self._cols.get(grid)
        if c is None:
            c = self._cols[grid] = np.arange(grid.ni * grid.nj, dtype=np.int64)
        return c

    def _bytes(self, recs, gpu):
        """(buffer, Records) to decode recs from: a local file's own, or a
        remote one's messages fetched into a buffer of their own and
        scanned there (each Record numbered and placed as in the file)."""
        if self._fetch is None:
            return self._buffer(gpu), recs
        spans = sorted({(e.offset, e.end) for e in recs})
        remote = getattr(self._fetch, 'remote', True)

        def whole(off, m):                      # a message reusing an earlier one's bitmap
            if _remote.lacks_bitmap(m):
                return _remote.with_bitmap(m, self._bitmap_before(off))
            return m
        if not remote:                          # a local file's GRIB1: made in one buffer
            buf, starts, places = self._fetch.joined(spans, whole)
            if places is not None:              # every section's place known: no scan
                out = []
                for e in recs:
                    c = copy.copy(e)
                    c.at, c.s5, c.s6, c.s7, c.s7end, c.bitmap = places[e.offset]
                    out.append(c)
                return buf, out
        else:
            got = self._fetch.messages(spans)
            for off, _ in spans:
                if got[off][7:8] == bytes([1]):     # GRIB1: as the GRIB2 it would be
                    from . import _grib1
                    got[off] = _grib1.transcode(got[off], 0)[0]
                got[off] = whole(off, got[off])
            starts, parts, at = {}, [], 0
            for off, _ in spans:
                starts[at] = off
                parts.append(got[off])
                at += len(got[off])
            buf = np.frombuffer(b''.join(parts), np.uint8)
        real, seen, bad = {}, {}, []
        recs_in = G.scan(buf, bad, alone=True)      # neighbours here, not in the file
        if bad:
            raise IOError('%s: fetched bytes damaged: %s' % (self.name, bad[0]))
        for r in recs_in:
            off = starts[r.offset]
            k = seen[off] = seen.get(off, -1) + 1
            real[(off, k)] = r
        subs = self._subs
        if subs is None:                        # each message's records (the inventory's fixed)
            subs = {}
            for e in sorted(self.records, key=lambda e: (e.offset, e.sub)):
                subs.setdefault(e.offset, []).append(e)
            self._subs = subs
        out = []
        remote = getattr(self._fetch, 'remote', True)
        for e in recs:
            r = real[(e.offset, subs[e.offset].index(e))]
            if not remote:                      # GRIB1: as scanned, placed in these bytes
                c = copy.copy(e)
                c.s5, c.s6, c.s7, c.s7end, c.bitmap, c.at = r.s5, r.s6, r.s7, r.s7end, r.bitmap, r.at
                out.append(c)
                continue
            r.index, r.msg, r.offset = e.index, e.msg, e.offset
            _remote.learn(e, r)
            out.append(r)
        if remote:
            self._index.clear()                 # what entries say may have changed
        return buf, out

    def _bitmap_before(self, off):
        """The bitmap section (bytes) in force just before the remote
        message at off: the last one an earlier message defines."""
        order = sorted({(e.offset, e.end) for e in self.records})
        k = [s[0] for s in order].index(off)
        for k in range(k - 1, max(k - 65, -1), -1):
            m = self._fetch.messages([order[k]])[order[k][0]]
            sec = _remote.last_bitmap(m)
            if sec is not None:                 # else none defined there: further back
                return sec
        raise IOError('%s: the message at byte %d reuses a bitmap no earlier message '
                      'defines' % (self.name, off))

    def _buffer(self, gpu):
        """The bytes to decode from: the map, or for a GPU job the file in
        RAM when it fits comfortably (min 2 GB, an eighth of RAM) -- the
        card then pulls records from page-locked memory, twice as fast as
        through staging from a map (NBM: 0.40 ms a field against 0.81)."""
        if self._ram is not None:
            return self._ram
        if gpu and isinstance(self._map, np.memmap) and \
                self._map.nbytes <= min(2 << 30, G.memory()[0] // 8):
            self._ram = np.fromfile(self.name, np.uint8)
            return self._ram
        return self._map


# a u component's name -> its v's (NCEP's names, as records carry them)
_V_NAME = {PARAMS[u][0]: PARAMS[v][0] for u, v in _winds.PAIRS.items() if u in PARAMS and v in PARAMS}
_U_NAME = {v: u for u, v in _V_NAME.items()}


def _target(to):
    """(lat, lon, shape) of a regrid() target."""
    if isinstance(to, GribFile):
        to = to.grid()
    elif isinstance(to, (G.Record, _remote.Entry)):
        to = to.grid
    if isinstance(to, _grids.Grid):
        lat, lon = _grids.latlon(to)
        return lat, lon, lat.shape
    try:
        lats, lons = to
    except (TypeError, ValueError):
        raise TypeError('regrid to a Grid, a GribFile, a Record or (lats, lons) axes, not %r'
                        % (to,))
    lats, lons = np.asarray(lats, np.float64), np.asarray(lons, np.float64)
    if lats.ndim != 1 or lons.ndim != 1:
        raise ValueError('(lats, lons): the 1-D axes of a lat-lon grid (at(lat, lon) takes '
                         'points)')
    lat, lon = np.meshgrid(lats, lons, indexing='ij')
    return lat, lon, lat.shape


def _host(a):
    """a as numpy (from the card, if there)."""
    return a.get() if hasattr(a, 'get') else a


def _matches(x, want):
    if callable(want):
        return bool(want(x))
    if isinstance(want, (list, tuple, set, frozenset)):
        return any(_matches(x, w) for w in want)
    if isinstance(want, float) and isinstance(x, float):
        return abs(x - want) <= 1e-6 * max(1.0, abs(want))
    return x == want


def _values(recs):
    """(values, fields) of recs the GPU would decode, as far as known: the
    complex-packed ones (the card's; the rest decode on the CPU anyway,
    PNG, JPEG2000, CCSDS, simple); remote fields not yet fetched by their
    bytes."""
    n = k = 0
    for r in recs:
        if r.grid is not None:
            if getattr(r, 'packing', 3) in (2, 3):
                n += r.grid.ni * r.grid.nj
                k += 1
        elif getattr(r, 'end', None) is not None:
            n += (r.end - r.offset) * 2         # ~2 values a packed byte
            k += 1
    return n, k


def _batches(recs, size):
    """Remote recs in consecutive batches of about `size` bytes."""
    out, cur, held = [], [], 0
    for r in recs:
        n = (r.end - r.offset + 1) if r.end is not None else size
        if cur and held + n > size:
            out.append(cur)
            cur, held = [], 0
        cur.append(r)
        held += n
    if cur:
        out.append(cur)
    return out


def _grid_of(recs):
    """The grid all of recs share (as read() lays them out)."""
    g = _grid_rec(recs).grid
    lay = _grids.layout(g)
    if any(r.grid is not g and _grids.layout(r.grid) != lay for r in recs):
        raise ValueError('these fields are on different grids: read them '
                         'apart, or go through fields()')
    return lay


def _grid_rec(recs):
    for r in recs:
        G._check_grid(r)
    return recs[0]


def _runs(recs):
    """recs cut into runs of consecutive fields on one grid."""
    run = []
    for r in recs:
        G._check_grid(r)
        if run and r.grid is not run[0].grid and \
                _grids.layout(r.grid) != _grids.layout(run[0].grid):
            yield run
            run = []
        run.append(r)
    if run:
        yield run


_GPU_STATE = None             # None not tried, else whether the GPU works


def _want_gpu(recs, job):
    """Whether a numpy job -- 'fresh' (a new result), 'stream' (fields()),
    or the reused out= array itself -- goes to the GPU, as measured:
    - a fresh result never: page-locking new memory costs the GPU more
      than it saves (NBM 2.7-3.3 ms a field against 0.9-1.8 on the CPU,
      GFS 0.25 deg 0.75 against 0.36);
    - into a reused array, for big fields (GPU_MIN_FIELD values or more:
      GFS 0.25 deg 0.14 ms a field against 0.21; GFS 0.5 deg is the
      CPU's) when the array can stay page-locked (min(2 GB, RAM/8)) and
      the job is worth the GPU's set-up (GPU_MIN_VALUES), or it is up;
    - a fields() stream of big fields when it is long and the GPU already
      up (all 769 NBM fields: 0.37 ms a field against 0.55, at a
      fourteenth of the CPU time; 96 of them lose to the GPU's per-call
      set-up, and a first set-up of the GPU costs more than it saves)."""
    if DEVICE != 'auto':
        return DEVICE == 'gpu'
    if isinstance(job, str) and job == 'fresh' or _GPU_STATE is False:
        return False
    n, k = _values(recs)
    if not k or n < GPU_MIN_FIELD * k:
        return False
    if isinstance(job, str):                    # a stream
        return bool(_GPU_STATE) and n >= GPU_MIN_STREAM
    if job.nbytes > min(2 << 30, G.memory()[0] // 8):
        return False                            # too big to stay page-locked
    if n < GPU_MIN_VALUES and not _GPU_STATE:
        return False
    return _try_gpu()


def _try_gpu():
    global _GPU_STATE
    if _GPU_STATE is None:
        try:
            _gpu()
        except Exception:
            _GPU_STATE = False
    return bool(_GPU_STATE)


def _gpu():
    """The GPU decoder, set up (raising if there is no usable GPU)."""
    from . import _gpu as gg
    gg._device()
    global _GPU_STATE
    _GPU_STATE = True
    return gg
