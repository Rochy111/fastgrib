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
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import _cpu as G
from . import _encode
from . import _grids
from . import _remote
from . import _winds
from .tables.params import PARAMS

_HEAD = 8 << 10               # a partial remote read's first request: a message's sections 0-5
                              # (a bitmap past it: fetched whole next, its length known)
_PREFIX_FETCH = 0.7           # complex packing remote: its values up to the last wanted when
                              # that is below this share of the message (else whole: fewer trips)
_GAP = 32 << 10               # ... then its values' byte ranges, joined this close together (a
                              # round trip costs more than fetching ~32 KB more: ~1 ms at 40 MB/s)
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


def open(src, max_records=None, byte_range=None, index=None, cache=None):    # noqa: A001
    """A GRIB2 file -- a path, an http(s):// or s3:// URL, or bytes -- as a
    GribFile. max_records: only its first fields (as many); byte_range:
    (start, stop), only the messages starting at byte start or after and
    before stop (None: the end). A remote file without an index then
    fetches only those messages (a window at a time, in parallel ranges),
    not the whole file. index: a remote file's index (a path or URL: wgrib2
    -s lines as GribFile.write_index() writes them, NOAA's .idx, ECMWF's
    .index or wgrib's .inv) rather than the one beside it. cache: a folder
    (or True: FASTGRIB_REMOTE_CACHE, else the user's cache folder) where
    a remote file's messages are kept once fetched, for any later read --
    this process's or another's -- to take from disk."""
    return GribFile(src, max_records, byte_range, index, cache)


class GribFile:
    """A GRIB2 file's fields: .records (its inventory, in file order),
    find(), select(), read(), fields(), latlon(), to_xarray(); .damaged
    lists messages skipped as damaged. Usable as a context manager."""

    def __init__(self, src, max_records=None, byte_range=None, index=None, cache=None):
        start, stop = _check_limits(max_records, byte_range)
        self._args = (src, max_records, byte_range, index, cache)
        if index is not None and not _remote.is_url(src):
            raise ValueError('index: for a remote file (a local one is scanned whole, in '
                             'milliseconds)')
        self._fetch = None
        self._base = 0                # the file offset of self._map's first byte
        self._subs = None             # _bytes(): {message offset: its records}
        self._orders = {}             # encode(): {grid: its points' layout index, file order}
        self._ram = None
        self._cols = {}
        self._lines = None            # find()'s inventory lines, made on first use
        self._index = {}              # select()'s: attribute -> {value: [positions]}
        self._heads = {}              # remote: message offset -> _head()'s, False for none
        self._closed = False
        self.damaged = []             # messages skipped as damaged (or GRIB1), described
        if _remote.is_url(src):
            self.name = src
            url = _remote.https(src)
            entries = _remote.inventory(url, index)
            if entries is not None:
                self._fetch = _remote.Fetcher(url, cache)
                self._map = None
                self.records = _limited(entries, max_records, start, stop)
                return
            if max_records is None and byte_range is None:      # no index: all of it
                self._map = np.frombuffer(_unpacked(_remote.fetch_all(url)), np.uint8)
            else:                               # no index: the messages wanted, walked to
                enough = None
                if max_records is not None:
                    enough = lambda b: _count(b) >= max_records    # noqa: E731
                try:
                    data, self._base = _remote.walk(url, start, stop, enough)
                except IOError:                 # no lengths or ranges: all of it
                    data = _remote.fetch_all(url)
                self._map = np.frombuffer(data, np.uint8)
        elif isinstance(src, (str, os.PathLike)):
            self.name = os.fspath(src)
            with builtins.open(self.name, 'rb') as fh:
                head = fh.read(6)
            if _packer(head) is not None:       # gzip, bzip2, xz: unpacked into memory
                with builtins.open(self.name, 'rb') as fh:
                    self._map = np.frombuffer(_unpacked(fh.read()), np.uint8)
            else:
                self._map = np.memmap(self.name, np.uint8, 'r')
        else:
            self.name = '<buffer>'
            if isinstance(src, (bytes, bytearray, memoryview)) and _packer(bytes(src[:6])):
                src = _unpacked(bytes(src))
            self._map = G._u8(src)
        g1 = []
        self.records = G.scan(self._map, self.damaged, grib1=g1)
        base = self._base
        if g1:                        # GRIB1 messages: read as the GRIB2 they would be,
            from . import _grib1      # scanned from their sections, each turned when read
            self.records, plans = _grib1.records(self._map, g1, self.records, self.damaged)
            self._fetch = _grib1.Local(self._map, {k + base: p for k, p in plans.items()}, base)
        if base:                      # part of a remote file: placed as in it
            for r in self.records:
                r.offset += base
                if getattr(r, 'end', None) is not None:
                    r.end += base
        if max_records is not None or byte_range is not None:
            self.records = _limited(self.records, max_records, start, stop)
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

    def __contains__(self, pattern):
        """Whether any field's inventory line matches `pattern` (a regular
        expression, as find()): ':TMP:2 m above ground:' in f."""
        if not isinstance(pattern, str):
            return pattern in self.records
        return bool(self.find(pattern))

    def to_pandas(self, which=None, **criteria):
        """The fields picked (all by default) as a pandas DataFrame, a row
        each: index, name, short_name, long_name, units, level, level_kind,
        level_value, fcst, fcst_hours, ref_time, valid_time, member,
        ensemble, extra, offset -- for looking a file over and filtering:
        any pick takes its rows back (f.read(df.query('level_value >
        500'))). Needs pandas."""
        import pandas as pd
        recs, _ = self._pick(which, criteria)
        rows = []
        for r in recs:
            d = r.as_dict()
            d['fcst_hours'] = r.fcst_hours
            d['offset'] = r.offset
            rows.append(d)
        cols = ['index', 'name', 'short_name', 'long_name', 'units', 'level', 'level_kind',
                'level_value', 'fcst', 'fcst_hours', 'ref_time', 'valid_time', 'member',
                'ensemble', 'extra', 'offset']
        return pd.DataFrame(rows, columns=cols)

    def write_index(self, path=None):
        """The inventory as an index, wgrib2 -s lines (NOAA's .idx: each
        message's offset, name, level, times), written to path or returned
        as text. Beside a remote file (name.idx), or given to open(url,
        index=...), it saves fetching to find the messages -- as a remote
        file without an index must, whole or a window at a time."""
        text = ''.join(repr(r) + '\n' for r in self.records)
        if path is None:
            return text
        with builtins.open(path, 'w') as out:
            out.write(text)
        return None

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

    def read(self, which=None, gpu=False, out=None, expand=False, box=None, **criteria):
        """The fields `which` picks (or select(**criteria) does; all of them
        by default): (ny, nx) float32 for one, (n, ny, nx) for several, NaN
        where a field has no value; cupy arrays on the card with gpu=True.
        out: an array of that shape to decode into instead (numpy, or cupy
        on the card) -- reused, it saves a big field's fresh pages (NBM:
        1.5 ms a field against 3.3). For a whole file, fields() keeps
        memory flat. expand: a reduced grid's fields (else (points,)) on
        its regular grid, (rows, longest row), as expand() makes them.
        box: (west, east, south, north) in degrees, as wgrib2 -small_grib
        takes it -- only the rows and columns of the points in it (a
        projected grid's: the rectangle round them), as latlon(box=...)
        places them."""
        recs, one = self._pick(which, criteria)
        if box is not None:
            if out is not None or expand:
                raise TypeError('box= with out= or expand=: read the box, then copy it')
            from . import _partial, _subset
            b = _subset.check(box)
            if self._fetch is None:
                g = _grid_of(recs)
                win = _subset.window(g, b)
                if not gpu and all(_partial.worth(r, win.ni * win.nj) for r in recs):
                    lay = _box_points(g, win)   # a small box: its points alone read (or
                    res = np.stack(_partial.values_many(        # up to its last, in order)
                        [(self._map, r, lay) for r in recs]))
                    return res[0] if one else res
                return _subset.cut(self._read(recs, one, gpu, None, True), win)

            def want(r):                        # remote: its box's points, if a small part
                g = _grids.layout(r.grid)
                lay = _box_points(g, _subset.window(g, b))
                return lay if _partial.worth(r, lay.size) else None
            part = {} if gpu else (self._partial_remote(recs, want)
                                   if getattr(self._fetch, 'remote', True)
                                   else self._grib1_part(recs, want))
            rest = [r for r in recs if (r.offset, r.sub) not in part]
            whole = iter(self._read(rest, False, gpu, None, True)) if rest else iter(())
            res = []
            for r in recs:
                if (r.offset, r.sub) in part:
                    buf, rr, whole = part[(r.offset, r.sub)]
                    g = _grids.layout(rr.grid)
                    lay = want(rr) if _partial.can(rr) else None
                    if lay is not None:
                        res.append(_partial.values(buf, rr, lay, prefix=not whole))
                    else:                       # fetched whole: decoded so, then cut
                        v = G.decode(buf, [rr], self._grid_cols(g), undef=NAN)[0]
                        res.append(_subset.cut(v.reshape(g.shape), _subset.window(g, b)))
                else:
                    g = _grids.layout(r.grid)
                    res.append(_subset.cut(next(whole), _subset.window(g, b)))
            if gpu:
                import cupy as cp
                res = cp.stack(res)
            else:
                res = np.stack(res)
            return res[0] if one else res
        if expand and _expandable(recs):
            v = self._read(recs, one, False, None, True)
            host = out if isinstance(out, np.ndarray) else None
            res = expand_grid(v, _grid_of(recs), out=host)
            if gpu or (out is not None and host is None):       # onto the card
                return _to_card(res, None if host is not None else out)
            return res
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

    def fields(self, which=None, gpu=False, expand=False, box=None, **criteria):
        """(record, field) for each field picked (all by default), in file
        order, as read() gives them -- decoded a batch at a time into
        reused memory, so a whole file takes no more than one batch (a
        remote one fetched a batch ahead). A field stays as it is until the
        next is asked for: copy() one to keep it. expand, box: as read()'s."""
        if box is not None:
            from . import _subset
            b = _subset.check(box)
            for r, v in self.fields(which, gpu, expand, **criteria):
                yield r, _subset.cut(v, _subset.window(_grids.layout(r.grid), b))
            return
        if expand:
            yield from self._expanded_fields(which, gpu, criteria)
            return
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

    def _expanded_fields(self, which, gpu, criteria):
        """fields(expand=True): each reduced grid's fields widened into one
        array kept for that grid."""
        kept = {}
        for r, v in self.fields(which, False, **criteria):
            if not _regrid().expandable(r.grid):
                yield r, (_to_card(v, None) if gpu else v)
                continue
            lay = _grids.layout(r.grid)
            if lay not in kept:
                kept[lay] = expand_grid(v, lay)
            else:
                expand_grid(v, lay, out=kept[lay])
            yield r, (_to_card(kept[lay], None) if gpu else kept[lay])

    def latlon(self, which=None, expand=False, box=None):
        """(lat, lon) of the grid of the fields picked (the first field's by
        default), each (ny, nx) like read()'s fields. expand: those of a
        reduced grid's regular one, as read(expand=True) lays it out; box:
        those of the box read(box=...) cuts."""
        r = _grid_rec(self._records(which))
        if box is not None:
            from . import _subset
            g = _grids.layout(r.grid)
            return _subset.latlon(g, _subset.window(g, _subset.check(box)))
        if expand and _regrid().expandable(r.grid):
            return _regrid().expanded_latlon(_grids.layout(r.grid))
        return G.latlon2d(r)

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

    def to_xarray(self, which=None, earth_winds=False, lazy=False, **criteria):
        """The fields picked (all by default) as an xarray Dataset: a
        variable each name and level kind, stacked along their levels,
        with latitude and longitude, units and the times (needs xarray).
        earth_winds: vector components given along the grid's axes turned
        to east and north (each u with its v, both picked). lazy: each
        variable read when indexed, only the fields indexed (as
        xr.open_dataset(path, engine='fastgrib') does) -- a local file's."""
        from ._xarray import dataset
        recs, _ = self._pick(which, criteria)
        return dataset(self, recs, earth_winds, lazy)

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

        def plan_of(r):
            g = _grids.layout(r.grid)
            if g is None:
                raise ValueError('%r: its grid is not one fastgrib reads' % r)
            if g not in plans and g.template not in _grids._SPECTRAL:
                plans[g] = _regrid.plan(g, lat, lon, method)
            return g, plans.get(g)

        def put(k, r, values):
            g, p = plan_of(r)
            if p is None:                       # spherical harmonics: summed at the points
                from . import _spectral
                out[k] = _spectral.synthesise(_host(values), *g.params[:3],
                                              *_grids.spectral_points(g, lat, lon)).ravel()
                return
            _regrid.apply(*p, _host(values), out[k])
        key = lambda r: (r.offset, r.sub)       # noqa: E731 -- (fields() may give remote
        pairs = {}                              # records anew) slot -> (u, v, whether the v)
        self._learn(recs)                       # (remote: which winds lie along the grid's axes)
        for k, r in enumerate(recs):
            if r.uv_relative and r.grid is not None and \
                    r.grid.template not in _grids._SPECTRAL:
                p = self._pair(r)
                if p is not None:
                    pairs[k] = (p[0], r if p[1] else self._v_of(r), p[1])
        where, need = {}, {}
        for k, r in enumerate(recs):
            need.setdefault(key(r), r)
            if k not in pairs:
                where.setdefault(key(r), []).append(k)
        both = set()
        for u, v, _ in pairs.values():
            for c in (u, v):
                need.setdefault(key(c), c)
                both.add(key(c))
        # every field read once; a wind component along the grid's axes kept
        # only at the corners its points are interpolated from, turned there
        # with its partner (rotation is point by point: the same values as
        # turning the whole field, as winds() does, without reading each
        # pair again and turning every point for each of its components)
        corners = {}
        # a simple or IEEE field of which the plan takes a small part: only
        # those points read (the values a whole read gives; _partial)
        from . import _partial
        counts = {}

        def want(r):                            # its plan's corners, if a small part of it
            if not _partial.can(r):
                return None
            g, p = plan_of(r)
            if p is None:
                return None
            if g not in counts:
                counts[g] = _corner_count(p[0])
            return np.maximum(p[0], 0) if _partial.worth(r, counts[g]) else None
        if self._fetch is None:                 # a local file: in place
            part = {kk: (self._map, r, True) for kk, r in need.items() if want(r) is not None}
        elif not getattr(self._fetch, 'remote', True):  # a local file's GRIB1: made GRIB2
            part = self._grib1_part(list(need.values()), want)
        else:                                   # remote: those bytes alone fetched
            part = self._partial_remote(list(need.values()), want)
        jobs, fulls = [], []
        for kk, (buf, r, whole) in part.items():
            need.pop(kk, None)
            g, p = plan_of(r)
            if p is not None and _partial.can(r):
                jobs.append((kk, buf, r, p, whole))
            else:                               # (fetched whole: JPEG2000, spectral ...)
                fulls.append((kk, buf, r))
        got = _partial.values_many([(buf, r, np.maximum(p[0], 0), not whole)
                                    for _, buf, r, p, whole in jobs])
        for (kk, buf, r, p, whole), vals in zip(jobs, got):
            for k in where.get(kk, ()):
                _regrid.apply(_own(p[0]), p[1], vals.ravel(), out[k])
            if kk in both:
                corners[kk] = vals
        for kk, buf, r in fulls:
            g = _grids.layout(r.grid)
            values = G.decode(buf, [r], self._grid_cols(g), undef=NAN)[0].reshape(g.shape)
            for k in where.get(kk, ()):
                put(k, r, values)
            if kk in both:
                corners[kk] = values.ravel()[np.maximum(plan_of(r)[1][0], 0)]
        for r, values in self.fields(list(need.values())) if need else ():
            for k in where.get(key(r), ()):
                put(k, r, values)
            if key(r) in both:
                g, p = plan_of(r)
                corners[key(r)] = _host(values).ravel()[np.maximum(p[0], 0)]
        for k, (u, v, is_v) in pairs.items():
            g, (idx, w) = plan_of(u)
            cu, cv = corners[key(u)], corners[key(v)]
            ang = _winds.angle(g)
            if ang is not None:
                at = np.maximum(idx, 0)
                cu, cv = _winds._turned(ang[0].ravel()[at], ang[1].ravel()[at], cu, cv, False)
            _regrid.apply(_own(idx), w, cv if is_v else cu, out[k])
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
            self._map[r.offset - self._base + 7] == 1
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
        return _encode.encode(buf, r, _host(values), self._order(buf, r), g.ni * g.nj, packing,
                              decimals, bits, **changes)

    def write(self, path, which=None, append=False, box=None, packing=None, decimals=None,
              bits=None, **criteria):
        """The fields picked (all by default), each as its own GRIB2
        message (see message()), into a file -- a subset of this one, from
        a remote file only what is picked, a GRIB1 file as GRIB2. Returns
        how many were written. box: (west, east, south, north), each field
        cut to it as read(box=...) cuts it and packed again as it was (its
        packing and precision: the values read back unchanged), on the
        smaller grid -- wgrib2's -small_grib. packing ('simple', 'complex',
        'ccsds', 'jpeg', 'png', 'ieee'), decimals, bits: each field packed
        again so, as encode() packs it (wgrib2's -set_grib_type), the
        fields packed on every core at once."""
        recs, _ = self._pick(which, criteria)
        with builtins.open(path, 'ab' if append else 'wb') as out:
            if box is not None or packing is not None or decimals is not None or \
                    bits is not None:
                self._write_packed(out, recs, box, packing, decimals, bits)
            elif self._fetch is None:
                for r in recs:
                    out.write(_encode.message(self._map, r))
            else:
                for batch in _batches(recs, getattr(self._fetch, 'batch', REMOTE_BATCH)):
                    buf, real = self._bytes(batch, False)
                    for r in real:
                        out.write(_encode.message(buf, r))
        return len(recs)

    def _write_packed(self, out, recs, box, packing, decimals, bits):
        """write()'s messages packed anew (cut to a box, or packed another
        way): each field decoded in turn (the decoders on every core), and
        packed on a thread of its own while the next decodes -- the packers
        release Python's lock, so fields pack side by side (HRRR, complex
        packing: 22 ms a field alone, 5 on 24 threads), written in file order;
        as many held at once as keeps memory to ~256 MB."""
        from collections import deque
        from . import _subset
        b = _subset.check(box) if box is not None else None
        batches = [recs] if self._fetch is None else \
            _batches(recs, getattr(self._fetch, 'batch', REMOTE_BATCH))
        n = G.threads()

        local = threading.local()

        def pack(buf, r, g, win):
            # decoded on this thread, whole (not split across the cores: the
            # other threads have them), into this thread's own reused row --
            # decoding each in turn on one thread, split, held packing to 11 ms
            # a field however many threads (HRRR); its fresh rows' pages faulted
            # in one at a time
            npts = g.ni * g.nj
            row = getattr(local, 'row', None)
            if row is None or row.size < npts:
                row = local.row = np.empty(npts, np.float32)
                _encode.crowded(n > 1)          # (packing side by side: its passes over
                                                # memory fewer, at some arithmetic's cost)
            v = G.decode(buf, [r], self._grid_cols(g), row[:npts].reshape(1, npts),
                         undef=NAN, _min_run=1 << 62).reshape(g.shape)
            if win is None:
                return _encode.encode(buf, r, v, self._order(buf, r), npts, packing,
                                      decimals, bits)
            v = _subset.cut(v, win)
            p, ln = _encode.sections(buf, r)[3]
            s3 = _subset.section3(bytes(buf[p:p + ln]), g, win)
            return _encode.encode(buf, r, v, None, win.ni * win.nj, packing, decimals, bits,
                                  grid=(s3, win.nj, win.ni))
        with ThreadPoolExecutor(n, initializer=G._fast_thread) as pool:
            held = deque()
            for batch in batches:
                buf, real = self._bytes(batch, False)
                for r in real:
                    G._check_grid(r)
                    g = _grids.layout(r.grid)
                    win = _subset.window(g, b) if b is not None else None
                    self._order(buf, r) if win is None else None       # (made once, here)
                    held.append(pool.submit(pack, buf, r, g, win))
                    while len(held) >= 2 * n:
                        out.write(held.popleft().result())
            while held:
                out.write(held.popleft().result())

    def _order(self, buf, r):
        """The layout index of each of r's grid's points in file order
        (None: the same order), kept a grid -- what encode() needs to put
        values laid out as read() gives them back in the file's order."""
        g = r.grid
        if g not in self._orders:
            full = copy.copy(r)
            full.bitmap = -1
            self._orders[g] = None if g.scan & 0xF0 == 0x40 else G._defined(buf, full)
        return self._orders[g]

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

    def __reduce__(self):
        """Pickled as how to open it again -- a path or URL, not its bytes
        (a process pool's worker, dask's: each opens the file itself, its
        map rather than a copy); a file opened from bytes, with them."""
        if self._closed:
            raise ValueError('%s is closed' % self.name)
        src, *rest = self._args
        if isinstance(src, (str, os.PathLike)):
            return GribFile, (src, *rest)
        return GribFile, (bytes(memoryview(self._map)), *rest)

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
        if hasattr(which, 'columns') and 'index' in getattr(which, 'columns', ()):
            by = {r.index: r for r in self.records}     # to_pandas()'s rows: by record number
            recs = [by[int(i)] for i in which['index']]
            if not recs:
                raise KeyError('nothing picked (no rows)')
            return recs, False
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

    def _partial_remote(self, entries, want):
        """{entry's (offset, sub): (buffer, Record, whether whole)} of remote
        entries read in part: each message's sections to 7's start (_head),
        then what want(record) -- its read() layout indices wanted, or None
        for most of it -- needs: a simple or IEEE field's values' byte
        ranges; a complex one's group headers and its values up to the last
        group wanted -- into a buffer of the message's length that holds
        nothing else (never read). Where that saves little (most of it
        wanted, a complex field's last groups, PNG and CCSDS, which say
        where a value ends only once decoded up to it) the rest is fetched
        into the same buffer, whole. Entries it cannot (a message of several
        fields or reusing a bitmap, GRIB1, already fetched) are left out."""
        from . import _partial
        fe = self._fetch
        if fe is None or not getattr(fe, 'remote', True):
            return {}

        def fill(buf, e, a, b):
            buf[a:b] = np.frombuffer(_remote.fetch(fe.url, e.offset + a, e.offset + b - 1),
                                     np.uint8)

        def one(e):
            if e.sub or e.offset in fe.cache:
                return None
            got = self._head(e)
            if got is None:
                return None
            head, total, r = got
            buf = np.zeros(total, np.uint8)
            buf[:len(head)] = np.frombuffer(head, np.uint8)
            buf[total - 4:] = np.frombuffer(b'7777', np.uint8)
            r = copy.copy(r)
            have = len(head)
            lay = want(r) if _partial.can(r) else None
            spans, whole, gap = None, False, _GAP
            if lay is not None:
                k, has = _partial.indices(buf, r, lay)
                if r.packing in (2, 3):         # complex: in order -- its group headers,
                    q = _partial.data_start(buf, r) - r.at      # then its values up to the
                    if q > have:                                # last group wanted
                        fill(buf, e, have, q)
                        have = q
                    stop = _partial.prefix_stop(buf, r, int(k[has].max()) + 1
                                                if has.any() else 0)
                    if stop is not None and stop - r.at <= _PREFIX_FETCH * total:
                        spans = [(have, stop - r.at)] if stop - r.at > have else []
                elif r.packing in _partial.DIRECT:
                    spans = [s for s in _partial.spans(buf, r, k[has]) if s[1] > have]
                elif r.packing in (41, 42):     # PNG, CCSDS: in order, ending who knows where
                    n_need = int(k[has].max()) + 1 if has.any() else 0
                    q = r.s7 + 5 - r.at                         # (their bytes' start)
                    nval = int.from_bytes(buf[r.s5 + 5:r.s5 + 9].tobytes(), 'big')
                    limit = int(_PREFIX_FETCH * total)
                    reach = q + int((total - q) * 1.15 * n_need / max(nval, 1)) + (16 << 10)
                    while True:                 # a guess from the share wanted, then on
                        reach = min(reach, limit)
                        if reach > have:        # by half as much again each time short
                            fill(buf, e, have, reach)
                            have = reach
                        if _partial.reaches(buf, r, n_need, r.at + have):
                            spans = []
                            break
                        if have >= limit:
                            break
                        reach = have + max(have - q, 64 << 10) // 2
                elif r.packing == 40 and has.any():     # JPEG2000: its headers, then the
                    spans = self._jpeg2000_part(buf, e, r, total, have, k[has], fill)
                    gap = 4 << 10                       # code-blocks' bodies a read decodes
                    #                                     (a few dozen: joined only if close)
            if spans is None:                   # the rest of it, after what is in hand
                spans, whole = [(have, total - 4)], True
            runs = []                           # the spans merged, ~32 KB apart at most
            for a, b in sorted(spans):
                if runs and a <= runs[-1][1] + gap:
                    runs[-1][1] = max(runs[-1][1], b)
                elif b > have:
                    runs.append([max(a, have), b])
            if whole and runs:                  # one request of it a connection
                a, b = runs[0]
                step = max(_remote.RUN_BYTES, -(-(b - a) // _remote.CONNECTIONS))
                runs = [[x, min(x + step, b)] for x in range(a, b, step)]
            if len(runs) > 1:                   # (side by side: each a round trip)
                with ThreadPoolExecutor(min(_remote.CONNECTIONS, len(runs))) as ex:
                    list(ex.map(lambda ab: fill(buf, e, *ab), runs))
            elif runs:
                fill(buf, e, *runs[0])
            if whole:                           # (kept: a later whole read fetches nothing)
                fe.keep(e.offset, e.offset + total - 1, buf.tobytes())
            r.index, r.msg, r.offset = e.index, e.msg, e.offset
            _remote.learn(e, r)
            return buf, r, whole
        with ThreadPoolExecutor(min(_remote.CONNECTIONS, max(1, len(entries)))) as ex:
            got = list(ex.map(one, entries))
        self._index.clear()
        return {(e.offset, e.sub): g for e, g in zip(entries, got) if g is not None}

    @staticmethod
    def _jpeg2000_part(buf, e, r, total, have, kk, fill):
        """_partial_remote()'s spans for a JPEG2000 field's values kk: its
        codestream's main and tile-part headers, then each packet's header
        (after the bodies before it: one by one, 4 KB each and twice as
        much while one runs on, the bodies between skipped), then the
        bodies of the code-blocks decode(want=) decodes -- or None (the
        rest whole) where that would be most of the message, or the
        codestream is not what it should be."""
        from . import _jpeg2000
        start, stop = r.s7 + 5 - r.at, total - 4
        held = [[0, have]]                      # the message's bytes in hand: [start, stop)s
        limit = int(_PREFIX_FETCH * total)
        last, step = -1, 0

        def taken():
            return sum(b - a for a, b in held)
        for _ in range(64):
            spans_cs = [(max(a, start) - start, min(b, stop) - start) for a, b in held
                        if b > start and a < stop]
            try:
                rng = _jpeg2000.wanted(buf[start:stop], spans_cs or [(0, 0)], kk)
            except _jpeg2000._Need as n:
                a = start + n.args[0]
                step = 2 * step if a == last else 4 << 10       # (a header a few hundred
                last = a                                        # bytes: more if it runs on)
                b = min(stop, a + step)
                if a >= stop or taken() + b - a > limit:
                    return None
                fill(buf, e, a, b)
                held.append([a, b])
                held.sort()
                merged = [held[0]]
                for x in held[1:]:
                    if x[0] <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], x[1])
                    else:
                        merged.append(x)
                held = merged
                continue
            except Exception:                   # (damaged, or a JP2 file: read whole)
                return None
            spans = [(start + a, start + b) for a, b in rng
                     if not any(ha <= start + a and start + b <= hb for ha, hb in held)]
            if taken() + sum(b - a for a, b in spans) > limit:
                return None
            return spans
        return None

    def _grib1_part(self, recs, want):
        """_partial_remote()'s answer for a local file's GRIB1 fields:
        each simply packed one (its packed values copied as they are) made
        its GRIB2 message -- the headers made, the values one copy, no
        decoding -- for its points alone to be read from (ERA5, a point from
        each field: 0.5 ms rather than 4.7)."""
        fe = self._fetch
        if fe is None or getattr(fe, 'remote', True):
            return {}
        from . import _grib1
        out = {}
        for e in recs:
            plan = fe.plans.get(e.offset)
            if plan is None or plan[4][0] != 'copy' or e.sub:
                continue
            m = np.frombuffer(_grib1._assemble(fe._mv, plan), np.uint8)
            rs = G.scan(m, [], alone=True)
            if len(rs) != 1 or want(rs[0]) is None:
                continue
            out[(e.offset, e.sub)] = (m, rs[0], True)
        return out

    def _head(self, e):
        """(the first bytes of remote entry e's message, to section 7's
        start at least, its length, its Record as scanned from them) --
        kept, so learning a field (its grid, its winds' axes) and then
        reading it fetch its head once; None for a message of several
        fields, one reusing an earlier bitmap, or GRIB1."""
        hit = self._heads.get(e.offset)
        if hit is not None:
            return hit or None
        fe = self._fetch
        total = None if e.end is None else e.end - e.offset + 1
        if total is None:                       # (the file's last: its length from section 0)
            sec0 = _remote.fetch(fe.url, e.offset, e.offset + 15)
            if sec0[:4] != b'GRIB' or sec0[7] != 2:
                self._heads[e.offset] = False
                return None
            total = int.from_bytes(sec0[8:16], 'big')
        head = _remote.fetch(fe.url, e.offset, e.offset + min(total, _HEAD) - 1)
        out = False
        if head[:4] == b'GRIB' and head[7] == 2:
            total = total or int.from_bytes(head[8:16], 'big')
            p = 16
            while True:                         # to section 7's start, fetching on if need be
                if p + 6 > len(head):
                    more = min(total, max(p + 6, len(head) * 2))
                    if more <= len(head):
                        break
                    head += _remote.fetch(fe.url, e.offset + len(head), e.offset + more - 1)
                    continue
                ln, num = int.from_bytes(head[p:p + 4], 'big'), head[p + 4]
                if num == 6 and head[p + 5] == 254 or ln < 5:
                    break                       # (an earlier message's bitmap)
                if num == 7:
                    if p + ln == total - 4:     # (else a message of several fields)
                        buf = np.zeros(total, np.uint8)
                        buf[:len(head)] = np.frombuffer(head, np.uint8)
                        buf[total - 4:] = np.frombuffer(b'7777', np.uint8)
                        bad = []
                        recs = G.scan(buf, bad, alone=True)
                        if not bad and len(recs) == 1:
                            out = (head, total, recs[0])
                    break
                p += ln
        self._heads[e.offset] = out
        return out or None

    def _learn(self, entries):
        """Remote entries' grids and winds' axes, from their heads (fetched
        side by side, kept): what an index does not say, and at() and
        regrid() need before reading -- which components to turn."""
        fe = self._fetch
        todo = [e for e in entries if isinstance(e, _remote.Entry) and e.grid is None and
                not e.sub and e.offset not in self._heads]
        if fe is None or not getattr(fe, 'remote', True) or not todo:
            return
        with ThreadPoolExecutor(min(_remote.CONNECTIONS, len(todo))) as ex:
            heads = list(ex.map(self._head, todo))
        for e, h in zip(todo, heads):
            if h is not None:
                r = copy.copy(h[2])
                r.index, r.msg, r.offset = e.index, e.msg, e.offset
                _remote.learn(e, r)
        self._index.clear()

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


def _box_points(g, win):
    """The read() layout indices (rows, columns) of a box's window of grid g."""
    rows = np.arange(win.rows.start, win.rows.stop, dtype=np.int64)
    cols = (np.arange(win.cols.start, win.cols.stop, dtype=np.int64)
            if isinstance(win.cols, slice) else np.asarray(win.cols, np.int64))
    return rows[:, None] * g.ni + cols[None, :]


def _own(idx):
    """A plan's corners as indices into its own corner values ((n, 4)
    raveled), -1 kept: what apply() takes with values read at the corners."""
    return np.where(idx >= 0, np.arange(idx.size, dtype=np.int64).reshape(idx.shape), -1)


def _corner_count(idx):
    """How many of the grid's points a plan's corners take."""
    return int(np.unique(idx[idx >= 0]).size)


def expand_grid(values, grid, nx=None, out=None):
    """Fields on a reduced grid (a Grid, or a Record or GribFile's), as
    read() gives them ((points,) or (n, points)), on its regular grid:
    (rows, nx) a field, nx by default its longest row's points (a reduced
    Gaussian grid's regular Gaussian one), south to north -- each row
    interpolated along itself as pygrib's redtoreg does, the same values
    but for float32's rounding (linear, round the globe; beside a missing
    value the nearer point's). Values on any other grid: as they are.
    GribFile.latlon(expand=True) gives its points."""
    if isinstance(grid, GribFile):
        grid = grid.grid()
    elif isinstance(grid, (G.Record, _remote.Entry)):
        grid = grid.grid
    if not isinstance(grid, _grids.Grid):
        raise TypeError('expand onto the regular grid of a Grid, Record or GribFile, not %r'
                        % (grid,))
    if hasattr(values, 'get') and not isinstance(values, np.ndarray):    # cupy
        return _to_card(_regrid().expand(values.get(), _grids.layout(grid), nx), None)
    return _regrid().expand(values, _grids.layout(grid), nx, out)


def _regrid():
    from . import _regrid as rg
    return rg


def _expandable(recs):
    return any(_regrid().expandable(r.grid) for r in recs if r.grid is not None)


def _to_card(a, out):
    """a (numpy) on the GPU: into out (a cupy array) when given."""
    import cupy as cp
    if out is None:
        return cp.asarray(a)
    out.set(a)
    return out


def _packer(head):
    """The module that unpacks data starting with these bytes (gzip, bzip2,
    xz: GRIB archives some centres compress whole), else None."""
    if head[:2] == b'\x1f\x8b':
        import gzip
        return gzip
    if head[:3] == b'BZh':
        import bz2
        return bz2
    if head[:6] == b'\xfd7zXZ\x00':
        import lzma
        return lzma
    return None


def _unpacked(data):
    """data, decompressed if it is gzip, bzip2 or xz (concatenated streams
    too, as gzip writes appended files); else as it is."""
    head = bytes(data[:6])
    mod = _packer(head)
    return data if mod is None else mod.decompress(bytes(data))


def _check_limits(max_records, byte_range):
    """(start, stop) of open()'s byte_range, its arguments checked."""
    if max_records is not None and (not isinstance(max_records, (int, np.integer))
                                    or max_records < 1):
        raise ValueError('max_records: a whole number, 1 or more')
    if byte_range is None:
        return 0, None
    try:
        start, stop = byte_range
    except (TypeError, ValueError):
        raise TypeError('byte_range: (start, stop), stop None for the end')
    start = int(start or 0)
    if start < 0 or (stop is not None and int(stop) <= start):
        raise ValueError('byte_range: 0 <= start < stop')
    return start, None if stop is None else int(stop)


def _limited(recs, max_records, start, stop):
    """recs whose messages start in [start, stop), the first max_records."""
    if start or stop is not None:
        recs = [r for r in recs if r.offset >= start and (stop is None or r.offset < stop)]
    return recs[:max_records] if max_records is not None else recs


def _count(b):
    """How many fields the whole messages in b hold (GRIB1's one each)."""
    g1 = []
    return len(G.scan(np.frombuffer(b, np.uint8), [], grib1=g1)) + len(g1)


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
