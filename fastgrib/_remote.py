"""GRIB2 over HTTP(S) a field at a time: the inventory NOAA (.idx, wgrib2
-s lines) or ECMWF (.index, JSON lines) publishes beside each file says
where every message lies, so only the messages picked are fetched, by
byte range. s3://bucket/key is read as the bucket's public HTTPS address.
"""
import json
import os
import re
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from . import _meta
from .tables import levels as _levels

CONNECTIONS = 8               # parallel range requests
CACHE_BYTES = 256 << 20       # fetched messages kept for later reads
RUN_BYTES = 4 << 20           # adjacent messages fetched as one request up to this, or
                              # a CONNECTIONS-th of what is asked for when that is more
_UA = {'User-Agent': 'fastgrib'}
http = urllib = None          # http.client and urllib, imported by _net() when first needed


def _net():
    """Python's HTTP and URL modules, imported with the first request rather
    than with fastgrib (a third of its own import time: local files never
    need them)."""
    global http, urllib
    if urllib is None:
        import http.client
        import urllib.error
        import urllib.parse
        import urllib.request


def https(url):
    """s3://bucket/key as HTTPS; anything else as it is."""
    m = re.match(r's3://([^/]+)/(.*)', url)
    return 'https://%s.s3.amazonaws.com/%s' % m.groups() if m else url


def is_url(src):
    return isinstance(src, str) and re.match(r'(https?|s3)://', src) is not None


def fetch(url, start=None, end=None, retries=3, timeout=60):
    """The bytes of url, or of [start, end] (end None: to the end)."""
    _net()
    head = dict(_UA)
    if start is not None:
        head['Range'] = 'bytes=%d-%s' % (start, '' if end is None else end)
    err = None
    for _ in range(retries):
        try:
            if _proxied():
                with urllib.request.urlopen(urllib.request.Request(url, headers=head),
                                            timeout=timeout) as r:
                    data = r.read()
            else:
                data = _get(url, head, timeout)
            if start is not None and end is not None and len(data) != end - start + 1:
                raise IOError('%s: asked for %d bytes, got %d' % (url, end - start + 1, len(data)))
            return data
        except urllib.error.HTTPError as e:
            if e.code in (403, 404, 416):
                raise
            err = e
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            err = e
    raise IOError('%s: %s' % (url, err))


# Connections kept open between requests, a few a host: a range request on a
# new one pays its TCP and TLS set-up (~100 ms to S3) before any byte
_IDLE = {}                    # (scheme, host) -> [connections]
_IDLE_LOCK = threading.Lock()
_PROXIES = None


def _forked():
    """In a child of fork(): the kept connections are the parent's (their
    sockets shared with it), so the child makes its own."""
    global _IDLE, _IDLE_LOCK
    _IDLE = {}
    _IDLE_LOCK = threading.Lock()


if hasattr(os, 'register_at_fork'):
    os.register_at_fork(after_in_child=_forked)


def _proxied():
    """Whether proxies are set (then urllib, which goes through them)."""
    global _PROXIES
    _net()
    if _PROXIES is None:
        _PROXIES = bool(urllib.request.getproxies())
    return _PROXIES


def _get(url, head, timeout, hops=5):
    """GET url on a kept connection (redirects followed); HTTPError for
    an error status. A kept connection the server has closed meanwhile is
    replaced once."""
    _net()
    u = urllib.parse.urlsplit(url)
    key = (u.scheme, u.netloc)
    path = (u.path or '/') + ('?' + u.query if u.query else '')
    with _IDLE_LOCK:
        idle = _IDLE.setdefault(key, [])
        c = idle.pop() if idle else None
    kept = c is not None
    if c is None:
        cls = http.client.HTTPSConnection if u.scheme == 'https' else http.client.HTTPConnection
        c = cls(u.netloc, timeout=timeout)
    try:
        c.request('GET', path, headers=head)
        r = c.getresponse()
        body = r.read()
    except (http.client.HTTPException, ConnectionError, OSError):
        c.close()
        if kept:                                # stale: once more, on a new one
            return _get(url, head, timeout, hops)
        raise
    if r.will_close:
        c.close()
    else:
        with _IDLE_LOCK:
            if len(_IDLE[key]) < 2 * CONNECTIONS:
                _IDLE[key].append(c)
            else:
                c.close()
    if r.status in (301, 302, 303, 307, 308) and hops and r.getheader('Location'):
        return _get(urllib.parse.urljoin(url, r.getheader('Location')), head, timeout, hops - 1)
    if r.status >= 400:
        raise urllib.error.HTTPError(url, r.status, r.reason, r.headers, None)
    return body


class Entry:
    """A field as an index lists it, before its bytes are fetched: what a
    Record says that the index says too (name, level, fcst, times), and
    where its message lies (offset, end)."""
    __slots__ = ('index', 'msg', 'sub', 'offset', 'end', 'date', 'center', 'tables', 'ckeys',
                 'name', 'long_name', 'units', 'level', 'level_type', 'level_value',
                 'level_type2', 'level_value2', 'fcst', 'extra', 'ref_time', 'valid_time',
                 'member', 'ensemble', 'grid', 'uv_relative', 'line', '_short')

    def __repr__(self):
        return self.line

    @property
    def short_name(self):
        """ecCodes' shortName: an ECMWF index's own; for a NOAA one, by
        ecCodes' concepts from what its line says."""
        if self._short is not None:
            return self._short
        hit = _meta.concept(self.ckeys, self.center, self.tables)
        return hit[0] if hit else None

    level_kind = _meta.Record.level_kind
    fcst_hours = _meta.Record.fcst_hours
    as_dict = _meta.Record.as_dict


def inventory(url):
    """The fields of a remote GRIB2 from its index (NOAA's .idx or
    ECMWF's .index; for GRIB1, wgrib's .inv), None when there is none."""
    grib1 = re.search(r'\.(grb|grib|grb1|grib1)$', url) is not None
    for suffix, parse in (('.idx', _noaa), ('.index', _ecmwf), ('.inv', _wgrib)):
        if suffix == '.inv' and not grib1:
            continue
        base = (re.sub(r'\.grib2?$', '', url) if suffix == '.index' else
                re.sub(r'\.(grb|grib|grb1|grib1)$', '', url) if suffix == '.inv' else url)
        try:
            text = fetch(base + suffix).decode('utf-8', 'replace')
        except (urllib.error.HTTPError, IOError):
            continue
        entries = parse(text)
        if entries:
            ends = sorted({e.offset for e in entries})
            nxt = dict(zip(ends, ends[1:] + [None]))
            for e in entries:
                if e.end is None and nxt[e.offset] is not None:
                    e.end = nxt[e.offset] - 1
            return entries
    return None


def _noaa(text):
    out = []
    for line in text.splitlines():
        line = re.sub(r'^(\d+(?:\.\d+)?):\1:', r'\1:', line.strip())   # NBM repeats some numbers
        f = line.split(':')
        if len(f) < 6 or not f[1].isdigit():
            continue
        e = Entry()
        num = f[0].split('.')
        e.index = len(out) + 1
        e.msg, e.sub = int(num[0]), int(num[1]) if len(num) > 1 else 0
        e.offset, e.end = int(f[1]), None
        e.date = f[2][2:] if f[2].startswith('d=') else f[2]
        e.name, e.level, e.fcst = f[3], f[4], f[5]
        rest = [x for x in f[6:] if x.strip()]
        e.extra = ''.join(x + ':' for x in rest)
        e.line = '%s:%d:d=%s:%s:%s:%s:%s' % (f[0], e.offset, e.date, e.name, e.level, e.fcst,
                                             e.extra)
        e.center, e.tables, e._short = _meta.NCEP, (2, 1), None
        _describe(e)
        out.append(e)
    return out


def _wgrib(text):
    """wgrib's GRIB1 inventory with the PDS numbers (NCEI's NARR has
    them: n:offset:D=YYYYMMDDHH:NAME:level:kpds=param,type,value:time:...),
    worded as the records read from it will be."""
    from . import _grib1
    lines = []
    for line in text.splitlines():
        f = line.strip().split(':')
        k = next((i for i, x in enumerate(f) if x.startswith('kpds=')), None)
        if k is None or k + 1 >= len(f) or len(f) < 7 or not f[1].isdigit():
            continue
        try:
            par, lt, lv = (int(x) for x in f[k][5:].split(','))
        except ValueError:
            continue
        t1, f1, t2, f2 = _grib1._level(lt, lv >> 8, lv & 0xFF)
        level = _meta._level(_meta.NCEP, t1, f1[0], f1[1], t2, f2[0], f2[1])
        fcst = f[k + 1]
        m = re.match(r'^(?:(\d+)-)?(\d+)hr (fcst|acc|ave)$', fcst)
        if m:
            a, b, kind = m.groups()
            fcst = ('%s hour fcst' % b if a is None else
                    '%s-%s hour %s fcst' % (a, b, kind))
        date = f[2][2:] if f[2][:2] in ('D=', 'd=') else f[2]
        lines.append('%s:%s:d=%s:%s:%s:%s:' % (f[0], f[1], date, _grib1.RENAMED.get(f[3], f[3]),
                                               level, fcst))
    return _noaa('\n'.join(lines))


def _ecmwf(text):
    out = []
    for i, line in enumerate(text.splitlines()):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if '_offset' not in d:
            continue
        e = Entry()
        e.index, e.msg, e.sub = len(out) + 1, len(out) + 1, 0
        e.offset, e.end = int(d['_offset']), int(d['_offset']) + int(d['_length']) - 1
        e.date = '%s%s' % (d.get('date', ''), d.get('time', '')[:2])
        e.center, e.tables = _meta.ECMWF, (0, 0)
        e._short = d.get('param', '?')
        lev = d.get('levelist')
        e.level = '%s %s' % (lev, d.get('levtype', '')) if lev else d.get('levtype', '')
        step = d.get('step', '0')
        e.fcst = 'anl' if step in ('0', 0) else '%s hour fcst' % step
        e.extra = ''.join('%s=%s:' % (k, d[k]) for k in ('number', 'type', 'stream') if k in d)
        e.line = '%d:%d:d=%s:%s:%s:%s:%s' % (e.index, e.offset, e.date, e._short, e.level,
                                             e.fcst, e.extra)
        num = d.get('number')
        e.member = int(num) if num is not None else None
        e.ensemble = {'cf': 'control', 'pf': 'perturbed', 'em': 'ens mean',
                      'es': 'ens std dev'}.get(d.get('type'))
        _ecmwf_param(e, d.get('levtype'), lev)
        e.ref_time = _date(e.date)
        try:
            e.valid_time = e.ref_time + timedelta(hours=float(step)) if e.ref_time else None
        except ValueError:
            e.valid_time = None
        e.grid, e.uv_relative = None, None
        out.append(e)
    return out


# the level wordings _meta writes, read back: [(regex, type)], and the layers
# of one type [(regex, type)]
_LEVELS, _LAYERS = [], []
_READY = set()                # the tables below made whole (each once, under _TABLES)
_TABLES = threading.Lock()


def _table(name, build):
    """build() the table called name, once, before anyone reads it."""
    if name not in _READY:
        with _TABLES:
            if name not in _READY:
                build()
                _READY.add(name)


def _rx(fmt):
    return re.compile('^%s$' % re.escape(fmt).replace(re.escape('%g'), r'(-?[\d.]+(?:e[-+]?\d+)?|inf)')
                      .replace(re.escape('%%'), '%'))


def _level_words():
    seen = set()
    for t, fmt in list(enumerate(_levels.WMO)) + [(192 + k, f) for k, f in enumerate(_levels.NCEP)]:
        if fmt != 'reserved' and fmt not in seen:
            seen.add(fmt)
            _LEVELS.append((_rx(fmt), t))
    for t, (fmt, _) in _meta._LAYER.items():
        _LAYERS.append((_rx(fmt), t))


def _parse_level(text):
    """(type, value, type2, value2) of a level as wgrib2 words it; type 255
    where it cannot be told."""
    _table('levels', _level_words)
    if text == 'atmos col':
        return 1, None, 8, None
    for rx, t in _LAYERS:
        m = rx.match(text)
        if m:
            return t, float(m.group(1)), t, float(m.group(2))
    parts = text.split(' - ')
    out = []
    for p in parts[:2]:
        for rx, t in _LEVELS:
            m = rx.match(p)
            if m:
                out += [t, float(m.group(1)) if m.groups() else None]
                break
        else:
            return 255, None, 255, None
    return tuple(out) + (255, None) * (2 - len(parts[:2]))


_BY_NAME = {}


def _describe(e):
    """What the index line says about e: parameter, level, times."""
    _table('names', lambda: [_BY_NAME.setdefault(v[0], v) for v in _meta.PARAMS.values()])
    m = re.match(r'^var discipline=(\d+) (?:master_table=\d+|center=\d+ local_table=\d+) '
                 r'parmcat=(\d+) parm=(\d+)$', e.name)
    if m:                                       # one wgrib2's tables do not know
        e.name = 'var%s_%s_%s' % m.groups()
    hit = _BY_NAME.get(e.name)
    e.long_name, e.units = (hit[1], hit[2]) if hit else ('unknown', 'unknown')
    if 'prob fcst' in e.extra:                  # a probability: percent, as its record says
        e.units = '%'
    e.level_type, e.level_value, e.level_type2, e.level_value2 = _parse_level(e.level)
    e.ref_time = _date(e.date)
    e.valid_time = None
    m = re.match(r'^(?:(\d+)-)?(\d+) (min|hour|day|sec) .*fcst$', e.fcst)
    if e.fcst == 'anl':
        e.valid_time = e.ref_time
    elif m and e.ref_time:
        step = {'min': 60, 'hour': 3600, 'day': 86400, 'sec': 1}[m.group(3)]
        e.valid_time = e.ref_time + timedelta(seconds=int(m.group(2)) * step)
    e.member, e.ensemble = _members(e.extra)
    e.ckeys = _entry_keys(e, hit)
    e.grid, e.uv_relative = None, None


_STAT_CODE = {v: k for k, v in _meta._STAT.items()}
_UNIT_CODE = {v: k for k, v in _meta._UNIT.items()}
_BY_DCN = {}


def _encoded(v):
    """(scale factor, scaled value) a producer would write v as."""
    if v is None:
        return -1, -1
    for sf in range(0, 7):
        sv = v * 10 ** sf
        if abs(sv - round(sv)) < 1e-6 * max(1.0, abs(sv)):
            return sf, int(round(sv))
    return -1, -1


def _entry_keys(e, hit):
    """ecCodes concept keys (as _meta's) from what a NOAA line says: the
    parameter's numbers, the level types and values (written as a
    producer would), the statistic over how long; None if the name is
    not in NCEP's table."""
    _table('numbers', lambda: [_BY_DCN.setdefault(v[0], k) for k, v in _meta.PARAMS.items()])
    m = re.match(r'^var(\d+)_(\d+)_(\d+)$', e.name)
    dcn = tuple(map(int, m.groups())) if m else _BY_DCN.get(e.name)
    if dcn is None:
        return None
    out = list(dcn) + [None] * 15 + [None] * 12 + [0] * 5 + [None]
    if e.level_type != 255:
        v1 = e.level_value * 100 if e.level_type in (100, 108) and e.level_value is not None \
            else e.level_value
        v2 = e.level_value2 * 100 if e.level_type2 in (100, 108) and e.level_value2 is not None \
            else e.level_value2
        out[3:9] = (e.level_type,) + _encoded(v1) + (e.level_type2,) + _encoded(v2)
    m = re.match(r'^(\d+)-(\d+) (\S+) (\S+) fcst$', e.fcst)
    if m and m.group(4) in _STAT_CODE and m.group(3) in _UNIT_CODE:
        out[9] = _STAT_CODE[m.group(4)]
        out[10] = int(m.group(2)) - int(m.group(1))
        out[11] = _UNIT_CODE[m.group(3)]
        if out[11] == 2:                        # wgrib2 shows whole days; written in hours
            out[10], out[11] = out[10] * 24, 1
    out[12] = e.tables[1]
    return tuple(out)


_BY_SHORT = {}


def _short_names():
    """ecCodes' shortNames -> ((discipline, category, number), concept)."""
    from .tables import eccodes as _eccodes
    rank = {}
    for dcn, ents in _eccodes.CONCEPTS.items():
        for ent in ents:                    # ECMWF's own numbers first, as it mostly
            r = next((k for k, bit in enumerate((8, 4, 2, 1)) if ent[3] & bit), None)
            if r is None:                   # writes them; its GRIB1-coded ones last
                continue
            r += 10 * (dcn[0] >= 192)
            if r < rank.get(ent[0], 99):
                rank[ent[0]] = r
                _BY_SHORT[ent[0]] = (dcn, ent)


def _ecmwf_param(e, levtype, levelist):
    """An ECMWF index entry named, levelled and described as its record
    will be: its shortName's concept gives the parameter's numbers (and
    NCEP's name for them where NCEP's table has one), and often the level."""
    _table('shorts', _short_names)
    got = _BY_SHORT.get(e._short)
    e.ckeys = None
    e.level_type = e.level_type2 = 255
    e.level_value = e.level_value2 = None
    e.name, e.long_name, e.units = e._short, 'unknown', 'unknown'
    if got is None:
        return
    (d, c, n), ent = got
    name, long_name, units = _meta.param(d, c, n, _meta.ECMWF)
    e.name, e.long_name, e.units = ((name, long_name, units) if long_name != 'unknown'
                                    else ent[:3])
    cond = dict(ent[4])
    t1 = cond.get(3)
    if t1 is not None:                          # the concept fixes the level
        sf1, sv1 = cond.get(4, -1), cond.get(5, -1)
        t2, sf2, sv2 = cond.get(6, 255), cond.get(7, -1), cond.get(8, -1)
    elif levtype in _LEVTYPE and (levelist is not None or levtype == 'sfc'):
        t1, t2 = _LEVTYPE[levtype]
        lv = int(levelist) if levelist is not None else None
        sf1, sv1, sf2, sv2 = -1, -1, -1, -1
        if levtype == 'pl':
            sf1, sv1 = 0, lv * 100
        elif levtype == 'sol':                  # soil layer n: levels n-1 to n
            sf1, sv1, sf2, sv2 = 0, lv - 1, 0, lv
        elif levtype in ('ml', 'pt'):
            sf1, sv1 = 0, lv
    else:
        return
    raw = lambda x, n: (1 << 8 * n) - 1 if x == -1 else x   # noqa: E731
    sf1, sv1, sf2, sv2 = raw(sf1, 1), raw(sv1, 4), raw(sf2, 1), raw(sv2, 4)
    e.level = _meta._level(_meta.ECMWF, t1, sf1, sv1, t2, sf2, sv2)
    e.level_type, e.level_type2 = t1, t2
    e.level_value, e.level_value2 = _meta._surface(sf1, sv1, t1), _meta._surface(sf2, sv2, t2)


# ECMWF's levtype: (first surface type, second) of its levels
_LEVTYPE = {'pl': (100, 255), 'sol': (151, 151), 'ml': (105, 255), 'pt': (107, 255),
            'sfc': (1, 255)}


_DERIVED_WORDS = {w: w for w in list(_meta._DERIVED.values()) + list(_meta._DERIVED_NCEP.values())}


def _members(extra):
    """(member, ensemble) from the ensemble wording _meta writes."""
    for item in extra.split(':'):
        m = re.match(r'^(MM-)?ENS=(?:(hi-res|low-res) ctl|([-+]?)(\d+))$', item)
        if m:
            if m.group(2):
                return 0, 'control'
            n = int(m.group(4))
            return ((n, 'multi-model') if m.group(1) else
                    (-n if m.group(3) == '-' else n, 'perturbed'))
        if item in _DERIVED_WORDS:
            return None, item
    return None, None


def _date(s):
    try:
        return datetime.strptime(s[:10], '%Y%m%d%H')
    except ValueError:
        return None


def learn(e, r):
    """What an index entry says, set right by its fetched record (an
    index names some fields ambiguously): later picks are then exact."""
    for k in ('name', 'long_name', 'units', 'level', 'level_type', 'level_value',
              'level_type2', 'level_value2', 'member', 'ensemble', 'valid_time', 'grid',
              'uv_relative'):
        setattr(e, k, getattr(r, k))
    e.center, e.tables, e.ckeys = r.center, r.tables, r.ckeys
    if e._short is None:
        e._short = r.short_name


def _sections(m):
    """(position, length, number) of each section of a GRIB2 message past
    section 0, while they make sense."""
    p, end = 16, len(m) - 4
    while p + 6 <= end:
        ln, num = int.from_bytes(m[p:p + 4], 'big'), m[p + 4]
        if ln < 5 or p + ln > end:
            return
        yield p, ln, num
        p += ln


def _bitmaps(m):
    return [(p, ln, m[p + 5]) for p, ln, num in _sections(m) if num == 6]


def lacks_bitmap(m):
    """Whether a message's first bitmap section says 254 ('the one given
    before'), so the bitmap is an earlier message's."""
    bm = _bitmaps(m)
    return bool(bm) and bm[0][2] == 254


def last_bitmap(m):
    """The last bitmap section a message defines (indicator 0), as bytes;
    None if it defines none."""
    got = [(p, ln) for p, ln, ind in _bitmaps(m) if ind == 0]
    return bytes(m[got[-1][0]:sum(got[-1])]) if got else None


def with_bitmap(m, sec):
    """The message with its first bitmap section (254) replaced by sec, an
    earlier message's -- whole on its own, as it reads within its file."""
    p, ln, _ = _bitmaps(m)[0]
    out = bytearray(m[:p]) + sec + m[p + ln:]
    out[8:16] = len(out).to_bytes(8, 'big')
    return bytes(out)


class Fetcher:
    """Messages of one remote file, fetched by byte range and kept (up to
    CACHE_BYTES, oldest out) for later reads."""

    def __init__(self, url):
        self.url = url
        self.cache = OrderedDict()                  # offset -> bytes
        self.held = 0
        self.lock = threading.Lock()

    def messages(self, spans):
        """{offset: bytes} for [(offset, end)] -- adjacent messages in one
        request, CONNECTIONS requests at once."""
        out, todo = {}, []
        with self.lock:
            for s in sorted(set(spans)):
                if s[0] in self.cache:
                    self.cache.move_to_end(s[0])
                    out[s[0]] = self.cache[s[0]]
                else:
                    todo.append(s)
        # adjacent messages in one request, but no request so big the others
        # idle: a whole file goes as CONNECTIONS requests at once
        known = sum(b - a + 1 for a, b in todo if b is not None)
        most = max(RUN_BYTES, -(-known // CONNECTIONS))
        runs = []                                   # [[start, end, [spans]]]
        for s in todo:
            last = runs[-1] if runs else None
            if last and last[1] is not None and s[0] == last[1] + 1 and                     (s[1] is None or s[1] - last[0] < most):
                last[1] = s[1]
                last[2].append(s)
            else:
                runs.append([s[0], s[1], [s]])
        with ThreadPoolExecutor(CONNECTIONS) as ex:
            got = list(ex.map(lambda r: fetch(self.url, r[0], r[1]), runs))
        for (start, _, parts), data in zip(runs, got):
            for a, b in parts:
                piece = data[a - start:None if b is None else b - start + 1]
                out[a] = piece
                with self.lock:
                    old = self.cache.pop(a, None)       # another thread fetched it too
                    if old is not None:
                        self.held -= len(old)
                    self.cache[a] = piece
                    self.held += len(piece)
                    while self.held > CACHE_BYTES and len(self.cache) > 1:
                        _, old = self.cache.popitem(last=False)
                        self.held -= len(old)
        return out
