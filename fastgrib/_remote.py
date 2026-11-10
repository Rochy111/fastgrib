"""GRIB2 over HTTP(S) a field at a time: the inventory NOAA (.idx, wgrib2
-s lines) or ECMWF (.index, JSON lines) publishes beside each file says
where every message lies, so only the messages picked are fetched, by
byte range. s3://bucket/key is read as the bucket's public HTTPS address.
"""
import json
import os
import random
import re
import threading
import time
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
BACKOFF = 0.5                 # seconds before the first retry, doubling (with jitter) after
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


def _wait(attempt, err):
    """Seconds before retry number attempt + 1: the server's Retry-After if it
    gave one (S3's 503 Slow Down, a 429), else BACKOFF doubling, jittered so
    that parallel requests do not all come back at once."""
    after = getattr(err, 'headers', None) and err.headers.get('Retry-After')
    if after and str(after).strip().isdigit():
        return min(float(after), 30.0)
    return BACKOFF * 2 ** attempt * (0.5 + random.random())


def fetch(url, start=None, end=None, retries=4, timeout=60):
    """The bytes of url, or of [start, end] (end None: to the end). A server
    error, a throttled request or a dropped connection is tried again after a
    growing wait, up to `retries` attempts in all."""
    _net()
    head = dict(_UA)
    if start is not None:
        head['Range'] = 'bytes=%d-%s' % (start, '' if end is None else end)
    err = None
    for attempt in range(retries):
        if attempt:
            time.sleep(_wait(attempt - 1, err))
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


def _get(url, head, timeout, hops=5, method='GET', info=None):
    """GET url on a kept connection (redirects followed); HTTPError for
    an error status. A kept connection the server has closed meanwhile is
    replaced once. info: a dict given the response's headers."""
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
        c.request(method, path, headers=head)
        r = c.getresponse()
        body = r.read()
    except (http.client.HTTPException, ConnectionError, OSError):
        c.close()
        if kept:                                # stale: once more, on a new one
            return _get(url, head, timeout, hops, method, info)
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
        return _get(urllib.parse.urljoin(url, r.getheader('Location')), head, timeout, hops - 1,
                    method, info)
    if r.status >= 400:
        raise urllib.error.HTTPError(url, r.status, r.reason, r.headers, None)
    if info is not None:
        info.update(r.headers.items())
    return body


def size(url, timeout=30):
    """The length in bytes of the file at url, as a HEAD request's
    Content-Length says, and whether the server takes byte ranges; (None,
    False) when it does not say or the request fails."""
    _net()
    info = {}
    try:
        if _proxied():
            req = urllib.request.Request(url, headers=_UA, method='HEAD')
            with urllib.request.urlopen(req, timeout=timeout) as r:
                info.update(r.headers.items())
        else:
            _get(url, dict(_UA), timeout, method='HEAD', info=info)
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return None, False
    low = {k.lower(): v for k, v in info.items()}
    n = low.get('content-length', '')
    return (int(n) if n.strip().isdigit() else None,
            low.get('accept-ranges', '').strip().lower() == 'bytes')


def fetch_into(url, start, out, piece=None):
    """Bytes start, start + 1, ... of url into out (a writable uint8 array,
    filled whole), as CONNECTIONS range requests at once: one request's
    bandwidth is a server's per-connection limit (S3: some tens of MB/s),
    not the network's."""
    n = len(out)
    piece = piece or max(RUN_BYTES, -(-n // CONNECTIONS))
    cuts = [(a, min(a + piece, n)) for a in range(0, n, piece)]

    def one(cut):
        a, b = cut
        out[a:b] = memoryview(fetch(url, start + a, start + b - 1))
    if len(cuts) == 1:
        one(cuts[0])
        return
    with ThreadPoolExecutor(min(CONNECTIONS, len(cuts))) as ex:
        list(ex.map(one, cuts))


def fetch_all(url):
    """The whole file at url: in parallel ranges when the server takes
    them and says how long it is, else one request."""
    n, ranges = size(url)
    if not ranges or n is None or n <= RUN_BYTES:
        return fetch(url)
    import numpy as np
    out = np.empty(n, np.uint8)
    fetch_into(url, 0, out)
    return out


WALK_BYTES = 8 << 20          # walk(): bytes fetched first, twice as many each time after


def _length(head):
    """A GRIB message's length from its first 16 octets (GRIB1's in 3,
    GRIB2's in 8); None for an ECMWF large GRIB1 (its length rounded up
    to 120 bytes)."""
    if head[:4] != b'GRIB':
        raise IOError('no GRIB message where the index says one starts')
    if head[7] == 1:
        n = int.from_bytes(head[4:7], 'big')
        return None if n & 0x800000 else n      # (ECMWF's large: rounded up, to the end)
    return int.from_bytes(head[8:16], 'big')


def _spans(b):
    """([(start, end)] of the whole GRIB messages in b, laid end to end as
    the scan lays them, (start, end) of one begun but not whole or None,
    where scanning stopped)."""
    out, pos, size = [], 0, len(b)
    while True:
        k = b.find(b'GRIB', pos)
        if k < 0:
            return out, None, max(pos, size - 3)    # (a 'GRIB' may straddle the end)
        if k + 16 > size:
            return out, (k, None), k
        if b[k + 7] == 1:                       # GRIB1: 3 octets (ECMWF: 120-byte units)
            n = int.from_bytes(b[k + 4:k + 7], 'big')
            n = (n & 0x7FFFFF) * 120 if n & 0x800000 else n
        else:
            n = int.from_bytes(b[k + 8:k + 16], 'big')
        if n < 16:                              # not a message: on to the next 'GRIB'
            pos = k + 4
            continue
        if k + n > size:
            return out, (k, k + n), k
        out.append((k, k + n))
        pos = k + n


def walk(url, start=0, stop=None, enough=None):
    """(bytes, the file offset of their first) of the whole messages of
    the file at url from the first at or after byte `start`, up to those
    starting before `stop` (None: the end), or fewer once enough(bytes)
    says so -- fetched a window at a time, WALK_BYTES first and twice as
    many each time, each window in parallel ranges; a message the window
    cuts short fetched to its end. Not a request a message: a message's
    length says where the next starts, so a request each is a round trip
    each (seconds for a file of hundreds)."""
    import numpy as np
    total, ranges = size(url)
    if total is None or not ranges:
        raise IOError('%s: the server does not say how long the file is or take byte ranges'
                      % url)
    end = total if stop is None else min(stop, total)
    buf = bytearray()
    want = min(WALK_BYTES, end - start) if enough is not None else end - start
    while True:
        lo = start + len(buf)
        hi = min(lo + max(want, 0), total)
        if hi > lo:
            got = np.empty(hi - lo, np.uint8)
            fetch_into(url, lo, got)
            buf += got.data
        spans, cut, at = _spans(buf)
        spans = [s for s in spans if start + s[0] < end]
        if start + len(buf) >= total or start + at >= end or \
                (enough is not None and spans and enough(bytes(buf[spans[0][0]:spans[-1][1]]))):
            break
        need = cut[1] - len(buf) if cut is not None and cut[1] is not None else 0
        want = max(min(2 * want, 32 * WALK_BYTES, end - start - len(buf)), need, 1 << 16)
    if not spans:
        return b'', start
    return bytes(buf[spans[0][0]:spans[-1][1]]), start + spans[0][0]


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


def inventory(url, index=None):
    """The fields of a remote GRIB2 from its index (NOAA's .idx or
    ECMWF's .index; for GRIB1, wgrib's .inv), None when there is none;
    index: one given (a path or URL), whichever kind it is."""
    if index is not None:
        if is_url(index):
            text = fetch(https(index)).decode('utf-8', 'replace')
        else:
            with open(index, encoding='utf-8', errors='replace') as fh:
                text = fh.read()
        first = next((ln.strip() for ln in text.splitlines() if ln.strip()), '')
        parse = _ecmwf if first.startswith('{') else _wgrib if 'kpds=' in first else _noaa
        entries = parse(text)
        if not entries:
            raise ValueError('%s: no index lines (wgrib2 -s, ECMWF .index or wgrib .inv)'
                             % index)
        return _ended(entries)
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
            return _ended(entries)
    return None


def _ended(entries):
    """entries, each ending where the next message starts (the last: as its
    own length says, once fetched)."""
    ends = sorted({e.offset for e in entries})
    nxt = dict(zip(ends, ends[1:] + [None]))
    for e in entries:
        if e.end is None and nxt[e.offset] is not None:
            e.end = nxt[e.offset] - 1
    return entries


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
                                             e.extra[:-1])     # (as wgrib2: no ':' last)
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
    m = re.match(r'^var discipline=(\d+) (?:master_table=\d+|center=(\d+) local_table=\d+) '
                 r'parmcat=(\d+) parm=(\d+)$', e.name)
    if m:                                       # one wgrib2's tables do not know: named as
        disc, center, cat, num = m.groups()     # the message itself is, once read
        e.name = _meta.param(int(disc), int(cat), int(num), int(center or -1))[0]
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
    if e.ckeys is not None and e.ckeys[9] == 1 and e.units != '%':
        e.units = _meta.accumulated(e.units)    # as the record will say
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


def cache_dir(cache):
    """The folder open(url, cache=...) keeps fetched messages in: cache
    itself (a path), or for True the environment's FASTGRIB_REMOTE_CACHE,
    else fastgrib/remote in the user's cache folder; None for none."""
    if cache is None or cache is False:
        return None
    if cache is True:
        cache = os.environ.get('FASTGRIB_REMOTE_CACHE')
        if not cache:
            base = (os.environ.get('LOCALAPPDATA') if os.name == 'nt' else
                    os.environ.get('XDG_CACHE_HOME')) or os.path.join(os.path.expanduser('~'),
                                                                      '.cache')
            cache = os.path.join(base, 'fastgrib', 'remote')
    return os.fspath(cache)


class Fetcher:
    """Messages of one remote file, fetched by byte range and kept (up to
    CACHE_BYTES, oldest out) for later reads -- and, given a folder, on
    disk: a message fetched once is read from there by any later process
    (a file a message, under a folder a URL; files written whole, then
    renamed into place, so a reader never sees half of one)."""

    def __init__(self, url, cache=None):
        self.url = url
        self.cache = OrderedDict()                  # offset -> bytes
        self.held = 0
        self.lock = threading.Lock()
        self.disk = None
        folder = cache_dir(cache)
        if folder is not None:
            import hashlib
            name = re.sub(r'[^A-Za-z0-9._-]+', '_', url.rsplit('/', 1)[-1])[-60:]
            self.disk = os.path.join(folder, hashlib.sha1(url.encode()).hexdigest()[:16] +
                                     '-' + name)

    def _disk_path(self, a, b):
        return os.path.join(self.disk, '%d-%s' % (a, 'end' if b is None else b))

    def _from_disk(self, a, b):
        if self.disk is None:
            return None
        path = self._disk_path(a, b)
        if b is None:                               # (its end found when it was fetched)
            try:
                hits = [n for n in os.listdir(self.disk)
                        if n.startswith('%d-' % a) and not n.endswith('.part')]
            except OSError:
                return None
            if not hits:
                return None
            path = os.path.join(self.disk, hits[0])
        try:
            with open(path, 'rb') as fh:
                data = fh.read()
        except OSError:
            return None
        end = path.rsplit('-', 1)[1]
        return data if end == 'end' or len(data) == int(end) - a + 1 else None

    def _to_disk(self, a, b, data):
        if self.disk is None:
            return
        try:
            os.makedirs(self.disk, exist_ok=True)
            path = self._disk_path(a, b)
            tmp = '%s.%d.%d.part' % (path, os.getpid(), threading.get_ident())
            with open(tmp, 'wb') as fh:
                fh.write(data)
            os.replace(tmp, path)
        except OSError:                             # a full or read-only disk: fetched all the same
            pass

    def keep(self, a, b, piece):
        """A message fetched otherwise (whole: a partial read's end) kept
        as messages() keeps one, in memory and on disk."""
        with self.lock:
            old = self.cache.pop(a, None)
            if old is not None:
                self.held -= len(old)
            self.cache[a] = piece
            self.held += len(piece)
            while self.held > CACHE_BYTES and len(self.cache) > 1:
                _, old = self.cache.popitem(last=False)
                self.held -= len(old)
        self._to_disk(a, b, piece)

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
        if self.disk is not None:                   # fetched before, by this or another process
            left = []
            for s in todo:
                data = self._from_disk(*s)
                if data is None:
                    left.append(s)
                else:
                    out[s[0]] = data
            todo = left
        # a message whose end the index does not give (its last): its own
        # length, from its first 16 octets -- not to the end of the file, which
        # an index of part of a file would make the rest of it
        for i, (a, b) in enumerate(todo):
            if b is None:
                n = _length(fetch(self.url, a, a + 15))
                todo[i] = (a, None if n is None else a + n - 1)
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
                self._to_disk(a, b, piece)
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
