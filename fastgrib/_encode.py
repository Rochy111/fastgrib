"""Writing GRIB2: a field's own message (cut from a message of several,
its bitmap made explicit, a GRIB1 one turned into GRIB2), with its
metadata set where asked -- its packed values untouched; and new values
packed as a field of the same grid and product: simple (5.0), complex
with second-order spatial differencing (5.3, as NCEP and ECMWF pack) or
IEEE (5.4, lossless), to the precision of the field they replace unless
asked otherwise."""
import math
import re
import struct
import threading
from datetime import datetime, timedelta

import numpy as np
from ._jit import njit

from . import _meta
from ._octets import octets, sm_bytes
from .tables.params import PARAMS

_SCALED = (0, 2, 3, 40, 41, 42)                 # packings with R, E, D and bits a value
_F32_MAX = float(np.finfo(np.float32).max)


# --- a field's sections

def sections(b, r):
    """{section number: (start, length)} of record r's message as it
    stands for r: the last section 1-4 before its packing (a message of
    several fields repeats some), its own 5, 6 and 7."""
    b = octets(b)
    pos = r.at                                  # (r.offset: its place in the file)
    end = pos + struct.unpack_from('>Q', b, pos + 8)[0]
    p, out = pos + 16, {}
    while p < r.s5:
        ln, num = struct.unpack_from('>IB', b, p)
        out[num] = (p, ln)
        p += ln
        if ln < 5 or p > end:
            raise ValueError('%r: its message does not hold together' % r)
    out[5] = (r.s5, r.s6 - r.s5)
    out[6] = (r.s6, r.s7 - r.s6)
    out[7] = (r.s7, r.s7end - r.s7)
    return out


def _bitmap_section(b, r, secs):
    """r's section 6 standing alone: 'the bitmap before' (254) made the
    bitmap it means."""
    p, ln = secs[6]
    if b[p + 5] != 254:
        return bytes(b[p:p + ln])
    n = (r.grid.ni * r.grid.nj + 7) // 8 if r.grid is not None else \
        (struct.unpack_from('>I', b, secs[3][0] + 6)[0] + 7) // 8
    return struct.pack('>IBB', 6 + n, 6, 0) + bytes(b[r.bitmap:r.bitmap + n])


def message(b, r, **changes):
    """The bytes of r's own GRIB2 message: as it is in b when it is one
    field's and nothing is changed, else made of its sections."""
    b = octets(b)                               # (its octets Python ints, numpy 1 or 2)
    secs = sections(b, r)
    if not changes and r.sub == 0 and b[secs[6][0] + 5] != 254:
        n = struct.unpack_from('>Q', b, r.at + 8)[0]
        return bytes(b[r.at:r.at + n])
    disc = b[r.at + 6]
    parts = {k: bytearray(b[p:p + ln]) for k, (p, ln) in secs.items() if k != 6}
    parts[6] = bytearray(_bitmap_section(b, r, secs))
    disc = _apply(parts, disc, r, changes)
    return _join(disc, parts)


def _join(disc, parts):
    """The message of sections {number: bytes, or a tuple of bytes-likes}
    -- each copied once."""
    pieces = []
    for k in (1, 2, 3, 4, 5, 6, 7):
        if k in parts:
            p = parts[k]
            pieces += p if isinstance(p, tuple) else (p,)
    n = 16 + sum(len(memoryview(p).cast('B')) for p in pieces) + 4
    return b''.join([b'GRIB', bytes([0, 0, disc, 2]), struct.pack('>Q', n)] + pieces +
                    [b'7777'])


# --- metadata set

_UNIT_CODE = {'min': 0, 'hour': 1, 'day': 2, 'sec': 13}
_UNIT_STEP = {0: timedelta(minutes=1), 1: timedelta(hours=1), 2: timedelta(days=1),
              13: timedelta(seconds=1)}
_STAT_CODE = {'ave': 0, 'acc': 1, 'max': 2, 'min': 3}
_BY_NAME = {}


def _param_of(name):
    if not _BY_NAME:
        for k in sorted(PARAMS, key=lambda k: (k[1] >= 192 or k[2] >= 192, k)):
            _BY_NAME.setdefault(PARAMS[k][0], k)
    if isinstance(name, tuple):
        return name
    hit = _BY_NAME.get(name)
    if hit is None:
        raise ValueError('parameter %r: not a name in the GRIB2 tables (give its '
                         '(discipline, category, number) instead)' % name)
    return hit


def _scaled_int(v):
    """(scale factor, scaled value) of a fixed surface's value."""
    if v is None:
        return 255, 0xFFFFFFFF
    for sf in range(10):
        s = v * 10 ** sf
        if abs(s - round(s)) < 1e-6 * max(1.0, abs(s)) and abs(round(s)) < 2 ** 31:
            return sf, int(round(s))
    raise ValueError('level value %r cannot be written' % v)


def _surface_bytes(t, v):
    if t in (100, 108) and v is not None:        # given in mb (as Records give it), stored Pa
        v = v * 100.0
    sf, sv = _scaled_int(v)
    if sv != 0xFFFFFFFF and sv < 0:
        sv = 0x80000000 | -sv
    return bytes([t, sf]) + struct.pack('>I', sv)


def _level_of(level):
    if isinstance(level, str):
        from ._remote import _parse_level
        t1, v1, t2, v2 = _parse_level(level)
        if t1 == 255:
            raise ValueError('level %r: not a level as wgrib2 words them' % level)
        return t1, v1, t2, v2
    t = tuple(level) + (255, None) * ((4 - len(level)) // 2)
    return t[:4]


def _fcst_of(fcst):
    """(statistic or None, start, length or 0, unit) of a forecast time as
    wgrib2 words it."""
    if fcst == 'anl':
        return None, 0, 0, 1
    m = re.match(r'^(\d+) (min|hour|day|sec) fcst$', fcst)
    if m:
        return None, int(m.group(1)), 0, _UNIT_CODE[m.group(2)]
    m = re.match(r'^(\d+)-(\d+) (min|hour|day|sec) (ave|acc|max|min) fcst$', fcst)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return _STAT_CODE[m.group(4)], a, b - a, _UNIT_CODE[m.group(3)]
    raise ValueError("forecast time %r: give it as wgrib2 words them ('anl', '6 hour fcst', "
                     "'0-6 hour acc fcst')" % fcst)


def _ref_time(s1):
    y, mo, d, h, mi, s = struct.unpack_from('>HBBBBB', s1, 12)
    return datetime(y, mo, d, h, mi, s)


def _apply(parts, disc, r, changes):
    """Changes made to the sections: name or param, level, date, fcst."""
    changes = dict(changes)
    s1, s4 = parts[1], parts[4]
    tmpl = struct.unpack_from('>H', s4, 7)[0]
    if 'name' in changes or 'param' in changes:
        d, c, n = _param_of(changes.pop('name', None) or changes.pop('param'))
        disc, s4[9], s4[10] = d, c, n
    if 'level' in changes:
        if tmpl not in _meta._WITH_LEVEL:
            raise ValueError('%r: product template 4.%d has no level to set' % (r, tmpl))
        t1, v1, t2, v2 = _level_of(changes.pop('level'))
        s4[22:34] = _surface_bytes(t1, v1) + _surface_bytes(t2, v2)
    old_ref = _ref_time(s1)
    ref = changes.pop('date', None)
    if ref is not None:
        if not isinstance(ref, datetime):
            ref = datetime.strptime(str(ref), '%Y%m%d%H')
        s1[12:19] = struct.pack('>HBBBBB', ref.year, ref.month, ref.day, ref.hour, ref.minute,
                                ref.second)
        end = _STAT_END.get(tmpl)
        if end is not None:                     # a statistic's end moves with it
            t = datetime(*struct.unpack_from('>HBBBBB', s4, end)) + (ref - old_ref)
            s4[end:end + 7] = struct.pack('>HBBBBB', t.year, t.month, t.day, t.hour, t.minute,
                                          t.second)
    else:
        ref = old_ref
    if 'fcst' in changes:
        parts[4] = s4 = _set_fcst(s4, tmpl, ref, changes.pop('fcst'), r)
    if changes:
        raise TypeError('cannot set %s (name, param, level, date and fcst can be)'
                        % ', '.join(sorted(changes)))
    return disc


_STAT_END = {8: 34, 11: 37}                     # where 4.8 / 4.11 give the statistic's end


def _set_fcst(s4, tmpl, ref, fcst, r):
    if tmpl not in (0, 1, 8, 11):
        raise ValueError('%r: forecast times of product template 4.%d cannot be set'
                         % (r, tmpl))
    stat, start, length, unit = _fcst_of(fcst)
    ens = tmpl in (1, 11)
    base = bytearray(s4[:37 if ens else 34])     # 4.0 / 4.1's octets
    base[17] = unit
    base[18:22] = struct.pack('>I', start)
    if stat is None:
        out = base
        new = 1 if ens else 0
    else:
        t = ref + _UNIT_STEP[unit] * (start + length)
        out = base + struct.pack('>HBBBBB', t.year, t.month, t.day, t.hour, t.minute,
                                 t.second) + bytes([1]) + struct.pack('>I', 0) + \
            bytes([stat, 2, unit]) + struct.pack('>I', length) + bytes([255]) + \
            struct.pack('>I', 0)
        new = 11 if ens else 8
    out[7:9] = struct.pack('>H', new)
    out[0:4] = struct.pack('>I', len(out))
    return out


# --- new values packed

def encode(b, r, values, order, npts, packing=None, decimals=None, bits=None, grid=None,
           **changes):
    """r's message with `values` (as read() lays them out) packed in place
    of its own. order: the layout index of each of the grid's npts points
    in the file's order (None: the same order). packing: 'simple',
    'complex' or 'ieee' (default: as r's -- complex for any compressed
    one, IEEE for IEEE); decimals and bits: the precision, by default r's
    (its decimal and binary scales: each value to within half of 2^E /
    10^D). grid: (section 3, ny, nx) of another grid the values lie on
    (a box cut from r's), in place of r's."""
    b = octets(b)                               # (its octets Python ints, numpy 1 or 2)
    if bits is not None and not 1 <= bits <= 30:
        raise ValueError('bits %r: 1 to 30 bits a value' % bits)
    if decimals is not None and not -20 <= decimals <= 20:
        raise ValueError('decimals %r: -20 to 20' % decimals)
    if r.grid is not None and r.grid.template in (50, 51, 52, 53):
        return _encode_spectral(b, r, values, packing, decimals, bits, changes)
    secs = sections(b, r)
    s5p = secs[5][0]
    drt = struct.unpack_from('>H', b, s5p + 9)[0]
    inherited = packing is None
    if packing is None:
        packing = {0: 'simple', 4: 'ieee', 200: 'ieee', 40: 'jpeg', 41: 'png',
                   42: 'ccsds'}.get(drt, 'complex')
    packing = _ALIAS.get(packing, packing)
    if packing not in _PACKINGS:
        raise ValueError('packing %r: %s' % (packing, ', '.join(repr(p) for p in _PACKINGS)))
    v = np.asarray(values)
    if v.dtype not in (np.float32, np.float64):
        v = v.astype(np.float64)
    v = np.ascontiguousarray(v).ravel()
    if v.size != npts:
        raise ValueError('%d values for a grid of %d points' % (v.size, npts))
    own = decimals is None and bits is None
    if drt in _SCALED:
        e0 = _meta._sm(struct.unpack_from('>H', b, s5p + 15)[0], 2)
        d0 = _meta._sm(struct.unpack_from('>H', b, s5p + 17)[0], 2)
        r0 = struct.unpack_from('>f', b, s5p + 11)[0] if own else None
    else:                                       # IEEE or run-length: float32's precision
        e0, d0, r0 = 0, 0, None
        if own:
            bits = 24
    d = decimals if decimals is not None else 0 if bits is not None and own is False else d0
    n_out = order.size if order is not None and order.size else v.size
    scale = 1.0 if packing == 'ieee' else 10.0 ** d
    holes = -1
    if (order is None or not order.size) and packing != 'ieee':
        holes, lo, hi = _extent(v, scale)       # no gaps: packed from v itself, times scale
    if holes == 0:
        dv, k = v, v.size
    else:                                       # its values, scaled, gaps squeezed out
        dv, ok = _scratch('dv', n_out, np.float64), _scratch('ok', n_out, np.bool_)
        k, lo, hi = _gather(v, np.empty(0, np.int64) if order is None else order, scale,
                            dv, ok)
        dv, scale = dv[:k], 1.0
    if packing != 'ieee' and not (math.isfinite(lo) and math.isfinite(hi)):
        raise ValueError('values must be finite (NaN for none) to pack as %s' % packing)
    if max(abs(lo), abs(hi)) > _F32_MAX and math.isfinite(lo) and math.isfinite(hi):
        raise ValueError('values%s beyond float32 (%.3g): %s' % (
            '' if packing == 'ieee' else ' times 10^%d' % d, max(abs(lo), abs(hi)),
            'GRIB2 holds no more' if packing == 'ieee' else 'fewer decimals'))
    if k == npts:
        s6 = struct.pack('>IBB', 6, 6, 255)
    else:
        bm = np.packbits(ok)
        s6 = struct.pack('>IBB', 6 + bm.size, 6, 0) + bm.tobytes()
    if packing == 'ieee':
        s5 = struct.pack('>IHB', k, 4, 1)
        s7 = dv.astype('>f4')
    else:
        g = r.grid
        rect = (g.nj, g.ni) if g is not None and k == npts and not g.pl else None
        if grid is not None:
            rect = grid[1:] if k == npts else None
        try:
            s5, s7 = _pack(dv, lo, hi, packing, e0, d, bits, e_given=own, ref0=r0, rect=rect,
                           scale=scale)
        except ValueError:
            if not (inherited and packing == 'jpeg'):
                raise                           # (JPEG2000 the field's: complex where it can't)
            s5, s7 = _pack(dv, lo, hi, 'complex', e0, d, bits, e_given=own, ref0=r0,
                           scale=scale)
    parts = {k: bytearray(b[p:p + ln]) for k, (p, ln) in secs.items() if k < 5}
    if grid is not None:
        parts[3] = bytearray(grid[0])
    parts[5] = bytearray(struct.pack('>IB', 5 + len(s5), 5) + s5)
    parts[6] = s6
    s7 = memoryview(s7).cast('B')               # (the packed bits: copied once, joining)
    parts[7] = (struct.pack('>IB', 5 + len(s7), 7), s7)
    disc = _apply(parts, b[r.at + 6], r, changes)
    return _join(disc, parts)


def _encode_spectral(b, r, values, packing, decimals, bits, changes):
    """encode() of spherical harmonics' coefficients (as read() gives
    them): 'simple' (5.50: the (0, 0) real part as it is, the rest packed)
    or 'complex' (5.51: a subset of the lowest wavenumbers as IEEE floats,
    the rest times (n (n + 1))^P, packed). By default as r's: its subset,
    Laplacian power and precision; from the other packing, 16 bits a value
    (and, to 5.51, the subset up to wavenumber 20 and P = 0.5)."""
    from ._spectral import count
    secs = sections(b, r)
    s5p = secs[5][0]
    drt = struct.unpack_from('>H', b, s5p + 9)[0]
    J, K, M = r.grid.params[:3]
    packing = {'spectral_simple': 'simple', 'spectral_complex': 'complex'}.get(
        packing, packing or ('complex' if drt == 51 else 'simple'))
    if packing not in ('simple', 'complex'):
        raise ValueError("spherical harmonics pack 'simple' (5.50) or 'complex' (5.51), "
                         'not %r' % packing)
    v = np.ascontiguousarray(values, np.float64).ravel()
    if v.size != count(J, K, M):
        raise ValueError('%d coefficients for a J=%d K=%d M=%d truncation (%d)'
                         % (v.size, J, K, M, count(J, K, M)))
    if not np.isfinite(v).all():
        raise ValueError('spherical harmonics must be finite (no gaps)')
    tmpl = 51 if packing == 'complex' else 50
    own = drt == tmpl and decimals is None and bits is None
    if own:
        e0 = _meta._sm(struct.unpack_from('>H', b, s5p + 15)[0], 2)
        d0 = _meta._sm(struct.unpack_from('>H', b, s5p + 17)[0], 2)
        r0 = struct.unpack_from('>f', b, s5p + 11)[0]
    else:
        e0, d0, r0 = 0, 0, None
        if bits is None and decimals is None:
            bits = 16
    d = decimals if decimals is not None else d0
    if tmpl == 50:
        rest, sub, tail = v[1:], b'', struct.pack('>f', v[0])
    else:
        if drt == 51:
            lap = _meta._sm(struct.unpack_from('>I', b, s5p + 20)[0], 4) * 1e-6
            js, ks, ms = struct.unpack_from('>HHH', b, s5p + 24)
            prec = b[s5p + 34]
        else:
            lap, js, ks, ms, prec = 0.5, min(20, J), min(20, K), min(20, M), 1
        n, m = _wavenumbers(J, K, M)
        inside = (m <= ms) & (n <= np.minimum(js + m, ks))
        rest = v[~inside] * (n[~inside] * (n[~inside] + 1.0)) ** lap
        sub = v[inside].astype('>f4' if prec == 1 else '>f8').tobytes()
        tail = sm_bytes(round(lap * 1e6), 4) + struct.pack('>HHHIB', js, ks, ms,
                                                           int(inside.sum()), prec)
    dv = rest * 10.0 ** d
    lo, hi = (float(dv.min()), float(dv.max())) if dv.size else (0.0, 0.0)
    if max(abs(lo), abs(hi)) > _F32_MAX:
        raise ValueError('coefficients times 10^%d beyond float32: fewer decimals' % d)
    s5, s7 = _pack(dv, lo, hi, 'simple', e0, d, bits, e_given=own, ref0=r0)
    s5 = struct.pack('>IH', v.size, tmpl) + s5[6:15] + tail     # (R, E, D, its bits)
    s7 = sub + np.asarray(s7, np.uint8).tobytes()
    parts = {k: bytearray(b[p:p + ln]) for k, (p, ln) in secs.items() if k < 5}
    parts[5] = bytearray(struct.pack('>IB', 5 + len(s5), 5) + s5)
    parts[6] = struct.pack('>IBB', 6, 6, 255)
    parts[7] = (struct.pack('>IB', 5 + len(s7), 7), s7)
    disc = _apply(parts, b[r.at + 6], r, changes)
    return _join(disc, parts)


def _wavenumbers(J, K, M):
    """(n, m) of each value of a J, K, M truncation, in read()'s order
    (each a real and an imaginary part)."""
    m = np.concatenate([np.full(min(J + mm, K) - mm + 1, mm) for mm in range(M + 1)])
    n = np.concatenate([np.arange(mm, min(J + mm, K) + 1) for mm in range(M + 1)])
    return np.repeat(n, 2), np.repeat(m, 2)


def _bitlen(x):
    return int(x).bit_length()


@njit(cache=True, nogil=True)
def _gather(v, order, scale, out, ok):
    """(how many of values v -- laid out as read() gives them -- are
    defined, their least and greatest times scale): those values times
    scale, in the file's order, into out from its start, and which points
    are defined into ok. order: each file point's index in v (empty: the
    same order). In the same order, every value scaled first and the
    undefined then squeezed out: one loop with no store held back by a
    branch, which vectorises (HRRR: 7.7 -> ~2 ms a field)."""
    n = order.size if order.size else v.size
    lo, hi = np.inf, -np.inf
    holes = 0
    if order.size:
        for i in range(n):
            x = v[order[i]]
            out[i] = x * scale
    else:
        for i in range(n):
            out[i] = v[i] * scale
    for i in range(n):
        s = out[i]
        d = s == s
        ok[i] = d
        holes += 0 if d else 1
        if s < lo:
            lo = s
        if s > hi:
            hi = s
    k = n
    if holes:                                   # the undefined squeezed out, the rest in order
        k = 0
        for i in range(n):
            if ok[i]:
                out[k] = out[i]
                k += 1
    if k == 0:
        lo = hi = 0.0
    return k, lo, hi


@njit(cache=True, nogil=True)
def _extent(v, scale):
    """(how many of values v are undefined, the least and greatest of the
    rest times scale) -- _gather's, read through once, nothing written:
    a field with no gaps, in the file's order, is packed from v itself."""
    lo, hi = np.inf, -np.inf
    holes = 0
    for i in range(v.size):
        s = v[i] * scale
        holes += 0 if s == s else 1
        if s < lo:
            lo = s
        if s > hi:
            hi = s
    return holes, lo, hi


@njit(cache=True, nogil=True)
def _quantize(dv, scale, ref, inv, x):
    """The integers X of values dv times scale: round((value - R) / 2^E),
    >= 0, into x (as long as dv)."""
    for i in range(dv.size):
        q = np.int64(np.rint((dv[i] * scale - ref) * inv))
        x[i] = max(q, 0)
    return x[:dv.size]


@njit(cache=True, nogil=True)
def _qchunk(dv, scale, ref, inv, xb, b0, c1):
    for i in range(b0, c1):
        q = np.int64(np.rint((dv[i] * scale - ref) * inv))
        xb[i - b0] = max(q, 0)


@njit(cache=True, nogil=True)
def _qdiff(dv, scale, ref, inv, xb, h, twice):
    """_quantize's integers' second-order differences less their least
    (the first two: 0) into h, without the integers stored: made a chunk
    at a time into xb (in L1) and differenced from there, twice -- for
    their least, then less it into h (each a loop that vectorises: a
    running least kept in the loop making them did not); (that least, the
    first two integers). The integers written out and read back, the
    differences read again for their least and then again to take it off,
    were 75 MB a HRRR field of memory's bandwidth (making the integers
    twice: 15 MB, of float32s) -- what packing on many threads at once
    waits on. twice=False (a field packed by itself, memory to spare): the
    integers made once, the differences into h as they are and their least
    taken off after (HRRR: 1.3 ms a field less on one core)."""
    n = dv.size
    ch = xb.size - 2
    m = np.int64(1) << 62
    if not twice:
        for c0 in range(0, n, ch):
            c1 = min(n, c0 + ch)
            b0 = max(0, c0 - 2)
            _qchunk(dv, scale, ref, inv, xb, b0, c1)
            if c0 == 0:
                z1, z2 = xb[0], xb[1]
            for i in range(max(c0, 2), c1):
                j = i - b0
                h[i] = xb[j] - 2 * xb[j - 1] + xb[j - 2]
            for i in range(max(c0, 2), c1):
                m = min(m, h[i])
        h[0] = h[1] = 0
        for i in range(2, n):
            h[i] -= m
        return m, z1, z2
    for c0 in range(0, n, ch):                  # their least
        c1 = min(n, c0 + ch)
        b0 = max(0, c0 - 2)                     # (the two before the chunk, again)
        _qchunk(dv, scale, ref, inv, xb, b0, c1)
        for i in range(max(c0, 2), c1):
            j = i - b0
            m = min(m, xb[j] - 2 * xb[j - 1] + xb[j - 2])
    _qchunk(dv, scale, ref, inv, xb, 0, 2)
    z1, z2 = xb[0], xb[1]
    for c0 in range(0, n, ch):                  # less it, into h
        c1 = min(n, c0 + ch)
        b0 = max(0, c0 - 2)
        _qchunk(dv, scale, ref, inv, xb, b0, c1)
        for i in range(max(c0, 2), c1):
            j = i - b0
            h[i] = xb[j] - 2 * xb[j - 1] + xb[j - 2] - m
    h[0] = h[1] = 0
    return m, z1, z2


_LOCAL = threading.local()


def crowded(on):
    """Whether this thread packs while others pack too (write()'s): then
    the packers spare memory's bandwidth, which they share, at some
    arithmetic's cost (_qdiff's twice)."""
    _LOCAL.crowded = bool(on)


def _scratch(name, n, dtype):
    """This thread's reused array of n items called name (a fresh one a
    field: its pages faulted in one at a time, which no thread count
    speeds up -- packing many fields on many threads waited on it). Its
    contents are this call's until the next on the thread."""
    a = getattr(_LOCAL, name, None)
    if a is None or a.size < n:
        a = np.empty(max(n, 1), dtype)
        setattr(_LOCAL, name, a)
    return a[:n]


_PACKINGS = ('simple', 'complex', 'ieee', 'png', 'jpeg', 'ccsds')
_ALIAS = {'jpeg2000': 'jpeg', 'aec': 'ccsds', 'grid_simple': 'simple', 'grid_ieee': 'ieee',
          'grid_complex_spatial_differencing': 'complex', 'grid_png': 'png',
          'grid_jpeg': 'jpeg', 'grid_ccsds': 'ccsds'}


def _pack(dv, lo, hi, packing, e, d, bits, e_given, ref0=None, rect=None, scale=1.0):
    """(section 5's body past its number, section 7's) of values dv
    times scale (all defined; times 10^d then; lo, hi: their least and
    greatest so scaled) packed with binary scale e and decimal scale d (or e chosen
    to give `bits` bits a value). ref0: the reference value of the field
    replaced, kept when still at or below every value -- its own values,
    unchanged, then pack to the same bits."""
    ref = np.float32(lo)
    if float(ref) > lo:                         # R at or below every value
        ref = np.nextafter(ref, np.float32(-np.inf))
    if ref0 is not None and ref0 - 2.0 ** (e - 1) <= lo and (hi - ref0) * 2.0 ** -e < 2 ** 30:
        ref = np.float32(ref0)                  # (values a float32 rounding below it: its own)
    span = hi - float(ref)
    if bits is not None:
        e = math.ceil(math.log2(span / (2 ** bits - 1))) if span > 0 else 0
    elif not e_given:                           # decimals given: values to 10^-d
        e = 0
    while span > 0 and span / 2.0 ** e >= 2 ** 30:  # at most 30 bits a value
        e += 1
    head = struct.pack('>f', ref) + sm_bytes(e, 2) + sm_bytes(d, 2)
    if packing not in ('png', 'jpeg', 'ccsds', 'simple') and dv.size >= 3:
        return _complex(dv, scale, head, float(ref), 2.0 ** -e)
    x = _quantize(dv, scale, float(ref), 2.0 ** -e, _scratch('x', dv.size, np.int64))
    if packing in ('png', 'jpeg', 'ccsds'):
        return _image(x, head, packing, rect)
    nb = _bitlen(x.max()) if x.size else 0      # simple (complex of under three values too)
    out = np.zeros((x.size * nb + 7) // 8, np.uint8)
    _put_fixed(x, nb, out, 0)
    return struct.pack('>IH', x.size, 0) + head + bytes([nb, 0]), out


def _image(x, head, packing, rect):
    """(section 5's body, section 7's) of integers x packed as an image:
    PNG (5.41: 8, 16, 24 or 32 bits a sample -- grey, RGB, RGBA), JPEG2000
    (5.40, lossless, up to 31 bits) or CCSDS (5.42, as ECMWF's: blocks of
    32, a reference sample every 128 blocks). rect: (rows, columns) of the
    image when the values fill the grid, else one row of them."""
    n = x.size
    nb = _bitlen(int(x.max())) if n else 0
    h, w = rect if rect is not None and rect[0] * rect[1] == n else (1, n)
    if packing == 'png':
        depth = 0 if nb == 0 else next(k for k in (8, 16, 24, 32) if nb <= k)
        s5 = struct.pack('>IH', n, 41) + head + bytes([depth, 0])
        if nb == 0:
            return s5, np.zeros(0, np.uint8)
        out = np.zeros(n * depth // 8, np.uint8)
        _put_fixed(x, depth, out, 0)
        return s5, np.frombuffer(_png_file(out.reshape(h, w * depth // 8), w, h, depth),
                                 np.uint8)
    if packing == 'jpeg':
        s5 = struct.pack('>IH', n, 40) + head + bytes([nb, 0, 0, 255])  # lossless
        if nb == 0:
            return s5, np.zeros(0, np.uint8)
        if nb > 31:
            raise ValueError('JPEG2000 of %d bits a value: 31 at most' % nb)
        from ._jpeg2000 import encode
        return s5, np.frombuffer(encode(x, w, h, nb), np.uint8)
    s5 = struct.pack('>IH', n, 42) + head + bytes([nb, 0, _CCSDS_FLAGS, 32]) + \
        struct.pack('>H', 128)
    if nb == 0:
        return s5, np.zeros(0, np.uint8)
    return s5, _aec_encode(x, nb, 32, 128)


def _png_file(rows, w, h, depth):
    """A PNG of `rows` (h x bytes a row): each row filtered as it packs
    best (none, sub or up: the least sum of the bytes' distances from 0)."""
    import zlib
    bpp = max(1, depth // 8)
    r = rows.astype(np.int16)
    sub = r.copy()
    sub[:, bpp:] -= r[:, :-bpp]
    up = r.copy()
    up[1:] -= r[:-1]
    cands = [r & 0xFF, sub & 0xFF, up & 0xFF]
    cost = np.stack([np.minimum(c, 256 - c).sum(1) for c in cands])
    pick = cost.argmin(0)
    body = np.empty((h, rows.shape[1] + 1), np.uint8)
    body[:, 0] = pick
    for k in range(3):
        sel = pick == k
        body[sel, 1:] = cands[k][sel]
    color = {8: 0, 16: 0, 24: 2, 32: 6}[depth]
    bitdepth = 16 if depth == 16 else 8

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + \
            struct.pack('>I', zlib.crc32(kind + data) & 0xFFFFFFFF)
    return b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, bitdepth, color,
                                                           0, 0, 0)) + \
        chunk(b'IDAT', zlib.compress(body.tobytes(), 6)) + chunk(b'IEND', b'')


# CCSDS 121.0 (5.42) as ECMWF writes it: libaec's flags preprocess (8), most significant
# byte first (4), 3 bytes for 17-24 bits (2) -- the last two say how a decoder lays the
# samples out, not how they are coded
_CCSDS_FLAGS = 14


@njit(cache=True, nogil=True)
def _aec_encode(x, nbits, J, R):
    """A CCSDS 121.0 stream of samples x (unsigned, nbits each): unit-delay
    prediction (a reference sample opening every R blocks), then each
    block of J coded the cheapest way -- a run of zero blocks (split at the
    64-block segments, the rest of one as ROS), second extension, a
    split-sample code (k = 0: the fundamental sequence) or raw."""
    n = x.size
    idlen = 5 if nbits > 16 else 4 if nbits > 8 else 3
    kmax = (1 << idlen) - 3
    raw_id = (1 << idlen) - 1
    xmax = (np.int64(1) << nbits) - 1
    every = R * J
    nb = (n + J - 1) // J
    d = np.zeros(nb * J, np.int64)              # mapped residuals (padded with 0)
    for i in range(n):
        if i % every == 0:
            d[i] = x[i]
            continue
        xh = x[i - 1]
        delta = x[i] - xh
        th = min(xh, xmax - xh)
        if 0 <= delta <= th:
            d[i] = 2 * delta
        elif -th <= delta < 0:
            d[i] = -2 * delta - 1
        else:
            d[i] = th + abs(delta)
    out = np.zeros(nb * (J * nbits + idlen + 2) // 8 + 16, np.uint8)
    st = np.zeros(3, np.int64)                  # write position (bytes), accumulator, bits held
    b = 0
    while b < nb:
        inrsi = b % R
        ref = 1 if inrsi == 0 else 0
        s = b * J
        if not ref and _zero(d, s, J):          # a run of zero blocks
            c = 1
            seg_end = min(b - inrsi + R, b - inrsi % 64 + 64, nb)   # (segments: from the RSI's start)
            while b + c < seg_end and _zero(d, (b + c) * J, J):
                c += 1
            _bits(out, st, 0, idlen + 1)
            if c <= 4:
                _fs(out, st, c - 1)
            elif b + c == seg_end:
                _fs(out, st, 4)                 # ROS: the rest of the segment
            else:
                _fs(out, st, c)
            b += c
            continue
        m = J - ref
        raw = m * nbits
        best, bestk = raw, -1
        tot = 0
        top = 0
        for t in range(s + ref, s + J):
            tot += d[t]
            if d[t] > top:
                top = d[t]
        for k in range(0, kmax + 1):           # split-sample: FS of v >> k, then k bits each
            cost = 0
            for t in range(s + ref, s + J):
                cost += (d[t] >> k) + 1
            cost += k * m
            if cost < best:
                best, bestk = cost, k
            if (tot >> k) < m:                  # higher k only adds bits
                break
        se = -1
        if top < (1 << 14):                     # second extension: pairs, FS-coded
            cost = 1
            t = s + ref
            first = ref == 1
            while t < s + J:
                a = 0 if first else d[t]
                bb = d[t] if first else d[t + 1]
                cost += (a + bb) * (a + bb + 1) // 2 + bb + 1
                t += 1 if first else 2
                first = False
            if cost < best:
                best, se = cost, 1
        if se == 1:
            _bits(out, st, 0, idlen)
            _bits(out, st, 1, 1)
            if ref:
                _bits(out, st, d[s], nbits)
            t = s + ref
            first = ref == 1
            while t < s + J:
                a = 0 if first else d[t]
                bb = d[t] if first else d[t + 1]
                _fs(out, st, (a + bb) * (a + bb + 1) // 2 + bb)
                t += 1 if first else 2
                first = False
        elif bestk >= 0:
            _bits(out, st, bestk + 1, idlen)
            if ref:
                _bits(out, st, d[s], nbits)
            for t in range(s + ref, s + J):
                _fs(out, st, d[t] >> bestk)
            if bestk:
                mask = (np.int64(1) << bestk) - 1
                for t in range(s + ref, s + J):
                    _bits(out, st, d[t] & mask, bestk)
        else:
            _bits(out, st, raw_id, idlen)
            for t in range(s, s + J):
                _bits(out, st, d[t], nbits)
        b += 1
    if st[2]:                                   # the last bits, to a whole byte
        out[st[0]] = (st[1] << (8 - st[2])) & 0xFF
        st[0] += 1
    return out[:st[0]]


@njit(cache=True, nogil=True)
def _zero(d, s, J):
    for t in range(s, s + J):
        if d[t]:
            return False
    return True


@njit(cache=True, nogil=True)
def _bits(out, st, v, n):
    """v's low n bits (n up to 32) onto the stream."""
    acc, held = st[1], st[2]
    acc = (acc << n) | (v & ((np.int64(1) << n) - 1))
    held += n
    while held >= 8:
        held -= 8
        out[st[0]] = (acc >> held) & 0xFF
        st[0] += 1
    acc &= (np.int64(1) << held) - 1
    st[1], st[2] = acc, held


@njit(cache=True, nogil=True)
def _fs(out, st, v):
    """The fundamental-sequence codeword of v: v zeros, then a one."""
    while v >= 32:
        _bits(out, st, 0, 32)
        v -= 32
    _bits(out, st, 1, v + 1)


def _complex(dv, scale, head, ref, inv):
    """5.3 (second-order spatial differencing, groups of the size packing
    tightest) of values dv times scale as _quantize makes them integers
    (three or more)."""
    n = dv.size
    h = _scratch('h', n, np.int64)
    many = getattr(_LOCAL, 'crowded', False)
    minsd, z1, z2 = _qdiff(dv, scale, ref, inv, _scratch('xb', 4098, np.int64), h, many)
    minsd, z1, z2 = int(minsd), int(z1), int(z2)    # (the first two: given aside, packed as 0)
    # groups: wgrib2's search (_groups), from blocks of 8 -- 9% smaller than
    # merging blocks of 16 into the group before (HRRR 20.7 MB a sample of
    # fields against 22.8; GFS's flux 41.9 against 45.8) for twice its time
    ng, lens, refs, widths = _groups(h, 8, _scratch('groups', 12 * (-(-n // 8)), np.int64),
                                     many)
    nref, wmin = _bitlen(int(refs.max())), int(widths.min())
    nw = _bitlen(int(widths.max()) - wmin)
    lref = int(lens[:-1].min()) if ng > 1 else int(lens[0])
    nl = _bitlen(int(lens[:-1].max()) - lref) if ng > 1 else 0
    top = max(z1, z2, abs(minsd))
    nd = max(1, -(-(_bitlen(top) + 1) // 8))     # octets each extra descriptor
    desc = z1.to_bytes(nd, 'big') + z2.to_bytes(nd, 'big') + \
        ((1 << (8 * nd - 1)) | -minsd if minsd < 0 else minsd).to_bytes(nd, 'big')
    nbytes = len(desc) + (ng * nref + 7) // 8 + (ng * nw + 7) // 8 + (ng * nl + 7) // 8 + \
        (int((widths * lens).sum()) + 7) // 8
    out = np.zeros(nbytes, np.uint8)
    out[:len(desc)] = np.frombuffer(desc, np.uint8)
    q = len(desc)
    _put_fixed(refs, nref, out, q * 8)
    q += (ng * nref + 7) // 8
    _put_fixed(widths - wmin, nw, out, q * 8)
    q += (ng * nw + 7) // 8
    scaled = lens - lref
    scaled[-1] = 0                              # the last's: its true length, in section 5
    _put_fixed(scaled, nl, out, q * 8)
    q += (ng * nl + 7) // 8
    _put_groups(h, lens, refs, widths, out, q * 8)
    s5 = struct.pack('>IH', n, 3) + head + bytes([nref, 0, 1, 0]) + \
        struct.pack('>ffIBBIBIBBB', 0.0, 0.0, ng, wmin, nw, lref, 1, int(lens[-1]), nl, 2, nd)
    return s5, out


@njit(cache=True, nogil=True)
def _bitsof(v):
    """Bits to hold v >= 0, by halving (wgrib2's find_nbits): a few steps,
    not one a bit -- and no math helper, which the kernel image has not."""
    j = 0
    while v > 65535:
        v >>= 16
        j += 16
    if v > 255:
        v >>= 8
        j += 8
    if v > 15:
        v >>= 4
        j += 4
    if v > 3:
        v >>= 2
        j += 2
    return j + 2 if v >= 2 else j + v


@njit(cache=True, nogil=True)
def _gsize(mn, mx, n, rb, wb):
    """Bits of a group of n values from mn to mx, its description included."""
    return rb + wb if mn == mx else _bitsof(mx - mn) * n + rb + wb


@njit(cache=True, nogil=True)
def _gmerge(i0, i1, mn, mx, nxt, prv, head, rb, wb, span, lmax):
    """wgrib2's merge_j: along the groups h -> m -> t, m merged into h or t
    -- whichever saves more bits, their range at most span, their length
    below lmax -- stepping back after merging into h (it may merge on).
    (Each group's bits kept rather than worked out again came out slower:
    at a pass's start for every group, most never wanted, 22 -> 28 ms a
    HRRR field; when first wanted, through a helper of ten arguments, 50.)"""
    h = head
    while h >= 0 and nxt[h] >= 0:
        m = nxt[h]
        t = nxt[m]
        shm = smt = -1
        lo0 = hi0 = lo1 = hi1 = 0
        if i1[m] - i0[h] < lmax:
            lo0, hi0 = min(mn[h], mn[m]), max(mx[h], mx[m])
            if hi0 - lo0 <= span:
                shm = (_gsize(mn[h], mx[h], i1[h] - i0[h] + 1, rb, wb) +
                       _gsize(mn[m], mx[m], i1[m] - i0[m] + 1, rb, wb) -
                       _gsize(lo0, hi0, i1[m] - i0[h] + 1, rb, wb))
        if t >= 0 and i1[t] - i0[m] < lmax:
            lo1, hi1 = min(mn[m], mn[t]), max(mx[m], mx[t])
            if hi1 - lo1 <= span:
                smt = (_gsize(mn[m], mx[m], i1[m] - i0[m] + 1, rb, wb) +
                       _gsize(mn[t], mx[t], i1[t] - i0[t] + 1, rb, wb) -
                       _gsize(lo1, hi1, i1[t] - i0[m] + 1, rb, wb))
        if shm >= smt and shm >= 0:
            i1[h], mn[h], mx[h], nxt[h] = i1[m], lo0, hi0, nxt[m]
            if nxt[h] >= 0:
                prv[nxt[h]] = h
            if prv[h] >= 0:
                h = prv[h]
        elif smt >= 0:
            i1[m], mn[m], mx[m], nxt[m] = i1[t], lo1, hi1, nxt[t]
            if nxt[m] >= 0:
                prv[nxt[m]] = m
        else:
            h = nxt[h]


@njit(cache=True, nogil=True)
def _gbytes(i0, i1, mn, mx, nxt, head, rb, wb):
    bits = 0
    s = head
    while s >= 0:
        bits += _gsize(mn[s], mx[s], i1[s] - i0[s] + 1, rb, wb)
        s = nxt[s]
    return (bits + 7) // 8


@njit(cache=True, nogil=True)
def _grange(v, a, b):
    lo = hi = v[a]
    for i in range(a + 1, b + 1):
        x = v[i]
        if x < lo:
            lo = x
        elif x > hi:
            hi = x
    return lo, hi


@njit(cache=True, nogil=True)
def _gexchange(v, i0, i1, mn, mx, nxt, prv, head, lmax):
    """wgrib2's exchange: a value at a boundary moved into the neighbour
    of narrower width where it already fits that neighbour's range."""
    s = head
    while s >= 0 and nxt[s] >= 0:
        t = nxt[s]
        bs, bt = _bitsof(mx[s] - mn[s]), _bitsof(mx[t] - mn[t])
        if bs == bt:
            s = t
            continue
        v0, v1 = v[i1[s]], v[i0[t]]
        if bs < bt and i1[s] - i0[s] < lmax and mn[s] <= v1 <= mx[s]:
            i1[s] += 1                          # t's first into s
            i0[t] += 1
            if i0[t] > i1[t]:
                nxt[s] = nxt[t]
                if nxt[s] >= 0:
                    prv[nxt[s]] = s
            elif v1 == mn[t] or v1 == mx[t]:
                mn[t], mx[t] = _grange(v, i0[t], i1[t])
        elif bs > bt and i1[t] - i0[t] < lmax and mn[t] <= v0 <= mx[t]:
            i1[s] -= 1                          # s's last into t
            i0[t] -= 1
            if i0[s] > i1[s]:                   # s emptied: t in its place
                i0[s], i1[s], mn[s], mx[s], nxt[s] = i0[t], i1[t], mn[t], mx[t], nxt[t]
                if nxt[s] >= 0:
                    prv[nxt[s]] = s
            elif v0 == mn[s] or v0 == mx[s]:
                mn[s], mx[s] = _grange(v, i0[s], i1[s])
        else:
            s = t


@njit(cache=True, nogil=True)
def _gcompact(i0, i1, mn, mx, nxt, prv, head, t0, t1, tn, tx):
    """The groups along the list from head (0) moved, in order, to places
    0, 1, ... (through t0 to tx, as long): each pass's walk along them
    then a walk along memory, not a jump to wherever a group began among
    all the blocks' (most merge away in the first passes; a jump a group,
    six arrays a jump, was what held packing on many threads at once to
    memory's speed). Where a group lies merges nothing differently. How
    many there are."""
    s, j = head, 0
    while s >= 0:
        t0[j], t1[j], tn[j], tx[j] = i0[s], i1[s], mn[s], mx[s]
        s = nxt[s]
        j += 1
    for i in range(j):
        i0[i], i1[i], mn[i], mx[i] = t0[i], t1[i], tn[i], tx[i]
        nxt[i] = i + 1 if i + 1 < j else -1
        prv[i] = i - 1
    return j


@njit(cache=True, nogil=True)
def _groups(v, blk, work, compact):
    """(groups, their lengths, minima, widths) of non-negative integers v,
    by wgrib2's search (complex_pk.c, public domain), from blocks of blk:
    neighbours merged where that saves bits, allowing a range of 1, 3, 7
    ... in turn; group lengths given more bits while that saves; values at
    boundaries moved to the narrower side; merged once more, any range.
    (wgrib2 starts from runs of equal values: 0.4% smaller, 4x the time.)
    work: 12 x the blocks' int64s, the groups' arrays and their copies (the
    caller's, reused: fresh ones a field, faulted in a page at a time, held
    packing on many threads to one field at a time). compact: the groups
    left moved together after each pass (_gcompact) -- for packing side by
    side, where it halves the search's time; by itself, a little dearer."""
    n = v.size
    vmx = 0
    for i in range(n):
        if v[i] > vmx:
            vmx = v[i]
    rb = _bitsof(vmx)                           # a group's reference, its width, its length
    lbits, lmax, est = 7, 127, 6
    m0 = (n + blk - 1) // blk
    i0, i1, mn = work[0:m0], work[m0:2 * m0], work[2 * m0:3 * m0]
    mx, nxt, prv = work[3 * m0:4 * m0], work[4 * m0:5 * m0], work[5 * m0:6 * m0]
    k0, k1, kn = work[6 * m0:7 * m0], work[7 * m0:8 * m0], work[8 * m0:9 * m0]
    kx, kt, kp = work[9 * m0:10 * m0], work[10 * m0:11 * m0], work[11 * m0:12 * m0]
    for b in range(m0):
        a, e = b * blk, min(n, (b + 1) * blk) - 1
        i0[b], i1[b] = a, e
        mn[b], mx[b] = _grange(v, a, e)
        nxt[b] = b + 1 if b + 1 < m0 else -1
        prv[b] = b - 1
    head = 0
    live = m0
    span = 1
    while span < vmx // 2:
        _gmerge(i0, i1, mn, mx, nxt, prv, head, rb, lbits + est, span, lmax)
        if compact:
            live = _gcompact(i0, i1, mn, mx, nxt, prv, head, k0, k1, kn, kx)
        span = 2 * span + 1
    size = _gbytes(i0, i1, mn, mx, nxt, head, rb, lbits + est)
    last = size + 1
    while size < last and lbits < 25:           # longer groups, while that saves
        last = size
        lbits += 1
        lmax = 2 * lmax + 1
        for j in range(live):                   # kept, to go back to
            k0[j], k1[j], kn[j], kx[j], kt[j], kp[j] = i0[j], i1[j], mn[j], mx[j], nxt[j], prv[j]
        _gmerge(i0, i1, mn, mx, nxt, prv, head, rb, lbits + est, span, lmax)
        size = _gbytes(i0, i1, mn, mx, nxt, head, rb, lbits + est)
        if size > last:                         # (the copies become the groups, and the
            i0, k0 = k0, i0                     # groups the room for the next copies)
            i1, k1 = k1, i1
            mn, kn = kn, mn
            mx, kx = kx, mx
            nxt, kt = kt, nxt
            prv, kp = kp, prv
            lbits -= 1
            lmax = (lmax - 1) // 2
        elif compact:
            live = _gcompact(i0, i1, mn, mx, nxt, prv, head, k0, k1, kn, kx)
    _gexchange(v, i0, i1, mn, mx, nxt, prv, head, lmax)
    _gmerge(i0, i1, mn, mx, nxt, prv, head, rb, lbits + est, vmx, lmax)
    ng = 0
    s = head
    while s >= 0:
        ng += 1
        s = nxt[s]
    lens = np.empty(ng, np.int64)
    refs = np.empty(ng, np.int64)
    widths = np.empty(ng, np.int64)
    s, g = head, 0
    while s >= 0:
        lens[g], refs[g], widths[g] = i1[s] - i0[s] + 1, mn[s], _bitsof(mx[s] - mn[s])
        g += 1
        s = nxt[s]
    return ng, lens, refs, widths


@njit(cache=True, nogil=True)
def _put_fixed(x, nb, out, bit):
    """x's values (each below 2^nb), nb bits each, into out from bit `bit`
    on (a byte's start)."""
    if nb == 0:
        return
    q, acc, held = bit >> 3, np.uint64(0), 0
    for v in x:
        acc = (acc << np.uint64(nb)) | np.uint64(v)
        held += nb
        while held >= 8:
            held -= 8
            out[q] = (acc >> np.uint64(held)) & np.uint64(0xFF)
            q += 1
        acc &= (np.uint64(1) << np.uint64(held)) - np.uint64(1)
    if held:
        out[q] = (acc << np.uint64(8 - held)) & np.uint64(0xFF)


@njit(cache=True, nogil=True)
def _put_groups(h, lens, refs, widths, out, bit):
    """Each group's values less its minimum, in its width's bits, one after
    another from bit `bit` on (a byte's start)."""
    q, acc, held, s = bit >> 3, np.uint64(0), 0, 0
    for k in range(refs.size):
        w, r = widths[k], refs[k]
        if w:
            sw = np.uint64(w)
            for i in range(s, s + lens[k]):
                acc = (acc << sw) | np.uint64(h[i] - r)
                held += w
                while held >= 8:
                    held -= 8
                    out[q] = (acc >> np.uint64(held)) & np.uint64(0xFF)
                    q += 1
                acc &= (np.uint64(1) << np.uint64(held)) - np.uint64(1)
        s += lens[k]
    if held:
        out[q] = (acc << np.uint64(8 - held)) & np.uint64(0xFF)
