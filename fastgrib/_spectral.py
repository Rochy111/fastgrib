"""Spherical harmonics: GRIB2's coefficient fields (grid 3.50, packing
5.50 simple and 5.51 complex) unpacked, and synthesised at any points.

Coefficients come as ecCodes and pygrib give them: each (n, m) a real
and an imaginary part, m outermost, n from m up (to J + m, capped at K:
triangular when J = K = M). Synthesis is ECMWF's (the IFS's): the field
the sum over m of e^(i m lon) times the sum over n of the coefficient and
the normalised associated Legendre function of sin(lat) -- normalised to
2 over [-1, 1] (so the (0, 0) coefficient is the field's global mean),
with no Condon-Shortley phase, the m < 0 terms the m > 0 ones' conjugates."""
import math
import struct

import numpy as np
from ._jit import njit
from ._octets import take as _take

from ._octets import ints, octets, sm


def count(J, K, M):
    """The values (reals) of a J, K, M truncation."""
    return 2 * sum(min(J + m, K) - m + 1 for m in range(M + 1))


def unpack(b, s5, s7, s7end, J, K, M):
    """The coefficients (float64) of the field whose sections 5 and 7 are
    at s5 and s7 (to s7end) in b."""
    b = octets(b)                               # (its octets Python ints, numpy 1 or 2)
    drt = struct.unpack_from('>H', b, s5 + 9)[0]
    n = struct.unpack_from('>I', b, s5 + 5)[0]
    if n != count(J, K, M):
        raise ValueError('%d values for a J=%d K=%d M=%d truncation (%d)'
                         % (n, J, K, M, count(J, K, M)))
    ref = struct.unpack_from('>f', b, s5 + 11)[0]
    e = sm(struct.unpack_from('>H', b, s5 + 15)[0], 2)
    d = sm(struct.unpack_from('>H', b, s5 + 17)[0], 2)
    nbits = b[s5 + 19]
    if nbits > 32:
        raise ValueError('%d bits a value' % nbits)
    data = np.frombuffer(bytes(b[s7 + 5:s7end]), np.uint8)
    scale, dec = 2.0 ** e, 10.0 ** -d
    if drt == 50:
        first = struct.unpack_from('>f', b, s5 + 20)[0]     # (0, 0)'s real part, as it is
        _room(data, (n - 1) * nbits)
        x = ints(data, 0, nbits, n - 1)
        out = np.empty(n, np.float64)
        out[0] = first
        out[1:] = (ref + x * scale) * dec
        return out
    if drt != 51:
        raise ValueError('spherical harmonics packed as 5.%d (5.50, 5.51 are read)' % drt)
    p = sm(struct.unpack_from('>I', b, s5 + 20)[0], 4) * 1e-6
    js, ks, ms = struct.unpack_from('>HHH', b, s5 + 24)
    ts = struct.unpack_from('>I', b, s5 + 30)[0]
    prec = b[s5 + 34]
    if prec not in (1, 2):
        raise ValueError('unpacked subset of precision %d (1 and 2 are read)' % prec)
    sub = count(js, ks, ms) if js <= J and ks <= K and ms <= M else -1
    if sub != ts:
        raise ValueError('an unpacked subset of %d values for JS=%d KS=%d MS=%d' % (ts, js, ks, ms))
    width = 4 if prec == 1 else 8
    if data.size < ts * width:
        raise ValueError('its unpacked subset is cut short')
    head = data[:ts * width].view('>f4' if prec == 1 else '>f8').astype(np.float64)
    _room(data, ts * width * 8 + (n - ts) * nbits)
    x = ints(data, 8 * ts * width, nbits, n - ts)
    return _complex(head, x, ref, scale, dec, p, J, K, M, js, ks, ms, n)


def _room(data, nbits):
    if (nbits + 7) // 8 > data.size:
        raise ValueError('its packed coefficients are cut short')


@njit(cache=True, nogil=True)
def _complex(head, x, ref, scale, dec, p, J, K, M, js, ks, ms, n):
    """5.51: the subset (n <= JS, m <= MS, as the truncation's) given
    as it is, the rest (R + X 2^E) 10^-D (n (n + 1))^-P."""
    out = np.empty(n, np.float64)
    k = h = i = 0
    for m in range(M + 1):
        top = min(J + m, K)
        stop = min(js + m, ks)
        for nn in range(m, top + 1):
            if m <= ms and nn <= stop:
                out[k] = head[h]
                out[k + 1] = head[h + 1]
                h += 2
            else:
                f = (nn * (nn + 1.0)) ** -p
                out[k] = (ref + x[i] * scale) * dec * f
                # m = 0's imaginary parts: a real field's are 0 (packed: the nearest value)
                out[k + 1] = 0.0 if m == 0 else (ref + x[i + 1] * scale) * dec * f
                i += 2
            k += 2
    return out


def synthesise(coef, J, K, M, lat, lon):
    """The field of coefficients `coef` (ecCodes' order) at points (lat,
    lon in degrees, any shape alike): float32, NaN where lat is not one."""
    lat = np.asarray(lat, np.float64)
    lon = np.asarray(lon, np.float64)
    shape = np.broadcast(lat, lon).shape
    lat, lon = np.broadcast_to(lat, shape).ravel(), np.broadcast_to(lon, shape).ravel()
    ok = (np.abs(lat) <= 90.0) & np.isfinite(lon)
    out = np.full(lat.size, np.nan, np.float32)
    if not ok.any():
        return out.reshape(shape)
    u, inv = np.unique(lat[ok], return_inverse=True)
    from . import _cpu
    c = np.ascontiguousarray(coef, np.float64)
    mu = np.sin(np.radians(u))
    a, b = np.zeros((mu.size, M + 1)), np.zeros((mu.size, M + 1))
    start = np.zeros(M + 2, np.int64)
    for m in range(M + 1):
        start[m + 1] = start[m] + 2 * (min(J + m, K) - m + 1)
    ctr = np.zeros(2, np.int64)
    _cpu.share(lambda: _sums(c, mu, J, K, M, start, a, b, ctr), mu.size)
    inv, lo = inv.astype(np.int64), np.radians(lon[ok])
    res = np.empty(lo.size, np.float32)
    _cpu.share(lambda: _along(a, b, inv, lo, M, res, ctr), -(-lo.size // _ALONG))
    out[ok] = res
    return out.reshape(shape)


_ALONG = 4096                           # _along's points a take


@njit(cache=True, nogil=True)
def _sums(c, mu, J, K, M, start, a, b, ctr):
    """For each latitude (mu = its sine) and m: the sums over n of the real
    and imaginary parts times the Legendre function (m > 0 doubled: its
    conjugate term too), into a and b; latitudes taken off ctr[0] until
    none is left (_cpu.share's threads; start: each m's first coefficient)."""
    nl = mu.size
    while True:
        t = _take(ctr, 0)
        if t >= nl:
            return
        x = mu[t]
        s = math.sqrt(max(0.0, 1.0 - x * x))
        pmm = 1.0                               # P(m, m), from P(0, 0) = 1
        for m in range(M + 1):
            if m > 0:
                pmm *= math.sqrt((2.0 * m + 1.0) / (2.0 * m)) * s
            top = min(J + m, K)
            k = start[m]
            p0 = pmm
            sa = c[k] * p0
            sb = c[k + 1] * p0
            if top > m:
                p1 = math.sqrt(2.0 * m + 3.0) * x * pmm
                sa += c[k + 2] * p1
                sb += c[k + 3] * p1
                for n in range(m + 2, top + 1):
                    an = math.sqrt((4.0 * n * n - 1.0) / (n * n - m * m))
                    bn = math.sqrt(((n - 1.0) * (n - 1.0) - m * m) / (4.0 * (n - 1.0) * (n - 1.0) - 1.0))
                    p2 = an * (x * p1 - bn * p0)
                    q = k + 2 * (n - m)
                    sa += c[q] * p2
                    sb += c[q + 1] * p2
                    p0, p1 = p1, p2
            w = 1.0 if m == 0 else 2.0
            a[t, m] = w * sa
            b[t, m] = w * sb
            if pmm == 0.0:                      # underflowed towards the pole: no more
                break


@njit(cache=True, nogil=True)
def _along(a, b, inv, lon, M, out, ctr):
    """The sums over m at each point: a cos(m lon) - b sin(m lon), into
    out; _ALONG points taken off ctr[1] at a time."""
    while True:
        i0 = _take(ctr, 1) * _ALONG
        if i0 >= lon.size:
            return
        for i in range(i0, min(i0 + _ALONG, lon.size)):
            _along1(a, b, inv, lon, M, out, i)


@njit(cache=True, nogil=True)
def _along1(a, b, inv, lon, M, out, i):
    """_along at point i."""
    t = inv[i]
    c1, s1 = math.cos(lon[i]), math.sin(lon[i])
    cm, sm = 1.0, 0.0
    acc = 0.0
    for m in range(M + 1):
        acc += a[t, m] * cm - b[t, m] * sm
        cm, sm = cm * c1 - sm * s1, sm * c1 + cm * s1
    out[i] = acc
