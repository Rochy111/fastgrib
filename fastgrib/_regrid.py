"""Regridding: fields interpolated onto a lat-lon grid, another file's
grid or any points -- bilinear (IPOLATES' rule for gaps: the missing
corners weighed out, at least half the weight defined) or nearest. Each
target point is placed in the source grid by the grid's own forward
projection (rotated grids: the point rotated into the grid's frame;
reduced grids: along their own rows), as a plan of four corners and
their weights, made once and applied to each field."""
import math

import numpy as np
from ._jit import njit
from ._octets import take as _take

from . import _grids

D2R = math.pi / 180.0


def plan(g, lat, lon, method=None):
    """(corners (n, 4) int64 -- indices into a field of grid g as read()
    lays it out, raveled, -1 for none -- and their weights (n, 4)) of the
    target points (lat, lon), any shapes, raveled. method: 'bilinear' or
    'nearest'; None: bilinear, nearest on points listed one by one."""
    if method is None:
        method = 'nearest' if g.template == 130 else 'bilinear'
    if method not in ('bilinear', 'nearest'):
        raise ValueError("method %r: 'bilinear' or 'nearest'" % method)
    lat = np.ascontiguousarray(lat, np.float64).ravel()
    lon = np.ascontiguousarray(lon, np.float64).ravel()
    if lat.size != lon.size:
        raise ValueError('%d latitudes for %d longitudes' % (lat.size, lon.size))
    bad = ~(np.abs(lat) <= 90.0) | ~np.isfinite(lon)    # no such point: no value
    if bad.any():
        lat, lon = lat.copy(), lon.copy()
        lat[bad] = lon[bad] = np.nan
    if g.template in _grids._ROTATED:
        lat, lon = rotate(lat, lon, *g.params[6:9])
    near = method == 'nearest'
    if g.template == 130:                       # points listed one by one: the nearest
        if not near:
            raise ValueError("%r: its points are listed one by one (3.130): method='nearest'"
                             % (g,))
        return _nearest_listed(g, lat, lon)
    if g.pl or _grids.stagger(g) is not None:  # by rows: reduced, or staggered
        rows, start, k, lo1, step, glob = _grids._rows(g)
        return _reduced_corners(lat, lon, rows, start, k, lo1, step, glob, near)
    fi, fj, wrap = _fractional(g, lat, lon)
    # nearest: a tie goes the way the file counts (IPOLATES' NINT of its 1-based index)
    return _corners(fi, fj, g.ni, g.nj, wrap, near, bool(g.scan & 0x80),
                    not g.scan & 0x40)


def _nearest_listed(g, lat, lon):
    """The plan of nearest points among a grid's listed ones: by a k-d tree
    (scipy's) or, without scipy, a search in blocks."""
    plat, plon = _grids.latlon(g)

    def unit(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    p, t = unit(plat, plon), unit(lat, lon)
    ok = np.isfinite(t).all(1)
    idx = np.full((lat.size, 4), -1, np.int64)
    w = np.zeros((lat.size, 4))
    w[:, 0] = 1.0
    try:
        from scipy.spatial import cKDTree
        idx[ok, 0] = cKDTree(p).query(t[ok])[1]
    except ImportError:
        tt = t[ok]
        near = np.empty(tt.shape[0], np.int64)
        step = max(1, (1 << 22) // max(1, p.shape[0]))
        for k in range(0, tt.shape[0], step):
            near[k:k + step] = np.argmax(tt[k:k + step] @ p.T, axis=1)
        idx[ok, 0] = near
    return idx, w


def apply(corners, weights, values, out=None):
    """The plan applied to one field (as read() gives it): (n,) float32."""
    v = np.ascontiguousarray(values).ravel()
    if out is None:
        out = np.empty(corners.shape[0], np.float32)
    _apply(corners, weights, v, out)
    return out


def rotate(lat, lon, splat, splon, angle=0.0):
    """unrotate()'s inverse: the rotated grid's own (lat, lon) of
    geographic points."""
    la, lo = np.radians(lat), np.radians(lon)
    x, y, z = np.cos(lo) * np.cos(la), np.sin(lo) * np.cos(la), np.sin(la)
    t, o = np.radians(-(90.0 + splat)), np.radians(-splon)
    st, ct, so, co = np.sin(t), np.cos(t), np.sin(o), np.cos(o)
    xd = ct * co * x - ct * so * y - st * z
    yd = so * x + co * y
    zd = st * co * x - st * so * y + ct * z
    return (np.degrees(np.arcsin(np.clip(zd, -1, 1))),
            (np.degrees(np.arctan2(yd, xd)) - angle) % 360.0)


def _fractional(g, lat, lon):
    """(fractional column, row of each point in g as read() lays it out,
    whether its rows go all the way round)."""
    t = g.template
    if t in (0, 1, 40, 41):
        la, lo = _grids._axes(g)
        if g.nj == 1:
            fj = np.where(np.abs(lat - la[0]) < 1e-6, 0.0, np.nan)
        elif t in (0, 1):
            fj = (lat - la[0]) / (la[1] - la[0])
        else:                                   # Gaussian: rows unevenly spaced
            fj = np.interp(lat, la, np.arange(g.nj, dtype=np.float64), left=np.nan,
                           right=np.nan)
        dl = (lo[-1] - lo[0]) / (g.ni - 1) if g.ni > 1 else 360.0
        fi = ((lon - lo[0]) % 360.0) / dl
        return fi, fj, abs(g.ni * dl - 360.0) < 0.5 * dl
    if t == 90:                                 # space view: by its own navigation
        fi, fj = _grids.space_view_steps(g, lat, lon)
        return fi, fj, False
    if t not in (10, 20, 30):
        raise ValueError('%r: no regridding from this grid' % (g,))
    p = g.params
    a, b = g.earth or (6371229.0, 6371229.0)
    dx, dy = (p[4], p[5]) if t == 20 else (p[5], p[6])
    if abs(a - b) > 1e-6:                       # an ellipsoid: by pyproj, as latlon()
        from pyproj import Proj
        pr = Proj(_grids.proj4(g))
        x0, y0 = pr(p[1], p[0])
        x, y = pr(lon, lat)
        x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
        bad = ~np.isfinite(x) | ~np.isfinite(y) | (np.abs(x) > 1e20)
        x[bad] = y[bad] = np.nan
    else:
        if t == 30:
            x0, y0 = _grids._lambert_fwd(p[0], p[1], p[2], p[3], p[4], a, p[7])
            x, y = _grids._lambert_fwd(lat, lon, p[2], p[3], p[4], a, p[7])
        elif t == 20:
            x0, y0 = _grids._stereo_fwd(p[0], p[1], p[2], p[3], a, p[6])
            x, y = _grids._stereo_fwd(lat, lon, p[2], p[3], a, p[6])
        else:                                   # Mercator: x from the first longitude
            k = a * math.cos(p[2] * D2R)
            x0, y0 = 0.0, k * math.log(math.tan(math.pi / 4 + p[0] * D2R / 2))
            x = k * ((lon - p[1]) % 360.0) * D2R
            with np.errstate(divide='ignore', invalid='ignore'):
                y = k * np.log(np.tan(np.pi / 4 + lat * D2R / 2))
    if g.scan & 0x80:                           # from the south-west corner, as read() lays out
        x0 -= (g.ni - 1) * dx
    if not g.scan & 0x40:
        y0 -= (g.nj - 1) * dy
    return (x - x0) / dx, (y - y0) / dy, False


@njit(cache=True, nogil=True)
def _corners(fi, fj, ni, nj, wrap, near, west_ties, south_ties):
    n = fi.size
    idx = np.full((n, 4), -1, np.int64)
    w = np.zeros((n, 4), np.float64)
    eps = 1e-6
    # IPOLATES' edges: a corner off the grid counts as one with no value -- weighed out,
    # the point defined while half its weight is on the grid (up to half a cell beyond)
    for t in range(n):
        x, y = fi[t], fj[t]
        if not (abs(x) < 1e15 and abs(y) < 1e15):
            continue                            # (NaN, or beyond any grid)
        if wrap:
            x = x % ni
        if near:
            i = int(math.ceil(x - 0.5)) if west_ties else int(math.floor(x + 0.5))
            j = int(math.ceil(y - 0.5)) if south_ties else int(math.floor(y + 0.5))
            if wrap:
                i %= ni
            w[t, 0] = 1.0
            if 0 <= i < ni and 0 <= j < nj:
                idx[t, 0] = j * ni + i
            continue
        i0, j0 = int(math.floor(x + eps)), int(math.floor(y + eps))   # (on a line: its cell)
        wx, wy = max(x - i0, 0.0), max(y - j0, 0.0)
        for c in range(4):
            i, j = i0 + (c & 1), j0 + (c >> 1)
            w[t, c] = (wx if c & 1 else 1 - wx) * (wy if c >> 1 else 1 - wy)
            if wrap:
                i %= ni
            if 0 <= i < ni and 0 <= j < nj:
                idx[t, c] = j * ni + i
    return idx, w


@njit(cache=True, nogil=True)
def _row_pos(lon, lo1, step, k, glob):
    """(the point at or west of lon in a row, the way to the next), -1 off it."""
    d = (lon - lo1) % 360.0
    if d > 360.0 - 1e-6 * step:                 # (a hair west of the first point: on it)
        d -= 360.0
    x = d / step
    if glob:
        i = int(math.floor(x))
        return i % k, x - i
    if x > k - 1 + 1e-6 or x < -1e-6:
        return -1, 0.0
    i = max(0, min(int(math.floor(x)), k - 2))
    return i, min(max(x - i, 0.0), 1.0)


@njit(cache=True, nogil=True)
def _reduced_corners(lat, lon, rows, start, k, lo1, step, glob, near):
    """_corners() on a reduced grid: two points on each of the rows either
    side, each row along its own longitudes."""
    n = lat.size
    idx = np.full((n, 4), -1, np.int64)
    w = np.zeros((n, 4), np.float64)
    m = rows.size
    for t in range(n):
        y = lat[t]
        if not (y >= rows[0] - 1e-6 and y <= rows[m - 1] + 1e-6):
            continue
        j0 = 0
        while j0 < m - 2 and rows[j0 + 1] <= y:
            j0 += 1
        j1 = min(j0 + 1, m - 1)
        wy = 0.0 if j1 == j0 else min(max((y - rows[j0]) / (rows[j1] - rows[j0]), 0.0), 1.0)
        if near:
            j = j1 if wy >= 0.5 else j0
            i, wx = _row_pos(lon[t], lo1[j], step[j], k[j], glob)
            if i >= 0:
                if wx >= 0.5:
                    i = (i + 1) % k[j] if glob else min(i + 1, k[j] - 1)
                idx[t, 0] = start[j] + i
                w[t, 0] = 1.0
            continue
        for side in range(2):
            j = j0 if side == 0 else j1
            wr = 1.0 - wy if side == 0 else wy
            i, wx = _row_pos(lon[t], lo1[j], step[j], k[j], glob)
            if i < 0:                           # off the row: weighed, no value
                w[t, 2 * side] = wr
                continue
            i1 = (i + 1) % k[j] if glob else min(i + 1, k[j] - 1)
            idx[t, 2 * side], idx[t, 2 * side + 1] = start[j] + i, start[j] + i1
            w[t, 2 * side], w[t, 2 * side + 1] = wr * (1 - wx), wr * wx
    return idx, w


def expandable(g):
    """Whether g is a reduced grid expand() widens (rows of points listed,
    not staggered)."""
    return bool(g is not None and g.pl) and _grids.stagger(g) is None


_EXPANDS = {}                                   # grid -> (latitudes, start, points, lon1, round)


def _expansion(g):
    hit = _EXPANDS.get(g)
    if hit is None:
        lat, start, k, lo1, _, glob = _grids._rows(g)
        hit = (lat, start, k, float(lo1[0]) if lo1.size else 0.0, glob)
        if len(_EXPANDS) > 64:
            _EXPANDS.clear()
        _EXPANDS[g] = hit
    return hit


def expand(values, g, nx=None, out=None):
    """Fields on reduced grid g, as read() lays them out ((points,) or (n,
    points)), on the regular grid of the same rows, nx points each (by
    default as many as its longest row: a reduced Gaussian grid's own
    regular one), south to north as read() lays a regular grid -- each
    row interpolated along itself as pygrib's redtoreg does it (linear,
    round the globe on a global grid), the same values but for float32's
    rounding; beside a missing value, the nearer point's (redtoreg's own
    rule, though 2.1.8 interpolates its missval in). A grid that is not
    reduced: the values as they are."""
    if not expandable(g):
        return values
    lat, start, k, _, glob = _expansion(g)
    nx = int(k.max()) if nx is None else int(nx)
    if nx < 1:
        raise ValueError('nx: 1 or more')
    v = np.ascontiguousarray(values, np.float32)
    one = v.ndim == 1
    v = v.reshape(-1, v.shape[-1])
    if v.shape[1] != g.ni:
        raise ValueError('%d values a field, the grid has %d points' % (v.shape[1], g.ni))
    shape = (v.shape[0], lat.size, nx)
    if out is None:
        out = np.empty(shape, np.float32)
    elif out.shape != (shape[1:] if one else shape) or out.dtype != np.float32:
        raise ValueError('out: float32 of shape %s' % (shape[1:] if one else shape,))
    from ._cpu import share
    flat, ctr = out.reshape(shape), np.zeros(1, np.int64)
    rows = shape[0] * shape[1]                  # (fields x rows: each a thread's item)
    share(lambda: _expand_rows(v, start, k, glob, flat, ctr), -(-rows * nx // _EXPAND_ITEM))
    return out[0] if one and out.ndim == 3 else out


_EXPAND_ITEM = 1 << 18                          # points a thread at least (a row ~5 us)


def expanded_latlon(g, nx=None):
    """(lat, lon), each (rows, nx), of expand()'s regular grid."""
    lat, _, k, lo1, glob = _expansion(g)
    nx = int(k.max()) if nx is None else int(nx)
    if glob:
        lon = lo1 + np.arange(nx) * (360.0 / nx)
    else:
        lo2 = g.params[3]
        lon = lo1 + np.arange(nx) * (((lo2 - lo1) % 360.0) / max(nx - 1, 1))
    lat2, lon2 = np.meshgrid(lat, lon % 360.0, indexing='ij')
    if g.template in _grids._ROTATED:
        return _grids.unrotate(lat2, lon2, *g.params[6:9])
    return lat2, lon2


@njit(cache=True, nogil=True)
def _expand_rows(v, start, k, glob, out, ctr):
    """expand(): pygrib's redtoreg on each field, from float32 -- its
    arithmetic exactly (i * k / nx, in double), less its two modulos a point
    (the index is below k by then: only the last point's neighbour wraps).
    Rows (of every field) taken off ctr[0] until none is left."""
    nf, nrows, nx = out.shape
    while True:
        t = _take(ctr, 0)
        if t >= nf * nrows:
            return
        f, j = t // nrows, t % nrows
        kj, s = k[j], start[j]
        fl = float(kj)
        span = fl if glob else fl - 1.0
        den = float(nx if glob else max(nx - 1, 1))
        for i in range(nx):
            zxi = i * span / den
            im = min(int(zxi), kj - 1)
            zdx = zxi - im
            ip = im + 1
            if ip >= kj:
                ip = 0 if glob else kj - 1
            a = np.float64(v[f, s + im])
            b = np.float64(v[f, s + ip])
            if a != a or b != b:                # a missing neighbour: the nearer one
                out[f, j, i] = a if zdx < 0.5 else b
            else:
                out[f, j, i] = a * (1.0 - zdx) + b * zdx


@njit(cache=True, nogil=True)
def _apply(idx, w, v, out):
    for t in range(idx.shape[0]):
        s = sv = tot = 0.0
        for c in range(4):
            k, wc = idx[t, c], w[t, c]
            tot += wc                           # (a corner off the grid: weight, no value)
            if k < 0:
                continue
            x = v[k]
            if x == x:
                s += wc * x
                sv += wc
        out[t] = s / sv if sv > 0.0 and sv >= 0.5 * tot - 1e-12 else np.nan
