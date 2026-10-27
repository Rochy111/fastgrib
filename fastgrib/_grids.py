"""Grid definitions (section 3) and where their points are.

A Grid is a tuple (ni, nj, scan, template, params, earth): ni points along
a row, nj rows, the scanning mode flags, the template number, its
parameters (angles in degrees, lengths in m) and the earth as (a, b) semi-
axes in m (a == b: a sphere; () when the file does not say). Grids compare
and hash as tuples, so records on one grid group together.

Every grid is laid out west to east, south to north (wgrib2's -order
we:sn) however the file scans it; latlon() gives (lat, lon) in that layout.
"""
import math
import struct
import threading
from collections import OrderedDict
from typing import NamedTuple

import numpy as np
from ._jit import njit
from ._octets import take as _take

from ._octets import int_power as _int_power, sm as _sm   # noqa: F401

D2R = np.pi / 180.0


class Grid(NamedTuple):
    ni: int
    nj: int
    scan: int
    template: int
    params: tuple
    earth: tuple
    pl: tuple = ()      # a reduced grid's points a row, in the file's order (ni: all its points)

    @property
    def shape(self):
        """(ny, nx) of a field on this grid; (points,) on a reduced one, and
        on one whose points fastgrib does not place (spherical harmonics'
        coefficients, an unstructured grid's values)."""
        flat = self.pl or self.template not in _PLACED or self.template == 130
        return (self.ni,) if flat else (self.nj, self.ni)

    @property
    def kind(self):
        return _KIND.get(self.template, 'template 3.%d' % self.template)

    def __repr__(self):
        return '<%s grid %d x %d>' % (self.kind, self.ni, self.nj)


_KIND = {0: 'lat-lon', 1: 'rotated lat-lon', 10: 'Mercator', 20: 'polar stereographic',
         30: 'Lambert conformal', 40: 'Gaussian', 41: 'rotated Gaussian',
         50: 'spherical harmonics', 51: 'rotated spherical harmonics',
         52: 'stretched spherical harmonics', 53: 'stretched rotated spherical harmonics',
         90: 'space view', 101: 'unstructured', 130: 'irregular (points listed)',
         204: 'curvilinear'}
_PLACED = frozenset((0, 1, 10, 20, 30, 40, 41, 90, 130))   # the templates fastgrib places
_SPECTRAL = frozenset((50, 51, 52, 53))     # spherical harmonics: coefficients, synthesised
_POINTS = {}                                    # 3.130's points, one copy each (grids share them)
_ROTATED = {1: 0, 41: 40}           # a rotated grid's unrotated kind


def layout(g):
    """g as read() lays its fields out: without the scanning flags that only
    order the points in the file (0x20 columns first, 0x10 alternate rows),
    so fields that differ in those alone stack together."""
    return g._replace(scan=g.scan & 0xCF) if g is not None and g.scan & 0x30 else g


def parse(b, p, centre=None):
    """Section 3 at byte p as a Grid, None for a template not covered;
    centre: the message's (NCEP's local templates read when it is 7)."""
    tmpl = struct.unpack_from('>H', b, p + 12)[0]
    ni, nj = struct.unpack_from('>II', b, p + 30)
    deg = lambda v: _sm(v, 4) / 1e6                         # noqa: E731
    earth = _earth(b, p)
    if tmpl in (32768, 32769) and centre == 7:
        return _ncep_rotated(b, p, tmpl, ni, nj, earth)
    if tmpl in (0, 1, 40, 41):
        basic, sub = struct.unpack_from('>II', b, p + 38)
        if basic not in (0, 0xFFFFFFFF) and sub not in (0, 0xFFFFFFFF):
            deg = lambda v: _sm(v, 4) * basic / sub         # noqa: E731
        la1, lo1 = struct.unpack_from('>II', b, p + 46)
        la2, lo2, di, dj = struct.unpack_from('>IIII', b, p + 55)
        gauss = tmpl in (40, 41)
        params = (deg(la1), deg(lo1), deg(la2), deg(lo2),
                  None if di == 0xFFFFFFFF else deg(di),    # a reduced grid's: missing
                  dj if gauss else deg(dj))                 # 3.40: N, parallels pole-equator
        if tmpl in (1, 41):                                 # the southern pole, a rotation
            splat, splon = struct.unpack_from('>II', b, p + 72)
            angle = struct.unpack_from('>f', b, p + 80)[0]
            params += (deg(splat), deg(splon), angle)
        pl = ()
        scan = b[p + 71]
        if tmpl in (0, 1) and scan & 0x01 and scan & 0x0E:  # staggered, offset rows short a point
            f = np.arange(nj - (scan >> 1 & 1))             # (offset along j: a row short too)
            short = np.where(f % 2 == 0, scan >> 3 & 1, scan >> 2 & 1)
            return Grid(int((ni - short).sum()), 1, scan, tmpl, params, earth,
                        tuple(int(v) for v in ni - short))
        if ni == 0xFFFFFFFF:                                # reduced: points a row listed
            width, kind = b[p + 10], b[p + 11]
            if kind != 1 or width not in (1, 2, 4):
                return None
            at = p + (84 if tmpl in (1, 41) else 72)
            fmt = {1: 'B', 2: 'H', 4: 'I'}[width]
            pl = struct.unpack_from('>%d%s' % (nj, fmt), b, at)
            ni = sum(pl)
            nj = 1
        return Grid(ni, nj, b[p + 71], tmpl, params, earth, tuple(pl))
    if tmpl in _SPECTRAL:                       # spherical harmonics: J, K, M; their values
        jkm = struct.unpack_from('>III', b, p + 14)
        npts = struct.unpack_from('>I', b, p + 6)[0]
        rot, stretch, q = (-90.0, 0.0, 0.0), (90.0, 0.0, 1.0), p + 28
        if tmpl in (51, 53):                    # rotated: the southern pole, an angle
            rot = (_sm(struct.unpack_from('>I', b, q)[0], 4) / 1e6,
                   struct.unpack_from('>I', b, q + 4)[0] / 1e6, struct.unpack_from('>f', b, q + 8)[0])
            q += 12
        if tmpl in (52, 53):                    # stretched: its pole, the factor
            la, lo, c = struct.unpack_from('>III', b, q)
            stretch = (_sm(la, 4) / 1e6, _sm(lo, 4) / 1e6, c / 1e6)
        return Grid(npts, 1, 0x40, tmpl, jkm + (b[p + 26], b[p + 27]) + rot + stretch, earth)
    if tmpl == 130:                             # irregular: each point's latitude, longitude
        npts = struct.unpack_from('>I', b, p + 6)[0]
        pts = bytes(b[p + 30:p + 30 + 8 * npts])
        if len(pts) < 8 * npts:
            return None
        return Grid(npts, 1, 0x40, tmpl, (_POINTS.setdefault(pts, pts),), earth)
    if tmpl == 90:                              # space view: a geostationary satellite's image
        lap, lop = struct.unpack_from('>II', b, p + 38)
        dx, dy, xp, yp = struct.unpack_from('>IIII', b, p + 47)
        ori, nr, xo, yo = struct.unpack_from('>IIII', b, p + 64)
        return Grid(ni, nj, b[p + 63], tmpl,
                    (deg(lap), deg(lop), dx, dy, xp / 1e3, yp / 1e3, deg(ori),
                     math.inf if nr == 0xFFFFFFFF else nr / 1e6, xo, yo),   # (missing: orthographic)
                    earth or (6371229.0, 6371229.0))
    if tmpl == 10:
        la1, lo1 = struct.unpack_from('>II', b, p + 38)
        lad, la2, lo2 = struct.unpack_from('>III', b, p + 47)
        di, dj = struct.unpack_from('>II', b, p + 64)
        return Grid(ni, nj, b[p + 59], tmpl,
                    (deg(la1), deg(lo1), deg(lad), deg(la2), deg(lo2), di / 1e3, dj / 1e3), earth)
    if tmpl == 20:
        la1, lo1 = struct.unpack_from('>II', b, p + 38)
        lad, lov, dx, dy = struct.unpack_from('>IIII', b, p + 47)
        return Grid(ni, nj, b[p + 64], tmpl,
                    (deg(la1), deg(lo1), deg(lad), deg(lov), dx / 1e3, dy / 1e3,
                     bool(b[p + 63] & 0x80)), earth)
    if tmpl == 30:
        la1, lo1 = struct.unpack_from('>II', b, p + 38)
        lad, lov, dx, dy = struct.unpack_from('>IIII', b, p + 47)
        lt1, lt2 = struct.unpack_from('>II', b, p + 65)
        return Grid(ni, nj, b[p + 64], tmpl,
                    (deg(la1), deg(lo1), deg(lov), deg(lt1), deg(lt2), dx / 1e3, dy / 1e3,
                     bool(b[p + 63] & 0x80), deg(lad)), earth)
    npts = struct.unpack_from('>I', b, p + 6)[0]    # any other: its values, not placed
    return Grid(npts, 1, 0x40, tmpl, (), earth) if npts else None


def _earth(b, p):
    """The earth's (a, b) semi-axes in m (code table 3.2); () unknown."""
    shape = b[p + 14]

    def val(q):                                 # scale factor, scaled value
        sf, sv = b[q], struct.unpack_from('>I', b, q + 1)[0]
        return sv * _int_power(10.0, -sf)
    if shape == 1:
        r = val(p + 15)
        return (r, r)
    if shape in (3, 7):                         # ellipsoid given (3: in km)
        a, c = val(p + 20), val(p + 25)
        k = 1e3 if shape == 3 else 1.0
        return (a * k, c * k)
    return {0: (6367470.0, 6367470.0), 2: (6378160.0, 6356775.0),
            4: (6378137.0, 6356752.314), 5: (6378137.0, 6356752.3142),
            6: (6371229.0, 6371229.0), 8: (6371200.0, 6371200.0),
            9: (6377563.396, 6356256.909)}.get(shape, ())


def check(rec):
    """rec's grid, if decode() can lay its values out."""
    g = rec.grid
    if g is None:
        raise ValueError('%r: grid template not supported (lat-lon, Gaussian -- regular, '
                         'reduced or rotated --, Mercator, polar stereographic and Lambert '
                         'conformal are)' % rec)
    return g


def axes(g):
    """(lat, lon) axes of a lat-lon or Gaussian grid, south-first and
    west-first; None for a projected, rotated, reduced or staggered grid."""
    if g.template not in (0, 40) or g.pl or stagger(g) is not None:
        return None
    return _axes(g)


def stagger(g):
    """A staggered lat-lon grid's offsets (WMO's scanning flags: 0x08 the
    file's odd rows, 0x04 its even rows, half a grid length along i; 0x02
    every point half a grid length along j), as read() lays rows out
    (south to north): (each row's, in grid lengths east, the points', in
    grid lengths north); None for one not staggered."""
    s = g.scan
    if g.template not in (0, 1) or not s & 0x0E:
        return None
    n = len(g.pl) if g.pl else g.nj
    f = np.arange(n) if s & 0x40 else np.arange(n)[::-1]   # each row's place in the file
    off = np.where(f % 2 == 0, 0.5 * (s >> 3 & 1), 0.5 * (s >> 2 & 1))
    return (-off if s & 0x80 else off), (0.0 if not s & 0x02 else 0.5 if s & 0x40 else -0.5)


def _ncep_rotated(b, p, tmpl, ni, nj, earth):
    """NCEP's 3.32768 (rotated E-grid) and 3.32769 (rotated B-grid): a
    rotated lat-lon grid (3.1) whose equator and prime meridian meet at
    the centre (La2, Lo2), its first point La1, Lo1 geographic -- 3.32769
    spaced by its last point (as NCEP advises), 3.32768 by Di, Dj, its
    rows staggered as the E-grid's (every other row half a step east)
    unless its scanning flags say how."""
    from ._regrid import rotate
    la1, lo1 = _sm(struct.unpack_from('>I', b, p + 46)[0], 4) / 1e6, \
        struct.unpack_from('>I', b, p + 50)[0] / 1e6
    lac, loc = _sm(struct.unpack_from('>I', b, p + 55)[0], 4) / 1e6, \
        struct.unpack_from('>I', b, p + 59)[0] / 1e6
    di, dj = (v / 1e6 for v in struct.unpack_from('>II', b, p + 63))
    scan = b[p + 71]
    pole = (lac - 90.0, loc)
    (r1,), (o1,) = rotate(np.array([la1]), np.array([lo1]), *pole)
    if tmpl == 32769:
        la2, lo2 = _sm(struct.unpack_from('>I', b, p + 72)[0], 4) / 1e6, \
            struct.unpack_from('>I', b, p + 76)[0] / 1e6
        (r2,), (o2,) = rotate(np.array([la2]), np.array([lo2]), *pole)
        di = ((o2 - o1) % 360.0) / max(1, ni - 1)
        dj = abs(r2 - r1) / max(1, nj - 1)
    else:
        if not scan & 0x0C:                     # (the E-grid's: the file's even rows offset)
            scan |= 0x04
        r2 = r1 + (nj - 1) * dj * (1 if scan & 0x40 else -1)
        o2 = o1 + (ni - 1) * di * (-1 if scan & 0x80 else 1)
    o1, o2 = (o1 + 180.0) % 360.0 - 180.0, (o2 + 180.0) % 360.0 - 180.0
    return Grid(ni, nj, scan, 1, (float(r1), o1, float(r2), o2, di, dj) + pole + (0.0,),
                earth)


def _axes(g):
    """axes() of a regular lat-lon / Gaussian grid, or a rotated one's in
    its own (rotated) coordinates."""
    la1, lo1, la2, lo2, di, n = g.params[:6]
    if g.template in (0, 1):
        lat = np.linspace(la1, la2, g.nj)
        if lat[0] > lat[-1]:
            lat = lat[::-1]
    else:
        lat = _gaussian(int(n), min(la1, la2), max(la1, la2), g.nj)
    w, e = (lo2, lo1) if g.scan & 0x80 else (lo1, lo2)
    if e < w:
        e += 360.0
    return lat, np.linspace(w, e, g.ni)


def _gaussian(n, lo, hi, nj):
    """The nj Gaussian latitudes (2n of them pole to pole) from lo to hi."""
    lat = np.degrees(np.arcsin(np.polynomial.legendre.leggauss(2 * n)[0]))
    i0 = int(np.argmin(abs(lat - lo)))
    if abs(lat[i0] - lo) > 1e-3 or i0 + nj > lat.size:
        raise ValueError('Gaussian latitudes do not match the grid (%g to %g, %d rows)'
                         % (lo, hi, nj))
    return lat[i0:i0 + nj]


# latlon() of the grids asked for lately, by layout: a Lambert grid's
# inverse projection costs far more than a copy (NBM: 164 ms against 8)
_LL = OrderedDict()
_LL_KEEP, _LL_BYTES = 4, 256 << 20
_LL_LOCK = threading.Lock()


def latlon(g):
    """(lat, lon) of every point of grid g, each (nj, ni) float64, rows
    south to north and columns west to east; longitudes in [0, 360) as
    wgrib2 prints them. Yours to change: the grid's last few are kept,
    and copies handed out."""
    key = layout(g)
    with _LL_LOCK:
        hit = _LL.get(key)
        if hit is not None:
            _LL.move_to_end(key)
    if hit is None:
        hit = _latlon(g)
        if 2 * hit[0].nbytes <= _LL_BYTES // _LL_KEEP:
            with _LL_LOCK:
                _LL[key] = hit
                while len(_LL) > _LL_KEEP:
                    _LL.popitem(last=False)
    return hit[0].copy(), hit[1].copy()


def spectral_points(g, lat, lon):
    """Points (lat, lon) in the frame of a spherical-harmonic field's
    coefficients: rotated as rotated grids are (its southern pole, then
    the angle), then stretched about its pole as Schmidt's transform
    stretches them (ARPEGE's; WMO's note): the angle d from the pole to
    2 atan(c tan(d/2)), along the great circle through the pole."""
    rot, (plat, plon, c) = g.params[5:8], g.params[8:11]
    rotated = g.template in (51, 53)
    if rotated:
        from ._regrid import rotate
        lat, lon = rotate(lat, lon, *rot)
    if g.template not in (52, 53) or c == 1.0:
        return lat, lon
    if rotated:                                 # (its pole in the rotated frame too)
        plat, plon = (float(v[0]) for v in rotate(np.array([plat]), np.array([plon]), *rot))

    def vec(la, lo):
        la, lo = np.radians(la), np.radians(lo)
        return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
    x, p = vec(lat, lon), vec(plat, plon)
    cd = np.clip(x @ p, -1.0, 1.0)
    d = np.arccos(cd)
    sd = np.sin(d)
    d1 = 2.0 * np.arctan(c * np.tan(d / 2.0))
    with np.errstate(invalid='ignore', divide='ignore'):
        t = np.where(sd[..., None] > 1e-15, (x - cd[..., None] * p) / sd[..., None], 0.0)
    y = np.cos(d1)[..., None] * p + np.sin(d1)[..., None] * t
    y = np.where((d < np.pi - 1e-12)[..., None], y, x)     # (the antipode stays)
    return (np.degrees(np.arcsin(np.clip(y[..., 2], -1.0, 1.0))),
            np.degrees(np.arctan2(y[..., 1], y[..., 0])) % 360.0)


def _latlon(g):
    if g.template not in _PLACED:
        raise ValueError(_UNPLACED.get(g.template, 'grid template 3.%d: fastgrib does not '
                                       'place its points' % g.template))
    if g.template == 90:
        return space_view(g)
    if g.template == 130:
        raw = np.frombuffer(g.params[0], '>u4').reshape(-1, 2).astype(np.int64)
        ll = np.where(raw & 0x80000000, -(raw & 0x7FFFFFFF), raw) / 1e6   # (sign and magnitude)
        return np.ascontiguousarray(ll[:, 0]), ll[:, 1] % 360.0
    if g.pl:
        return _reduced_latlon(g)
    st = stagger(g)
    if st is not None:                          # staggered: each row's own offset
        la, lo = _axes(g)
        di = (lo[-1] - lo[0]) / (g.ni - 1) if g.ni > 1 else g.params[4] or 0.0
        dj = (la[-1] - la[0]) / (g.nj - 1) if g.nj > 1 else g.params[5] or 0.0
        lat = np.broadcast_to((la + st[1] * dj)[:, None], (g.nj, g.ni))
        lon = lo[None, :] + st[0][:, None] * di
        if g.template in _ROTATED:
            return unrotate(lat, lon, *g.params[6:9])
        return np.array(lat), lon % 360.0
    ax = axes(g)
    if ax is not None:
        lat, lon = np.meshgrid(ax[0], ax[1], indexing='ij')
        return lat, lon % 360.0
    if g.template in _ROTATED:
        lat, lon = np.meshgrid(*_axes(g), indexing='ij')
        return unrotate(lat, lon, *g.params[6:9])
    x, y = _xy(g)
    a, b = g.earth or (6371229.0, 6371229.0)
    if abs(a - b) > 1e-6:
        return _latlon_proj(g, x, y)
    r = a
    if g.template in (20, 30):                  # by rows on every core
        p = g.params
        x, y = np.ascontiguousarray(x.ravel()), np.ascontiguousarray(y.ravel())
        lat = np.empty((y.size, x.size))
        lon = np.empty_like(lat)
        if g.template == 30:
            lov, lt1, lt2, south = p[2], p[3], p[4], bool(p[7])
            if south:
                lt1, lt2 = -lt1, -lt2
            n, t = _cone(lt1, lt2)
            args = (lov, n, r * np.cos(lt1 * D2R) * t(lt1) ** n / n, south)
            kernel = _lambert_rows
        else:
            lad, lov, south = p[2], p[3], bool(p[6])
            args = (lov, r * (1 + np.sin((-lad if south else lad) * D2R)), south)
            kernel = _stereo_rows
        from . import _cpu
        step = max(1, -(-y.size // (4 * _cpu.threads())))
        _cpu._pool_map(lambda j: kernel(x, y, j, min(j + step, y.size), *args, lat, lon),
                       list(range(0, y.size, step)))
        return lat, lon
    else:
        la1, lo1, lad, la2, lo2, dx, dy = g.params
        k = r * np.cos(lad * D2R)
        lat = np.degrees(2 * np.arctan(np.exp(y / k)) - np.pi / 2)
        lon = lo1 + np.degrees(x / k)
    lat, lon = np.broadcast_arrays(lat, lon)     # Mercator's are (nj, 1) and (1, ni)
    return np.array(lat), np.array(lon) % 360.0


def unrotate(lat, lon, splat, splon, angle=0.0):
    """Geographic (lat, lon) of points given in a rotated grid's own
    coordinates, its southern pole at (splat, splon) and turned by angle
    about its polar axis (GRIB2 3.1, as ecCodes reads it); longitudes in
    [0, 360)."""
    lat, lon = np.radians(lat), np.radians(np.asarray(lon) + angle)
    xd, yd, zd = np.cos(lon) * np.cos(lat), np.sin(lon) * np.cos(lat), np.sin(lat)
    t, o = np.radians(-(90.0 + splat)), np.radians(-splon)
    st, ct, so, co = np.sin(t), np.cos(t), np.sin(o), np.cos(o)
    x = ct * co * xd + so * yd + st * co * zd
    y = -ct * so * xd + co * yd - st * so * zd
    z = -st * xd + ct * zd
    return np.degrees(np.arcsin(np.clip(z, -1, 1))), np.degrees(np.arctan2(y, x)) % 360.0


def rotated(g):
    """A rotated grid's points in its own rotated (lat, lon), laid out as
    latlon()'s."""
    if g.pl:
        return _reduced_latlon(g, rotate=False)
    return np.meshgrid(*_axes(g), indexing='ij')


def _reduced_latlon(g, rotate=True):
    """(lat, lon) of a reduced grid's points, each (points,): rows south to
    north, each west to east -- the order read() gives its values in. A
    row of n points spans the grid's longitudes, all 360 degrees in n
    steps on a global one, else n - 1 from the first to the last."""
    rows, start, k, lo1, step, _ = _rows(g)
    lat = np.repeat(rows, k)
    lon = (np.repeat(lo1, k) + np.repeat(step, k) * (np.arange(k.sum()) - np.repeat(start, k))) \
        % 360.0
    if g.template in _ROTATED and rotate:
        return unrotate(lat, lon, *g.params[6:9])
    return lat, lon


def _rows(g):
    """A reduced or staggered grid's rows as read() lays them out, south to
    north, those with points: (their latitudes, where each starts, its
    points, its first longitude, its step, whether the rows go round)."""
    st = stagger(g)
    if st is not None:
        if g.pl:                                # (rows a point short: those offset)
            k = np.asarray(g.pl, np.int64)[::1 if g.scan & 0x40 else -1]
            n, ni = k.size, int(k.max())
            la, lo = _axes(g._replace(ni=ni, nj=n + (g.scan >> 1 & 1), pl=()))
            la = la[:n]
        else:
            la, lo = _axes(g)
            n, k = g.nj, np.full(g.nj, g.ni, np.int64)
        di = (lo[-1] - lo[0]) / (lo.size - 1) if lo.size > 1 else g.params[4] or 0.0
        dj = (la[-1] - la[0]) / (n - 1) if n > 1 else g.params[5] or 0.0
        start = np.concatenate(([0], np.cumsum(k)[:-1])).astype(np.int64)
        return (np.asarray(la + st[1] * dj, np.float64), start, k,
                np.asarray(lo[0] + st[0] * di, np.float64), np.full(n, di, np.float64), False)
    la1, lo1, la2, lo2, di, n = g.params[:6]
    nrows = len(g.pl)
    if g.template in (40, 41):
        lat = _gaussian(int(n), min(la1, la2), max(la1, la2), nrows)[::-1]  # north first
    else:
        lat = np.linspace(max(la1, la2), min(la1, la2), nrows)
    pl = np.asarray(g.pl, np.int64)
    if g.scan & 0x40:                           # the file's rows go south to north
        lat = lat[::-1]
    order = range(nrows)[::-1] if not g.scan & 0x40 else range(nrows)
    span = (lo2 - lo1) % 360.0
    widest = 360.0 / max(1, pl.max())           # global: the last longitude a step short
    glob = abs(span + widest - 360.0) < widest / 2
    js = [j for j in order if pl[j] > 0]
    k = pl[js]
    step = 360.0 / k if glob else span / np.maximum(1, k - 1)
    start = np.concatenate(([0], np.cumsum(k)[:-1])).astype(np.int64)
    return (np.asarray(lat[js], np.float64), start, k, np.full(len(js), float(lo1)),
            np.asarray(step, np.float64), bool(glob))


def _xy(g):
    """Projection x (1, ni) and y (nj, 1) in m of a projected grid's points,
    we:sn, from its first point -- the corner the file starts at."""
    p = g.params
    la1, lo1 = p[0], p[1]
    a, b = g.earth or (6371229.0, 6371229.0)
    if g.template == 30:
        lov, lt1, lt2, dx, dy, south = p[2], p[3], p[4], p[5], p[6], p[7]
        x0, y0 = _lambert_fwd(la1, lo1, lov, lt1, lt2, a, south)
    elif g.template == 20:
        lad, lov, dx, dy, south = p[2], p[3], p[4], p[5], p[6]
        x0, y0 = _stereo_fwd(la1, lo1, lad, lov, a, south)
    else:
        lad, dx, dy = p[2], p[5], p[6]
        k = a * np.cos(lad * D2R)
        x0, y0 = 0.0, k * np.log(np.tan(np.pi / 4 + la1 * D2R / 2))
    if abs(a - b) > 1e-6:
        return x0, y0                           # _latlon_proj works from these
    if g.scan & 0x80:                           # step back to the south-west corner
        x0 -= (g.ni - 1) * dx
    if not g.scan & 0x40:
        y0 -= (g.nj - 1) * dy
    return ((x0 + dx * np.arange(g.ni))[None, :], (y0 + dy * np.arange(g.nj))[:, None])


def _cone(lt1, lt2):
    t = lambda phi: np.tan(np.pi / 4 + phi * D2R / 2)       # noqa: E731
    if abs(lt1 - lt2) < 1e-9:
        return np.sin(lt1 * D2R), t
    return np.log(np.cos(lt1 * D2R) / np.cos(lt2 * D2R)) / np.log(t(lt2) / t(lt1)), t


def _lambert_fwd(lat, lon, lov, lt1, lt2, r, south):
    """Lambert conformal conic on a sphere (Snyder 1987, 15-1 to 15-5)."""
    if south:
        lat, lt1, lt2 = -lat, -lt1, -lt2
    n, t = _cone(lt1, lt2)
    rf = r * np.cos(lt1 * D2R) * t(lt1) ** n / n
    rho = rf / t(lat) ** n
    th = n * ((lon - lov + 180.0) % 360.0 - 180.0) * D2R
    x, y = rho * np.sin(th), -rho * np.cos(th)
    return (x, -y) if south else (x, y)


def _lambert_inv(x, y, lov, lt1, lt2, r, south):
    if south:
        y, lt1, lt2 = -y, -lt1, -lt2
    n, t = _cone(lt1, lt2)
    rf = r * np.cos(lt1 * D2R) * t(lt1) ** n / n
    s = np.sign(n)
    rho = s * np.hypot(x, y)
    with np.errstate(divide='ignore'):          # a pole on the grid
        lat = 2 * np.arctan((rf / rho) ** (1 / n)) / D2R - 90.0
    lon = lov + np.arctan2(s * x, -s * y) / (n * D2R)
    return (-lat if south else lat), lon


@njit(cache=True, nogil=True)
def _lambert_rows(x, y, j0, j1, lov, n, rf, south, lat, lon):
    """_lambert_inv for rows j0 to j1, into lat and lon."""
    s = 1.0 if n > 0 else -1.0
    e, d2r = 1.0 / n, math.pi / 180.0
    for j in range(j0, j1):
        yy = -y[j] if south else y[j]
        for i in range(x.size):
            rho = s * math.hypot(x[i], yy)
            la = 90.0 if rho == 0.0 else 2.0 * math.atan((rf / rho) ** e) / d2r - 90.0
            lat[j, i] = -la if south else la
            lon[j, i] = (lov + math.atan2(s * x[i], -s * yy) / (n * d2r)) % 360.0


@njit(cache=True, nogil=True)
def _stereo_rows(x, y, j0, j1, lov, k, south, lat, lon):
    """_stereo_inv for rows j0 to j1 (k = r (1 + sin lad)), into lat and lon."""
    deg = 180.0 / math.pi
    for j in range(j0, j1):
        yy = -y[j] if south else y[j]
        for i in range(x.size):
            la = 90.0 - 2.0 * (math.atan(math.hypot(x[i], yy) / k) * deg)
            lat[j, i] = -la if south else la
            lon[j, i] = (lov + math.atan2(x[i], -yy) * deg) % 360.0


def _stereo_fwd(lat, lon, lad, lov, r, south):
    """Polar stereographic on a sphere, true at latitude lad (Snyder 21-4)."""
    if south:
        lat, lad = -lat, -lad
    rho = r * (1 + np.sin(lad * D2R)) * np.tan(np.pi / 4 - lat * D2R / 2)
    th = (lon - lov) * D2R
    x, y = rho * np.sin(th), -rho * np.cos(th)
    return (x, -y) if south else (x, y)


def _stereo_inv(x, y, lad, lov, r, south):
    if south:
        y, lad = -y, -lad
    rho = np.hypot(x, y)
    lat = 90.0 - 2 * np.degrees(np.arctan(rho / (r * (1 + np.sin(lad * D2R)))))
    lon = lov + np.degrees(np.arctan2(x, -y))
    return (-lat if south else lat), lon


def proj4(g):
    """A PROJ string for a projected grid (for pyproj, cartopy), None for
    lat-lon and Gaussian grids."""
    a, b = g.earth or (6371229.0, 6371229.0)
    ell = '+a=%r +b=%r' % (a, b)
    p = g.params
    if g.template == 30:
        lat0 = p[8] if len(p) > 8 else p[3]
        return ('+proj=lcc +lat_1=%r +lat_2=%r +lat_0=%r +lon_0=%r %s +units=m +no_defs'
                % (p[3], p[4], lat0, p[2], ell))
    if g.template == 20:
        return ('+proj=stere +lat_0=%d +lat_ts=%r +lon_0=%r %s +units=m +no_defs'
                % (-90 if p[6] else 90, p[2], p[3], ell))
    if g.template == 10:
        return '+proj=merc +lat_ts=%r +lon_0=%r %s +units=m +no_defs' % (p[2], p[1], ell)
    if g.template in _ROTATED:                  # x, y: rotated longitude, latitude (degrees)
        return ('+proj=ob_tran +o_proj=longlat +o_lon_p=%r +o_lat_p=%r +lon_0=%r %s '
                '+to_meter=0.0174532925199433 +no_defs' % (-p[8], -p[6], p[7], ell))
    return None


_UNPLACED = {
    **{t: 'spherical harmonics have no points: regrid() or at() synthesise them where you want '
          'them' for t in _SPECTRAL},
    101: "an unstructured grid (ICON's): its points are in the grid file keys() names "
         '(uuidOfHGrid, numberOfGridUsed), not in the message',
    204: "a curvilinear grid: its points' latitudes and longitudes are fields of their own"}


def _view(g):
    """A space view's camera: (equatorial and polar radii, its position,
    the unit vectors toward it from the Earth's centre and along the
    image's x and y -- all Earth-centred -- orthographic or not, and what
    a grid length in x and in y is: radians of scan, or metres across an
    orthographic image). The camera looks at the Earth's centre from over
    the sub-satellite point; the image's y axis at the orientation's
    bearing from the meridian there (as PROJ's tilted perspective turns
    its plane), north along it toward increasing latitude (at a pole:
    along 180 E at the North, 0 at the South)."""
    lap, lop, dx, dy, xp, yp, ori, nr, xo, yo = g.params
    if not (nr > 1 and dx > 0 and dy > 0):
        raise ValueError('a space view seen from %g Earth radii: not placed' % nr)
    a, b = g.earth
    la, lo = math.radians(lap), math.radians(lop)
    e2 = 1.0 - (b / a) ** 2
    n = a / math.sqrt(1.0 - e2 * math.sin(la) ** 2)
    up = np.array([n * math.cos(la) * math.cos(lo), n * math.cos(la) * math.sin(lo),
                   n * (1.0 - e2) * math.sin(la)])
    up /= np.linalg.norm(up)
    if abs(lap) >= 90.0:
        lo = math.pi if lap > 0 else 0.0
    north = np.array([-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)])
    north -= north.dot(up) * up                 # (the geodetic north, square to the view)
    north /= np.linalg.norm(north)
    east = np.cross(north, up)
    t = math.radians(ori)
    xax, yax = east * math.cos(t) - north * math.sin(t), north * math.cos(t) + east * math.sin(t)
    if math.isinf(nr):                          # from infinitely far: the Earth's diameter is dx
        return a, b, up * 0.0, up, xax, yax, True, 2.0 * a / dx, 2.0 * a / dy
    size = 2.0 * math.asin(1.0 / nr)            # the Earth's apparent diameter, radians
    return a, b, up * (nr * a), up, xax, yax, False, size / dx, size / dy


def _view_steps(g):
    """The grid lengths along the image's x (of each column) and y (each
    row) from the sub-satellite point, of each point as read() lays them
    out (WMO's scanning: as wgrib2; ecCodes takes j scanning positively as
    the rows still running north to south)."""
    lap, lop, dx, dy, xp, yp, ori, nr, xo, yo = g.params
    c = np.arange(g.ni, dtype=np.float64)       # columns west to east, rows south to north
    r = np.arange(g.nj, dtype=np.float64)
    i = g.ni - 1 - c if g.scan & 0x80 else c    # the file's column and row of each
    j = r if g.scan & 0x40 else g.nj - 1 - r
    x = (xp - (i + xo)) if g.scan & 0x80 else ((i + xo) - xp)
    y = ((j + yo) - yp) if g.scan & 0x40 else (yp - (j + yo))
    return x, y


def space_view(g):
    """(lat, lon) of a space view's points (CGMS's navigation, on its
    ellipsoid, from wherever the camera is and however the image is
    turned; orthographic views too); NaN where it sees past the Earth."""
    a, b, sat, up, xax, yax, ortho, rx, ry = _view(g)
    x, y = _view_steps(g)
    lat = np.empty((g.nj, g.ni))
    lon = np.empty((g.nj, g.ni))
    from . import _cpu
    x, y, ctr = x * rx, y * ry, np.zeros(1, np.int64)
    _cpu.share(lambda: _navigate(x, y, a, b, sat, up, xax, yax, ortho, lat, lon, ctr), y.size)
    return lat, lon


@njit(cache=True, nogil=True)
def _navigate(x, y, a, b, sat, up, xax, yax, ortho, lat, lon, ctr):
    """Each point's (lat, lon): where its ray meets the ellipsoid. x, y:
    scan angles (radians) or, orthographic, distances (m) in the image.
    Rows taken off ctr[0] until none is left (_cpu.share's threads)."""
    ia, ib = 1.0 / (a * a), 1.0 / (b * b)
    ux, uy, uz = up[0], up[1], up[2]
    xx, xy, xz = xax[0], xax[1], xax[2]
    yx, yy, yz = yax[0], yax[1], yax[2]
    while True:
        j = _take(ctr, 0)
        if j >= y.size:
            return
        cy, sy = math.cos(y[j]), math.sin(y[j])
        for i in range(x.size):
            if ortho:                           # parallel rays, toward the Earth along -up
                ox = x[i] * xx + y[j] * yx + 4.0 * a * ux
                oy = x[i] * xy + y[j] * yy + 4.0 * a * uy
                oz = x[i] * xz + y[j] * yz + 4.0 * a * uz
                dx, dy, dz = -ux, -uy, -uz
            else:
                cx, sx = math.cos(x[i]), math.sin(x[i])
                ox, oy, oz = sat[0], sat[1], sat[2]
                p, q = cx * cy, sx * cy
                dx = -ux * p + xx * q + yx * sy
                dy = -uy * p + xy * q + yy * sy
                dz = -uz * p + xz * q + yz * sy
            qa = (dx * dx + dy * dy) * ia + dz * dz * ib
            qb = 2.0 * ((ox * dx + oy * dy) * ia + oz * dz * ib)
            qc = (ox * ox + oy * oy) * ia + oz * oz * ib - 1.0
            disc = qb * qb - 4.0 * qa * qc
            if disc < 0.0:                      # past the Earth's edge
                lat[j, i] = lon[j, i] = np.nan
                continue
            t = (-qb - math.sqrt(disc)) / (2.0 * qa)    # the nearer meeting
            px, py, pz = ox + t * dx, oy + t * dy, oz + t * dz
            lat[j, i] = math.degrees(math.atan(a * a * ib * pz / math.hypot(px, py)))
            lon[j, i] = math.degrees(math.atan2(py, px)) % 360.0


def space_view_steps(g, lat, lon):
    """space_view()'s inverse: each point's fractional (column, row) as
    read() lays them out; NaN where the camera cannot see it."""
    a, b, sat, up, xax, yax, ortho, rx, ry = _view(g)
    la, lo = np.radians(lat), np.radians(lon)
    e2 = 1.0 - (b / a) ** 2
    n = a / np.sqrt(1.0 - e2 * np.sin(la) ** 2)
    p = np.stack([n * np.cos(la) * np.cos(lo), n * np.cos(la) * np.sin(lo),
                  n * (1.0 - e2) * np.sin(la)], -1)
    normal = p * np.array([1.0 / a ** 2, 1.0 / a ** 2, 1.0 / b ** 2])
    if ortho:
        seen = normal @ up > 0
        x, y = p @ xax / rx, p @ yax / ry
    else:
        v = p - sat                             # from the camera to each point
        c1, c2, c3 = v @ -up, v @ xax, v @ yax
        seen = (np.einsum('...k,...k->...', normal, -v) > 0) & (c1 > 0)
        x = np.arctan2(c2, c1) / rx
        y = np.arcsin(np.clip(c3 / np.linalg.norm(v, axis=-1), -1, 1)) / ry
    lap, lop, dx, dy, xp, yp, ori, nr, xo, yo = g.params
    i = xp - x - xo if g.scan & 0x80 else x + xp - xo
    j = y + yp - yo if g.scan & 0x40 else yp - y - yo
    c = g.ni - 1 - i if g.scan & 0x80 else i
    rr = j if g.scan & 0x40 else g.nj - 1 - j
    return np.where(seen, c, np.nan), np.where(seen, rr, np.nan)


def _latlon_proj(g, x0, y0):
    """An ellipsoidal earth: the projection's inverse by pyproj."""
    try:
        from pyproj import Proj
    except ImportError:
        raise ValueError('%r: an ellipsoidal earth needs pyproj (pip install pyproj)' % g)
    pr = Proj(proj4(g))
    p = g.params
    x0, y0 = pr(p[1], p[0])
    dx, dy = (p[4], p[5]) if g.template == 20 else (p[5], p[6])
    if g.scan & 0x80:
        x0 -= (g.ni - 1) * dx
    if not g.scan & 0x40:
        y0 -= (g.nj - 1) * dy
    x, y = np.meshgrid(x0 + dx * np.arange(g.ni), y0 + dy * np.arange(g.nj))
    lon, lat = pr(x, y, inverse=True)
    return lat, lon % 360.0                     # [0, 360), as every grid's
