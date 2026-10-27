"""Vector fields: u and v components found in pairs, and turned from the
grid's axes to east and north where the grid section says they lie along
its x and y (resolution and component flags, bit 0x08) -- HRRR's, NAM's,
RAP's Lambert winds, polar stereographic ones -- as wgrib2's
-new_grid_winds earth does.

On a conformal projection the grid's x axis lies at the convergence angle
from east, -n (lon - lov) on a Lambert cone of constant n and (n = 1) a
polar stereographic plane -- +n (lon - lov) for one about the south pole;
nothing on lat-lon, Gaussian and Mercator grids. The same on an ellipsoid:
a conformal projection's convergence is its cone constant times the
longitude difference. Checked against wgrib2's own earth/grid pairs
(Lambert, north polar) and, both hemispheres, against the x axis's
bearing measured from latlon().
"""
import threading
from collections import OrderedDict

import numpy as np
from ._jit import njit

from . import _grids

# (discipline, category, number) of a u component -> its v's
PAIRS = {(0, 2, 2): (0, 2, 3),          # UGRD, VGRD
         (0, 2, 23): (0, 2, 24),        # UGUST, VGUST
         (0, 2, 27): (0, 2, 28),        # USTM, VSTM (storm motion)
         (0, 2, 17): (0, 2, 18),        # UFLX, VFLX (momentum flux)
         (0, 3, 194): (0, 3, 195),      # U-GWD, V-GWD (NCEP)
         (10, 1, 2): (10, 1, 3),        # UOGRD, VOGRD (ocean current)
         (10, 2, 4): (10, 2, 5)}        # UICE, VICE (sea ice drift)
V_OF_U = PAIRS
U_OF_V = {v: u for u, v in PAIRS.items()}

_FLAGS_AT = {0: 54, 1: 54, 40: 54, 41: 54, 10: 46, 20: 46, 30: 46}


def grid_relative(b, p):
    """Whether section 3 at byte p puts vector components along the grid's
    x and y (False: east and north, or a template not covered)."""
    tmpl = int.from_bytes(bytes(b[p + 12:p + 14]), 'big')
    q = _FLAGS_AT.get(tmpl)
    return q is not None and bool(b[p + q] & 0x08)


# angle()s made, by grid: every point's lat-lon and a cosine and sine each
# took longer than turning the winds (RAP, 152k points: 3.6 ms a call
# against 0.2). Bounded, oldest out; read-only, as they are shared.
_ANGLES = OrderedDict()
_ANGLES_BYTES = 64 << 20
_ANGLES_LOCK = threading.Lock()


def angle(g):
    """(cos b, sin b), each (nj, ni) float64 we:sn, of the direction b of
    the grid's x axis at each point, counterclockwise from east; None where
    the axes are east and north already. Kept (read-only) for the grids
    used last."""
    with _ANGLES_LOCK:
        if g in _ANGLES:
            _ANGLES.move_to_end(g)
            return _ANGLES[g]
    ang = _angle(g)
    if ang is not None:
        for a in ang:
            a.flags.writeable = False
    size = 0 if ang is None else ang[0].nbytes * 2
    if size <= _ANGLES_BYTES // 2:
        with _ANGLES_LOCK:
            _ANGLES[g] = ang
            while sum(0 if a is None else a[0].nbytes * 2 for a in _ANGLES.values())                     > _ANGLES_BYTES:
                _ANGLES.popitem(last=False)
    return ang


def _angle(g):
    if g.template in _grids._ROTATED:
        return _rotated_axis(g)
    if g.template not in (20, 30):
        return None
    p = g.params
    lat, lon = _grids.latlon(g)
    if g.template == 30:
        lov, lt1, lt2, south = p[2], p[3], p[4], bool(p[7])
        n = _grids._cone(-lt1 if south else lt1, -lt2 if south else lt2)[0]
    else:
        lov, south, n = p[3], bool(p[6]), 1.0
    d = (lon - lov + 180.0) % 360.0 - 180.0
    b = np.radians(n * d if south else -n * d)
    return np.cos(b), np.sin(b)


def _rotated_axis(g):
    """angle() on a rotated grid: the bearing of its rotated east, measured
    a hair's breadth along it (the rotation keeps angles)."""
    rlat, rlon = _grids.rotated(g)
    sp = g.params[6:9]
    lat0, lon0 = _grids.unrotate(rlat, rlon, *sp)
    eps = 1e-5
    lat1, lon1 = _grids.unrotate(rlat, rlon + eps, *sp)
    de = np.radians((lon1 - lon0 + 180.0) % 360.0 - 180.0) * np.cos(np.radians(lat0))
    dn = np.radians(lat1 - lat0)
    b = np.arctan2(dn, de)
    return np.cos(b), np.sin(b)


def to_earth(u, v, g):
    """u, v along grid g's axes, turned to east and north (float32 out,
    float64 arithmetic; NaN stays NaN). u and v may be (ny, nx) or (n, ny,
    nx); grids whose axes are east and north come back as they are."""
    ang = angle(g)
    if ang is None:
        return u, v
    c, s = ang                                  # the grid's axes turned by b onto east
    return _turned(c, s, u, v, False)


def to_grid(u, v, g):
    """East and north components turned to grid g's axes (to_earth's
    inverse)."""
    ang = angle(g)
    if ang is None:
        return u, v
    c, s = ang
    return _turned(c, s, u, v, True)


def _turned(c, s, u, v, back):
    """u, v turned by the angles (c, s): to east and north, or back to the
    grid's axes -- in one pass (_turn), u and v the grid's shape or a stack
    of fields of it; numpy's broadcasting otherwise."""
    u, v = np.asarray(u), np.asarray(v)
    if u.shape == v.shape and u.shape[-c.ndim:] == c.shape and u.dtype == v.dtype             and u.dtype in (np.float32, np.float64):
        u, v = np.ascontiguousarray(u), np.ascontiguousarray(v)
        ue, ve = np.empty(u.shape, np.float32), np.empty(u.shape, np.float32)
        _turn(c.ravel(), s.ravel(), u.ravel(), v.ravel(), ue.ravel(), ve.ravel(), back)
        return ue, ve
    u64, v64 = np.asarray(u, np.float64), np.asarray(v, np.float64)
    if back:
        return (c * u64 + s * v64).astype(np.float32), (-s * u64 + c * v64).astype(np.float32)
    return (c * u64 - s * v64).astype(np.float32), (s * u64 + c * v64).astype(np.float32)


@njit(cache=True, nogil=True)
def _turn(c, s, u, v, ue, ve, back):
    """_turned's pass: numpy's arithmetic (float64, float32 out) without
    its six temporaries (RAP: 1.6 ms a pair against 0.2)."""
    m = c.size
    for k in range(u.size // m):
        o = k * m
        for i in range(m):
            uu, vv = np.float64(u[o + i]), np.float64(v[o + i])
            if back:
                ue[o + i] = np.float32(c[i] * uu + s[i] * vv)
                ve[o + i] = np.float32(-s[i] * uu + c[i] * vv)
            else:
                ue[o + i] = np.float32(c[i] * uu - s[i] * vv)
                ve[o + i] = np.float32(s[i] * uu + c[i] * vv)
