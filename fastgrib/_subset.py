"""A box cut out of fields, as wgrib2's -small_grib: the rows and columns
of a grid whose points lie in (west, east, south, north) -- a lat-lon or
Gaussian grid's own (round the globe where the grid goes round), a
projected grid's (Mercator, polar stereographic, Lambert) smallest
rectangle holding every point in the box -- and the section 3 of the
smaller grid, to write it as GRIB2 (its first point the south-west one,
rows south to north: as read() lays fields out)."""
import struct
from typing import NamedTuple

import numpy as np

from . import _grids
from ._octets import sm_bytes

_CUT = frozenset((0, 40, 10, 20, 30))           # the templates a box is cut from


class Window(NamedTuple):
    rows: slice         # of read()'s rows (south to north)
    cols: object        # a slice of its columns (west to east), or a tuple of them (round the seam)
    ni: int
    nj: int


def check(box):
    """(west, east, south, north) of a box, as floats, checked."""
    try:
        w, e, s, n = (float(x) for x in box)
    except (TypeError, ValueError):
        raise TypeError('box: (west, east, south, north) in degrees, as wgrib2 -small_grib '
                        'and cartopy give it')
    if not (-90.0 <= s <= n <= 90.0):
        raise ValueError('box: -90 <= south <= north <= 90, not %g, %g' % (s, n))
    return w, e, s, n


def _inside_lon(lon, w, e):
    """Whether longitudes lie from w east to e (round the antimeridian when
    e is west of w; all of them when the box is a full circle)."""
    width = (e - w) % 360.0
    if width == 0.0 and e != w:
        width = 360.0
    return ((lon - w) % 360.0) <= width + 1e-9


_WINDOWS = {}


def window(g, box):
    """The Window of grid g (a layout) a box covers; ValueError if none of
    its points do, or the grid is not one a box is cut from."""
    key = (g, box)
    hit = _WINDOWS.get(key)
    if hit is not None:
        return hit
    if g.template not in _CUT or g.pl or _grids.stagger(g) is not None:
        raise ValueError('%r: a box is cut from lat-lon, Gaussian, Mercator, polar '
                         'stereographic and Lambert grids (not reduced, rotated or staggered)'
                         % (g,))
    w, e, s, n = box
    ax = _grids.axes(g)
    if ax is not None:                          # lat-lon, Gaussian: rows and columns apart
        lat, lon = ax
        rows = np.nonzero((lat >= s - 1e-9) & (lat <= n + 1e-9))[0]
        cols = np.nonzero(_inside_lon(lon % 360.0, w, e))[0]
        if not rows.size or not cols.size:
            raise ValueError('box %s: no point of %r in it' % (box, g))
        step = (lon[-1] - lon[0]) / (g.ni - 1) if g.ni > 1 else 360.0
        round_ = abs(g.ni * step - 360.0) < 0.5 * step
        if round_ and cols.size < g.ni and np.any(np.diff(cols) > 1):
            # across the grid's seam: from the box's west edge on
            cols = cols[np.argsort((lon[cols] - w) % 360.0, kind='stable')]
            colsel = tuple(int(c) for c in cols)        # (a tuple: windows are dict keys)
        else:
            colsel = slice(int(cols[0]), int(cols[-1]) + 1)
        win = Window(slice(int(rows[0]), int(rows[-1]) + 1), colsel, int(cols.size),
                     int(rows.size))
    else:                                       # projected: the rectangle round the points in it
        lat, lon = _grids.latlon(g)
        inside = (lat >= s - 1e-9) & (lat <= n + 1e-9) & _inside_lon(lon % 360.0, w, e)
        rr, cc = np.nonzero(np.any(inside, 1))[0], np.nonzero(np.any(inside, 0))[0]
        if not rr.size:
            raise ValueError('box %s: no point of %r in it' % (box, g))
        win = Window(slice(int(rr[0]), int(rr[-1]) + 1), slice(int(cc[0]), int(cc[-1]) + 1),
                     int(cc[-1] - cc[0] + 1), int(rr[-1] - rr[0] + 1))
    if len(_WINDOWS) > 256:
        _WINDOWS.clear()
    _WINDOWS[key] = win
    return win


def cut(a, win):
    """Fields (..., ny, nx) as read() lays them out, cut to the window (a
    copy: the whole field's memory let go)."""
    a = a[..., win.rows, :]
    a = a[..., win.cols] if isinstance(win.cols, slice) else \
        a.take(np.asarray(win.cols, np.intp), axis=-1)
    if hasattr(a, 'get'):                       # cupy
        import cupy as cp
        return cp.ascontiguousarray(a)
    return np.ascontiguousarray(a)


def latlon(g, win):
    lat, lon = _grids.latlon(g)
    return cut(lat, win), cut(lon, win)


_CORNERS = {}


def _corners(g, win):
    """(lat, lon) of the window's first (south-west) and last points, kept:
    the same for every field on the grid (the whole grid's points are not
    worked out again for each)."""
    def part(x):                                # (a slice is hashable from Python 3.12 only)
        return (x.start, x.stop, x.step) if isinstance(x, slice) else x
    key = (g, part(win.rows), part(win.cols), win.ni, win.nj)
    hit = _CORNERS.get(key)
    if hit is None:
        lat, lon = latlon(g, win)
        hit = (lat[0, 0], lon[0, 0], lat[-1, -1], lon[-1, -1])
        if len(_CORNERS) > 256:
            _CORNERS.clear()
        _CORNERS[key] = hit
    return hit


def _udeg(x):
    return struct.pack('>I', int(round((x % 360.0) * 1e6)) % 360000000)


def section3(sec, g, win):
    """The section 3 (bytes) of the window cut from grid g, whose own is
    sec: its points, size, first (south-west) and last points, and
    scanning (rows south to north, columns west to east); the rest -- the
    earth, the spacing, the projection -- as it was."""
    b = bytearray(sec)
    if g.template in (0, 40):
        basic, sub = struct.unpack_from('>II', b, 38)
        if basic not in (0, 0xFFFFFFFF) and sub not in (0, 0xFFFFFFFF):
            raise ValueError('%r: its angles in other units than microdegrees' % (g,))
    struct.pack_into('>I', b, 6, win.ni * win.nj)
    struct.pack_into('>II', b, 30, win.ni, win.nj)
    la1, lo1, la2, lo2 = _corners(g, win)
    if g.template in (0, 40):
        b[46:50], b[50:54] = sm_bytes(int(round(la1 * 1e6)), 4), _udeg(lo1)
        b[55:59], b[59:63] = sm_bytes(int(round(la2 * 1e6)), 4), _udeg(lo2)
        b[71] = 0x40
    elif g.template == 10:                      # Mercator: first and last points
        b[38:42], b[42:46] = sm_bytes(int(round(la1 * 1e6)), 4), _udeg(lo1)
        b[51:55], b[55:59] = sm_bytes(int(round(la2 * 1e6)), 4), _udeg(lo2)
        b[59] = 0x40
    else:                                       # polar stereographic, Lambert: the first
        b[38:42], b[42:46] = sm_bytes(int(round(la1 * 1e6)), 4), _udeg(lo1)
        b[64] = 0x40
    return bytes(b)
