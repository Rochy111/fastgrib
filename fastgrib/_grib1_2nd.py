"""GRIB1 second-order ('complex') grid-point packing, unpacked to the
integers simple packing would hold (value = (R + X 2^E) 10^-D), for
_grib1 to give as GRIB2 simple packing. The forms (the BDS's extended
flags, octet 14):

- WMO's: groups of points, each a first-order value (its reference) and
  its points' second-order values in its width -- the groups the rows
  (row by row), or marked in a secondary bitmap, the widths one for all
  (constant width) or one each (general);
- ECMWF's general extended form (GRIBEX, ecCodes' grid_second_order):
  the groups' widths and lengths packed, and spatial differencing of
  order 1 to 3 undone (its first values and a bias given);
- either with the rows boustrophedonic (every other row reversed).

Every count and place is checked against the section before it is read:
a damaged message raises ValueError, never a read past its bytes."""
import numpy as np
from ._jit import njit

from ._octets import bits as _bits, ints, u


def unpack(bds, ext, nval, rows, bitmapped):
    """The integers X (int64, nval of them, in the file's point order) of a
    second-order BDS (bytes, its first octet at 0). ext: its extended
    flags; rows: the grid's points a row, file order; bitmapped: whether a
    bitmap leaves points out (then the coded values are the defined
    ones)."""
    b = np.frombuffer(bds, np.uint8)
    if ext & 0x40:
        raise ValueError('a matrix of values at each point (not read)')
    n1, n2 = u(b, 11, 2), u(b, 14, 2)
    ng = u(b, 16, 2) + 65536 * int(b[20])      # (Python ints: numpy 2 keeps uint8 math 8-bit)
    wf = int(b[10])
    if ext & 0x08:                              # ECMWF's general extended
        order = ext & 0x03
        ww, wl, nl = int(u(b, 21, 1)), int(u(b, 22, 1)), u(b, 23, 2)
        if ww > 32 or wl > 32:
            raise ValueError("group widths or lengths of more than 32 bits")
        at = 25
        spd = np.zeros(4, np.int64)
        if order:
            wspd = int(u(b, 25, 1))
            if wspd > 32:
                raise ValueError('spatial differencing values of %d bits' % wspd)
            at = 26
            _room(b, at, wspd * (order + 1))
            for k in range(order + 1):
                v = _bits(b, 8 * at + k * wspd, wspd)
                if k == order and wspd and v >> (wspd - 1):     # the bias: sign and magnitude
                    v = -(v & ((1 << (wspd - 1)) - 1))
                spd[k] = v
            at += (wspd * (order + 1) + 7) // 8
        _room(b, at, ng * ww)
        _room(b, nl - 1, ng * wl)
        _room(b, n1 - 1, ng * wf)
        widths = ints(b, 8 * at, ww, ng)
        lens = ints(b, 8 * (nl - 1), wl, ng)
        if lens.sum() != nval - order:
            raise ValueError('its groups hold %d values, not %d' % (lens.sum(), nval - order))
        x = _groups(b, n1, n2, wf, widths, lens, nval - order)
        x = _undifference(x, spd, order, nval)
    else:
        constant, marked = not ext & 0x10, bool(ext & 0x20)
        nw = 1 if constant else ng
        _room(b, 21, 8 * nw)
        widths = np.full(ng, b[21], np.int64) if constant else b[21:21 + ng].astype(np.int64)
        if marked:                              # a secondary bitmap: each group's first point
            _room(b, 21 + nw, nval)
            starts = np.flatnonzero(np.unpackbits(b[21 + nw:21 + nw + (nval + 7) // 8],
                                                  count=nval))
            if starts.size != ng or (ng and starts[0] != 0):
                raise ValueError('its secondary bitmap marks %d groups, not %d'
                                 % (starts.size, ng))
            lens = np.diff(np.append(starts, nval))
        else:                                   # row by row
            if bitmapped:
                raise ValueError('row-by-row packing with a bitmap (not read)')
            lens = np.asarray(rows, np.int64)
            if lens.size != ng or lens.sum() != nval:
                raise ValueError('its %d groups are not the grid\'s %d rows' % (ng, lens.size))
        _room(b, n1 - 1, ng * wf)
        x = _groups(b, n1, n2, wf, widths, lens, nval)
    if ext & 0x04:                              # boustrophedonic: every other row reversed
        if bitmapped:
            raise ValueError('boustrophedonic rows with a bitmap (not read)')
        x = _unturn(x, np.asarray(rows, np.int64))
    return x


def _room(b, at, nbits):
    """That nbits from octet `at` (from 0) lie in the section."""
    if at < 0 or at + (int(nbits) + 7) // 8 > b.size:
        raise ValueError('its second-order data run past the section')


def _groups(b, n1, n2, wf, widths, lens, n):
    if widths.size and widths.max() > 32 or wf > 32:
        raise ValueError('second-order widths beyond 32 bits')
    _room(b, n2 - 1, int((widths * lens).sum()))
    return _unpack_groups(b, 8 * (n1 - 1), wf, 8 * (n2 - 1), widths, lens, n)


@njit(cache=True, nogil=True)
def _unpack_groups(b, fbit, wf, sbit, widths, lens, n):
    x = np.empty(n, np.int64)
    k = 0
    for g in range(widths.size):
        ref = _bits(b, fbit + g * wf, wf)
        w = widths[g]
        for i in range(lens[g]):
            x[k] = ref + _bits(b, sbit, w)
            sbit += w
            k += 1
    return x


@njit(cache=True, nogil=True)
def _undifference(d, spd, order, n):
    """Spatial differencing of `order` undone: the first values spd[:order],
    each next one its difference (d + the bias, spd[order]) on top of the
    extrapolation of those before it."""
    x = np.empty(n, np.int64)
    if order == 0:
        x[:] = d
        return x
    bias = spd[order]
    for i in range(order):
        x[i] = spd[i]
    for i in range(order, n):
        v = d[i - order] + bias
        if order == 1:
            v += x[i - 1]
        elif order == 2:
            v += 2 * x[i - 1] - x[i - 2]
        else:
            v += 3 * x[i - 1] - 3 * x[i - 2] + x[i - 3]
        x[i] = v
    return x


@njit(cache=True, nogil=True)
def _unturn(x, rows):
    out = x.copy()
    s = 0
    for j in range(rows.size):
        k = rows[j]
        if j % 2 == 1:
            for i in range(k):
                out[s + i] = x[s + k - 1 - i]
        s += k
    return out
