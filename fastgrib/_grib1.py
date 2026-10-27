"""GRIB1, read as the GRIB2 it would be: each message turned into the
equivalent GRIB2 message -- its product (PDS), grid (GDS), bitmap (BMS)
and simple-packed data (BDS) as GRIB2's sections 1, 3, 4, 5, 6 and 7, the
packed bits copied as they are -- so every decoder, the CPU's and the
GPU's, and every name, level and time wording apply to it unchanged.

What GRIB1 says and GRIB2 says otherwise:
- parameters: NCEP's (table versions 2 and 128-141) by its ON388 tables,
  ECMWF's (128 and up) by ecCodes', each with the GRIB2 numbers of the
  same parameter where there is one (else named by its GRIB1 table);
- levels (table 3) as code table 4.5's, in SI units (hPa to Pa, cm to m,
  layers' two sides);
- time ranges (table 5): a forecast or analysis (0, 1, 10) a point in
  time, a range, average, accumulation or difference (2-5) a statistic
  (4.8); others as their P1, with TR=n said;
- the reference value: an IBM float, exactly an IEEE one (24 bits);
- grids: lat-lon, Gaussian (thinned too), Mercator, Lambert, polar
  stereographic (true at 60 deg), rotated lat-lon and Gaussian.
Second-order (complex) packing is unpacked when read (_grib1_2nd); not
read: spherical harmonics, predefined grids (no GDS) and bitmaps.
"""
import calendar
import struct
from datetime import datetime, timedelta

import numpy as np

from .tables.params1 import PARAMS1
from ._octets import ibm, ibm_array, s, sm_bytes, u

NCEP, ECMWF = 7, 98
# NCEP's GRIB1 abbreviations its GRIB2 table spells otherwise (the same quantity and units)
RENAMED = {'CLWMR': 'CLMR', 'UGWD': 'U-GWD', 'VGWD': 'V-GWD', 'PDMX24': 'PDMAX24'}


class Unread(Exception):
    """A GRIB1 message this module does not turn into GRIB2 (why)."""


def length(b, pos):
    """(total length, whether ECMWF's large-message rule applies)."""
    n = int.from_bytes(bytes(b[pos + 4:pos + 7]), 'big')
    if n & 0x800000:                            # ECMWF: length in 120-byte units
        return (n & 0x7FFFFF) * 120, True
    return n, False


# WMO's international exchange grids (GRIB1 Table B): (Ni, Nj, Lo1, Lo2, La1, La2, Di, Dj),
# millidegrees -- quarter globes whose pole row is a single value
_WMO_GRIDS = {21: (37, 37, 0, 180000, 0, 90000, 5000, 2500),
              22: (37, 37, -180000, 0, 0, 90000, 5000, 2500),
              23: (37, 37, 0, 180000, -90000, 0, 5000, 2500),
              24: (37, 37, -180000, 0, -90000, 0, 5000, 2500),
              25: (72, 19, 0, 355000, 0, 90000, 5000, 5000),
              26: (72, 19, 0, 355000, -90000, 0, 5000, 5000),
              61: (91, 46, 0, 180000, 0, 90000, 2000, 2000),
              62: (91, 46, -180000, 0, 0, 90000, 2000, 2000),
              63: (91, 46, 0, 180000, -90000, 0, 2000, 2000),
              64: (91, 46, -180000, 0, -90000, 0, 2000, 2000)}


def _predefined(centre, grid):
    """The GDS (bytes) of a predefined grid: WMO's international ones, or
    NCEP's (ON388 Table B) for its own messages; None if not known."""
    if grid in _WMO_GRIDS:                      # thinned: the pole row's one value
        ni, nj, lo1, lo2, la1, la2, di, dj = _WMO_GRIDS[grid]
        pl = [ni] * (nj - 1) + [1] if la2 == 90000 else [1] + [ni] * (nj - 1)
        body = bytes([0, 33, 0]) + struct.pack('>HH', 0xFFFF, nj) + sm_bytes(la1, 3) + sm_bytes(lo1, 3) + \
            bytes([0x80]) + sm_bytes(la2, 3) + sm_bytes(lo2, 3) + struct.pack('>HH', 0xFFFF, dj) + \
            bytes([0x40]) + bytes(4) + b''.join(struct.pack('>H', k) for k in pl)
        return (3 + len(body)).to_bytes(3, 'big') + body
    if centre != NCEP:
        return None
    from .tables.ncep_grids import GRIDS
    d = GRIDS.get(grid)
    if d is None:
        return None
    kind, ni, nj, la1, lo1, res = d[:6]
    head = struct.pack('>HH', ni, nj) + sm_bytes(la1 * 1000, 3) + sm_bytes(lo1 * 1000, 3) + bytes([res])
    if kind == 'latlon':
        la2, lo2, di, dj, scan = d[6:]
        body = bytes([0, 255, 0]) + head + sm_bytes(la2 * 1000, 3) + sm_bytes(lo2 * 1000, 3) + \
            struct.pack('>HH', round(di * 1000), round(dj * 1000)) + bytes([scan]) + bytes(4)
    elif kind in ('polar', 'lambert'):
        lov, dx, dy, proj, scan = d[6:11]
        body = bytes([0, 255, 5 if kind == 'polar' else 3]) + head + sm_bytes(lov * 1000, 3) + \
            sm_bytes(dx * 1000, 3) + sm_bytes(dy * 1000, 3) + bytes([proj, scan])
        if kind == 'polar':
            body += bytes(4)
        else:
            body += sm_bytes(d[11] * 1000, 3) + sm_bytes(d[12] * 1000, 3) + sm_bytes(-90000, 3) + sm_bytes(0, 3) + bytes(2)
    else:                                       # Mercator
        la2, lo2, latin, scan, di, dj = d[6:]
        body = bytes([0, 255, 1]) + head + sm_bytes(la2 * 1000, 3) + sm_bytes(lo2 * 1000, 3) + \
            sm_bytes(latin * 1000, 3) + bytes([0, scan]) + sm_bytes(di * 1000, 3) + sm_bytes(dj * 1000, 3) + bytes(8)
    return (3 + len(body)).to_bytes(3, 'big') + body


def _gribex_edge(js, ks, ms, lap):
    """Factors for a GRIB1 spectral subset's values: GRIBEX wrote its last
    wavenumber (n = JS) multiplied by the Laplacian's (n (n + 1))^P as
    the packed ones are -- ecCodes takes every GRIB1 field so, and the same
    field in GRIB2 agrees -- so those are divided back."""
    f = []
    for m in range(ms + 1):
        top = min(js + m, ks)
        for n in range(m, top + 1):
            f += [(n * (n + 1.0)) ** -lap if n == top and n > 0 else 1.0] * 2
    return np.array(f)


def _sec(num, body):
    return struct.pack('>IB', 5 + len(body), num) + body


# --- the parameter

def _param(table, num, centre):
    """(GRIB2 (discipline, category, number) or None, (name, description,
    units) of GRIB1's own, {GRIB2 key: value} of the level and statistic
    the parameter implies -- ECMWF's 2t is at 2 m, its tp accumulated)."""
    hit = None
    if centre == ECMWF and table >= 128:
        from .tables.params1_ecmwf import PARAMS1_EC
        hit = PARAMS1_EC.get((table, num))
    elif table <= 3 and num < 128:               # the WMO part: NCEP's names
        hit = PARAMS1.get((2, num))
    elif centre == NCEP:                         # a local table; below 128 WMO's unless it says
        hit = PARAMS1.get((table, num)) or (PARAMS1.get((2, num)) if num < 128 else None)
    if hit is None:
        return None, ('var%d_%d' % (table, num), 'unknown', 'unknown'), {}
    return hit[3], hit[:3], dict(hit[4] or ()) if len(hit) > 4 else {}


# --- levels: GRIB1 table 3 -> (type1, sf1, v1, type2, sf2, v2), GRIB2's

_SINGLE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 102: 101, 200: 200,
           201: 201, 204: 204, 206: 206, 207: 207, 209: 209, 210: 210, 211: 211, 212: 212,
           213: 213, 214: 214, 215: 215, 220: 220, 222: 222, 223: 223, 224: 224, 232: 232,
           233: 233, 234: 234, 235: 235, 242: 242, 243: 243, 244: 244, 245: 245, 246: 246,
           247: 247, 248: 248, 249: 249, 251: 251, 252: 252}
MISS = (255, 0xFFFFFFFF)


def _level(t, o11, o12):
    """GRIB2's (type1, (sf, sv), type2, (sf, sv)) of a GRIB1 level."""
    v = o11 << 8 | o12
    none = (255, MISS, 255, MISS)
    if t in _SINGLE:
        return _SINGLE[t], MISS, 255, MISS
    if t == 20:                                 # isothermal, 1/100 K
        return 20, (2, v), 255, MISS
    if t == 100:                                # isobaric hPa
        return 100, (0, v * 100), 255, MISS
    if t == 101:                                # isobaric layer, kPa
        return 100, (0, o11 * 1000), 100, (0, o12 * 1000)
    if t == 103:                                # m above MSL
        return 102, (0, v), 255, MISS
    if t == 104:                                # layer, hm above MSL
        return 102, (0, o11 * 100), 102, (0, o12 * 100)
    if t == 105:                                # m above ground
        return 103, (0, v), 255, MISS
    if t == 106:                                # layer, hm above ground
        return 103, (0, o11 * 100), 103, (0, o12 * 100)
    if t == 107:                                # sigma, 1/10000
        return 104, (4, v), 255, MISS
    if t == 108:                                # sigma layer, 1/100
        return 104, (2, o11), 104, (2, o12)
    if t == 109:                                # hybrid
        return 105, (0, v), 255, MISS
    if t == 110:
        return 105, (0, o11), 105, (0, o12)
    if t == 111:                                # cm below land
        return 106, (2, v), 255, MISS
    if t == 112:                                # layer, cm below land
        return 106, (2, o11), 106, (2, o12)
    if t == 113:                                # isentropic K
        return 107, (0, v), 255, MISS
    if t == 114:                                # layer, 475 - K
        return 107, (0, 475 - o11), 107, (0, 475 - o12)
    if t == 115:                                # hPa above ground
        return 108, (0, v * 100), 255, MISS
    if t == 116:                                # layer, hPa above ground
        return 108, (0, o11 * 100), 108, (0, o12 * 100)
    if t == 117:                                # PV, 1e-9 K m2/kg/s (signed)
        pv = -(v & 0x7FFF) if v & 0x8000 else v
        return 109, (9, pv), 255, MISS
    if t == 119:                                # eta, 1/10000
        return 111, (4, v), 255, MISS
    if t == 120:
        return 111, (2, o11), 111, (2, o12)
    if t == 121:                                # isobaric layer, 1100 - hPa
        return 100, (0, (1100 - o11) * 100), 100, (0, (1100 - o12) * 100)
    if t == 125:                                # cm above ground
        return 103, (2, v), 255, MISS
    if t == 141:                                # isobaric layer, kPa / 1100 - hPa
        return 100, (0, o11 * 1000), 100, (0, (1100 - o12) * 100)
    if t == 160:                                # m below sea level
        return 160, (0, v), 255, MISS
    return none


def _implied_level(implied):
    """_level()'s answer from a parameter's GRIB2 level keys."""
    out = []
    for side in ('First', 'Second'):
        t = implied.get('typeOf%sFixedSurface' % side, 255)
        sv = implied.get('scaledValueOf%sFixedSurface' % side)
        out += [t, MISS if sv is None else (implied.get('scaleFactorOf%sFixedSurface' % side, 0),
                                            sv)]
    return tuple(out)


def _fixed(t, sv):
    sf, v = sv
    return bytes([t, sf & 0xFF if sf >= 0 else 0x80 | -sf]) + (
        struct.pack('>I', v) if (sf, v) == MISS else sm_bytes(v, 4))


# --- time

_UNIT = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 10: 10, 11: 11, 12: 12, 254: 13}


def _add(t, unit, n):
    """t plus n of GRIB1 time unit `unit`."""
    if unit in (3, 4, 5, 6, 7):                 # months and up
        months = n * {3: 1, 4: 12, 5: 120, 6: 360, 7: 1200}[unit]
        y, m = divmod(t.month - 1 + months, 12)
        y += t.year
        return t.replace(year=y, month=m + 1, day=min(t.day, calendar.monthrange(y, m + 1)[1]))
    step = {0: 60, 1: 3600, 2: 86400, 10: 10800, 11: 21600, 12: 43200, 254: 1}.get(unit)
    if step is None:
        raise Unread('time unit %d' % unit)
    return t + timedelta(seconds=n * step)


# --- the grid: GDS -> GRIB2 section 3

_EARTH = {0: 0, 1: 2}                           # GRIB1 resolution bit 0x40: IAU 1965 oblate


def _deg(b, q):
    return s(b, q, 3) * 1000                     # millidegrees -> microdegrees


def _ulon(v):
    return struct.pack('>I', v % 360000000)     # GRIB2's longitudes: unsigned, 0-360


def _grid(b, g, glen):
    """Section 3's body (after length and number) of a GDS at g."""
    nv, pvl, kind = b[g + 3], b[g + 4], b[g + 5]
    res = b[g + 16]
    shape = _EARTH[1 if res & 0x40 else 0]
    flags = (0x30 if res & 0x80 else 0) | (0x08 if res & 0x08 else 0)
    earth = bytes([shape]) + bytes([255]) + b'\xff' * 4 + bytes([255]) + b'\xff' * 4 + \
        bytes([255]) + b'\xff' * 4
    ni, nj = struct.unpack_from('>HH', b, g + 6)
    pl = None
    if kind in (0, 4, 10, 14) and ni == 0xFFFF:                    # thinned rows
        at = g + (pvl - 1 if pvl not in (0, 255) else 32) + 4 * nv
        if nv == 0 and pvl in (0, 255):
            at = g + 32
        pl = struct.unpack_from('>%dH' % nj, b, at)
    npts = sum(pl) if pl else ni * nj
    head = lambda tmpl: (bytes([0]) + struct.pack('>I', npts) +    # noqa: E731
                         (bytes([2, 1]) if pl else bytes([0, 0])) + struct.pack('>H', tmpl))
    if kind in (50, 60, 70, 80):                # spherical harmonics (rotated, stretched, both)
        from ._spectral import count
        j, k, m = struct.unpack_from('>HHH', b, g + 6)
        n = count(j, k, m)
        tmpl = {50: 50, 60: 51, 70: 52, 80: 53}[kind]
        body = bytes([0]) + struct.pack('>I', n) + bytes([0, 0]) + struct.pack('>H', tmpl) + \
            struct.pack('>III', j, k, m) + bytes([b[g + 12], b[g + 13]])
        q = g + 32
        if kind in (60, 80):                    # the southern pole, the angle of rotation
            body += sm_bytes(_deg(b, q), 4) + _ulon(_deg(b, q + 3)) + struct.pack('>f', ibm(b, q + 6))
            q += 10
        if kind in (70, 80):                    # the pole of stretching, the factor
            body += sm_bytes(_deg(b, q), 4) + sm_bytes(_deg(b, q + 3), 4) + \
                struct.pack('>I', round(ibm(b, q + 6) * 1e6))
        return body, n, ()
    if kind == 90:                              # space view: GRIB2's 3.90 (Xp, Yp to 1e-3)
        nr = u(b, g + 31, 3)
        body = head(90) + earth + struct.pack('>II', ni, nj) + sm_bytes(_deg(b, g + 10), 4) + \
            sm_bytes(_deg(b, g + 13), 4) + bytes([flags]) + \
            struct.pack('>IIII', u(b, g + 17, 3), u(b, g + 20, 3), u(b, g + 23, 2) * 1000,
                        u(b, g + 25, 2) * 1000) + bytes([b[g + 27]]) + \
            sm_bytes(u(b, g + 28, 3) * 1000, 4) + \
            struct.pack('>III', 0xFFFFFFFF if nr == 0xFFFFFF else nr, u(b, g + 34, 2),
                        u(b, g + 36, 2))
        return body, npts, (ni,) * nj
    if kind in (0, 4, 10, 14):
        la1, lo1, la2, lo2 = _deg(b, g + 10), _deg(b, g + 13), _deg(b, g + 17), _deg(b, g + 20)
        di = struct.unpack_from('>H', b, g + 23)[0]
        dj = struct.unpack_from('>H', b, g + 25)[0]
        di = 0xFFFFFFFF if di == 0xFFFF or pl else di * 1000
        gauss = kind in (4, 14)
        dj = dj if gauss else (0xFFFFFFFF if dj == 0xFFFF else dj * 1000)
        body = head({0: 0, 4: 40, 10: 1, 14: 41}[kind]) + earth + struct.pack(
            '>II', 0xFFFFFFFF if pl else ni, nj) + struct.pack('>II', 0, 0xFFFFFFFF) + \
            sm_bytes(la1, 4) + _ulon(lo1) + bytes([flags]) + sm_bytes(la2, 4) + _ulon(lo2) + \
            struct.pack('>II', di, dj) + bytes([b[g + 27]])
        if kind in (10, 14):                    # rotated: the southern pole, an angle
            body += sm_bytes(_deg(b, g + 32), 4) + _ulon(_deg(b, g + 35)) + \
                struct.pack('>f', ibm(b, g + 38))
        if pl:
            body += b''.join(struct.pack('>H', k) for k in pl)
        return body, npts, (tuple(pl) if pl else (ni,) * nj)
    if kind == 1:                               # Mercator
        la1, lo1, la2, lo2 = _deg(b, g + 10), _deg(b, g + 13), _deg(b, g + 17), _deg(b, g + 20)
        lad = _deg(b, g + 23)
        di, dj = u(b, g + 28, 3) * 1000, u(b, g + 31, 3) * 1000
        body = head(10) + earth + struct.pack('>II', ni, nj) + sm_bytes(la1, 4) + _ulon(lo1) + \
            bytes([flags]) + sm_bytes(lad, 4) + sm_bytes(la2, 4) + _ulon(lo2) + bytes([b[g + 27]]) + \
            struct.pack('>III', 0, di, dj)
        return body, npts, (tuple(pl) if pl else (ni,) * nj)
    if kind in (3, 5):                          # Lambert, polar stereographic
        la1, lo1, lov = _deg(b, g + 10), _deg(b, g + 13), _deg(b, g + 17)
        dx, dy = u(b, g + 20, 3) * 1000, u(b, g + 23, 3) * 1000       # m -> mm
        centre, scan = b[g + 26], b[g + 27]
        if kind == 5:
            lad = -60000000 if centre & 0x80 else 60000000           # GRIB1: true at 60
            body = head(20) + earth + struct.pack('>II', ni, nj) + sm_bytes(la1, 4) + \
                _ulon(lo1) + bytes([flags]) + sm_bytes(lad, 4) + _ulon(lov) + \
                struct.pack('>II', dx, dy) + bytes([centre, scan])
            return body, npts, (tuple(pl) if pl else (ni,) * nj)
        lt1, lt2 = _deg(b, g + 28), _deg(b, g + 31)
        spl, splo = _deg(b, g + 34), _deg(b, g + 37)
        body = head(30) + earth + struct.pack('>II', ni, nj) + sm_bytes(la1, 4) + _ulon(lo1) + \
            bytes([flags]) + sm_bytes(lt1, 4) + _ulon(lov) + struct.pack('>II', dx, dy) + \
            bytes([centre, scan]) + sm_bytes(lt1, 4) + sm_bytes(lt2, 4) + sm_bytes(spl, 4) + _ulon(splo)
        return body, npts, (tuple(pl) if pl else (ni,) * nj)
    if kind in (203, 205):                      # NCEP's rotated E- and B-grids: 3.32768, 3.32769
        la1, lo1, lac, loc = _deg(b, g + 10), _deg(b, g + 13), _deg(b, g + 17), _deg(b, g + 20)
        if kind == 203:                         # spaced as iplib spaces it: the first point a
            from ._regrid import rotate         # corner of a domain centred on (lac, loc)
            (r1,), (o1,) = rotate(np.array([la1 / 1e6]), np.array([lo1 / 1e6]),
                                  lac / 1e6 - 90.0, loc / 1e6)
            o1 = (o1 + 180.0) % 360.0 - 180.0
            di, dj = round(2e6 * abs(o1) / max(1, ni - 1)), round(2e6 * abs(r1) / max(1, nj - 1))
        else:
            di, dj = u(b, g + 23, 2) * 1000, u(b, g + 25, 2) * 1000
        body = head(32768 if kind == 203 else 32769) + earth + struct.pack('>IIII', ni, nj, 0, 0xFFFFFFFF) + \
            sm_bytes(la1, 4) + _ulon(lo1) + bytes([flags]) + sm_bytes(lac, 4) + _ulon(loc) + \
            struct.pack('>II', di, dj) + bytes([b[g + 27]])
        if kind == 205:                         # its last point
            body += sm_bytes(_deg(b, g + 28), 4) + _ulon(_deg(b, g + 31))
        return body, npts, (ni,) * nj
    if kind in (201, 202):
        raise Unread('grid type %d (an E-grid without its centre: NCEP retired it)' % kind)
    raise Unread('grid type %d (GRIB1 table 6)' % kind)


# --- the message

def transcode(b, pos, data=True):
    """(GRIB2 bytes, the GRIB1 message's length, (name, description,
    units) to give the field when GRIB2 has no number for its parameter,
    or None; extra wording or '') of the GRIB1 message at pos. Without
    `data`: its sections but the bitmap and packed values (what scanning
    reads), the message checked whole all the same."""
    lay = _layout(b, pos)
    return _assemble(memoryview(b).cast('B'), lay[0], data), lay[1], lay[2], lay[3]


def _assemble(b, plan, data=True):
    """The GRIB2 message of a plan (_layout's): its sections 1-5 made,
    the bitmap and packed bits taken from b (a memoryview) -- one copy. A
    matrix of values: a message of a field an element."""
    if plan[4][0] == 'matrix':
        from . import _grib1_matrix
        _, q, stop, ext, bitmap, npts = plan[4]
        try:
            elems = _grib1_matrix.fields(b, q, stop, npts, bitmap, plan[2][1], ext)
        except (ValueError, IndexError) as err:
            raise ValueError('GRIB1 matrix of values: %s' % err)
        return _grib1_matrix.message(plan[0], plan[1], plan[2], elems)
    return b''.join(_pieces(b, plan, data))


def _pieces(b, plan, data=True):
    """_assemble()'s message in its pieces: what is made, and views of b.
    Second-order data are unpacked here, to the integers simple packing
    holds, in as few bits as they need."""
    disc, head, (s5, nbits, suffix), bm, how = plan
    if data and bm is not None:
        s6, bits = struct.pack('>IBB', 6 + bm[1], 6, 0), b[bm[0]:bm[0] + bm[1]]
    else:
        s6, bits = struct.pack('>IBB', 6, 6, 255), b''
    packed = b''
    if data and how[0] == 'copy':
        packed = b[how[1]:how[1] + how[2]]
    elif data and how[0] == 'spec':             # the subset's IBM floats as IEEE, then bits
        _, at, ts, start, need, js, ks, ms, lap = how
        sub = ibm_array(b[at:at + 4 * ts]) * _gribex_edge(js, ks, ms, lap)
        packed = sub.astype('>f4').tobytes() + bytes(b[start:start + need])
    elif data:
        from . import _grib1_2nd
        from ._encode import _put_fixed
        _, q, stop, ext, rows, bitmapped = how
        nval = struct.unpack_from('>I', s5, 0)[0]
        try:
            x = _grib1_2nd.unpack(bytes(b[q:stop]), ext, nval, rows, bitmapped)
        except (ValueError, IndexError) as err:
            raise ValueError('GRIB1 second-order packing: %s' % err)
        nbits = int(x.max()).bit_length() if x.size else 0
        if nbits > 32 or (x.size and x.min() < 0):
            raise ValueError('GRIB1 second-order packing: values out of range (damaged)')
        out = np.zeros((x.size * nbits + 7) // 8, np.uint8)
        _put_fixed(x, nbits, out, 0)
        packed = memoryview(out)
    s5 = s5 + bytes([nbits]) + suffix
    n = 16 + len(head) + 5 + len(s5) + len(s6) + len(bits) + 5 + len(packed) + 4
    return (b''.join((b'GRIB', bytes([0, 0, disc, 2]), struct.pack('>Q', n), head,
                      struct.pack('>IB', 5 + len(s5), 5), s5, s6)), bits,
            struct.pack('>IB', 5 + len(packed), 7), packed, b'7777')


def _layout(b, pos):
    """((discipline, GRIB2 sections 1-5, the bitmap's (start, bytes) in b
    or None, the packed bits' (start, bytes)), the GRIB1 message's length,
    its own names or None, extra wording) of the message at pos: its
    GRIB2 made but for the bytes it shares with the GRIB1."""
    total, large = length(b, pos)
    end = pos + total
    if total < 8 + 28 + 11 + 4 or end > len(b):
        raise Unread('cut short: it says %d bytes, %d are there' % (total, len(b) - pos))
    b = memoryview(b).cast('B') if not isinstance(b, memoryview) else b   # octets as ints, no copy
    if bytes(b[end - 4:end]) != b'7777':
        hit = bytes(b[pos:end]).rfind(b'7777') if large else -1
        if hit < 0:
            raise Unread('no end marker where its length says it ends')
        end = pos + hit + 4
    p = pos + 8
    plen = u(b, p, 3)
    table, centre, grid_id, flag = b[p + 3], b[p + 4], b[p + 6], b[p + 7]
    par, ltype, o11, o12 = b[p + 8], b[p + 9], b[p + 10], b[p + 11]
    yc, mo, dy, hh, mi = b[p + 12:p + 17]
    unit, p1, p2, tri = b[p + 17], b[p + 18], b[p + 19], b[p + 20]
    nmiss = b[p + 23]
    century, subc = b[p + 24], b[p + 25]
    dscale = bytes(b[p + 26:p + 28])
    year = (century - 1) * 100 + yc
    try:
        ref = datetime(year, mo, dy, hh, mi)
    except ValueError:
        raise Unread('reference time %d-%d-%d %d:%d' % (year, mo, dy, hh, mi))
    q = p + plen
    if flag & 0x80:
        glen = u(b, q, 3)
        g3, npts, rows = _grid(b, q, glen)
        q += glen
    else:                                       # a predefined grid: its description made
        gds = _predefined(centre, grid_id)
        if gds is None:
            raise Unread('a predefined grid (%d) of centre %d, no grid description'
                         % (grid_id, centre))
        g3, npts, rows = _grid(memoryview(gds), 0, len(gds))
    bitmap = bm = None
    if flag & 0x40:
        blen = u(b, q, 3)
        if struct.unpack_from('>H', b, q + 4)[0] != 0:
            raise Unread('a predefined bitmap')
        bm = (q + 6, (npts + 7) // 8)
        bitmap = bytes(b[bm[0]:bm[0] + bm[1]])
        if len(bitmap) < bm[1] or bm[0] + bm[1] > end - 4:
            raise Unread('its bitmap is cut short')
        q += blen
    blen = u(b, q, 3)
    if large and blen < 120:                    # ECMWF's large message: to the end marker
        blen = end - 4 - q
    bflags, nbits = b[q + 3], b[q + 10]
    spectral = bool(bflags & 0x80)
    second = bool(bflags & 0x40) and not spectral   # second-order: unpacked when read
    e = bytes(b[q + 4:q + 6])
    r = ibm(b, q + 6)
    nval = npts if bitmap is None else \
        bin(int.from_bytes(bitmap, 'big') >> (8 * len(bitmap) - npts)).count('1')    # (int.bit_count: 3.10+)
    stop = min(q + blen, end - 4)
    suffix = b'\0'                              # 5.0's octet 21: floating-point values
    tmpl5 = 0
    if spectral:                                # 5.50 (simple), 5.51 (ECMWF's complex)
        if struct.unpack_from('>H', g3, 7)[0] not in (50, 51, 52, 53):
            raise Unread('spectral packing of a grid of points')
        if bitmap is not None:
            raise Unread('spherical harmonics with a bitmap')
        if bflags & 0x40:
            ip = struct.unpack_from('>H', b, q + 13)[0]
            ip = -(ip & 0x7FFF) if ip & 0x8000 else ip
            js, ks, ms = b[q + 15], b[q + 16], b[q + 17]
            from ._spectral import count
            ts = count(js, ks, ms)
            # the packed values follow the subset (GRIBEX's N counts from the message start)
            at = q + 18 + 4 * ts
            need = ((nval - ts) * nbits + 7) // 8
            if ts > nval or at + need > stop:
                raise Unread('its spectral data are cut short')
            tmpl5 = 51
            suffix = sm_bytes(ip * 1000, 4) + struct.pack('>HHHIB', js, ks, ms, ts, 1)
            data = ('spec', q + 18, ts, at, need, js, ks, ms, ip / 1000.0)
        else:
            need = ((nval - 1) * nbits + 7) // 8
            if q + 15 + need > stop:
                raise Unread('its spectral data are cut short')
            tmpl5 = 50
            suffix = struct.pack('>f', ibm(b, q + 11))     # (0, 0)'s real part
            data = ('copy', q + 15, need)
    elif second:
        if stop - q < 22 or b[q + 13] & 0x40:
            raise Unread('second-order packing %s' % ('cut short' if stop - q < 22 else
                                                        'of a matrix of values'))
        data = ('second', q, stop, b[q + 13], rows, bitmap is not None)
    elif bflags & 0x10 and b[q + 13] & 0x40:   # a matrix of values: its elements' fields
        from . import _grib1_matrix
        try:
            words = _grib1_matrix.words(b, q, stop)
        except (ValueError, IndexError) as err:
            raise Unread('its matrix of values: %s' % err)
        data = ('matrix', q, stop, b[q + 13], bitmap, npts)
    else:
        at = q + 11
        if bflags & 0x10:                       # extended flags, one value a point: past the
            from . import _grib1_matrix         # matrix description all the same (ecCodes)
            try:
                at = _grib1_matrix.header(b, q, stop)[6]
            except (ValueError, IndexError):
                at = q + 14
        need = (nval * nbits + 7) // 8          # GRIB1 pads to even; GRIB2's is exact
        if stop - at < need:
            raise Unread('its data are cut short (%d of %d bytes)' % (max(stop - at, 0), need))
        data = ('copy', at, need)
    dcn, own, implied = _param(table, par, centre)
    local = dcn is not None and (dcn[1] >= 192 or dcn[2] >= 192)
    disc, cat, num = dcn if dcn is not None else (255, 255, 255)
    s1 = struct.pack('>HHBBBHBBBBBBB', centre, subc, 2, 1 if local else 0, 1, year, mo, dy,
                     hh, mi, 0, 0, 1)
    t1, f1, t2, f2 = _level(ltype, o11, o12)
    if ltype == 1 and implied.get('typeOfFirstFixedSurface', 1) != 1:   # 'surface': the
        t1, f1, t2, f2 = _implied_level(implied)                       # parameter's level
    lev = _fixed(t1, f1) + _fixed(t2, f2)
    tunit = _UNIT.get(unit)
    if tunit is None:
        raise Unread('time unit %d' % unit)
    extra = ''
    ens = _ensemble(b, p, plen, centre)
    implied_stat = implied.get('typeOfStatisticalProcessing')
    if implied_stat == 1 and tri in (0, 1) and centre == ECMWF and p1 > 0:
        tri, p1, p2 = 4, 0, p1                  # ECMWF's accumulations: from the start
    if tri in (2, 3, 4, 5):                     # a statistic over P1..P2
        stat = {2: 255, 3: 0, 4: 1, 5: 4}[tri]
        if tri == 2 and implied_stat is not None:
            stat = implied_stat                 # a range: the parameter says of what
        t_end = _add(ref, unit, p2)
        tmpl = 11 if ens else 8
        st = struct.pack('>HBBBBB', t_end.year, t_end.month, t_end.day, t_end.hour,
                         t_end.minute, t_end.second)
        st += bytes([1]) + struct.pack('>I', nmiss) + bytes([stat, 2, tunit]) + \
            struct.pack('>I', max(0, p2 - p1)) + bytes([255]) + struct.pack('>I', 0)
        ft = p1
    else:
        tmpl = 1 if ens else 0
        ft = (p1 << 8 | p2) if tri == 10 else (0 if tri == 1 else p1)
        st = b''
        if tri not in (0, 1, 10):
            extra = 'TR=%d' % tri
    s4 = struct.pack('>HH', 0, tmpl) + bytes([cat, num, 2, 0, b[p + 5]]) + \
        struct.pack('>HB', 0, 0) + bytes([tunit]) + sm_bytes(ft, 4) + lev
    if ens:
        s4 += bytes(ens)
    s4 += st
    s5 = (struct.pack('>IH', nval, tmpl5) + struct.pack('>f', r) + e + dscale, nbits, suffix)
    head = _sec(1, s1) + _sec(3, g3) + _sec(4, s4)
    if data[0] == 'matrix':                     # each element's wording
        extra = [extra + ':' + w if extra else w for w in words]
    return (disc, head, s5, bm, data), end - pos, (own if dcn is None else None), extra


def _ensemble(b, p, plen, centre):
    """GRIB2's 4.1 octets 35-37 (kind, perturbation, forecasts) of a GRIB1
    ensemble member: NCEP's PDS extension, ECMWF's local definition 1."""
    if plen < 44:
        return None
    if centre == NCEP and b[p + 40] == 1:
        kind = {1: 0, 2: 2, 3: 3}.get(b[p + 41])
        if kind is not None:
            return (kind, b[p + 42], 255)
    if centre == ECMWF and b[p + 40] == 1 and plen >= 52:
        typ, number, total = b[p + 42], b[p + 49], b[p + 50]
        if typ in (10, 11):                     # cf, pf
            return (1 if typ == 10 else 3, number, total)
    return None


def records(b, places, recs, problems):
    """The Records of a file's GRIB1 messages at `places` (turned into
    GRIB2 sections, without their data, to scan) among those of its GRIB2
    messages `recs`, in file order, numbered as one file's; each with
    where its message ends -- and {place: its GRIB2's plan} for Local.
    GRIB1 messages not turned are described in problems."""
    from ._meta import scan
    parts, marks, at, plans = [], [], 0, {}
    mv = memoryview(b).cast('B')
    for pos in places:
        try:
            plan, n, own, extra = _layout(mv, pos)
            m = _assemble(mv, plan, data=False)     # (a matrix's: its fields' data made too)
        except (Unread, struct.error, IndexError, ValueError, OverflowError) as e:
            problems.append('message at byte %d: GRIB1, not read: %s' % (pos, e))
            continue
        plans[pos] = plan
        parts.append(m)
        marks.append((at, pos, own, extra, n))
        at += len(m)
    more = scan(np.frombuffer(b''.join(parts), np.uint8), problems) if parts else []
    patch(more, marks)
    for r in recs:                              # GRIB2's: its total length
        r.end = r.offset + struct.unpack('>Q', bytes(b[r.offset + 8:r.offset + 16]))[0] - 1
    out = sorted(recs + more, key=lambda r: (r.offset, r.sub))
    msg, last = 0, None
    for k, r in enumerate(out):
        if r.offset != last:
            msg, last = msg + 1, r.offset
        r.index, r.msg = k + 1, msg
    return out, plans


def patch(recs, marks):
    """Records scanned from records()' GRIB2 sections as their file's: offsets in
    the file (and where they end there), GRIB1's own names where GRIB2 has
    no number for the parameter, the extra wording."""
    import bisect
    starts = [m[0] for m in marks]
    for r in recs:
        k = bisect.bisect_right(starts, r.offset) - 1
        if k < 0 or starts[k] != r.offset:
            continue
        _, orig, own, extra, n = marks[k]
        r.offset, r.end = orig, orig + n - 1
        if own is not None:
            r.name, r.long_name, r.units = own
            r.ckeys = None                      # no concept: nothing to match
        if isinstance(extra, list):             # a matrix's: each element's own
            r.extra += extra[max(r.sub, 1) - 1] + ':'
        elif extra:
            r.extra += extra + ':'


class Local:
    """A local file holding GRIB1 messages as GribFile's fetcher: each
    message asked for turned into GRIB2 then -- the file's worth never
    held at once (GRIB2 messages in it given as they are); each GRIB1's
    sections as planned when the file was scanned."""
    remote = False
    batch = 16 << 20                            # fields(): bytes turned a batch (no latency to hide)

    def __init__(self, buf, plans):
        self.buf, self.plans = buf, plans
        self._mv = memoryview(buf).cast('B')

    def messages(self, spans):
        out, b = {}, self.buf
        for off, _ in spans:
            plan = self.plans.get(off)
            if plan is not None:
                out[off] = _assemble(self._mv, plan)
            else:
                n = struct.unpack('>Q', bytes(b[off + 8:off + 16]))[0]
                out[off] = bytes(b[off:off + n])
        return out

    def joined(self, spans, whole):
        """(one uint8 buffer of the messages at spans as GRIB2, {its place
        there: the message's in the file}, {the message's place in the
        file: (its start there, sections 5, 6 and 7, 7's end, the bitmap or
        -1)} -- known as made, so nothing need scan them -- or None when a
        GRIB2 message is among them): GRIB1's bitmaps and packed bits
        copied once, from the file; GRIB2 messages through whole(offset,
        bytes)."""
        chunks, starts, places, at = [], {}, {}, 0
        mixed = False
        for off, _ in spans:
            plan = self.plans.get(off)
            if plan is not None and plan[4][0] == 'matrix':     # (a message of several fields)
                c = (_assemble(self._mv, plan),)
                mixed = True
            elif plan is not None:
                c = _pieces(self._mv, plan)
                s5 = at + 16 + len(plan[1])
                s6 = s5 + 5 + len(plan[2][0]) + 1 + len(plan[2][2])
                s7 = at + len(c[0]) + len(c[1])
                places[off] = (at, s5, s6, s7, s7 + len(c[2]) + len(c[3]),
                               s6 + 6 if len(c[1]) else -1)
            else:
                c = (whole(off, self.messages([(off, None)])[off]),)
                mixed = True
            starts[at] = off
            at += sum(len(x) for x in c)
            chunks += c
        out = np.empty(at, np.uint8)
        k = 0
        for x in chunks:
            out[k:k + len(x)] = np.frombuffer(x, np.uint8)
            k += len(x)
        return out, starts, None if mixed else places
