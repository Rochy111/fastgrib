"""Scanning a GRIB2 file: each field's metadata as a Record, worded as
wgrib2 -s words it ("12:0:d=2026012000:TMP:2 m above ground:6 hour fcst:"),
so the .idx files NOAA publishes beside each GRIB select the same fields.
"""
import math
import struct
from datetime import datetime, timedelta

import numpy as np

from . import _grids, _winds
from .tables import levels as _levels
from .tables.params import PARAMS

NCEP = 7


class Record:
    """One field of a GRIB2 file.

    index                    its place among the file's fields, from 1
    msg, sub                 its message's number, and its place in the
                             message when that holds several (else 0) --
                             wgrib2's 5 or 5.2
    name, long_name, units   the parameter (NCEP's short names, as wgrib2)
    level                    wgrib2's level wording ("500 mb")
    level_type, level_value  code table 4.5 and the value in level's units
                             (level_type2, level_value2 for a layer's other side)
    level_kind               the level type's name, as ecCodes / pygrib say
    ref_time, valid_time     datetimes; fcst is wgrib2's wording
    member                   an ensemble member's number (0 a control, -n a
                             negative perturbation), else None
    ensemble                 what an ensemble field is: 'control', 'perturbed',
                             'ens mean', 'ens std dev', ... (else None)
    grid                     its Grid (shape, kind, latlon())
    uv_relative              whether a vector component lies along the grid's
                             axes (Lambert, polar winds), not east and north
    """
    __slots__ = ('index', 'msg', 'sub', 'offset', 'disc', 'center', 'date', 'cat', 'num',
                 'tables', 'ckeys', 'name', 'long_name', 'units', 'level', 'level_type', 'level_value',
                 'level_type2', 'level_value2', 'fcst', 'extra', 'ref_time', 'valid_time',
                 'member', 'ensemble', 'product', 'packing', 'grid', 'uv_relative',
                 's5', 's6', 's7', 's7end', 'bitmap', 'end', 'at')

    def __repr__(self):
        num = '%d.%d' % (self.msg, self.sub) if self.sub else '%d' % self.msg
        return '%s:%d:d=%s:%s:%s:%s:%s' % (num, self.offset, self.date, self.inv_name,
                                           self.level, self.fcst, self.extra)

    @property
    def inv_name(self):
        """The parameter as wgrib2's inventory names it: its name, or for
        one no table knows, its numbers spelt out."""
        if self.long_name != 'unknown':
            return self.name
        return unknown_name(self.disc, self.cat, self.num, self.center, self.tables)

    @property
    def level_kind(self):
        """The level type's name as ecCodes (pygrib, cfgrib) gives it."""
        return level_kind(self.level_type, self.level_type2, self.center, self.level_value)

    @property
    def short_name(self):
        """The parameter's ecCodes shortName ('2t', 'tp', 'mx2t3', ...), as
        pygrib and cfgrib name it; None where ecCodes has none."""
        hit = concept(self.ckeys, self.center, self.tables)
        return hit[0] if hit else None

    @property
    def fcst_hours(self):
        """Hours from the reference time to the valid time."""
        if self.valid_time is None or self.ref_time is None:
            return None
        return (self.valid_time - self.ref_time).total_seconds() / 3600.0

    def as_dict(self):
        """The metadata as a dict (for attrs, JSON, pandas)."""
        keys = ('index', 'name', 'short_name', 'long_name', 'units', 'level', 'level_kind',
                'level_value',
                'fcst', 'ref_time', 'valid_time', 'member', 'ensemble', 'extra')
        return {k: getattr(self, k) for k in keys}


# layers of one level type as wgrib2 words them (its level2), and what its
# values are divided by (in float) to be shown
_LAYER = {100: ('%g-%g mb', 100), 102: ('%g-%g m above mean sea level', None),
          103: ('%g-%g m above ground', None), 104: ('%g-%g sigma layer', None),
          105: ('%g-%g hybrid layer', None), 106: ('%g-%g m below ground', None),
          107: ('%g-%g K isentropic layer', None), 108: ('%g-%g mb above ground', 100),
          111: ('%g-%g Eta layer', None), 115: ('%g-%g sigma height layer', None),
          118: ('%g-%g hybrid height layer', None), 119: ('%g-%g hybrid pressure layer', None),
          150: ('%g-%g generalized vertical height coordinate', None),
          160: ('%g-%g m below sea level', None), 161: ('%g-%g m ocean layer', None)}
_LAYER_NCEP = {235: ('%g-%gC ocean isotherm layer', 10), 236: ('%g-%g m ocean layer', 0.1)}
KMA = 40
UNDEF_LEVEL = float(np.float32(9.999e20))      # wgrib2's value of a missing surface
# code table 4.10 (statistics) and 4.4 (time units) as wgrib2 names them
_STAT = {0: 'ave', 1: 'acc', 2: 'max', 3: 'min', 4: 'last-first', 5: 'RMS', 6: 'StdDev',
         7: 'covar', 8: 'first-last', 9: 'ratio', 10: 'standardized anomaly',
         11: 'summation', 12: 'return period', 100: 'severity', 101: 'mode', 255: 'missing'}
_UNIT = {0: 'min', 1: 'hour', 2: 'day', 3: 'month', 4: 'year', 5: 'decade', 6: 'normal',
         7: 'century', 10: '3-hours', 11: '6-hours', 12: '12-hours', 13: 'sec', 255: 'missing'}
# time-range units as timedeltas (code table 4.4); months and years have none
_STEP = {0: timedelta(minutes=1), 1: timedelta(hours=1), 2: timedelta(days=1),
         10: timedelta(hours=3), 11: timedelta(hours=6), 12: timedelta(hours=12),
         13: timedelta(seconds=1)}
# where a product template's statistics (4.8's octet 35 on) begin, past what it
# adds to 4.0: 4.9 a probability (13 octets), 4.10 a percentile (1), 4.11 an
# ensemble member (3), 4.12 an ensemble statistic (2), 4.13 and 4.14 clusters
_STAT_AT = {8: 0, 9: 13, 10: 1, 11: 3, 12: 2, 13: 34, 14: 30}
# code table 4.7 (derived forecasts, 4.2 and 4.12) in wgrib2's words; NCEP's
# local entries only for NCEP's files
_DERIVED = {0: 'ens mean', 1: 'wt ens mean', 2: 'ens std dev', 3: 'normalized ens std dev',
            4: 'ens spread', 5: 'ens large anom index', 6: 'unwt ens mean',
            7: '25%-75% range', 8: 'min all members', 9: 'max all members'}
_DERIVED_NCEP = {192: 'unwt mode all members', 193: '10% all members', 194: '50% all members',
                 195: '90% all members', 196: 'stat. weight for each members',
                 197: 'percentile from climate distribution',
                 198: 'deviation of ens mean from daily climo', 199: 'extreme forecast index',
                 200: 'equally weighted mean', 201: '5% all members', 202: '25% all members',
                 203: '75% all members', 204: '95% all members'}
# code table 4.6 (an ensemble member's kind, 4.1 and 4.11)
_MEMBER = {0: 'control', 1: 'control', 2: 'perturbed', 3: 'perturbed', 4: 'multi-model'}
_WITH_LEVEL = tuple(range(16))                  # 4.0 to 4.15: 4.0's level and times
ECMWF = 98


def _f32(x):
    """x as a C float: rounded to float32, inf past its range (silently)."""
    x = float(x)        # (a numpy float32's comparison would cast the bound to float32: inf)
    if abs(x) >= 3.4028235677973366e38:          # FLT_MAX and half an ulp: rounds to inf
        return math.copysign(math.inf, x)
    return float(np.float32(x))


def _scaled(sf, sv):
    """wgrib2's scaled2flt: a fixed surface's value, as float32."""
    if sf == 0:
        return _f32(sv)
    return _f32(sv * _grids._int_power(10.0, -sf))


_sm = _grids._sm


def param(disc, cat, num, center):
    """(name, description, units) of a parameter; local numbers (192 and
    up) by NCEP's table only for NCEP's files."""
    local = cat >= 192 or num >= 192
    hit = PARAMS.get((disc, cat, num)) if not local or center == NCEP else None
    return hit or ('var%d_%d_%d' % (disc, cat, num), 'unknown', 'unknown')


def unknown_name(disc, cat, num, center, tables):
    """wgrib2's inventory wording of a parameter no table knows."""
    if 192 <= num <= 254 or 192 <= cat <= 254 or 192 <= disc <= 254:
        return ('var discipline=%d center=%d local_table=%d parmcat=%d parm=%d'
                % (disc, center, tables[1], cat, num))
    return 'var discipline=%d master_table=%d parmcat=%d parm=%d' % (disc, tables[0], cat, num)


def _surface(sf, sv, t):
    """A fixed surface's value in the units wgrib2 shows it (mb for Pa)."""
    if sf == 255 or sv == 0xFFFFFFFF or t == 255:
        return None
    v = _scaled(_sm(sf, 1), _sm(sv, 4))
    return _f32(v * 0.01) if t in (100, 108) else v


def _value(sf, sv):
    """wgrib2's fixed_surfaces: a surface's value, UNDEF_LEVEL when missing."""
    if sf == 255 or sv == 0xFFFFFFFF:
        return UNDEF_LEVEL, True
    return _scaled(_sm(sf, 1), _sm(sv, 4)), False


def _level(center, t1, sf1, sv1, t2, sf2, sv2):
    """wgrib2's wording of a field's level or layer (its level2)."""
    v1, miss1 = _value(sf1, sv1)
    v2, miss2 = _value(sf2, sv2)
    if t1 == t2:
        hit = _LAYER.get(t1) or (_LAYER_NCEP.get(t1) if center == NCEP else None)
        if hit:
            fmt, div = hit
            if div == 0.1:                      # NCEP's obsolete 236: value*10, in float
                v1, v2 = _f32(np.float32(v1) * np.float32(10)), _f32(np.float32(v2) * np.float32(10))
            elif div is not None:               # value/100 (Pa to mb), /10: in float
                v1, v2 = _f32(np.float32(v1) / np.float32(div)), _f32(np.float32(v2) / np.float32(div))
            return fmt % (v1, v2)
    if t1 == 1 and t2 == 8:
        return 'atmos col'
    if t1 == 9 and t2 == 1:
        return 'ocean column'
    if t1 == 255 and t2 == 255:
        return 'no_level'
    s = _level1(center, t1, v1, miss1)
    if t2 != 255:
        s += ' - ' + _level1(center, t2, v2, miss2)
    return s


def _level1(center, t, v, miss):
    """wgrib2's wording of a single level (its level1)."""
    if t < 192:
        if t in (100, 108):
            v = _f32(v * 0.01)                  # Pa -> mb
        return _fmt(_levels.WMO[t], v)
    if t == 255:
        return ''
    if center in (NCEP, KMA):
        if t == 235:
            v = _f32(v * 0.1)                   # 0.1 C -> C
        return _fmt((_levels.NCEP if center == NCEP else _levels.KMA)[t - 192], v)
    return ('local level type %d' % t if miss
            else 'local level type %d %g' % (t, v))


def _fmt(fmt, v):
    """C's sprintf(fmt, v) for wgrib2's level formats."""
    return fmt % v if '%g' in fmt else fmt.replace('%%', '%')


def level_kind(t1, t2, center, value=None):
    """A level type's name as ecCodes gives it ('isobaricInhPa', 'soilLayer',
    'heightAboveGroundLayer', ...); 'level<type>' where it has none. A
    pressure level that is no whole number of hPa (value, in mb) is
    'isobaricInPa', as ecCodes has it."""
    if t1 == 100 and t2 == 255 and value is not None and not float(value).is_integer():
        return 'isobaricInPa'
    k = _levels.KINDS
    name = k.get((t1, t2)) or k.get((t1, None))
    if name is None and center == NCEP:
        name = _levels.KINDS_NCEP.get((t1, t2)) or _levels.KINDS_NCEP.get((t1, None))
    if name is None and t2 not in (255, None, t1):
        name = k.get((t1, 255))                 # a layer between kinds: its top's
    return name or 'level%d' % t1


_CONCEPT = {}


def concept(keys, center, tables):
    """(shortName, name, units) of a field by ecCodes' concepts: those of
    the WMO tables and of the centre's own (ECMWF's or NCEP's), then
    ECMWF's for any centre, then the legacy ones (retired numbers), as
    ecCodes falls back to them; ECMWF's fields of tables version 33 or
    before by the concepts of that time. The match: every condition true,
    the most conditions, ties as ecCodes breaks them -- the same as
    ecCodes' shortName on every field tried. None if no concept matches."""
    key = (keys, center, tables[0])
    if key in _CONCEPT:
        return _CONCEPT[key]
    from .tables import eccodes as _eccodes
    hit = None
    cands = _eccodes.CONCEPTS.get(keys[:3]) if keys else None
    if cands:
        old = center == ECMWF and tables[0] <= 33           # ecCodes' MTG2 switch
        wmo, ecmf = (2, 8) if old else (1, 4)
        own = ecmf if center == ECMWF else 16 if center == NCEP else 0
        for sets in (wmo | own, wmo | ecmf, 32 | 64, 128 | 256):   # (then the chemical split)
            best = None
            for e in cands:                     # most conditions first, in file order
                if not e[3] & sets:
                    continue
                if best is not None and len(e[4]) < len(best[4]):
                    break
                if all(keys[k] == v for k, v in e[4]):
                    # a tie as ecCodes breaks it (its list holds each file last
                    # entry first, the WMO file ahead): the WMO concept, else
                    # the later one
                    if best is None or e[3] & 163 or not best[3] & 163:   # (1|2|32|128: WMO)
                        best = e
            if best is not None:
                hit = best[:3]
                break
    if len(_CONCEPT) < 65536:
        _CONCEPT[key] = hit
    return hit


NKEYS = 36                                      # len(_eccodes.KEYS)


def _chem_keys(b, s4, tmpl):
    """_eccodes.KEYS 18 to 34 of a product (its own template, not the one
    it is read as): a chemical's constituent, an aerosol's type, size and
    wavelength, and ecCodes' is_chemical ... is_aerosol_optical."""
    out = [None] * 12 + [int(40 <= tmpl <= 43), int(tmpl in (57, 58, 67, 68)),
                         int(76 <= tmpl <= 79), int(44 <= tmpl <= 49 or tmpl == 85),
                         int(tmpl in (48, 49))]
    if not 40 <= tmpl <= 49:
        return out

    def sf(q):
        return -1 if b[s4 + q] == 255 else _sm(b[s4 + q], 1)

    def sv(q):
        v = struct.unpack_from('>I', b, s4 + q)[0]
        return -1 if v == 0xFFFFFFFF else _sm(v, 4)
    if tmpl <= 43:
        out[0] = struct.unpack_from('>H', b, s4 + 11)[0]
        return out
    out[1] = struct.unpack_from('>H', b, s4 + (12 if tmpl == 47 else 11))[0]
    out[2] = b[s4 + (14 if tmpl == 47 else 13)]     # (as _chem_words reads 4.47)
    out[3:7] = sf(14), sv(15), sf(19), sv(20)
    if tmpl in (48, 49):
        out[7:12] = b[s4 + 24], sf(25), sv(26), sf(30), sv(31)
    return out


def _keys(b, s4, tmpl, disc, cat, num, tables, chem=None, own=None, gdt=None):
    """What ecCodes' concepts look at, in _eccodes.KEYS' order (None where
    the product template has no such key; MISSING for all ones). chem:
    _chem_keys() of the product, own: its template number when read as
    another, gdt: the grid's template number."""
    out = [disc, cat, num] + [None] * 15 + (chem or [None] * 12 + [0] * 5) + [gdt]
    if tmpl in _WITH_LEVEL:
        t1, sf1, sv1, t2, sf2, sv2 = struct.unpack_from('>BBIBBI', b, s4 + 22)
        out[3:9] = (t1, -1 if sf1 == 255 else _sm(sf1, 1), -1 if sv1 == 0xFFFFFFFF else _sm(sv1, 4),
                    t2, -1 if sf2 == 255 else _sm(sf2, 1), -1 if sv2 == 0xFFFFFFFF else _sm(sv2, 4))
        out[13] = b[s4 + 11]
    if tmpl in _STAT_AT:
        q = s4 + _STAT_AT[tmpl]
        out[9], out[11] = b[q + 46], b[q + 48]
        out[10] = struct.unpack_from('>I', b, q + 49)[0]
    if tmpl in (5, 9):
        sf, sv = b[s4 + 37], struct.unpack_from('>I', b, s4 + 38)[0]
        out[14] = b[s4 + 36]
        out[15], out[16] = -1 if sf == 255 else _sm(sf, 1), -1 if sv == 0xFFFFFFFF else _sm(sv, 4)
    out[12] = tables[1]
    out[17] = tmpl if own is None else own
    return tuple(out)


def _fcst(tmpl, b, p4, center, disc, cat, num, tables):
    """wgrib2's forecast wording (its ftime2) for product templates 4.0 to
    4.15: a time ('6 hour fcst', 'anl') or a statistic's time ranges
    ('0-1 day acc fcst', '0-6 hour acc@(fcst,dt=450 sec),missing=0')."""
    if tmpl not in _WITH_LEVEL:
        return 'anl'
    unit = b[p4 + 17]
    ft = _sm(struct.unpack_from('>I', b, p4 + 18)[0], 4)
    if tmpl not in _STAT_AT:                    # a point in time
        return 'anl' if ft == 0 else _ptime(unit, ft, 0, 0, 1) + ' fcst'
    verf = p4 + _STAT_AT[tmpl] + 34
    end = p4 + struct.unpack_from('>I', b, p4)[0]
    out, missing = _ranges(b, verf, end, ft, unit, 0, (center, disc, cat, num, tables))
    if missing:
        m = struct.unpack_from('>I', b, verf + 8)[0]
        out += ',missing=-1' if m == 0xFFFFFFFF else ',missing=%d' % m
    return out


def _norm(unit, v):
    """wgrib2's normalize_time_range: 3, 6, 12-hour units in hours."""
    k = {10: 3, 11: 6, 12: 12}.get(unit)
    return (1, v * k) if k else (unit, v)


def _whole(unit, v):
    """wgrib2's simple_time_range: in the largest unit it divides."""
    unit, v = _norm(unit, v)
    for u, k, nxt in ((13, 60, 0), (0, 60, 1), (1, 24, 2)):
        if unit == u and v % k == 0 and v != 0:
            v, unit = _cdiv(v, k), nxt
    return unit, v


def _cdiv(a, b):
    return -(-a // b) if (a < 0) != (b < 0) else a // b     # C's truncating division


def _tr(unit):
    return _UNIT.get(unit, '?')


def _ptime(u1, v1, u2, v2, fmt):
    """wgrib2's print_ftime2: 'N unit' (1), 'N-M unit' (2), 'N:M unit' (3)."""
    u1, v1 = _norm(u1, v1)
    if fmt == 1:
        return '%d %s' % (v1, _tr(u1))
    dash = '-' if fmt == 2 else ':'
    u2, v2 = _norm(u2, v2)
    if v1 == 0:
        u1 = u2
    elif v2 == 0:
        u2 = u1
    if u1 != u2:
        return '%d %s%s(%d %s+%d %s)' % (v1, _tr(u1), dash, v1, _tr(u1), v2, _tr(u2))
    for u, k, nxt in ((13, 60, 0), (0, 60, 1), (1, 24, 2), (3, 12, 4)):
        if u1 == u and v1 % k == 0 and v2 % k == 0:
            v1, v2, u1 = _cdiv(v1, k), _cdiv(v2, k), nxt
    return '%d%s%d %s' % (v1, dash, v1 + v2, _tr(u1))


def _ranges(b, verf, end, ft, fu, n, who):
    """wgrib2's ftime2_tr: (wording of the time ranges from the n-th,
    whether the missing count follows)."""
    nmax = b[verf + 7]
    at = verf + 12 + n * 12
    if nmax in (0, 255) or at + 12 > end:
        return '', False
    stat, inc, ua = b[at], b[at + 1], b[at + 2]
    ta = _sm(struct.unpack_from('>I', b, at + 3)[0], 4)
    ub = b[at + 7]
    tb = _sm(struct.unpack_from('>I', b, at + 8)[0], 4)
    name = _STAT.get(stat, '???')
    if n == nmax - 1:                           # the innermost (or only) range
        if inc == 0:
            return 'Code Table 4.11=reserved', False
        if inc == 1 and tb == 0:
            s = _ptime(ua, 0, ua, ta, 2)
            return (s + (' %s anl' % name if ft == 0 else
                         ' %s(%d %s fcst)' % (name, ft, _tr(fu)))), False
        if inc == 1:
            s = ('%d@%d %s %s' % (_cdiv(ta, tb) + 1, tb, _tr(ub), name) if ua == ub else
                 '%d %ss@%d %s %s' % (ta, _tr(ua), tb, _tr(ub), name))
            s += '(anl)' if ft == 0 else '(%s fcst)' % _ptime(fu, ft, ua, ta, 1)
            return s, True
        if inc == 2 and tb == 0:
            center, disc, cat, num, tables = who
            s = _ptime(fu, ft, ua, ta, 2)
            if disc == 0 and center == NCEP and cat == 0 and tables[0] <= 5 \
                    and num in (4, 5) and stat == 255:      # NCEP's old TMAX, TMIN
                return s + ' fcst', False
            return s + ' %s fcst' % name, False
        if inc == 2:
            u, v = _norm(ub, tb)
            return _ptime(fu, ft, ua, ta, 2) + ' %s@(fcst,dt=%d %s)' % (name, v, _tr(u)), True
        if inc == 3 and tb == 0:
            return ('0-%d %s %s(' % (ta, _tr(ua), name) + _ptime(fu, ft, ua, -ta, 3)
                    + ' fcst)'), False
        if inc in (3, 4):
            return 'ensemble %s-%d valid %d %s' % (name, inc, ft, _tr(fu)), False
        if inc == 5:
            return _ptime(fu, ft, ua, ta, 2) + ' (subinterval) %s' % name, False
        return 'Code Table 4.11=%d' % inc, False
    if tb == 0 or ua == 255 or ub == 255:       # wgrib2: a fatal error
        return '', False
    if 1 <= inc <= 5:
        left, right = {1: ('(', ')'), 2: ('(', ')++'), 3: (' (', ')--'), 4: ('(', ')++'),
                       5: (' ?[', ']')}[inc]
        if ua == ub:
            u, v = _whole(ub, tb)
            s = '%d@%s%d %s %s%s' % (_cdiv(ta, tb) + 1, '-' if inc == 4 else '', v, _tr(u),
                                    name, left)
        else:
            s = '%d %ss@%d %s %s%s' % (ta, _tr(ua), tb, _tr(ub), name, left)
        inner, _ = _ranges(b, verf, end, ft, fu, n + 1, who)
        return s + inner + right, True
    inner, _ = _ranges(b, verf, end, ft, fu, n + 1, who)
    head = 'Code Table 4.11=reserved ?[' if inc == 0 else 'CodeTable 4.11=%d ?[' % inc
    return head + inner + ']', True


def _extra(tmpl, b, p4, center, reftype, more=()):
    """What wgrib2 adds after the forecast time (its f_misc), each item
    ':'-ended: an ensemble member or statistic, a cluster, a probability's
    limits and number, a spatial statistic, what kind of product it is (code
    table 4.3), a percentile, the reference time's kind when not the usual."""
    items = []
    if tmpl in (1, 11):
        kind, pert = _member(b, p4, center)
        word = {0: 'ENS=hi-res ctl', 1: 'ENS=low-res ctl', 2: 'ENS=-%d' % pert,
                3: 'ENS=+%d' % pert, 4: 'MM-ENS=%d' % pert}.get(kind)
        if word is None:
            word = ('ENS=%d' % pert if kind == 192 and center == NCEP
                    else 'ENS=? table4.6=%d pert=%d' % (kind, pert))
        items.append(word)
    if tmpl in _DERIVED_AT:
        items.append(_derived(b[p4 + 34], center, tmpl))
    if tmpl in (3, 4, 13, 14):
        items.append('cluster=%d' % b[p4 + 36])
    if tmpl in (5, 9):
        num, total, kind = b[p4 + 34], b[p4 + 35], b[p4 + 36]

        def limit(q):                           # wgrib2's scaled2flt, all ones too
            return _scaled(_sm(b[q], 1), _sm(struct.unpack_from('>I', b, q + 1)[0], 4))
        lo, hi = limit(p4 + 37), limit(p4 + 42)
        lim = {0: 'prob <%g' % lo, 1: 'prob >%g' % hi,
               2: 'prob =%g' % lo if lo == hi else 'prob >=%g <%g' % (lo, hi),
               3: 'prob >%g' % lo, 4: 'prob <%g' % hi, 5: 'prob =%g' % lo,
               6: 'prob above normal', 7: 'prob near normal', 8: 'prob below normal'}.get(
                   kind, 'prob ?? (code table 4.9=%d)' % kind)
        items.append('%s:prob fcst %d/%d' % (lim, num, total))
    if tmpl == 15:
        stat = b[p4 + 34]
        word = _STAT.get(stat)
        items.append(('spatial %s:' % word if stat != 255 and word else 'spatial none:')
                     + _INTERP.get(b[p4 + 35], ''))
    gen = b[p4 + 11]
    word = {5: 'probability forecast', 6: 'analysis/forecast error',
            7: 'analysis/forecast error', 9: 'climatological'}.get(gen)
    if word is None and center == NCEP:
        word = {192: 'Confidence Indicator', 194: 'Neighborhood Probability'}.get(gen)
    if word is None and 192 <= gen < 255:
        word = 'process=%d' % gen
    if word:
        items.append(word)
    if tmpl == 7:
        items.append('analysis/forecast error')
    elif tmpl in (6, 10):
        items.append('%d%% level' % b[p4 + 34])
    items += more                               # chemistry, aerosols (_chem_words)
    if reftype > 1:
        items.append('type_reftime=' + {2: 'verifying time of forecast', 3: 'observation time',
                                        4: 'local time'}.get(reftype, 'unknown'))
    return ''.join(x + ':' for x in items)


# where code table 4.7 (a derived forecast) is: 4.2-4.4 and 4.12-4.14's octet 35
_DERIVED_AT = (2, 3, 4, 12, 13, 14)
# code table 4.15 (spatial interpolation) as wgrib2 shortens it
_INTERP = {0: 'no interpolation', 1: 'bilinear interpolation', 2: 'bicubic interpolation',
           3: 'nearest neighbor interpolation', 4: 'budget interpolation',
           5: 'spectral interpolation', 6: 'neighbor-budget interpolation',
           255: 'missing interpolation'}


# templates that are another with octets inserted: (the one they extend,
# where the inserted octets start, how many) -- 4.40-4.43 a constituent
# type (2), 4.44-4.47 an aerosol type and size interval (13), 4.48/4.49
# those and a wavelength interval (24), 4.60/4.61 a model version date (7)
_EXTENDS = {40: (0, 11, 2), 41: (1, 11, 2), 42: (8, 11, 2), 43: (11, 11, 2),
            44: (0, 11, 13), 45: (1, 11, 13), 46: (8, 11, 13), 47: (11, 11, 13),
            48: (0, 11, 24), 49: (1, 11, 24), 60: (1, 37, 7), 61: (11, 37, 7)}


def _normal(b, s4, tmpl):
    """(bytes, 0, template) of section 4 read as the template it extends:
    its inserted octets cut out, its length and number set to match."""
    base, at, n = _EXTENDS[tmpl]
    ln = struct.unpack_from('>I', b, s4)[0]
    sec = bytearray(b[s4:s4 + ln])
    nv = struct.unpack_from('>H', b, s4 + 5)[0]             # vertical coordinates after
    short = tmpl == 44 and ln - 4 * nv == 45
    if len(sec) < max(34, at) + n - 2 * short:
        raise IndexError('section 4 too short for template 4.%d' % tmpl)
    del sec[at:at + n]
    if short:
        # WMO's 4.44 has a 2-octet forecast time (an error of WMO's that
        # ecCodes reads too; wgrib2 reads 4 octets there): widened to 4
        sec[18:18] = b'\0\0'
    sec[0:4] = struct.pack('>I', len(sec))
    sec[7:9] = struct.pack('>H', base)
    return bytes(sec), 0, base


def _chem_words(b, s4, tmpl, center, tables):
    """What wgrib2 says of a constituent or aerosol (its f_misc): chemical=,
    aerosol=, aerosol_size, aerosol_wavelength."""
    if not 40 <= tmpl <= 49:
        return []
    from .tables.chem import CONSTITUENTS
    out = []

    def name(at):
        v = struct.unpack_from('>H', b, s4 + at)[0]
        word = CONSTITUENTS.get(v)
        if word is None or (tables[0] <= 4 and center == ECMWF):
            word = '%d' % v
        return word
    if tmpl <= 43:
        out.append('chemical=' + name(11))
        return out
    out.append('aerosol=' + name(12 if tmpl == 47 else 11))

    def interval(kind, q1, q2):
        val = lambda q: _sm(struct.unpack_from('>I', b, s4 + q + 1)[0], 4) * \
            _grids._int_power(10.0, -_sm(b[s4 + q], 1))           # noqa: E731
        return _INTERVAL.get(kind, '??? code_table_4.91=%d' % kind).format(val(q1), val(q2))
    # an item each, empty when its interval is missing (wgrib2 still ends it)
    if tmpl <= 48:
        kind = b[s4 + (14 if tmpl == 47 else 13)]
        out.append('aerosol_size ' + interval(kind, 14, 19) if kind != 255 else '')
    if tmpl == 48:
        kind = b[s4 + 24]
        out.append('aerosol_wavelength ' + interval(kind, 25, 30) if kind != 255 else '')
    return out


# code table 4.91 (type of interval) as wgrib2 prints it, {0} and {1} the limits
_INTERVAL = {0: '<{0:g}', 1: '>{1:g}', 2: '>={0:g},<{1:g}', 3: '>{0:g}', 4: '<{1:g}',
             5: '<={0:g}', 6: '>={1:g}', 7: '>={0:g},<={1:g}', 8: '>={0:g}', 9: '<={1:g}',
             10: '>{0:g},<={1:g}', 11: '={0:g}'}


def _member(b, p4, center):
    """(code table 4.6, perturbation number) of a 4.1/4.11 member; ECMWF's
    members of an undefined kind (255) read as wgrib2 reads them."""
    kind, pert, n = b[p4 + 34], b[p4 + 35], b[p4 + 36]
    if center == ECMWF and kind == 255 and n > 0:
        kind = 1 if pert == 0 else 3
    return kind, pert


def _derived(kind, center, tmpl=2):
    """Code table 4.7 in wgrib2's words (for clusters, 4.3/4.4/4.13/4.14,
    some say 'cluster' for 'ens')."""
    word = _DERIVED.get(kind)
    if word is None and center == NCEP:
        word = _DERIVED_NCEP.get(kind)
    if word and tmpl not in (2, 12) and kind in (2, 3, 6):
        word = {2: 'cluster std dev', 3: 'normalized cluster std dev',
                6: 'unwt cluster mean'}[kind]
    return word or 'unknown derived fcst'


def _ensemble(tmpl, b, p4, center):
    """(member, ensemble) of a Record."""
    if tmpl in (1, 11):
        kind, pert = _member(b, p4, center)
        member = (0 if kind in (0, 1) else -pert if kind == 2 else
                  None if pert == 255 else pert)
        return member, _MEMBER.get(kind, 'member')
    if tmpl in _DERIVED_AT:
        return None, _derived(b[p4 + 34], center, tmpl)
    return None, None


def _times(tmpl, b, p4, ref):
    """The valid time: a statistic's end, else the reference time plus the
    forecast time; None when it cannot be told."""
    if ref is None:
        return None
    if tmpl in _STAT_AT:
        q = p4 + _STAT_AT[tmpl] + 34
        y, mo, d, h, mi, s = struct.unpack_from('>HBBBBB', b, q)
        try:
            return datetime(y, mo, d, h, mi, s)
        except ValueError:
            return None
    step = _STEP.get(b[p4 + 17])
    if step is None:
        return None
    try:
        return ref + step * _sm(struct.unpack_from('>I', b, p4 + 18)[0], 4)
    except OverflowError:                       # a forecast time past year 9999
        return None


def scan(buf, problems=None, alone=False, lacking=None, grib1=None):
    """Every field in a GRIB2 buffer (bytes, mmap or uint8 array), as
    Records in file order, numbered from 1 like wgrib2. A message holding
    several fields yields each of them. A field whose bitmap section says
    254 ('the bitmap given before') takes the last bitmap defined before
    it -- in its own message as the standard has it, or else in an earlier
    one (unless `alone`: messages not each other's neighbours, as fetched).
    Such a message with none before it is damaged; its place in buf goes
    in `lacking` too when that is a list. GRIB1 messages are skipped, their
    places in `grib1` when that is a list (else described as unread).

    A damaged message -- cut short, its length or sections inconsistent,
    no end marker -- yields nothing, and scanning goes on at the next
    'GRIB' after it, so one bad message (an interrupted download, bytes
    lost in a copy) costs only its own fields. Each is described in
    `problems` (a list) when one is given, else in one warning; GRIB1
    messages likewise."""
    b = memoryview(buf).cast('B') if not isinstance(buf, (bytes, bytearray)) else buf
    out, pos, size = [], 0, len(b)
    bitmap = None
    found = [] if problems is None else problems
    nmsg = 0
    while pos + 16 <= size:
        if b[pos:pos + 4] != b'GRIB':
            nxt = _find(b, pos, size)
            if nxt < 0:
                break
            pos = nxt
            continue
        if grib1 is not None and b[pos + 7] == 1:          # GRIB1: noted, for _grib1
            n = int.from_bytes(bytes(b[pos + 4:pos + 7]), 'big')
            n = (n & 0x7FFFFF) * 120 if n & 0x800000 else n
            if n >= 40:
                grib1.append(pos)
                pos += n if pos + n <= size else 4
                continue
        try:
            recs, bitmap, end = _message(b, pos, size, len(out), None if alone else bitmap,
                                         nmsg + 1)
        except _Damaged as e:
            found.append('message at byte %d: %s' % (pos, e))
            if lacking is not None and e.args[0] is _NO_BITMAP:
                lacking.append(pos)
            pos += 4                                # resynchronise on the next 'GRIB'
            continue
        out += recs
        nmsg += 1
        pos = end
    if found and problems is None:
        import warnings
        warnings.warn('%d damaged or unread GRIB messages skipped; the first: %s'
                      % (len(found), found[0]), stacklevel=2)
    return out


class _Damaged(Exception):
    pass


_NO_BITMAP = 'it reuses a bitmap from an earlier message, which is not here'


def _find(b, pos, size):
    """The next 'GRIB' at or after pos, -1 if none."""
    step = 1 << 20
    while pos < size:
        hit = bytes(b[pos:min(size, pos + step + 3)]).find(b'GRIB')
        if hit >= 0:
            return pos + hit
        pos += step
    return -1


def _message(b, pos, size, nrec, bitmap, msg):
    """The fields of the message at pos, checked whole before any is
    taken: (records, the last bitmap defined by its end, its end). bitmap:
    the last defined before it (None: none)."""
    if b[pos + 7] != 2:
        if b[pos + 7] == 1:
            n = int.from_bytes(bytes(b[pos + 4:pos + 7]), 'big')
            raise _Damaged('a GRIB1 message (%d bytes), not read' % n)
        raise _Damaged('edition %d, not GRIB2' % b[pos + 7])
    total = struct.unpack_from('>Q', b, pos + 8)[0]
    end = pos + total
    if total < 16 + 4 or end > size:
        raise _Damaged('cut short: it says %d bytes, %d are there' % (total, size - pos))
    if b[end - 4:end] != b'7777':
        raise _Damaged('no end marker where its length says it ends')
    disc = b[pos + 6]
    p = pos + 16
    center, date, ref, grid, s4 = 0, '', None, None, None
    uvrel = False                               # vectors along the grid's axes
    tables, reftype = (2, 0), 1
    s5 = s6 = -1
    last = 0
    npts = None
    recs = []
    cur = -1                                    # the bitmap of the field being read
    while p < end - 4:
        if p + 5 > end - 4:
            raise _Damaged('a section runs past the end marker')
        ln, num = struct.unpack_from('>IB', b, p)
        if ln < 5 or p + ln > end - 4:
            raise _Damaged('section %d at byte %d has a bad length (%d)' % (num, p, ln))
        if not 1 <= num <= 7 or (num < last and num not in (2, 3, 4)):
            raise _Damaged('section %d out of order at byte %d' % (num, p))
        last = num
        try:
            if num == 1:
                center = struct.unpack_from('>H', b, p + 5)[0]
                tables = (b[p + 9], b[p + 10])
                reftype = b[p + 11]
                y, mo, d, h, mi, s = struct.unpack_from('>HBBBBB', b, p + 12)
                date = '%04d%02d%02d%02d' % (y, mo, d, h)
                try:
                    ref = datetime(y, mo, d, h, mi, s)
                except ValueError:
                    ref = None
            elif num == 3:
                grid, uvrel = _grid_of(b, p, ln, center)
                npts = struct.unpack_from('>I', b, p + 6)[0]
                if grid is not None and grid.ni * grid.nj != npts:
                    raise _Damaged('its grid is %d x %d but says %d points'
                                   % (grid.ni, grid.nj, npts))
                if npts == 0:
                    raise _Damaged('its grid has no points')
            elif num == 4:
                s4 = p
            elif num == 5:
                s5 = p
            elif num == 6:
                ind = b[p + 5]
                if ind == 0:
                    bitmap = cur = p + 6
                elif ind == 255:
                    cur = -1
                elif ind == 254:
                    if bitmap is None:              # e.g. a message fetched alone
                        raise _Damaged(_NO_BITMAP)
                    cur = bitmap
                else:
                    raise _Damaged('predefined bitmap %d not supported' % ind)
                s6 = p
            elif num == 7:
                if s4 is None or s5 < 0:
                    raise _Damaged('a data section without its product or packing sections')
                if npts is not None and struct.unpack_from('>I', b, s5 + 5)[0] > npts:
                    raise _Damaged('more values than its grid has points')
                recs.append(_record(b, nrec + len(recs) + 1, pos, disc, center, tables,
                                    reftype, date, ref, grid, s4, s5, s6, p, ln, cur))
                recs[-1].uv_relative = uvrel
        except (struct.error, IndexError) as e:
            raise _Damaged('section %d at byte %d is too short (%s)' % (num, p, e))
        p += ln
    if not recs:
        raise _Damaged('no data section')
    for k, r in enumerate(recs):
        r.msg, r.sub = msg, (k + 1 if len(recs) > 1 else 0)
    return recs, bitmap, end


# Section 3s parsed, by (centre, its bytes): a file's messages mostly share
# one grid, parsed once rather than a message each (GFS, 743 messages: a
# quarter of opening it). The key holds every byte a parse reads -- the
# section, and the octets up to 96 after its start that a fixed template
# reads (a damaged short section's run on) -- so grids listing rows or
# points (reduced, 3.130), read further, are not kept.
_GRIDS = {}
_GRIDS_MAX = 4096


def _grid_of(b, p, ln, center):
    """(Grid, whether vectors lie along its axes) of section 3 at p."""
    if ln > 1024:
        return _grids.parse(b, p, center), _winds.grid_relative(b, p)
    key = (center, bytes(b[p:p + max(ln, 96)]))
    hit = _GRIDS.get(key)
    if hit is None:
        hit = (_grids.parse(b, p, center), _winds.grid_relative(b, p))
        g = hit[0]
        if g is not None and (g.pl or g.template == 130):
            return hit
        if len(_GRIDS) >= _GRIDS_MAX:
            _GRIDS.clear()
        _GRIDS[key] = hit
    return hit


def _record(b, index, pos, disc, center, tables, reftype, date, ref, grid, s4, s5, s6, p, ln,
            bitmap):
    r = Record()
    r.index, r.offset, r.disc, r.center, r.date, r.ref_time = index, pos, disc, center, date, ref
    r.at = pos                                  # the message's start in the bytes scanned
    r.tables = tables
    tmpl = struct.unpack_from('>H', b, s4 + 7)[0]
    r.product = tmpl
    r.packing = struct.unpack_from('>H', b, s5 + 9)[0] if s5 >= 0 else -1
    r.cat, r.num = b[s4 + 9], b[s4 + 10]
    more = _chem_words(b, s4, tmpl, center, tables)
    chem, own = _chem_keys(b, s4, tmpl), tmpl
    if tmpl in _EXTENDS:                        # read as the template it extends
        b, s4, tmpl = _normal(b, s4, tmpl)
    r.name, r.long_name, r.units = param(disc, r.cat, r.num, center)
    r.ckeys = _keys(b, s4, tmpl, disc, r.cat, r.num, tables, chem, own,
                    grid.template if grid is not None else None)
    if r.long_name == 'unknown':                # beyond NCEP's tables: ecCodes' names
        hit = concept(r.ckeys, center, tables)
        if hit:
            r.name, r.long_name, r.units = hit
    if tmpl in (5, 9):                          # a probability's values are percent, not
        r.units = '%'                           # the parameter's (wgrib2: [prob])
    r.level_type = r.level_type2 = 255
    r.level_value = r.level_value2 = None
    if tmpl in _WITH_LEVEL:
        t1, sf1, sv1, t2, sf2, sv2 = struct.unpack_from('>BBIBBI', b, s4 + 22)
        r.level = _level(center, t1, sf1, sv1, t2, sf2, sv2)
        r.level_type, r.level_type2 = t1, t2
        r.level_value, r.level_value2 = _surface(sf1, sv1, t1), _surface(sf2, sv2, t2)
    else:
        r.level = 'product template 4.%d' % tmpl
    r.fcst = _fcst(tmpl, b, s4, center, disc, r.cat, r.num, tables)
    r.extra = _extra(tmpl, b, s4, center, reftype, more)
    if r.product in (45, 47) and r.extra.startswith(('ENS=', 'MM-ENS=')):
        r.extra = r.extra.split(':', 1)[1]      # wgrib2 words no member for these
    r.member, r.ensemble = _ensemble(tmpl, b, s4, center)
    r.valid_time = _times(tmpl, b, s4, ref) if tmpl in _WITH_LEVEL else None
    r.grid, r.s5, r.s6, r.s7, r.s7end = grid, s5, s6, p, p + ln
    r.bitmap = bitmap if bitmap is not None else -1
    return r
