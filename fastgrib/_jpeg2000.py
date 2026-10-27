"""JPEG2000 (ITU-T T.800 | ISO/IEC 15444-1) for GRIB2's template 5.40,
fastgrib's own: a decoder of any Part-1 codestream of one component --
every progression order, tiles and tile-parts, quality layers,
precincts, every code-block style (bypass, reset, termination, causal
contexts, segmentation symbols), the 5/3 and 9/7 wavelets, regions of
interest -- and a lossless encoder, to 31 bits a sample (OpenJPEG, which
ecCodes and wgrib2 use, loses samples past 25 bits; Pillow's reading,
past 16).

decode(codestream) -> its samples (int64, row by row); encode(x, w, h,
prec) -> a codestream. Tier-1 (the MQ coder's passes over each
code-block) runs on code-blocks in parallel when asked; tier-2 (the
packets) and the wavelets serially, all numba.
"""
import struct
import threading

import numpy as np
from ._jit import njit

from . import _cpu
from ._octets import clz64 as _clz64, take as _take

# MQ coder (T.800 Table C.2): each state's Qe, its next state after an
# MPS and after an LPS, and whether an LPS there swaps the MPS
_QE = np.array([
    0x5601, 0x3401, 0x1801, 0x0AC1, 0x0521, 0x0221, 0x5601, 0x5401, 0x4801, 0x3801,
    0x3001, 0x2401, 0x1C01, 0x1601, 0x5601, 0x5401, 0x5101, 0x4801, 0x3801, 0x3401,
    0x3001, 0x2801, 0x2401, 0x2201, 0x1C01, 0x1801, 0x1601, 0x1401, 0x1201, 0x1101,
    0x0AC1, 0x09C1, 0x08A1, 0x0521, 0x0441, 0x02A1, 0x0221, 0x0141, 0x0111, 0x0085,
    0x0049, 0x0025, 0x0015, 0x0009, 0x0005, 0x0001, 0x5601], np.int64)
_NMPS = np.array([
    1, 2, 3, 4, 5, 38, 7, 8, 9, 10, 11, 12, 13, 29, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 45, 46], np.int64)
_NLPS = np.array([
    1, 6, 9, 12, 29, 33, 6, 14, 14, 14, 17, 18, 20, 21, 14, 14, 15, 16, 17, 18, 19, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 46], np.int64)
_SWITCH = np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1] + [0] * 32, np.int64)
# a context's state: 2 x its index + its MPS; Qe and the next states by it
_QE2 = np.repeat(_QE, 2)
_NM2 = (2 * np.repeat(_NMPS, 2) + np.tile([0, 1], 47)).astype(np.int64)
_NL2 = (2 * np.repeat(_NLPS, 2) + (np.tile([0, 1], 47) ^ np.repeat(_SWITCH, 2))).astype(np.int64)

# contexts: 0-8 zero coding, 9-13 sign, 14-16 refinement, 17 run length, 18 uniform
_RL, _UNI = 17, 18
BYPASS, RESET, TERMALL, VSC, PTERM, SEGSYM = 1, 2, 4, 8, 16, 32   # code-block styles


# --- a coefficient's flags: its own state, and its neighbours' significance
# and signs (set as each neighbour turns significant: contexts are lookups)

SIG, VIS, REF, NEG = 1, 2, 4, 8
_N, _S, _W, _E = 1 << 4, 1 << 5, 1 << 6, 1 << 7
_NW, _NE, _SW, _SE = 1 << 8, 1 << 9, 1 << 10, 1 << 11
_NN, _SN, _WN, _EN = 1 << 12, 1 << 13, 1 << 14, 1 << 15     # that neighbour negative
_NBR = 0xFF0                                    # any neighbour significant
_BELOW = _S | _SW | _SE | _SN                   # the next stripe's: not seen (VSC)


def _tables():
    """Zero-coding contexts by orientation and neighbour bits (Table
    D.1), sign contexts and their XOR bits (D.3) by neighbour bits."""
    zc = np.zeros((4, 256), np.int64)
    for o in range(4):
        for f in range(256):
            v = (f & 1) + (f >> 1 & 1)
            h = (f >> 2 & 1) + (f >> 3 & 1)
            d = (f >> 4 & 1) + (f >> 5 & 1) + (f >> 6 & 1) + (f >> 7 & 1)
            a, b = (v, h) if o == 1 else (h, v)
            if o == 3:
                hv = h + v
                n = (8 if d >= 3 else (7 if hv else 6) if d == 2 else
                     (5 if hv >= 2 else 4 if hv else 3) if d == 1 else
                     (2 if hv >= 2 else 1 if hv else 0))
            elif a == 2:
                n = 8
            elif a == 1:
                n = 7 if b else 6 if d else 5
            else:
                n = 4 if b == 2 else 3 if b else 2 if d >= 2 else 1 if d else 0
            zc[o, f] = n
    sc = np.zeros(256, np.int64)
    sx = np.zeros(256, np.int64)
    for f in range(256):                        # bits 0-3: N S W E significant, 4-7 negative
        def c(bit):
            return 0 if not f >> bit & 1 else (-1 if f >> (bit + 4) & 1 else 1)
        h = max(-1, min(1, c(2) + c(3)))
        v = max(-1, min(1, c(0) + c(1)))
        k = (h + 1) * 3 + v + 1
        sc[f] = (13, 12, 11, 10, 9, 10, 11, 12, 13)[k]
        sx[f] = (1, 1, 1, 1, 0, 0, 0, 0, 0)[k]
    return zc, sc, sx


_ZC, _SC, _SX = _tables()


# Tier-1 indexes its arrays with unsigned integers: numba then adds no
# negative-index fix-up to each access (RAP's fields: 57 -> 51 ns a value)
_U0, _U1, _U3 = np.uint64(0), np.uint64(1), np.uint64(3)


# --- the encoder's flags (above): _turn, _sign_of, _idle, _run, _unvisit

@njit(cache=True, nogil=True)
def _turn(fl, p, W, neg):
    """The coefficient at p turned significant: its neighbours told."""
    fl[p] |= SIG | (NEG if neg else 0)
    fl[p - W] |= _S | (_SN if neg else 0)
    fl[p + W] |= _N | (_NN if neg else 0)
    fl[p - _U1] |= _E | (_EN if neg else 0)
    fl[p + _U1] |= _W | (_WN if neg else 0)
    fl[p - W - _U1] |= _SE
    fl[p - W + _U1] |= _SW
    fl[p + W - _U1] |= _NE
    fl[p + W + _U1] |= _NW


@njit(cache=True, nogil=True)
def _sign_of(f):
    k = np.uint64(((f >> 4) & 0xF) | ((f >> 8) & 0xF0))
    return _SC[k], _SX[k]


@njit(cache=True, nogil=True)
def _idle(fl, p, W, kind):
    """Whether a full stripe column (from p) has nothing for a pass: kind 0
    significance (all significant, or none with a significant neighbour),
    1 refinement (none significant), 2 cleanup (all significant or
    visited)."""
    f0, f1, f2, f3 = fl[p], fl[p + W], fl[p + W + W], fl[p + W + W + W]
    if kind == 0:
        return (f0 & f1 & f2 & f3 & SIG) != 0 or not (f0 | f1 | f2 | f3) & _NBR
    if kind == 1:
        return not (f0 | f1 | f2 | f3) & SIG
    return ((f0 | f0 >> 1) & (f1 | f1 >> 1) & (f2 | f2 >> 1) & (f3 | f3 >> 1) & 1) != 0


@njit(cache=True, nogil=True)
def _unvisit(fl):
    for k in range(fl.size):
        fl[k] &= ~VIS


@njit(cache=True, nogil=True)
def _run(fl, p, W, vsc):
    """Whether a stripe's column (from p) is coded by run length: none of
    its four significant, visited, or with a significant neighbour."""
    if (fl[p] | fl[p + W] | fl[p + W + W]) & (SIG | VIS | _NBR):
        return False
    f = fl[p + W + W + W]
    if vsc:
        f &= ~_BELOW
    return not f & (SIG | VIS | _NBR)


# --- the decoder's flags: one 64-bit word a stripe column (OpenJPEG 2.3's
# design), (h + 3) // 4 + 2 rows of w + 2 words. Row r of the stripe (-1
# and 4: the stripes' above and below) at bits 3(r + 1): bits 0-17 the
# significance of 6 rows x 3 columns (west, its own, east), 32-49 its own
# column's sign (chi), refined (mu) and visited (pi) bits. Shifted right
# by 3r, a row's word has its 3x3 neighbourhood in bits 0-8 and its own
# bits at fixed places: a pass's tests are register operations on one word
# loaded a column (where _idle loaded 4), a turn sets 3 words, not 9, and
# the rows a pass codes are a mask whose set bits it visits (_low). Alone,
# with a row's tests as before, the layout gained nothing; with the masks,
# RAP's fields: tier-1 36 -> 31 ns a value.

_C_SIG = np.uint64(1 << 4)                      # in a row's window (word >> 3r)
_C_NBR = np.uint64(0x1EF)                       # its 8 neighbours
_C_CHI = np.uint64(1 << 35)
_C_MU = np.uint64(1 << 36)
_C_PI = np.uint64(1 << 37)
_C_BELOW = np.uint64(0x1C0 | 1 << 38)           # the south row (VSC: unseen)
_C_SIGS = np.uint64(0x3FFFF)                    # a word's: significance bits
_C_ROW = np.uint64(0x9249)                      # a row's bit 0, rows -1..4
_C_PIS = np.uint64(0x249 << 37)                 # its rows' pi bits
_C_LAST = np.uint64(0x38000 | 1 << 47)          # row 4 (VSC: unseen)
_C_ROWS = np.array([0, 0x1, 0x9, 0x49, 0x249], np.uint64)   # rows of a stripe of 0-4


def _tables9():
    """Zero-coding contexts by a row's 9-bit window, and sign contexts
    (twice over, + the XOR bit) by its N, W, E, S significance and signs."""
    zc = np.zeros((4, 512), np.int64)
    for f in range(512):
        b = [f >> i & 1 for i in range(9)]      # NW N NE W . E SW S SE
        zc[:, f] = _ZC[:, b[1] | b[7] << 1 | b[3] << 2 | b[5] << 3 | b[0] << 4
                       | b[2] << 5 | b[6] << 6 | b[8] << 7]
    scx = np.zeros(256, np.int64)
    for f in range(256):
        b = [f >> i & 1 for i in range(8)]      # chi/sigma: N N W W E E S S
        k = b[1] | b[7] << 1 | b[3] << 2 | b[5] << 3 | b[0] << 4 | b[6] << 5 | b[2] << 6 | b[4] << 7
        scx[f] = _SC[k] << 1 | _SX[k]
    return zc, scx


_ZC9, _SCX = _tables9()


@njit(cache=True, nogil=True)
def _turn_col(fl, p, W, f, sh, neg):
    """The row at shift sh (3 x the row) of the column at p (its word f, in
    hand) turned significant: f, and its neighbours' words, told."""
    n = np.uint64(neg)
    fl[p - _U1] |= np.uint64(1 << 5) << sh
    fl[p + _U1] |= np.uint64(1 << 3) << sh
    if sh == _U0:
        fl[p - W] |= np.uint64(1 << 16) | n << np.uint64(47)
        fl[p - W - _U1] |= np.uint64(1 << 17)
        fl[p - W + _U1] |= np.uint64(1 << 15)
    elif sh == np.uint64(9):
        fl[p + W] |= np.uint64(1 << 1) | n << np.uint64(32)
        fl[p + W - _U1] |= np.uint64(1 << 2)
        fl[p + W + _U1] |= _U1
    return f | (_C_SIG | n << np.uint64(35)) << sh


@njit(cache=True, nogil=True)
def _sign_cx(fl, p, g, sh):
    """Row (word g, shifted by sh)'s sign context x 2 + XOR bit."""
    u = ((g & np.uint64(0xAA)) | (g >> np.uint64(32) & np.uint64(0x41))
         | (fl[p - _U1] >> sh >> np.uint64(33) & np.uint64(4))
         | (fl[p + _U1] >> sh >> np.uint64(31) & np.uint64(0x10)))
    return _SCX[u]


# --- the MQ decoder: its registers A, C, CT and position carried along

@njit(cache=True, nogil=True)
def _byte_in(buf, c, ct, pos):
    if buf[pos] == 0xFF:
        if buf[pos + 1] > 0x8F:
            return c + 0xFF00, 8, pos
        return c + (buf[pos + 1] << 9), 7, pos + 1
    return c + (buf[pos + 1] << 8), 8, pos + 1


@njit(cache=True, nogil=True)
def _bytes_in(buf, c, pos):
    """_byte_in when CT ran out, two bytes at once where neither the last
    nor the next is 0xFF (no stuffing, no marker): C's bits 0-15 are then
    clear, the second byte waits in 0-7 unseen by any comparison (C >> 16)
    or carry until shifted where _byte_in would have put it -- the same
    decisions, half the byte-in branches, coin tosses each (RAP's fields:
    tier-1 -5%). It reads past the next byte only when that is not 0xFF:
    never past the 0xFF 0xFF that _byte_in needs at the end. (Four bytes,
    C held 2^16 times over, cost more: _mq no longer inlined, or A spilled
    around a call.)"""
    if (buf[pos] != 0xFF) & (buf[pos + 1] != 0xFF):
        return c + (buf[pos + 1] << 8) + buf[pos + 2], 16, pos + 2
    return _byte_in(buf, c, 0, pos)


@njit(cache=True, nogil=True)
def _mq_init(buf, pos):
    c, ct, pos = _byte_in(buf, buf[pos] << 16, 0, pos)
    return 0x8000, (c << 7) & 0xFFFFFFFF, ct - 7, pos


# the decoder's contexts: a state as Qe << 7 | its index (2 x the state + MPS),
# Qe with it in one load (a load off A and C's chain; RAP's fields: tier-1
# 42 -> 36 ns a value with _mq's selects); by its index, its next states
# after an MPS and after an LPS in one word (bits 0-31, 32-63), loaded as
# soon as the state is: a context's next decision waits on its state's
# store when the context repeats (41% of RAP's decisions)
_MQS = (_QE2 << 7) | np.arange(94)
_MQP = _MQS[_NM2] | _MQS[_NL2] << 32


@njit(cache=True, nogil=True)
def _mq_reset(cs):
    cs[:] = _MQS[0]
    cs[0] = _MQS[4 << 1]
    cs[_RL] = _MQS[3 << 1]
    cs[_UNI] = _MQS[46 << 1]


@njit(cache=True, nogil=True)
def _mq(buf, a, c, ct, pos, cs, cx):
    """One decision in context cx: (it, A, C, CT, position). Branch-free
    but for a byte in: LPS or MPS, exchange and renormalising or not are
    selects -- they are coin tosses to a branch predictor (RAP's fields:
    tier-1 47.5 -> 42 ns a value)."""
    u = np.uint64(cx)
    t = cs[u]
    qe = t >> 7
    nx = _MQP[np.uint64(t & 0x7F)]
    a -= qe
    lo = np.int64((c >> 16) < qe)               # the LPS's interval
    e = lo ^ np.int64(a < qe)                   # the LPS decoded (exchange or not)
    m = -lo
    a = (qe & m) | (a & ~m)
    c -= (qe << 16) & ~m
    # the state moves on when A renormalises (always in the LPS's interval):
    # selects of values, not of loads, else LLVM makes them a branch (a coin
    # toss again). RAP's decisions alone: 3.8 -> 3.5 ns each against the
    # next state loaded by index x 4 + renormalised x 2 + LPS (tier-1 as a
    # whole: within noise)
    nt = (nx >> (e << 5)) & 0xFFFFFFFF
    cs[u] = t ^ ((nt ^ t) & -np.int64(a < 0x8000))
    # renormalise: shift until A's top bit is set, a byte in whenever CT
    # runs out -- as many bits at once as CT holds, not one an iteration
    # (an LPS's interval can need 15)
    n = _clz64(a) - 48
    if n <= ct:
        return (t & 1) ^ e, a << n, (c << n) & 0xFFFFFFFF, ct - n, pos
    while True:
        if ct == 0:
            c, ct, pos = _bytes_in(buf, c, pos)
        s = min(n, ct)
        a <<= s
        c = (c << s) & 0xFFFFFFFF
        ct -= s
        n -= s
        if n == 0:
            return (t & 1) ^ e, a, c, ct, pos


@njit(cache=True, nogil=True)
def _mqb(buf, a, c, ct, pos, cs, cx):
    """_mq with a branch for its likeliest outcome, an MPS without
    renormalising: a short chain where predicted -- run-length decisions
    (RAP's: 96% 0; the cleanup pass -8%); on zero coding it lost."""
    t = cs[np.uint64(cx)]
    qe = t >> 7
    if ((c >> 16) >= qe) & (a - qe >= 0x8000):
        return t & 1, a - qe, c - (qe << 16), ct, pos
    return _mq(buf, a, c, ct, pos, cs, cx)


@njit(cache=True, nogil=True)
def _raw(buf, b, ct, pos):
    """One bit of a bypassed (raw) segment: (it, byte, CT, position)."""
    if ct == 0:
        if b == 0xFF:
            if buf[pos] > 0x8F:
                b, ct = 0xFF, 8
            else:
                b, ct, pos = buf[pos], 7, pos + 1
        else:
            b, ct, pos = buf[pos], 8, pos + 1
    ct -= 1
    return (b >> ct) & 1, b, ct, pos


@njit(cache=True, nogil=True)
def _reset(cs):
    """The encoder's contexts' states (the decoder's: _mq_reset)."""
    cs[:] = 0
    cs[0] = 4 << 1
    cs[_RL] = 3 << 1
    cs[_UNI] = 46 << 1


# --- tier-1 decoding: a code-block's passes, its coefficients twice over
# (OpenJPEG's form: each with half its least known bit); q: the coder's
# registers between passes; zc: the zero-coding contexts (_ZC9) of its
# orientation; fl: the stripe columns' words

@njit(cache=True, nogil=True)
def _low(m):
    """m's lowest set bit, and its place (a row's shift: 3 x the row)."""
    b = m & (_U0 - m)
    return b, np.uint64(63 - _clz64(b))


@njit(cache=True, nogil=True)
def _sig_rows(f, ym):
    """The rows (bits 3r) neither significant nor visited, with a significant
    neighbour: each row's 3 x 3 significance ORed bit-parallel."""
    x = f & _C_SIGS
    y = (x | x >> _U1 | x >> np.uint64(2)) & _C_ROW
    z = y | y >> _U3 | (y & ym) >> np.uint64(6)
    return z & ~(f >> np.uint64(4) | f >> np.uint64(37))


@njit(cache=True, nogil=True)
def _sig_pass(buf, q, cs, fl, v2, w, h, zc, bp1, vsc, raw):
    W = np.uint64(w + 2)
    wu = np.uint64(w)
    oph = (1 << bp1) | (1 << (bp1 - 1))
    ym = np.uint64(0x1249) if vsc else _C_ROW  # row 3's neighbourhood: not row 4 (VSC)
    a, c, ct, pos = q[0], q[1], q[2], q[3]
    for k in range(0, h, 4):
        rows = _C_ROWS[min(4, h - k)]
        p = np.uint64((k // 4 + 1) * (w + 2) + 1)
        o0 = np.uint64(k * w)
        for i in range(w):
            f = fl[p]
            m = _sig_rows(f, ym) & rows
            # the rows to code, lowest first: no test a row
            while m != _U0:
                b, sh = _low(m)
                g = f >> sh
                if vsc and sh == np.uint64(9):
                    g &= ~_C_BELOW
                if raw:
                    bit, c, ct, pos = _raw(buf, c, ct, pos)
                else:
                    bit, a, c, ct, pos = _mq(buf, a, c, ct, pos, cs, zc[g & np.uint64(0x1FF)])
                f |= _C_PI << sh
                m ^= b
                if bit:
                    if raw:
                        s, c, ct, pos = _raw(buf, c, ct, pos)
                    else:
                        t = _sign_cx(fl, p, g, sh)
                        s, a, c, ct, pos = _mq(buf, a, c, ct, pos, cs, t >> 1)
                        s ^= t & 1
                    ci = (sh * np.uint64(11)) >> np.uint64(5)   # sh / 3
                    v2[o0 + ci * wu] = -oph if s else oph
                    f = _turn_col(fl, p, W, f, sh, s)
                    # the rows below may have gained a significant neighbour
                    m = _sig_rows(f, ym) & rows & ~(b + b - _U1)
            fl[p] = f
            p += _U1
            o0 += _U1
    q[0], q[1], q[2], q[3] = a, c, ct, pos


@njit(cache=True, nogil=True)
def _ref_pass(buf, q, cs, fl, v2, w, h, bp1, vsc, raw):
    wu = np.uint64(w)
    half = (1 << bp1) >> 1
    a, c, ct, pos = q[0], q[1], q[2], q[3]
    for k in range(0, h, 4):
        rows = _C_ROWS[min(4, h - k)]
        p = np.uint64((k // 4 + 1) * (w + 2) + 1)
        o0 = np.uint64(k * w)
        for i in range(w):
            f = fl[p]
            # rows significant and not visited: refinement changes no
            # word, so they are known for the column at once
            m0 = f >> np.uint64(4) & ~(f >> np.uint64(37)) & rows
            m = m0
            while m != _U0:
                b, sh = _low(m)
                m ^= b
                if raw:
                    bit, c, ct, pos = _raw(buf, c, ct, pos)
                else:
                    g = f >> sh
                    if vsc and sh == np.uint64(9):
                        g &= ~_C_BELOW
                    cx = 16 if g & _C_MU else (15 if g & _C_NBR else 14)
                    bit, a, c, ct, pos = _mq(buf, a, c, ct, pos, cs, cx)
                o = o0 + ((sh * np.uint64(11)) >> np.uint64(5)) * wu
                t = v2[o]
                v2[o] = t + half if bit ^ (t < 0) else t - half
            fl[p] = f | m0 << np.uint64(36)
            p += _U1
            o0 += _U1
    q[0], q[1], q[2], q[3] = a, c, ct, pos


@njit(cache=True, nogil=True)
def _clean_pass(buf, q, cs, fl, v2, w, h, zc, bp1, vsc, segsym):
    W = np.uint64(w + 2)
    wu = np.uint64(w)
    oph = (1 << bp1) | (1 << (bp1 - 1))
    # a run: none of the 4 significant, visited or with a significant neighbour
    rm = ~_C_LAST if vsc else ~_U0
    a, c, ct, pos = q[0], q[1], q[2], q[3]
    for k in range(0, h, 4):
        n = min(4, h - k)
        rows = _C_ROWS[n]
        p = np.uint64((k // 4 + 1) * (w + 2) + 1)
        o0 = np.uint64(k * w)
        for i in range(w):
            f = fl[p]
            if n == 4 and (f & rm) == _U0:
                bit, a, c, ct, pos = _mqb(buf, a, c, ct, pos, cs, _RL)
                if not bit:
                    p += _U1
                    o0 += _U1
                    continue
                r, a, c, ct, pos = _mq(buf, a, c, ct, pos, cs, _UNI)
                r2, a, c, ct, pos = _mq(buf, a, c, ct, pos, cs, _UNI)
                ci = np.uint64(r << 1 | r2)
                sh = ci * _U3
                g = f >> sh
                if vsc and sh == np.uint64(9):
                    g &= ~_C_BELOW
                t = _sign_cx(fl, p, g, sh)
                s, a, c, ct, pos = _mq(buf, a, c, ct, pos, cs, t >> 1)
                s ^= t & 1
                v2[o0 + ci * wu] = -oph if s else oph
                f = _turn_col(fl, p, W, f, sh, s)
                b = _U1 << sh
                m = rows & ~(b + b - _U1)       # the rows after it
            else:
                m = ~(f >> np.uint64(4) | f >> np.uint64(37)) & rows
            # rows neither significant nor visited: a turn changes none
            while m != _U0:
                b, sh = _low(m)
                m ^= b
                g = f >> sh
                if vsc and sh == np.uint64(9):
                    g &= ~_C_BELOW
                bit, a, c, ct, pos = _mq(buf, a, c, ct, pos, cs, zc[g & np.uint64(0x1FF)])
                if bit:
                    t = _sign_cx(fl, p, g, sh)
                    s, a, c, ct, pos = _mq(buf, a, c, ct, pos, cs, t >> 1)
                    s ^= t & 1
                    ci = (sh * np.uint64(11)) >> np.uint64(5)
                    v2[o0 + ci * wu] = -oph if s else oph
                    f = _turn_col(fl, p, W, f, sh, s)
            fl[p] = f & ~_C_PIS                 # visited: for this bit-plane only
            p += _U1
            o0 += _U1
    if segsym:
        for _ in range(4):
            bit, a, c, ct, pos = _mq(buf, a, c, ct, pos, cs, _UNI)
    q[0], q[1], q[2], q[3] = a, c, ct, pos


@njit(cache=True, nogil=True)
def _block(buf, seg_at, seg_np, nseg, w, h, orient, numbps, roi, style, v2):
    fl = np.zeros(((h + 3) // 4 + 2) * (w + 2), np.uint64)
    cs = np.zeros(19, np.int64)
    _mq_reset(cs)
    q = np.zeros(4, np.int64)
    zc = _ZC9[orient]
    kind = 2                                    # 0 significance, 1 refinement, 2 cleanup
    bp1 = roi + numbps
    vsc = (style & VSC) != 0
    for s in range(nseg):
        raw = (style & BYPASS) != 0 and bp1 <= numbps - 4 and kind < 2
        if raw:
            q[0], q[1], q[2], q[3] = 0, 0, 0, seg_at[s]
        else:
            q[0], q[1], q[2], q[3] = _mq_init(buf, seg_at[s])
        for _ in range(seg_np[s]):
            if bp1 < 1:
                break
            if kind == 0:
                _sig_pass(buf, q, cs, fl, v2, w, h, zc, bp1, vsc, raw)
            elif kind == 1:
                _ref_pass(buf, q, cs, fl, v2, w, h, bp1, vsc, raw)
            else:
                _clean_pass(buf, q, cs, fl, v2, w, h, zc, bp1, vsc,
                            (style & SEGSYM) != 0)
            if style & RESET and not raw:
                _mq_reset(cs)
            kind += 1
            if kind == 3:
                kind = 0
                bp1 -= 1
    if roi:                                     # the region of interest's coefficients back down
        t = 1 << roi
        for k in range(w * h):
            m = abs(v2[k])
            if m >= t:
                v2[k] = -(m >> roi) if v2[k] < 0 else m >> roi


@njit(cache=True, nogil=True)
def _decode_block(c, buf, cb, seg_at, seg_np, cb_nseg, orient_of, roi, style, out):
    # cb: band, x0, y0, x1, y1 (in the band's array), bit-planes, its offset, its width
    w, h = cb[c, 3] - cb[c, 1], cb[c, 4] - cb[c, 2]
    numbps = cb[c, 5]
    if cb_nseg[c] == 0 or numbps <= 0 or w <= 0 or h <= 0:
        return
    v2 = np.zeros(w * h, np.int64)
    _block(buf, seg_at[c], seg_np[c], cb_nseg[c], w, h, orient_of[cb[c, 0]], numbps, roi,
           style, v2)
    for j in range(h):
        base = cb[c, 6] + (cb[c, 2] + j) * cb[c, 7] + cb[c, 1]
        for i in range(w):
            out[base + i] = v2[j * w + i]


@njit(cache=True, nogil=True)
def _blocks_serial(n, buf, cb, seg_at, seg_np, cb_nseg, orient_of, roi, style, out):
    for c in range(n):
        _decode_block(c, buf, cb, seg_at, seg_np, cb_nseg, orient_of, roi, style, out)


@njit(cache=True, nogil=True)
def _blocks_take(n, buf, cb, seg_at, seg_np, cb_nseg, orient_of, roi, style, out, ctr):
    """Code-blocks taken off ctr[0] until none is left (_cpu.share's threads)."""
    while True:
        c = _take(ctr, 0)
        if c >= n:
            return
        _decode_block(c, buf, cb, seg_at, seg_np, cb_nseg, orient_of, roi, style, out)


# --- tier-2: packets

@njit(cache=True, nogil=True)
def _bit(d, s):
    """A bit of a packet header: s = [buffer, CT, position]."""
    if s[1] == 0:
        s[0] = (s[0] << 8) & 0xFFFF
        s[1] = 7 if s[0] == 0xFF00 else 8
        if s[2] < d.size:
            s[0] |= d[s[2]]
            s[2] += 1
    s[1] -= 1
    return (s[0] >> s[1]) & 1


@njit(cache=True, nogil=True)
def _bits(d, s, n):
    v = 0
    for _ in range(n):
        v = (v << 1) | _bit(d, s)
    return v


@njit(cache=True, nogil=True)
def _tag(d, s, parent, val, low, leaf, threshold):
    """Tag-tree decoding (B.10.2): whether leaf's value is below threshold."""
    stack = np.empty(64, np.int64)
    n = 0
    node = leaf
    while parent[node] >= 0:
        stack[n] = node
        n += 1
        node = parent[node]
    lo = 0
    while True:
        if lo > low[node]:
            low[node] = lo
        else:
            lo = low[node]
        while lo < threshold and lo < val[node]:
            if _bit(d, s):
                val[node] = lo
            else:
                lo += 1
        low[node] = lo
        if n == 0:
            break
        n -= 1
        node = stack[n]
    return val[node] < threshold


@njit(cache=True, nogil=True)
def _floorlog2(n):
    k = 0
    while n > 1:
        n >>= 1
        k += 1
    return k


@njit(cache=True, nogil=True)
def _init_seg(seg_np, seg_max, c, g, style, prev):
    seg_np[c, g] = 0
    if style & TERMALL:
        seg_max[c, g] = 1
    elif style & BYPASS:
        seg_max[c, g] = 10 if prev < 0 else (2 if prev == 1 or prev == 10 else 1)
    else:
        seg_max[c, g] = 109


@njit(cache=True, nogil=True)
def _grow(a):
    b = np.empty((a.shape[0] * 2, a.shape[1]), a.dtype)
    b[:a.shape[0]] = a
    return b


@njit(cache=True, nogil=True)
def _tier2(d, packets, rp0, rp1, e_band, e_nw, e_nh, e_cb0, e_tt0, parent, band_mb,
           style, sop, eph, cb_numbps, cb_nseg, seg_np, seg_max, seg_len):
    """Every packet's header and body: each code-block's segments (passes
    and bytes) and its pieces of data, in order (code-block, segment,
    position, length). Returns (pieces, how many; -1: data run out)."""
    ncb = cb_numbps.size
    inc_val = np.full(parent.size, 1 << 30, np.int64)
    inc_low = np.zeros(parent.size, np.int64)
    zb_val = np.full(parent.size, 1 << 30, np.int64)
    zb_low = np.zeros(parent.size, np.int64)
    lblock = np.full(ncb, 3, np.int64)
    maxseg = seg_np.shape[1]
    new = np.empty((64, 3), np.int64)           # this packet's contributions: cb, segment, length
    pieces = np.empty((max(ncb, 16), 4), np.int64)
    npc = 0
    pos = 0
    s = np.zeros(3, np.int64)
    for t in range(packets.shape[0]):
        layer, rp = packets[t, 0], packets[t, 1]
        if pos >= d.size:
            break
        if sop and pos + 6 <= d.size and d[pos] == 0xFF and d[pos + 1] == 0x91:
            pos += 6
        s[0], s[1], s[2] = 0, 0, pos
        nnew = 0
        if _bit(d, s):
            for e in range(rp0[rp], rp1[rp]):
                for k in range(e_nw[e] * e_nh[e]):
                    c = e_cb0[e] + k
                    leaf = e_tt0[e] + k
                    if cb_nseg[c] == 0:
                        inc = _tag(d, s, parent, inc_val, inc_low, leaf, layer + 1)
                    else:
                        inc = _bit(d, s)
                    if not inc:
                        continue
                    if cb_nseg[c] == 0:         # first time in: its missing bit-planes
                        z = 0
                        while not _tag(d, s, parent, zb_val, zb_low, leaf, z):
                            z += 1
                            if z > 76:                  # (more than any band has: damaged)
                                return pieces, -1
                        cb_numbps[c] = band_mb[e_band[e]] + 1 - z
                    if not _bit(d, s):          # its new passes
                        n = 1
                    elif not _bit(d, s):
                        n = 2
                    else:
                        x = _bits(d, s, 2)
                        if x != 3:
                            n = 3 + x
                        else:
                            x = _bits(d, s, 5)
                            n = 6 + x if x != 31 else 37 + _bits(d, s, 7)
                    while _bit(d, s):
                        lblock[c] += 1
                    if cb_nseg[c] == 0:         # the passes into segments
                        g = 0
                        _init_seg(seg_np, seg_max, c, 0, style, -1)
                        cb_nseg[c] = 1
                    else:
                        g = cb_nseg[c] - 1
                        if seg_np[c, g] == seg_max[c, g]:
                            g += 1
                            if g >= maxseg:
                                return pieces, -1
                            _init_seg(seg_np, seg_max, c, g, style, seg_max[c, g - 1])
                            cb_nseg[c] = g + 1
                    while True:
                        take = min(seg_max[c, g] - seg_np[c, g], n)
                        if nnew == new.shape[0]:
                            new = _grow(new)
                        new[nnew, 0], new[nnew, 1] = c, g
                        new[nnew, 2] = _bits(d, s, lblock[c] + _floorlog2(take))
                        nnew += 1
                        seg_np[c, g] += take
                        n -= take
                        if n <= 0:
                            break
                        g += 1
                        if g >= maxseg:
                            return pieces, -1
                        _init_seg(seg_np, seg_max, c, g, style, seg_max[c, g - 1])
                        cb_nseg[c] = g + 1
        if (s[0] & 0xFF) == 0xFF and s[2] < d.size:     # the header ends a byte after an 0xFF
            s[2] += 1
        pos = s[2]
        if eph and pos + 2 <= d.size and d[pos] == 0xFF and d[pos + 1] == 0x92:
            pos += 2
        for k in range(nnew):                   # the body
            ln = new[k, 2]
            if ln < 0 or pos + ln > d.size:
                return pieces, -1
            if npc == pieces.shape[0]:
                pieces = _grow(pieces)
            pieces[npc, 0], pieces[npc, 1] = new[k, 0], new[k, 1]
            pieces[npc, 2], pieces[npc, 3] = pos, ln
            seg_len[new[k, 0], new[k, 1]] += ln
            npc += 1
            pos += ln
    return pieces, npc


@njit(cache=True, nogil=True)
def _gather(d, pieces, npc, cb_nseg, seg_len, buf):
    """Each segment's bytes together in buf (0xFF throughout, as long as
    the segments and two bytes after each, and two more: the coders read on
    past a segment's end); each segment's position."""
    ncb, maxseg = seg_len.shape
    seg_at = np.zeros((ncb, maxseg), np.int64)
    tot = 0
    for c in range(ncb):
        for g in range(cb_nseg[c]):
            seg_at[c, g] = tot
            tot += seg_len[c, g] + 2
    fill = seg_at.copy()
    for k in range(npc):
        c, g, at, ln = pieces[k, 0], pieces[k, 1], pieces[k, 2], pieces[k, 3]
        o = fill[c, g]
        for i in range(ln):
            buf[o + i] = d[at + i]
        fill[c, g] = o + ln
    return seg_at


# --- wavelets (T.800 F.3, F.4): lines in place, i0 their first sample's
# absolute coordinate (low-pass samples at even coordinates)

@njit(cache=True, nogil=True)
def _isr53(x, n, i0):
    if n == 1:
        if i0 & 1:
            x[0] >>= 1
        return
    e = i0 & 1                                  # the first even coordinate's index
    for k in range(e, n, 2):
        a = x[k - 1] if k > 0 else x[k + 1]
        b = x[k + 1] if k + 1 < n else x[k - 1]
        x[k] -= (a + b + 2) >> 2
    for k in range(1 - e, n, 2):
        a = x[k - 1] if k > 0 else x[k + 1]
        b = x[k + 1] if k + 1 < n else x[k - 1]
        x[k] += (a + b) >> 1


@njit(cache=True, nogil=True)
def _fsd53(x, n):
    """The forward 5/3 of a line starting at coordinate 0."""
    if n == 1:
        return
    for k in range(1, n, 2):
        b = x[k + 1] if k + 1 < n else x[k - 1]
        x[k] -= (x[k - 1] + b) >> 1
    for k in range(0, n, 2):
        a = x[k - 1] if k > 0 else x[k + 1]
        b = x[k + 1] if k + 1 < n else x[k - 1]
        x[k] += (a + b + 2) >> 2


_A97, _B97 = -1.586134342059924, -0.052980118572961
_G97, _D97 = 0.882911075530934, 0.443506852043971
_K97 = 1.230174104914001


@njit(cache=True, nogil=True)
def _lift(x, n, start, c):
    for k in range(start, n, 2):
        a = x[k - 1] if k > 0 else x[k + 1]
        b = x[k + 1] if k + 1 < n else x[k - 1]
        x[k] += c * (a + b)


@njit(cache=True, nogil=True)
def _isr97(x, n, i0):
    if n == 1:
        if i0 & 1:
            x[0] *= 0.5
        return
    e = i0 & 1
    for k in range(e, n, 2):
        x[k] *= _K97
    for k in range(1 - e, n, 2):
        x[k] *= 1.0 / _K97
    _lift(x, n, e, -_D97)
    _lift(x, n, 1 - e, -_G97)
    _lift(x, n, e, -_B97)
    _lift(x, n, 1 - e, -_A97)


@njit(cache=True, nogil=True)
def _interleave(lo, hl, lh, hh, out, x0, y0, x1, y1):
    """A resolution's samples from its four bands, a row at a time: its
    even columns from one band's row, its odd ones from the other's."""
    w = x1 - x0
    lx, ly = (x0 + 1) >> 1, (y0 + 1) >> 1
    hx, hy = x0 >> 1, y0 >> 1
    lw, hw = ((x1 + 1) >> 1) - lx, (x1 >> 1) - hx
    e = x0 & 1                                  # where the row's first even column is
    for y in range(y0, y1):
        o = out[(y - y0) * w:(y - y0 + 1) * w]
        if y & 1:
            r = (y >> 1) - hy
            ev, od = lh[r * lw:(r + 1) * lw], hh[r * hw:(r + 1) * hw]
        else:
            r = ((y + 1) >> 1) - ly
            ev, od = lo[r * lw:(r + 1) * lw], hl[r * hw:(r + 1) * hw]
        for u in range(lw):
            o[e + 2 * u] = ev[u]
        for u in range(hw):
            o[1 - e + 2 * u] = od[u]


@njit(cache=True, nogil=True)
def _idwt53(a, x0, y0, x1, y1):
    """The inverse 5/3 of a resolution in place: each row's, then each
    column's -- the columns' lifting steps a whole row at a time (the same
    operations on each sample, in the same order, as a column at a time,
    but along memory rather than across it)."""
    w, h = x1 - x0, y1 - y0
    for j in range(h):
        _isr53(a[j * w:(j + 1) * w], w, x0)
    if h == 1:
        if y0 & 1:
            for i in range(w):
                a[i] >>= 1
        return
    e = y0 & 1
    for k in range(e, h, 2):
        r, u, d = _nbr_rows(a, w, h, k)
        for i in range(w):
            r[i] -= (u[i] + d[i] + 2) >> 2
    for k in range(1 - e, h, 2):
        r, u, d = _nbr_rows(a, w, h, k)
        for i in range(w):
            r[i] += (u[i] + d[i]) >> 1


@njit(cache=True, nogil=True)
def _nbr_rows(a, w, h, k):
    """Row k of a and the rows above and below it, mirrored at the edges
    (as _isr53's and _lift's samples are)."""
    up = k - 1 if k > 0 else k + 1
    dn = k + 1 if k + 1 < h else k - 1
    return a[k * w:(k + 1) * w], a[up * w:(up + 1) * w], a[dn * w:(dn + 1) * w]


@njit(cache=True, nogil=True)
def _lift_rows(a, w, h, start, c):
    """_lift down the columns, a row at a time."""
    for k in range(start, h, 2):
        r, u, d = _nbr_rows(a, w, h, k)
        for i in range(w):
            r[i] += c * (u[i] + d[i])


@njit(cache=True, nogil=True)
def _idwt97(a, x0, y0, x1, y1):
    """The inverse 9/7 in place, as _idwt53."""
    w, h = x1 - x0, y1 - y0
    for j in range(h):
        _isr97(a[j * w:(j + 1) * w], w, x0)
    if h == 1:
        if y0 & 1:
            for i in range(w):
                a[i] *= 0.5
        return
    e = y0 & 1
    for k in range(h):
        s = _K97 if (k & 1) == e else 1.0 / _K97
        for i in range(w):
            a[k * w + i] *= s
    _lift_rows(a, w, h, e, -_D97)
    _lift_rows(a, w, h, 1 - e, -_G97)
    _lift_rows(a, w, h, e, -_B97)
    _lift_rows(a, w, h, 1 - e, -_A97)


@njit(cache=True, nogil=True)
def _halve(v):
    """Coefficients (twice over, as tier-1 leaves them) to the reversible
    wavelet's, in place: C's division by 2, toward 0."""
    for k in range(v.size):
        x = v[k]
        v[k] = (x - (x >> 63)) >> 1


@njit(cache=True, nogil=True)
def _place(img, w, h, ox, oy, shift, lo, hi, out, W):
    for j in range(h):
        for i in range(w):
            x = img[j * w + i] + shift
            out[(oy + j) * W + ox + i] = min(max(x, lo), hi)


# --- the codestream: markers

class _Coding:
    """A component's coding: decomposition levels, code-block exponents,
    style, reversible, each level's precinct exponents; and COD's own:
    progression order, layers, SOP and EPH markers."""
    __slots__ = ('nl', 'xcb', 'ycb', 'style', 'rev', 'pp', 'order', 'layers', 'sop', 'eph')


def _cod(B, p, base=None):
    """COD at p, or (base given) COC's coding past its component index."""
    c = _Coding()
    s = B[p]
    q = p + 1
    if base is None:
        c.order, c.layers = B[q], (B[q + 1] << 8) | B[q + 2]
        c.sop, c.eph = bool(s & 2), bool(s & 4)
        q += 4
    else:
        c.order, c.layers, c.sop, c.eph = base.order, base.layers, base.sop, base.eph
    c.nl, c.xcb, c.ycb, c.style, c.rev = B[q], B[q + 1] + 2, B[q + 2] + 2, B[q + 3], B[q + 4] == 1
    q += 5
    c.pp = ([(B[q + r] & 15, B[q + r] >> 4) for r in range(c.nl + 1)] if s & 1 else
            [(15, 15)] * (c.nl + 1))
    if c.xcb > 10 or c.ycb > 10 or c.xcb + c.ycb > 12 or c.nl > 32:
        raise ValueError('JPEG2000: code-blocks of 2^%d x 2^%d, %d levels'
                         % (c.xcb, c.ycb, c.nl))
    return c


def _qcd(B, p, n):
    """(guard bits, [(exponent, mantissa) of each subband], derived)."""
    s = B[p]
    style, g = s & 31, s >> 5
    if style == 0:
        return g, [(B[p + 1 + k] >> 3, 0) for k in range(n - 1)], False
    vals = struct.unpack_from('>%dH' % ((n - 1) // 2), B, p + 1)
    return g, [(v >> 11, v & 0x7FF) for v in vals], style == 1


class _Header:
    def __init__(self):
        self.cod = self.coc = self.qcd = self.qcc = self.rgn = None


def _parse(B):
    """(SIZ's fields, the main header, {tile: (its header, [its tile-parts'
    data spans])}) of codestream bytes B."""
    if len(B) < 4 or B[0] != 0xFF or B[1] != 0x4F:
        raise ValueError('JPEG2000: not a codestream')
    main = _Header()
    tiles = {}
    siz = None
    p = 2
    while p + 4 <= len(B):
        m = (B[p] << 8) | B[p + 1]
        if m == 0xFFD9:                         # EOC
            break
        ln = (B[p + 2] << 8) | B[p + 3]
        q = p + 4
        if m == 0xFF51:                         # SIZ
            v = struct.unpack_from('>HIIIIIIIIH', B, q)
            if v[9] != 1:
                raise ValueError('JPEG2000 of %d components (GRIB has one)' % v[9])
            ss = B[q + 36]
            siz = v[1:9] + ((ss & 0x7F) + 1, bool(ss & 0x80), B[q + 37], B[q + 38])
            if siz[8] > 38 or not B[q + 37] or not B[q + 38]:
                raise ValueError('JPEG2000: a component of %d bits' % siz[8])
            p = q + ln - 2
        elif m == 0xFF90:                       # SOT: a tile-part
            it, psot = struct.unpack_from('>HI', B, q)
            end = min(p + psot if psot else len(B) - 2, len(B))
            hdr, spans = tiles.setdefault(it, (_Header(), []))
            p = q + ln - 2
            while p + 2 <= end:                 # its header, to SOD
                m = (B[p] << 8) | B[p + 1]
                if m == 0xFF93:
                    p += 2
                    break
                ln = (B[p + 2] << 8) | B[p + 3]
                _marker(hdr, B, m, p + 4, ln - 2)
                p += 2 + ln
            spans.append((p, end))
            p = end
        else:
            _marker(main, B, m, q, ln - 2)
            p = q + ln - 2
    if siz is None or main.cod is None or main.qcd is None:
        raise ValueError('JPEG2000: its main header is incomplete')
    return siz, main, tiles


def _marker(hdr, B, m, q, n):
    if m == 0xFF52:
        hdr.cod = q
    elif m == 0xFF53:                           # COC (component 0's: there is one)
        hdr.coc = q + 1
    elif m == 0xFF5C:
        hdr.qcd = (q, n)
    elif m == 0xFF5D:
        hdr.qcc = (q + 1, n - 1)
    elif m == 0xFF5E:                           # RGN: the region of interest's shift
        hdr.rgn = B[q + 2]
    elif m == 0xFF5F:
        raise ValueError('JPEG2000 with progression order changes (POC): not read')
    elif m in (0xFF60, 0xFF61):
        raise ValueError('JPEG2000 with packed packet headers (PPM/PPT): not read')


def _ceil(a, s):
    return -(-a // s)


def decode(data, parallel=False, expect=None, scratch=False):
    """The samples of a JPEG2000 codestream (or JP2 file) of one
    component, row by row (int64). parallel: code-blocks on every core;
    expect: how many samples it should hold (refused before decoding
    otherwise). A damaged codestream raises ValueError."""
    try:
        B = bytes(data)
        if B[4:8] == b'jP  ':
            B = _jp2c(B)
        siz, main, tiles = _parse(B)
        w, h, x0, y0, tw, th, tx0, ty0, prec, signed, xr, yr = siz
        W, H = _ceil(w, xr) - _ceil(x0, xr), _ceil(h, yr) - _ceil(y0, yr)
        ntx, nty = _ceil(w - tx0, tw), _ceil(h - ty0, th)
        if W <= 0 or H <= 0 or tx0 > x0 or ty0 > y0 or ntx * nty > 1 << 16 or                 W * H > 1 << 32 or (expect is not None and W * H != expect):
            raise ValueError('JPEG2000: an image of %d x %d in %d tiles' % (W, H, ntx * nty))
        out = _scratch('out', H * W, np.int64, 0) if scratch else np.zeros(H * W, np.int64)
        d = np.frombuffer(B, np.uint8)
        for it, (hdr, spans) in sorted(tiles.items()):
            if it >= ntx * nty:
                raise ValueError('JPEG2000: tile %d of %d' % (it, ntx * nty))
            _tile(B, d, siz, main, hdr, spans, it % ntx, it // ntx, out, W, parallel)
        return out
    except (struct.error, IndexError, KeyError, TypeError, ZeroDivisionError) as e:
        raise ValueError('JPEG2000: a damaged codestream (%s: %s)' % (type(e).__name__, e))


def _jp2c(B):
    p = 0
    while p + 8 <= len(B):
        ln, kind = struct.unpack_from('>I4s', B, p)
        hd = 8
        if ln == 1:
            ln, hd = struct.unpack_from('>Q', B, p + 8)[0], 16
        if kind == b'jp2c':
            return B[p + hd:p + ln if ln else len(B)]
        if ln < 8:
            break
        p += ln
    raise ValueError('JPEG2000: a JP2 file without a codestream')


def _tile(B, d, siz, main, hdr, spans, px, py, out, W, parallel):
    w, h, x0, y0, tw, th, tx0, ty0, prec, signed, xr, yr = siz
    cod = _cod(B, hdr.cod if hdr.cod is not None else main.cod)
    coc = hdr.coc if hdr.coc is not None else (None if hdr.cod is not None else main.coc)
    c = _cod(B, coc, cod) if coc is not None else cod
    qq = hdr.qcc or hdr.qcd or main.qcc or main.qcd
    guard, steps, derived = _qcd(B, qq[0], qq[1])
    roi = hdr.rgn if hdr.rgn is not None else (main.rgn or 0)
    if roi > 37:
        raise ValueError('JPEG2000: a region of interest shifted %d bit-planes' % roi)
    ax0, ax1 = max(tx0 + px * tw, x0), min(tx0 + (px + 1) * tw, w)
    ay0, ay1 = max(ty0 + py * th, y0), min(ty0 + (py + 1) * th, h)
    cx0, cx1, cy0, cy1 = _ceil(ax0, xr), _ceil(ax1, xr), _ceil(ay0, yr), _ceil(ay1, yr)
    nl = c.nl
    bands = []                  # [r, orient, x0, y0, x1, y1, Mb, step, offset]
    res = []                    # (x0, y0, x1, y1, ppx, ppy, precincts across, down, bands)
    off = 0
    for r in range(nl + 1):
        sc = 1 << (nl - r)
        rx0, ry0, rx1, ry1 = _ceil(cx0, sc), _ceil(cy0, sc), _ceil(cx1, sc), _ceil(cy1, sc)
        ids = []
        for o in ((0,) if r == 0 else (1, 2, 3)):
            if r == 0:
                bx0, by0, bx1, by1, nb, k = rx0, ry0, rx1, ry1, nl, 0
            else:
                nb = nl - r + 1
                s2, h2 = 1 << nb, (1 << nb) >> 1
                bx0, bx1 = _ceil(cx0 - h2 * (o & 1), s2), _ceil(cx1 - h2 * (o & 1), s2)
                by0, by1 = _ceil(cy0 - h2 * (o >> 1), s2), _ceil(cy1 - h2 * (o >> 1), s2)
                k = 3 * (r - 1) + o
            if derived:
                e, mu = steps[0][0] - nl + nb, steps[0][1]
            elif k < len(steps):
                e, mu = steps[k]
            else:
                raise ValueError('JPEG2000: quantisation for %d of %d subbands'
                                 % (len(steps), 3 * nl + 1))
            step = (1.0 + mu / 2048.0) * 2.0 ** (prec + (0, 1, 1, 2)[o] - e)
            bands.append([r, o, bx0, by0, bx1, by1, guard + e - 1, step, off])
            ids.append(len(bands) - 1)
            off += (bx1 - bx0) * (by1 - by0)
        ppx, ppy = c.pp[r]
        npw = _ceil(rx1, 1 << ppx) - (rx0 >> ppx) if rx1 > rx0 else 0
        nph = _ceil(ry1, 1 << ppy) - (ry0 >> ppy) if ry1 > ry0 else 0
        res.append((rx0, ry0, rx1, ry1, ppx, ppy, npw, nph, ids))
    if c.order > 4:
        raise ValueError('JPEG2000 progression order %d' % c.order)
    cbs = []                    # band, x0, y0, x1, y1 (in its band), -, band offset, band width
    e_band, e_nw, e_nh, e_cb0, e_tt0, rp0, rp1, parent, key = [], [], [], [], [], [], [], [], []
    for r, (rx0, ry0, rx1, ry1, ppx, ppy, npw, nph, ids) in enumerate(res):
        bpx, bpy = ppx - (r > 0), ppy - (r > 0)         # precincts in the bands' coordinates
        xcb, ycb = min(c.xcb, bpx), min(c.ycb, bpy)
        for k in range(npw * nph):
            gx, gy = (rx0 >> ppx) + k % npw, (ry0 >> ppy) + k // npw
            rp = len(rp0)
            y, x = max(cy0, (gy << ppy) << (nl - r)), max(cx0, (gx << ppx) << (nl - r))
            for layer in range(c.layers):     # (sorted below into the progression's order)
                key.append(((layer, r, k), (r, layer, k), (r, y, x, layer), (y, x, r, layer),
                            (y, x, r, layer))[c.order] + (layer, rp))
            rp0.append(len(e_band))
            for bi in ids:
                bx0, by0, bx1, by1 = bands[bi][2:6]
                qx0, qy0 = max(bx0, gx << bpx), max(by0, gy << bpy)
                qx1, qy1 = min(bx1, (gx + 1) << bpx), min(by1, (gy + 1) << bpy)
                nw = nh = 0
                if qx1 > qx0 and qy1 > qy0:
                    kx0, ky0 = qx0 >> xcb, qy0 >> ycb
                    nw, nh = _ceil(qx1, 1 << xcb) - kx0, _ceil(qy1, 1 << ycb) - ky0
                e_band.append(bi)
                e_nw.append(nw)
                e_nh.append(nh)
                e_cb0.append(len(cbs))
                e_tt0.append(len(parent))
                for j in range(nh):
                    for i in range(nw):
                        cbs.append([bi, max(qx0, (kx0 + i) << xcb) - bx0,
                                    max(qy0, (ky0 + j) << ycb) - by0,
                                    min(qx1, (kx0 + i + 1) << xcb) - bx0,
                                    min(qy1, (ky0 + j + 1) << ycb) - by0, 0, bands[bi][8],
                                    bx1 - bx0])
                _tree(parent, nw, nh)
            rp1.append(len(e_band))
    key.sort()
    packets = np.array([kk[-2:] for kk in key], np.int64).reshape(-1, 2)
    data = (np.concatenate([d[a:b] for a, b in spans]) if spans else np.zeros(0, np.uint8))
    ncb = len(cbs)
    cb = np.array(cbs, np.int64).reshape(-1, 8)
    band_mb = np.array([bb[6] for bb in bands], np.int64)
    maxmb = int(band_mb.max()) + roi
    maxseg = 3 * maxmb + 2 if c.style & (BYPASS | TERMALL) else 3 * maxmb // 109 + 2
    cb_numbps = np.zeros(ncb, np.int64)
    cb_nseg = np.zeros(ncb, np.int64)
    seg_np = np.zeros((ncb, maxseg), np.int64)
    seg_max = np.zeros((ncb, maxseg), np.int64)
    seg_len = np.zeros((ncb, maxseg), np.int64)
    pieces, npc = _tier2(data, packets, np.array(rp0, np.int64), np.array(rp1, np.int64),
                         np.array(e_band, np.int64), np.array(e_nw, np.int64),
                         np.array(e_nh, np.int64), np.array(e_cb0, np.int64),
                         np.array(e_tt0, np.int64), np.array(parent, np.int64), band_mb,
                         c.style, c.sop, c.eph, cb_numbps, cb_nseg, seg_np, seg_max, seg_len)
    if npc < 0:
        raise ValueError('JPEG2000: its packets run past its data')
    if ncb and (cb_numbps > band_mb[cb[:, 0]]).any():
        raise ValueError('JPEG2000: more bit-planes than its quantisation allows')
    cb[:, 5] = cb_numbps
    buf = _scratch('gather', int(seg_len.sum() + 2 * cb_nseg.sum()) + 2, np.int64, 0xFF)
    seg_at = _gather(data, pieces, npc, cb_nseg, seg_len, buf)
    coef = _scratch('coef', off, np.int64, 0)
    orient = np.array([bb[1] for bb in bands], np.int64)
    if parallel and ncb > 1:
        ctr = np.zeros(1, np.int64)
        _cpu.share(lambda: _blocks_take(ncb, buf, cb, seg_at, seg_np, cb_nseg, orient, roi,
                                        c.style, coef, ctr), ncb)
    else:
        _blocks_serial(ncb, buf, cb, seg_at, seg_np, cb_nseg, orient, roi, c.style, coef)
    img = _synthesise(coef, bands, res, c.rev)
    if signed:
        lo, hi, shift = -(1 << (prec - 1)), (1 << (prec - 1)) - 1, 0
    else:
        lo, hi, shift = 0, (1 << prec) - 1, 1 << (prec - 1)
    _place(img, cx1 - cx0, cy1 - cy0, cx0 - _ceil(x0, xr), cy0 - _ceil(y0, yr), shift, lo,
           hi, out, W)


def _tree(parent, nw, nh):
    """A tag tree of nw x nh leaves: its nodes (the leaves row by row,
    then each level up) appended to parent (each node's, -1 the root)."""
    if nw * nh == 0:
        return
    levels = [(nw, nh)]
    while levels[-1][0] * levels[-1][1] > 1:
        levels.append(((levels[-1][0] + 1) // 2, (levels[-1][1] + 1) // 2))
    start = len(parent)
    starts = []
    for lw, lh in levels:
        starts.append(start)
        start += lw * lh
    for li, (lw, lh) in enumerate(levels):
        for j in range(lh):
            for i in range(lw):
                parent.append(starts[li + 1] + (j // 2) * levels[li + 1][0] + i // 2
                              if li + 1 < len(levels) else -1)


def _synthesise(coef, bands, res, rev):
    """The tile-component's samples (int64) from its subbands'
    coefficients (twice over, as tier-1 leaves them): dequantised, then
    the inverse wavelet, level by level."""
    if rev:
        _halve(coef)                            # every band at once, in place

    def band(bi):
        x0, y0, x1, y1, mb, step, off = bands[bi][2:]
        v = coef[off:off + (x1 - x0) * (y1 - y0)]
        return v if rev else v * (0.5 * step)
    a = band(res[0][8][0])
    for r in range(1, len(res)):
        rx0, ry0, rx1, ry1 = res[r][:4]
        hl, lh, hh = (band(bi) for bi in res[r][8])
        nxt = _scratch(('level', r & 1), (ry1 - ry0) * (rx1 - rx0), a.dtype)
        if nxt.size:
            _interleave(a, hl, lh, hh, nxt, rx0, ry0, rx1, ry1)
            (_idwt53 if rev else _idwt97)(nxt, rx0, ry0, rx1, ry1)
        a = nxt
    return a if rev else np.rint(a).astype(np.int64)


# Each thread's scratch arrays (decoded samples, coefficients, wavelet
# levels, gathered segments), kept between fields: a field's arrays new each
# time were fresh pages, faulted in one at a time -- a process's page faults
# do not run in parallel (Windows: ~0.45 ms each 4 MB on any number of
# threads), so with every core decoding they held them up (RAP, 152k
# points: 1.06 -> 0.85 ms a field on 24 threads, 49.5 -> 47.1 ns a value on
# one). At most _SCRATCH_BYTES a thread; release() lets them all go.
_SCRATCH = threading.local()
_SCRATCH_BYTES = 32 << 20
_SCRATCHES = []                         # every thread's, for release()
_SCRATCH_LOCK = threading.Lock()


def release():
    """Let go of every thread's scratch arrays."""
    with _SCRATCH_LOCK:
        for d in _SCRATCHES:
            d.clear()


def _scratch(key, n, dtype, fill=None):
    """This thread's array `key` of n items, filled with `fill` (if given)."""
    d = getattr(_SCRATCH, 'd', None)
    if d is None:
        d = _SCRATCH.d = {}
        with _SCRATCH_LOCK:
            _SCRATCHES.append(d)
    k = (key, np.dtype(dtype))
    a = d.get(k)
    if a is None or a.size < n:
        a = np.empty(n, dtype)
        d[k] = a
        if sum(x.nbytes for x in d.values()) > _SCRATCH_BYTES:
            d.clear()                           # too much kept: let it all go
    a = a[:n]
    if fill is not None:
        a.fill(fill)
    return a


# --- the encoder: lossless (5/3), one tile, one layer, 64 x 64 code-blocks

@njit(cache=True, nogil=True)
def _byte_out(out, c, bp):
    """(C, CT, position) after a byte out (out[0]: a byte before the data,
    for the coder's carries)."""
    if out[bp] == 0xFF:
        out[bp + 1] = c >> 20
        return c & 0xFFFFF, 7, bp + 1
    if not c & 0x8000000:
        out[bp + 1] = c >> 19
        return c & 0x7FFFF, 8, bp + 1
    out[bp] += 1
    if out[bp] == 0xFF:
        c &= 0x7FFFFFF
        out[bp + 1] = c >> 20
        return c & 0xFFFFF, 7, bp + 1
    out[bp + 1] = c >> 19
    return c & 0x7FFFF, 8, bp + 1


@njit(cache=True, nogil=True)
def _enc(out, a, c, ct, bp, cs, cx, bit):
    """Code one decision: (A, C, CT, position)."""
    t = cs[cx]
    qe = _QE2[t]
    a -= qe
    if bit == t & 1:
        if a & 0x8000:
            return a, c + qe, ct, bp
        if a < qe:
            a = qe
        else:
            c += qe
        cs[cx] = _NM2[t]
    else:
        if a < qe:
            c += qe
        else:
            a = qe
        cs[cx] = _NL2[t]
    while True:
        a <<= 1
        c <<= 1
        ct -= 1
        if ct == 0:
            c, ct, bp = _byte_out(out, c, bp)
        if a & 0x8000:
            return a, c, ct, bp


@njit(cache=True, nogil=True)
def _flush(out, a, c, ct, bp):
    """The coder's last bytes: the data's length."""
    t = c + a
    c |= 0xFFFF
    if c >= t:
        c -= 0x8000
    c, ct, bp = _byte_out(out, c << ct, bp)
    c, ct, bp = _byte_out(out, c << ct, bp)
    if out[bp] != 0xFF:
        bp += 1
    return bp - 1


@njit(cache=True, nogil=True)
def _encode_block(mag, neg, w, h, orient, out):
    """A code-block's passes (mag, neg: its coefficients' magnitudes and
    signs, h x w) into out[1:]: (bytes, bit-planes)."""
    m = 0
    for k in range(w * h):
        m = max(m, mag[k])
    nb = 0
    while m >> nb:
        nb += 1
    if nb == 0:
        return 0, 0
    W = w + 2
    fl = np.zeros((h + 2) * W, np.int32)
    cs = np.zeros(19, np.int64)
    _reset(cs)
    a, c, ct, bq = 0x8000, 0, 12, 0
    out[0] = 0
    for bp in range(nb - 1, -1, -1):
        if bp < nb - 1:
            for k in range(0, h, 4):            # significance propagation
                for i in range(w):
                    p = (k + 1) * W + i + 1
                    if k + 3 < h and _idle(fl, p, W, 0):
                        continue
                    for j in range(k, min(k + 4, h)):
                        f = fl[p]
                        if not (f & (SIG | VIS)) and f & _NBR:
                            o = j * w + i
                            bit = (mag[o] >> bp) & 1
                            a, c, ct, bq = _enc(out, a, c, ct, bq, cs,
                                                _ZC[orient, (f >> 4) & 0xFF], bit)
                            if bit:
                                cx, x = _sign_of(f)
                                a, c, ct, bq = _enc(out, a, c, ct, bq, cs, cx, neg[o] ^ x)
                                _turn(fl, p, W, neg[o])
                            fl[p] |= VIS
                        p += W
            for k in range(0, h, 4):            # refinement
                for i in range(w):
                    p = (k + 1) * W + i + 1
                    if k + 3 < h and _idle(fl, p, W, 1):
                        continue
                    for j in range(k, min(k + 4, h)):
                        f = fl[p]
                        if (f & (SIG | VIS)) == SIG:
                            cx = 16 if f & REF else (15 if f & _NBR else 14)
                            a, c, ct, bq = _enc(out, a, c, ct, bq, cs, cx,
                                                (mag[j * w + i] >> bp) & 1)
                            fl[p] = f | REF
                        p += W
        for k in range(0, h, 4):                # cleanup
            for i in range(w):
                p0 = (k + 1) * W + i + 1
                j0 = k
                if k + 3 < h and _idle(fl, p0, W, 2):
                    continue
                if k + 3 < h and _run(fl, p0, W, False):
                    r = 0
                    while r < 4 and not (mag[(k + r) * w + i] >> bp) & 1:
                        r += 1
                    if r == 4:
                        a, c, ct, bq = _enc(out, a, c, ct, bq, cs, _RL, 0)
                        continue
                    a, c, ct, bq = _enc(out, a, c, ct, bq, cs, _RL, 1)
                    a, c, ct, bq = _enc(out, a, c, ct, bq, cs, _UNI, r >> 1)
                    a, c, ct, bq = _enc(out, a, c, ct, bq, cs, _UNI, r & 1)
                    j = k + r
                    p = p0 + r * W
                    o = j * w + i
                    cx, x = _sign_of(fl[p])
                    a, c, ct, bq = _enc(out, a, c, ct, bq, cs, cx, neg[o] ^ x)
                    _turn(fl, p, W, neg[o])
                    j0 = j + 1
                p = p0 + (j0 - k) * W
                for j in range(j0, min(k + 4, h)):
                    f = fl[p]
                    if not f & (SIG | VIS):
                        o = j * w + i
                        bit = (mag[o] >> bp) & 1
                        a, c, ct, bq = _enc(out, a, c, ct, bq, cs,
                                            _ZC[orient, (f >> 4) & 0xFF], bit)
                        if bit:
                            cx, x = _sign_of(f)
                            a, c, ct, bq = _enc(out, a, c, ct, bq, cs, cx, neg[o] ^ x)
                            _turn(fl, p, W, neg[o])
                    p += W
        _unvisit(fl)
    return _flush(out, a, c, ct, bq), nb


@njit(cache=True, nogil=True)
def _encode_blocks(a, W, blocks, cap, out, lens, nbps, ctr):
    """The code-blocks (blocks: x0, y0, w, h, orientation in the wavelet's
    array a, W wide) taken off ctr[0] until none is left, each into out
    (cap bytes each; _cpu.share's threads)."""
    while True:
        c = _take(ctr, 0)
        if c >= blocks.shape[0]:
            return
        x0, y0, w, h = blocks[c, 0], blocks[c, 1], blocks[c, 2], blocks[c, 3]
        mag = np.empty(w * h, np.int64)
        neg = np.empty(w * h, np.int64)
        for j in range(h):
            for i in range(w):
                v = a[(y0 + j) * W + x0 + i]
                mag[j * w + i] = abs(v)
                neg[j * w + i] = 1 if v < 0 else 0
        buf = np.zeros(cap + 2, np.int64)
        n, nb = _encode_block(mag, neg, w, h, blocks[c, 4], buf)
        lens[c], nbps[c] = n, nb
        out[c * cap:c * cap + n] = buf[1:n + 1]


@njit(cache=True, nogil=True)
def _fdwt53(a, w, h, nl):
    """The forward 5/3, nl levels, in place: each level's low-pass to the
    top left of its region (columns first, then rows)."""
    cw, ch = w, h
    line = np.empty(max(w, h), np.int64)
    for _ in range(nl):
        lo = (ch + 1) >> 1
        for i in range(cw):
            for j in range(ch):
                line[j] = a[j * w + i]
            _fsd53(line, ch)
            for j in range(ch):
                a[((j >> 1) if not j & 1 else lo + (j >> 1)) * w + i] = line[j]
        lo = (cw + 1) >> 1
        for j in range(ch):
            for i in range(cw):
                line[i] = a[j * w + i]
            _fsd53(line, cw)
            for i in range(cw):
                a[j * w + ((i >> 1) if not i & 1 else lo + (i >> 1))] = line[i]
        cw, ch = (cw + 1) >> 1, (ch + 1) >> 1


class _BitWriter:
    """Packet-header bits, an 0xFF byte followed by one of 7 bits."""

    def __init__(self):
        self.out = bytearray()
        self.buf, self.ct = 0, 8

    def put(self, v, n=1):
        for k in range(n - 1, -1, -1):
            if self.ct == 0:
                self._byte()
            self.ct -= 1
            self.buf |= ((v >> k) & 1) << self.ct

    def _byte(self):
        self.buf = (self.buf << 8) & 0xFFFF
        self.ct = 7 if self.buf == 0xFF00 else 8
        self.out.append(self.buf >> 8)

    def flush(self):
        self._byte()
        if self.ct == 7:
            self._byte()
        return bytes(self.out)


def _tag_encode(bw, nodes, leaf, threshold):
    """Tag-tree encoding: nodes[k] = [value, low, known, parent]."""
    path = []
    k = leaf
    while k >= 0:
        path.append(k)
        k = nodes[k][3]
    lo = 0
    for k in reversed(path):
        nd = nodes[k]
        if lo > nd[1]:
            nd[1] = lo
        else:
            lo = nd[1]
        while lo < threshold:
            if lo >= nd[0]:
                if not nd[2]:
                    bw.put(1)
                    nd[2] = 1
                break
            bw.put(0)
            lo += 1
        nd[1] = lo


def _tree_nodes(values, nw, nh):
    parent = []
    _tree(parent, nw, nh)
    nodes = [[1 << 30, 0, 0, p] for p in parent]
    for k, v in enumerate(values):
        nodes[k][0] = v
    for k in range(len(nodes)):                 # each node the least of its children
        p = nodes[k][3]
        if p >= 0:
            nodes[p][0] = min(nodes[p][0], nodes[k][0])
    return nodes


def encode(x, w, h, prec):
    """A lossless JPEG2000 codestream of unsigned samples x (h rows of w,
    each below 2^prec; prec 1 to 31)."""
    if not 1 <= prec <= 31:
        raise ValueError('JPEG2000 of %d bits a sample: 1 to 31' % prec)
    a = np.ascontiguousarray(x, np.int64).ravel() - (1 << (prec - 1))
    if a.size != w * h:
        raise ValueError('%d samples for %d x %d' % (a.size, w, h))
    nl = 0
    while nl < 5 and min(w, h) >> (nl + 1):
        nl += 1
    _fdwt53(a, w, h, nl)
    guard = max(2, prec + 4 - 31)               # exponents fit 5 bits: guard bits take the rest
    # the bands (r, orientation, x0, y0, width, height in the wavelet's array), by resolution
    dims = [(w, h)]
    for _ in range(nl):
        dims.append(((dims[-1][0] + 1) >> 1, (dims[-1][1] + 1) >> 1))
    bands = [(0, 0, 0, 0) + dims[nl]]
    for r in range(1, nl + 1):
        lw, lh = dims[nl - r + 1]
        fw, fh = dims[nl - r]
        bands += [(r, 1, lw, 0, fw - lw, lh), (r, 2, 0, lh, lw, fh - lh),
                  (r, 3, lw, lh, fw - lw, fh - lh)]
    blocks, where = [], []
    for bi, (r, o, bx, by, bw_, bh) in enumerate(bands):
        nbx, nby = _ceil(bw_, 64), _ceil(bh, 64)
        where.append((len(blocks), nbx, nby))
        for j in range(nby):
            for i in range(nbx):
                blocks.append((bx + 64 * i, by + 64 * j, min(64, bw_ - 64 * i),
                               min(64, bh - 64 * j), o))
    blocks = np.array(blocks, np.int64).reshape(-1, 5)
    cap = 64 * 64 * (prec + 4) // 8 + 64
    out = np.zeros(max(len(blocks), 1) * cap, np.int64)
    lens = np.zeros(len(blocks), np.int64)
    nbps = np.zeros(len(blocks), np.int64)
    if len(blocks):
        ctr = np.zeros(1, np.int64)
        _cpu.share(lambda: _encode_blocks(a, w, blocks, cap, out, lens, nbps, ctr), len(blocks))
    exps = [prec + (0, 1, 1, 2)[b[1]] + 2 - guard for b in bands]
    mbs = [guard + e - 1 for e in exps]
    body = bytearray()
    for r in range(nl + 1):                     # one packet a resolution (LRCP, one layer)
        bw = _BitWriter()
        ids = [0] if r == 0 else [1 + 3 * (r - 1), 2 + 3 * (r - 1), 3 + 3 * (r - 1)]
        if not any(lens[where[bi][0]:where[bi][0] + where[bi][1] * where[bi][2]].any()
                   for bi in ids):
            bw.put(0)
            body += bw.flush()
            continue
        bw.put(1)
        data = []
        for bi in ids:
            c0, nbx, nby = where[bi]
            n = nbx * nby
            if n == 0:
                continue
            nb = nbps[c0:c0 + n]
            if (nb > mbs[bi]).any():
                raise ValueError('JPEG2000: coefficients past their bit-planes')
            inc = _tree_nodes([0 if v else 1 for v in nb], nbx, nby)
            zb = _tree_nodes([mbs[bi] - v for v in nb], nbx, nby)
            for k in range(n):
                _tag_encode(bw, inc, k, 1)
                if not nb[k]:
                    continue
                _tag_encode(bw, zb, k, 1 << 30)
                passes = 3 * int(nb[k]) - 2
                if passes == 1:
                    bw.put(0)
                elif passes == 2:
                    bw.put(2, 2)
                elif passes <= 5:
                    bw.put(0xC | (passes - 3), 4)
                elif passes <= 36:
                    bw.put(0x1E0 | (passes - 6), 9)
                else:
                    bw.put(0xFF80 | (passes - 37), 16)
                ln = int(lens[c0 + k])
                fl2 = (passes).bit_length() - 1
                lb = max(3, ln.bit_length() - fl2)
                bw.put((1 << (lb - 3 + 1)) - 2, lb - 3 + 1)     # Lblock's increase, then 0
                bw.put(ln, lb + fl2)
                data.append(out[(c0 + k) * cap:(c0 + k) * cap + ln])
        body += bw.flush()
        for dd in data:
            body += dd.astype(np.uint8).tobytes()
    siz = struct.pack('>HHHIIIIIIIIHBBB', 0xFF51, 41, 0, w, h, 0, 0, w, h, 0, 0, 1, prec - 1, 1, 1)
    cod = struct.pack('>HHBBHBBBBBB', 0xFF52, 12, 0, 0, 1, 0, nl, 4, 4, 0, 1)
    qcd = struct.pack('>HHB', 0xFF5C, 3 + 3 * nl + 1, guard << 5) + bytes(e << 3 for e in exps)
    sot = struct.pack('>HHHIBB', 0xFF90, 10, 0, 12 + 2 + len(body), 0, 1)
    return b'\xff\x4f' + siz + cod + qcd + sot + b'\xff\x93' + bytes(body) + b'\xff\xd9'
