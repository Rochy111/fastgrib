"""Partial reads: the values of some points of a field without decoding
the rest. Simple packing (5.0) and IEEE (5.4) put each value at a place
its index gives -- nbits a value, or 4 or 8 bytes -- so a point's value is
read where it lies: its place in read()'s layout turned into its place in
the file (the grid's scanning, undone), then, under a bitmap, into its
value's index (the bits set before it). The arithmetic is the decoders'
own, so each value is the same float32 a whole read gives.

at(), regrid() and read(box=...) use it when they need a small part of a
field. A remote field's message is fetched in part too: its sections up
to 7's start, then only the byte ranges of the values wanted, into a
buffer of the message's length (the rest left zero, never read) that the
scanner takes as the whole message.
"""
import struct
import threading

import numpy as np

from . import _cpu as G
from ._octets import int_power, octets, sm

DIRECT = (0, 4)                                 # simple, IEEE: each value where its index says
PREFIX = (2, 3, 41, 42)                         # complex, PNG, CCSDS: read in order, so up to
REGION = (40,)                                  # JPEG2000: the code-blocks reaching them
PACKINGS = DIRECT + PREFIX + REGION             # the last value wanted, and no further
FRACTION = 8                                    # partial when wanting below 1/8 of the points
PREFIX_SHARE = 0.8                              # a prefix read when it decodes below this share
_POP = np.array([bin(k).count('1') for k in range(256)], np.int64)
_INVERSE = {}                                   # grid -> read() layout index -> file position
_LOCK = threading.Lock()


def can(r):
    """Whether record r's values can be read point by point."""
    g = getattr(r, 'grid', None)
    return (g is not None and getattr(r, 'packing', -1) in PACKINGS and
            g.template not in (50, 51, 52, 53) and g.ni * g.nj > 0)


def worth(r, n):
    """Whether reading n of r's points by themselves beats decoding all."""
    return can(r) and n * FRACTION < r.grid.ni * r.grid.nj


def _inverse(arr, r):
    """The file position of each point of r's grid, by its index in
    read()'s layout; None where the two are the same (rows south to north,
    west to east: scanning 0x40). Kept a grid."""
    g = r.grid
    if g.scan & 0xF0 == 0x40 and not g.pl:
        return None
    hit = _INVERSE.get(g)
    if hit is None:
        import copy
        full = copy.copy(r)
        full.bitmap = -1
        lay = G._defined(arr, full)             # the layout index of each file position
        hit = np.empty(lay.size, np.int64)
        hit[lay] = np.arange(lay.size, dtype=np.int64)
        with _LOCK:
            if len(_INVERSE) > 32:
                _INVERSE.clear()
            _INVERSE[g] = hit
    return hit


def indices(arr, r, lay):
    """(each point's value index, whether it has a value) of the points
    `lay` (indices in read()'s layout, any shape) of record r in arr."""
    lay = np.asarray(lay, np.int64)
    inv = _inverse(arr, r)
    fp = lay if inv is None else inv[lay]
    if r.bitmap < 0:
        return fp, np.ones(fp.shape, bool)
    g = r.grid
    n = g.ni * g.nj
    bm = np.asarray(arr[r.bitmap:r.bitmap + (n + 7) // 8], np.uint8)
    byte, bit = fp >> 3, fp & 7
    has = (bm[byte] >> (7 - bit)) & 1 == 1
    before = _prefix(bm)                                               # bits set in earlier bytes
    within = _POP[bm[byte] & ((0xFF00 >> bit) & 0xFF)]                 # and in this one, before it
    return before[byte] + within, has


_PREFIX = []                                    # [(bitmap bytes, its running counts)], newest last


def _prefix(bm):
    """The bits set in a bitmap's bytes before each byte, kept for the last
    few bitmaps (the fields of a file mostly share one, each given anew:
    GFS's land-sea mask, 41 times) -- a hit found by its bytes compared."""
    for cand, pre in reversed(_PREFIX):
        if cand.size == bm.size and np.array_equal(cand, bm):
            return pre
    pre = np.empty(bm.size, np.int64)
    pre[0] = 0
    np.cumsum(_POP[bm[:-1]], out=pre[1:])
    with _LOCK:
        _PREFIX.append((bm.copy(), pre))
        del _PREFIX[:-4]
    return pre


def _params(arr, r):
    """(template, nbits or IEEE bytes, R, 2^E, 10^-D, the values' first byte,
    how many) of record r, as the decoders take them."""
    b = octets(arr)
    s5 = r.s5
    drt = struct.unpack_from('>H', b, s5 + 9)[0]
    nval = struct.unpack_from('>I', b, s5 + 5)[0]
    data = r.s7 + 5
    if drt == 4:
        return drt, 4 if b[s5 + 11] == 1 else 8, 0.0, 1.0, 1.0, data, nval
    ref0 = struct.unpack_from('>f', b, s5 + 11)[0]
    e = sm(struct.unpack_from('>H', b, s5 + 15)[0], 2)
    d = sm(struct.unpack_from('>H', b, s5 + 17)[0], 2)
    return drt, b[s5 + 19], ref0, int_power(2.0, e), int_power(10.0, -d), data, nval


def spans(arr, r, k):
    """[(start, stop)] byte ranges of r's message (from its start) that
    hold the values of indices k -- what a remote read fetches."""
    drt, nb, _, _, _, data, _ = _params(arr, r)
    k = np.unique(np.asarray(k, np.int64))
    if drt == 4:
        lo, hi = data + k * nb, data + (k + 1) * nb
    elif nb == 0:
        return []
    else:
        lo, hi = data + ((k * nb) >> 3), data + (((k + 1) * nb + 7) >> 3)
    return [(int(a) - r.at, int(b) - r.at) for a, b in zip(lo, hi)]


def values_many(jobs, undef=np.float32(np.nan)):
    """[values(arr, r, lay, prefix=...)] of jobs [(arr, r, lay[, prefix])]:
    those read in order (complex, PNG, CCSDS) side by side, a core each
    (the kernels release Python's lock)."""
    crowd = len(jobs)

    def one(j):
        return values(j[0], j[1], j[2], undef, crowd, j[3] if len(j) > 3 else False,
                      whole=False)
    if crowd > 1 and any(getattr(j[1], 'packing', -1) in PREFIX + REGION for j in jobs):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(min(G.threads(), crowd), initializer=G._fast_thread) as ex:
            out = list(ex.map(one, jobs))
    else:
        out = [one(j) for j in jobs]
    # those worth no prefix read: decoded whole, together, as fields() decodes
    # (from inside the pool above each split its field across the cores, the
    # splits contending for them: HRRR, a point near its end, 330 ms against 120)
    rest = {}
    for i, (j, o) in enumerate(zip(jobs, out)):
        if o is None:
            rest.setdefault((id(j[0]), j[1].grid), []).append(i)
    for (_, g), ids in rest.items():
        arr = jobs[ids[0]][0]
        cols = np.arange(g.ni * g.nj, dtype=np.int64)
        recs = [jobs[i][1] for i in ids]
        for k, block in G.decode_iter(arr, recs, cols, undef=undef):
            for t in range(len(block)):
                out[ids[k + t]] = block[t][np.asarray(jobs[ids[k + t]][2], np.int64)]
    return out


def _complex_params(arr, r):
    p = np.zeros(18, np.int64)
    f = np.zeros(4, np.float64)
    G._params(arr, r, p, f)
    return p


def data_start(arr, r):
    """The byte (in arr) where a complex-packed record's packed values
    begin, past its group headers -- known from section 5 alone."""
    p = _complex_params(arr, r)
    return int(G._layout(np.asarray(arr, np.uint8), p)[3])


def prefix_stop(arr, r, n_need):
    """The byte (in arr) after the packed values of the groups holding a
    complex-packed record's first n_need values (its group headers in arr);
    None when they do not add up."""
    p = _complex_params(arr, r)
    if p[G._NG] == 0:
        return int(G._layout(np.asarray(arr, np.uint8), p)[3])
    ng, cnt, bit = G.prefix_groups(np.asarray(arr, np.uint8), p, n_need)
    return None if cnt < n_need else int((bit + 7) // 8)


def need(arr, r, lay):
    """How many of r's values, from its first, hold those of the points
    lay: what a read of them in order (complex, PNG, CCSDS) decodes."""
    k, has = indices(arr, r, lay)
    return int(k[has].max()) + 1 if has.any() else 0


def prefix_worth(r, n_need, crowd=1):
    """Whether decoding r's first n_need values on one core beats decoding
    it whole -- on every core when it is read alone (crowd: the fields
    read at once, a core each)."""
    total = r.grid.ni * r.grid.nj
    return n_need * max(1.0, G.threads() / max(1, crowd)) < PREFIX_SHARE * total


_SCRATCH = threading.local()


def _scratch(n):
    """A float32 row of at least n values, this thread's, reused: a fresh
    one a field is faulted in page by page, which no thread count speeds
    up (HRRR's 1.9M-point fields, 173 at once: 312 ms against 117 decoding
    them whole into the reused memory fields() keeps)."""
    row = getattr(_SCRATCH, 'row', None)
    if row is None or row.size < n:
        row = _SCRATCH.row = np.empty(max(n, 1 << 16), np.float32)
    return row[:n]


def _prefix_row(arr, r, n_need, undef):
    """r's first n_need values (at least), decoded in order as a whole read
    decodes them, or None when its packing did not add up."""
    p = np.zeros(18, np.int64)
    f = np.zeros(4, np.float64)
    f[3] = undef
    G._params(arr, r, p, f)
    p[G._DEST], p[G._FILL] = -1, 0
    dest = np.zeros(1, np.int32)
    drt = int(p[G._DRT])
    if drt in (2, 3):
        if p[G._NG] == 0:                       # a constant field
            return np.full(n_need, np.float32(f[0] * f[2]), np.float32)
        ng, cnt, _ = G.prefix_groups(np.asarray(arr, np.uint8), p, n_need)
        if cnt < n_need:
            return None
        row = _scratch(cnt)                     # (gathered from before this thread's next)
        return row if G.decode_prefix(np.asarray(arr, np.uint8), p, f, dest, row, ng) == 0 \
            else None
    from . import _images
    q = p.copy()
    q[G._NVAL] = n_need
    row = _scratch(n_need)
    data = np.asarray(arr, np.uint8)[p[G._DATA]:p[G._END]]
    nbits = int(p[G._NBITS])
    if drt == 42:
        if nbits == 0:
            return np.full(n_need, np.float32(f[0] * f[2]), np.float32)
        nb = (nbits + 7) // 8 * 8
        got = _images._aec(data, nbits, int(p[G._MGMT]), int(p[G._REFW]), int(p[G._BITSW]),
                           nb <= 25, q, f, dest, row)
        return row if got >= n_need else None
    if nbits == 0:                              # PNG of a constant field
        return np.full(n_need, np.float32(f[0] * f[2]), np.float32)
    stream = _images.png_prefix(data, int(p[G._NVAL]), n_need, nbits)
    q[G._DRT], q[G._DATA], q[G._END] = 0, 0, stream.size    # its samples: simple packing's
    return row if G._decode_one(stream, q, f, dest, row) == 0 else None


def reaches(arr, r, n_need, have):
    """Whether a PNG or CCSDS record's bytes in arr before `have` (from
    arr's start; the rest of the message not fetched, zeros) hold its first
    n_need values -- a remote read fetching on until they do. Each says
    where a value's bits end only once decoded up to it: CCSDS decoded that
    far, its bits read counted; PNG's stream inflated from the bytes in
    hand alone, its rows enough or not."""
    p = np.zeros(18, np.int64)
    f = np.zeros(4, np.float64)
    G._params(arr, r, p, f)
    p[G._DEST], p[G._FILL] = -1, 0
    start, end = int(p[G._DATA]), int(p[G._END])
    nbits = int(p[G._NBITS])
    if nbits == 0 or n_need == 0:
        return True
    if have >= end:
        return True
    if int(p[G._DRT]) == 42:
        from . import _images
        q = p.copy()
        q[G._NVAL] = n_need
        got = _images._aec(np.asarray(arr, np.uint8)[start:end], nbits, int(p[G._MGMT]),
                           int(p[G._REFW]), int(p[G._BITSW]), nbits <= 25, q, f,
                           np.zeros(1, np.int32), _scratch(n_need))
        return got >= n_need and start + (int(q[G._EXTRA]) + 7) // 8 <= have
    import zlib
    b = bytes(np.asarray(arr, np.uint8)[start:have])
    if b[:8] != b'\x89PNG\r\n\x1a\n':
        return False
    pos, idat, w, h, bpp = 8, [], 0, 0, 0
    while pos + 8 <= len(b):                    # the chunks in hand (the last maybe cut short)
        ln = int.from_bytes(b[pos:pos + 4], 'big')
        kind = b[pos + 4:pos + 8]
        if kind == b'IHDR' and pos + 8 + 13 <= len(b):
            body = b[pos + 8:pos + 21]
            w, h = int.from_bytes(body[0:4], 'big'), int.from_bytes(body[4:8], 'big')
            bpp = {0: 1, 2: 3, 4: 2, 6: 4}.get(body[9], 1) * body[8]
        elif kind == b'IDAT':
            idat.append(b[pos + 8:pos + 8 + ln])
        elif kind == b'IEND':
            return True
        pos += 12 + ln
    if not w or not bpp:
        return False
    rowbytes = (w * bpp + 7) // 8
    rows = min(h, -(-(n_need * nbits + 7) // 8 // rowbytes) + 1)
    want = rows * (rowbytes + 1)
    try:
        raw = zlib.decompressobj().decompress(b''.join(idat), want)
    except zlib.error:
        return False
    return len(raw) >= want


def _region(arr, r, lay, undef, crowd):
    """values() of a JPEG2000 field: its codestream's packets parsed, then
    only the code-blocks whose coefficients reach the samples wanted
    decoded, the inverse wavelet over windows around them (a point of
    HRRR: 5 ms, against 68 for the whole on a core; a few scattered all
    over the field cost what the whole does) -- on every core when read
    alone. The samples made floats as _images._ints makes them."""
    from . import _jpeg2000
    k, has = indices(arr, r, lay)
    p = np.zeros(18, np.int64)
    f = np.zeros(4, np.float64)
    G._params(arr, r, p, f)
    out = np.full(k.shape, undef, np.float32)
    if not has.any():
        return out
    if p[G._NBITS] == 0:                        # a constant field
        out[has] = np.float32(f[0] * f[2])
        return out
    kk = k[has]
    data = np.asarray(arr, np.uint8)[p[G._DATA]:p[G._END]]
    try:
        ints = _jpeg2000.decode(data, parallel=crowd == 1, expect=int(p[G._NVAL]),
                                scratch=True, want=kk)
    except ValueError:
        raise
    except Exception as e:                      # (as _images.decode words it)
        raise ValueError('damaged JPEG2000 data: %s: %s' % (type(e).__name__, e))
    x = ints[kk].astype(np.float64)
    out[has] = ((f[0] + f[1] * x) * f[2]).astype(np.float32)
    return out


def values(arr, r, lay, undef=np.float32(np.nan), crowd=1, prefix=False, whole=True):
    """float32 values of record r at the points lay (read() layout indices,
    any shape): each as decoding the whole field gives it; undef where the
    field has none. Simple and IEEE: read where they lie; complex, PNG and
    CCSDS: decoded up to the last of them where that is worth it (crowd:
    the fields read at once; prefix: always -- a remote message fetched
    only that far), else the field decoded whole (whole=False: None
    instead, for the caller to decode with others)."""
    drt = getattr(r, 'packing', -1)
    if drt in REGION:
        return _region(arr, r, lay, undef, crowd)
    if drt in PREFIX:
        k, has = indices(arr, r, lay)
        n_need = int(k[has].max()) + 1 if has.any() else 0
        row = (_prefix_row(arr, r, n_need, undef) if prefix or prefix_worth(r, n_need, crowd)
               else None)
        if row is None and prefix:
            raise ValueError('%r: damaged -- its packed values did not add up' % r)
        if row is None and not whole:
            return None
        if row is None:                         # whole, as read() decodes it
            g = r.grid
            whole = G.decode(arr, [r], np.arange(g.ni * g.nj, dtype=np.int64),
                             undef=undef)[0]
            return whole[np.asarray(lay, np.int64)]
        out = np.full(k.shape, undef, np.float32)
        out[has] = row[k[has]]
        return out
    k, has = indices(arr, r, lay)
    drt, nb, r0, f2, f10, data, nval = _params(arr, r)
    if drt == 0 and nb > 32:                    # (the decoders' own error: not supported)
        g = r.grid
        return G.decode(arr, [r], np.arange(g.ni * g.nj, dtype=np.int64),
                        undef=undef)[0][np.asarray(lay, np.int64)]
    out = np.full(k.shape, undef, np.float32)
    k = np.where(has & (k < nval), k, 0)
    buf = np.asarray(arr, np.uint8)
    if drt == 4:
        at = data + k * nb
        raw = np.zeros(k.shape + (nb,), np.uint8)
        for j in range(nb):
            raw[..., j] = buf[at + j]
        v = raw.view('>f4' if nb == 4 else '>f8')[..., 0].astype(np.float64)
        y = v.astype(np.float32)
        y = np.where(v == v, y, undef)
    elif nb == 0:                               # a constant field
        y = np.full(k.shape, np.float32(r0 * f10), np.float32)
    else:
        bit = data * 8 + k * nb
        byte = bit >> 3
        word = np.zeros(k.shape, np.uint64)
        top = buf.size
        for j in range(5):                      # 40 bits hold any 32 starting anywhere in a byte
            q = byte + j
            word = (word << np.uint64(8)) | np.where(q < top, buf[np.minimum(q, top - 1)],
                                                     0).astype(np.uint64)
        x = (word >> (np.uint64(40) - (bit & 7).astype(np.uint64) - np.uint64(nb))) & \
            np.uint64((1 << nb) - 1)
        x = x.astype(np.int64)
        xf = x.astype(np.float32).astype(np.float64) if nb <= 25 else x.astype(np.float64)
        y = ((r0 + f2 * xf) * f10).astype(np.float32)   # the decoders' order of operations
    out[has] = y[has]
    return out
