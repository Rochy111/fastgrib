"""Data templates whose values come as an image or a coded stream: 5.40
(JPEG2000), 5.41 (PNG) and 5.42 (CCSDS adaptive entropy coding, as libaec
writes it -- ECMWF's open data). Each decodes to the packed integers, then
to floats with wgrib2's arithmetic: PNG and CCSDS through its simple
unpacking (the integers as a bitstream of their own width), JPEG2000
straight, (R + X 2^E) 10^-D.

All three are decoded here, by no other library: PNG with zlib, CCSDS
and JPEG2000 (_jpeg2000: any Part-1 codestream, to 31 bits) in numba.
"""
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ._jit import njit

from . import _cpu as G
from ._octets import clz64 as _clz64

UNDEF_OK = 0


def decode(arr, p, f, dest, row, alone=False):
    """A record of template 5.40/41/42 into row; _decode_one's status.
    alone: the only record of its call (a JPEG2000 one then decodes on
    threads of its own)."""
    drt, nbits, n = int(p[G._DRT]), int(p[G._NBITS]), int(p[G._NVAL])
    q = p.copy()
    q[G._DRT] = 0
    if drt == 4:                                    # IEEE floats, 32 or 64 bits
        nb = nbits // 8
        data = arr[p[G._DATA]:p[G._END]]
        if data.size < n * nb:
            return 1
        vals = np.frombuffer(data[:n * nb], '>f4' if nb == 4 else '>f8').astype(np.float64)
        return _floats(vals, p, f, dest, row)
    if drt == 200:                                  # run-length (JMA)
        return _runlength(arr, p, f, dest, row)
    if drt in (50, 51):                             # spherical harmonics' coefficients
        from . import _spectral
        vals = _spectral.unpack(arr, int(p[G._REFL]), int(p[G._DATA]) - 5, int(p[G._END]),
                                int(p[G._NG]), int(p[G._REFW]), int(p[G._BITSW]))
        return _floats(vals, p, f, dest, row)
    if nbits == 0:                                  # a constant field
        return G._decode_one(arr, q, f, dest, row)
    data = arr[p[G._DATA]:p[G._END]]
    if drt == 41:
        try:
            if alone and n >= PNG_PIPE:
                code = _png_pipelined(data, p, f, dest, row)
                if code is not None:
                    return code
            stream = G._u8(_png(data, n))
        except (zlib.error, ValueError, IndexError) as e:
            raise ValueError('damaged PNG data: %s' % e)
        q[G._DATA], q[G._END] = 0, stream.size
        return G._decode_one(stream, q, f, dest, row)
    if drt == 42:
        nb = (nbits + 7) // 8 * 8                   # wgrib2 unpacks bytes of samples
        got = _aec(data, nbits, int(p[G._MGMT]), int(p[G._REFW]), int(p[G._BITSW]),
                   nb <= 25, p, f, dest, row)
        return 0 if got >= n else 1
    try:
        ints = _jpeg2000(data, n, 0 if alone else 1)
    except ValueError:
        raise
    except Exception as e:                          # (IndexError ...: a damaged codestream)
        raise ValueError('damaged JPEG2000 data: %s: %s' % (type(e).__name__, e))
    if ints.size != n:
        return 1
    return _ints(ints, False, p, f, dest, row)          # (int64 already: no copy)


@njit(cache=True, nogil=True)
def _ints(ints, widen, p, f, dest, row):
    """Packed integers to wgrib2's floats into their columns: (R + X 2^E)
    10^-D, X widened through float32 first where wgrib2's simple
    unpacking does (up to 25 bits)."""
    usedest = p[G._DEST] >= 0
    if p[G._FILL]:
        row[:] = np.float32(f[3])
    r0, f2, f10 = f[0], f[1], f[2]
    for k in range(ints.size):
        x = np.float64(np.float32(ints[k])) if widen else np.float64(ints[k])
        G._put(row, dest, usedest, k, np.float32((r0 + f2 * x) * f10))
    return 0


@njit(cache=True, nogil=True)
def _floats(vals, p, f, dest, row):
    """IEEE values into their columns, as float32 (wgrib2: NaN is
    undefined)."""
    usedest = p[G._DEST] >= 0
    undefv = np.float32(f[3])
    if p[G._FILL]:
        row[:] = undefv
    for k in range(vals.size):
        v = vals[k]
        G._put(row, dest, usedest, k, np.float32(v) if v == v else undefv)
    return 0


@njit(cache=True, nogil=True)
def _runlength(buf, p, f, dest, row):
    """Run-length packing (5.200, JMA's), wgrib2's way: nbits-wide codes,
    a code up to MV a level (0: undefined, else the level table's value,
    10^-D scaled), each followed by any number of codes above MV, the
    digits (base 2^nbits - 1 - MV, least first) of its repeat count less
    one. 1 when the codes do not add up to the field."""
    n, nbits, mv, mvl = p[G._NVAL], p[G._NBITS], p[G._MGMT], p[G._EXTRA]
    usedest = p[G._DEST] >= 0
    undefv = np.float32(f[3])
    if p[G._FILL]:
        row[:] = undefv
    nvals = ((p[G._END] - p[G._DATA]) * 8) // nbits
    if nvals == 0 or mv == 0:
        for k in range(n):
            G._put(row, dest, usedest, k, undefv)
        return 0
    levels = np.empty(mvl + 1, np.float32)
    levels[0] = undefv
    for i in range(1, mvl + 1):
        q = p[G._REFL] + 2 * (i - 1)
        raw = (np.int64(buf[q]) << 8) | np.int64(buf[q + 1])
        v = -(raw & 0x7FFF) if raw & 0x8000 else raw              # sign and magnitude
        levels[i] = np.float32(v * f[2])
    span = (1 << nbits) - 1 - mv
    if span <= 0:
        return 1
    bit = p[G._DATA] * 8
    i = j = 0
    while i < nvals:
        v = G._rd(buf, bit + i * nbits, nbits)
        i += 1
        if v > mv:
            return 1
        reps, factor = 1, 1
        while i < nvals:
            c = G._rd(buf, bit + i * nbits, nbits)
            if c <= mv:
                break
            reps += factor * (c - mv - 1)
            factor *= span
            i += 1
            if reps > n or factor > n * span:           # past the field: damaged
                return 1
        if j + reps > n:
            return 1
        y = levels[v]
        for k in range(reps):
            G._put(row, dest, usedest, j + k, y)
        j += reps
    return 0 if j == n else 1


# -- JPEG2000

def _jpeg2000(data, n, threads=1):
    """A JPEG2000 codestream's n samples, in its own precision (int64);
    threads 0: its code-blocks on every core."""
    from . import _jpeg2000
    return _jpeg2000.decode(data, parallel=threads == 0, expect=n, scratch=True)


# -- PNG

PNG_PIPE = 1 << 22              # values: a PNG field this big, read alone, decodes as a pipeline
PNG_BLOCK = 1 << 20             # bytes of image inflated a step (MRMS: 1 MB best of 0.25-8)


def _png_pipelined(data, p, f, dest, row):
    """A big PNG field's values into row, inflate overlapped with unfilter
    (a thread each: zlib and numba let go of the GIL) and the rows
    converted on every core as they come (MRMS, 24.5M points: 150 ms
    against 75, zlib alone 60). None when its samples are not whole bytes
    a value, or not a value a pixel -- the serial path then."""
    w, h, bpp, idat = _png_parse(bytes(data))
    n, nbits = int(p[G._NVAL]), int(p[G._NBITS])
    if bpp % 8 or bpp != nbits or w * h != n:
        return None
    rb, vb = w * bpp // 8, bpp // 8
    raw = np.empty(h * (rb + 1), np.uint8)
    out = np.empty(h * rb, np.uint8)
    step = max(1, PNG_BLOCK // (rb + 1))
    blocks = [(y, min(h, y + step)) for y in range(0, h, step)]
    z = zlib.decompressobj()
    feed = [b''.join(idat)]

    def inflate(k):
        y0, y1 = blocks[k]
        want, got = (y1 - y0) * (rb + 1), 0
        while got < want:
            part = z.decompress(feed[0], want - got)
            feed[0] = z.unconsumed_tail
            if not part:
                raise ValueError('PNG stream ends early')
            raw[y0 * (rb + 1) + got:y0 * (rb + 1) + got + len(part)] = np.frombuffer(part, np.uint8)
            got += len(part)

    if p[G._FILL]:
        row[:] = np.float32(f[3])
    seen = G._u8(raw)                           # read-only, as _png's: one compiled _unfilter
    conv, pool = [], G._pool()
    try:
        with ThreadPoolExecutor(1) as ex:
            nxt = ex.submit(inflate, 0)
            for k, (y0, y1) in enumerate(blocks):
                nxt.result()
                if k + 1 < len(blocks):
                    nxt = ex.submit(inflate, k + 1)
                if _unfilter(seen, out, y0, y1, rb, max(1, vb)):
                    raise ValueError('corrupt PNG stream')
                q = max(1, (y1 - y0) // 4)
                for a in range(y0, y1, q):
                    conv.append(pool.submit(_convert, out, a * w, min(y1, a + q) * w,
                                            vb, nbits <= 25, p, f, dest, row))
    finally:                                    # none still writing into row once we return
        for c in conv:
            c.result()
    return 0


@njit(cache=True, nogil=True)
def _convert(img, k0, k1, vb, widen, p, f, dest, row):
    """Values k0 to k1 of an unfiltered image of vb-byte big-endian samples
    into their columns, as _simple converts them."""
    usedest = p[G._DEST] >= 0
    r0, f2, f10 = f[0], f[1], f[2]
    for k in range(k0, k1):
        x = 0
        for j in range(vb):
            x = (x << 8) | np.int64(img[np.uint64(k * vb + j)])
        xf = np.float64(np.float32(x)) if widen else np.float64(x)
        G._put(row, dest, usedest, k, np.float32((r0 + f2 * xf) * f10))


def _png_parse(b):
    """(width, height, bits a pixel, [IDAT bodies]) of a PNG stream."""
    if b[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('not a PNG stream')
    pos, idat = 8, []
    w = h = depth = ctype = 0
    while pos + 8 <= len(b):
        ln = int.from_bytes(b[pos:pos + 4], 'big')
        kind = b[pos + 4:pos + 8]
        body = b[pos + 8:pos + 8 + ln]
        if kind == b'IHDR':
            w, h = int.from_bytes(body[0:4], 'big'), int.from_bytes(body[4:8], 'big')
            depth, ctype, interlace = body[8], body[9], body[12]
            if interlace:
                raise ValueError('interlaced PNG not supported')
        elif kind == b'IDAT':
            idat.append(body)
        elif kind == b'IEND':
            break
        pos += 12 + ln
    chans = {0: 1, 2: 3, 4: 2, 6: 4}.get(ctype)
    if chans is None:
        raise ValueError('PNG colour type %d not supported' % ctype)
    return w, h, chans * depth, idat


def _png(data, n):
    """A PNG's samples as rows of bytes, filters undone (what wgrib2's
    decoder hands its simple unpacking); n: the field's values, which the
    image must hold and not wildly exceed (a damaged header claims any
    size: nothing is allocated for one)."""
    w, h, bpp, idat = _png_parse(bytes(data))
    rowbytes = (w * bpp + 7) // 8
    if w * h < n or w * h > 2 * n + w + 64:
        raise ValueError('a %d x %d image for %d values' % (w, h, n))
    want = h * (rowbytes + 1)
    raw = np.frombuffer(zlib.decompressobj().decompress(b''.join(idat), want), np.uint8)
    if raw.size < want:
        raise ValueError('corrupt PNG stream')
    out = np.empty(h * rowbytes, np.uint8)
    if _unfilter(raw, out, 0, h, rowbytes, max(1, bpp // 8)):
        raise ValueError('corrupt PNG stream')
    return out


@njit(cache=True, nogil=True)
def _unfilter(raw, out, y0, y1, rb, bpp):
    """PNG's per-row filters undone for rows y0 to y1 (those before them
    done), a loop per filter kind (one checked a row, not a byte: None and
    Up vectorise)."""
    for y in range(y0, y1):
        kind = raw[y * (rb + 1)]
        src = raw[y * (rb + 1) + 1:(y + 1) * (rb + 1)]
        cur = out[y * rb:(y + 1) * rb]
        if y > 0:
            up = out[(y - 1) * rb:y * rb]
        else:
            up = np.zeros(rb, np.uint8)
        if kind == 0:
            cur[:] = src
        elif kind == 1:
            for i in range(rb):
                cur[i] = src[i] + (cur[i - bpp] if i >= bpp else 0)
        elif kind == 2:
            for i in range(rb):
                cur[i] = src[i] + up[i]
        elif kind == 3:
            for i in range(rb):
                a = np.int32(cur[i - bpp]) if i >= bpp else 0
                cur[i] = (np.int32(src[i]) + (a + np.int32(up[i])) // 2) & 0xFF
        elif kind == 4:
            for i in range(rb):
                a = np.int32(cur[i - bpp]) if i >= bpp else 0
                b = np.int32(up[i])
                c = np.int32(up[i - bpp]) if i >= bpp else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pr = a if pa <= pb and pa <= pc else b if pb <= pc else c
                cur[i] = (np.int32(src[i]) + pr) & 0xFF
        else:
            return 1
    return 0


# -- CCSDS 121.0 adaptive entropy coding (libaec's flags)

_AEC_SIGNED, _AEC_PREPROCESS, _AEC_RESTRICTED, _AEC_PAD_RSI = 1, 8, 16, 32


@njit(cache=True, nogil=True)
def _get(buf, pos, k):
    """k (0..56) bits from bit pos; past the end reads zeros."""
    if k == 0:
        return 0
    q = pos >> 3
    if q + 8 <= buf.size:                       # one load and a byte swap
        x = G._be64(buf, q)
    else:
        x = np.uint64(0)
        for j in range(8):
            x = (x << np.uint64(8)) | (np.uint64(buf[q + j]) if q + j < buf.size else np.uint64(0))
    x = (x << np.uint64(pos & 7)) >> np.uint64(64 - k)
    return np.int64(x)


@njit(cache=True, nogil=True)
def _fs(buf, pos):
    """A fundamental-sequence codeword at pos: (its value, bits used)."""
    z = 0
    end = buf.size * 8
    while pos + z < end:
        w = _get(buf, pos + z, 56)
        if w != 0:
            lz = _clz64(w) - 8                  # (w: 56 bits)
            return z + lz, z + lz + 1
        z += 56
    return -1, 0


@njit(cache=True, nogil=True)
def _aec_y(x, widen, r0, f2, f10):
    """A sample's value, as _simple converts it."""
    xf = np.float64(np.float32(x)) if widen else np.float64(x)
    return np.float32((r0 + f2 * xf) * f10)


@njit(cache=True, nogil=True)
def _aec(buf, nbits, flags, J, rsi, widen, p, f, dest, row):
    """A CCSDS 121.0 stream's samples, as wgrib2's floats, into their
    columns; how many came out (fewer than the field's: damaged). One
    pass: each block's coded values into a block-sized buffer (a function
    each kind of block -- in one big loop LLVM made it 3x slower), then
    the inverse preprocessing and the floats while they are in L1; a run
    of zero blocks is one value repeated. (The whole field's coded values
    into a buffer of their own first, then two passes over them: ECMWF's
    open data 8.6 ns a value on a core, against 5.2.)"""
    n = p[G._NVAL]
    usedest = p[G._DEST] >= 0
    if p[G._FILL]:
        row[:] = np.float32(f[3])
    if J <= 0 or rsi <= 0:
        return 0
    r0, f2, f10 = f[0], f[1], f[2]
    pp = flags & _AEC_PREPROCESS
    signed = flags & _AEC_SIGNED
    if nbits > 16:
        idlen = 5
    elif nbits > 8:
        idlen = 4
    elif flags & _AEC_RESTRICTED and nbits <= 4:
        idlen = 1 if nbits <= 2 else 2
    else:
        idlen = 3
    uncomp = (1 << idlen) - 1
    if signed:
        xmin, xmax = -(np.int64(1) << (nbits - 1)), (np.int64(1) << (nbits - 1)) - 1
    else:
        xmin, xmax = np.int64(0), (np.int64(1) << nbits) - 1
    top = np.int64(1) << (nbits - 1)
    full = np.int64(1) << nbits
    tmp = np.empty(J, np.int64)                 # a block's coded values
    pos = 0
    i = 0
    xh = np.int64(0)                            # the sample before (the prediction)
    while i < n:
        b = 0
        while b < rsi and i < n:
            ref = 1 if pp and b == 0 else 0
            opt = _get(buf, pos, idlen)
            pos += idlen
            low = 0
            if opt == 0:                    # low entropy: its selector bit, then the reference
                low = 1 + _get(buf, pos, 1)
                pos += 1
            if ref:                         # a reference sample, as it is
                d = _get(buf, pos, nbits)
                pos += nbits
                xh = d - full if signed and d & top else d
                G._put(row, dest, usedest, i, _aec_y(xh, widen, r0, f2, f10))
                i += 1
            if opt == 0 and low == 1:       # zero blocks: the prediction (or 0) throughout
                z, used = _fs(buf, pos)
                if z < 0:
                    return i
                pos += used
                z += 1
                if z == 5:                  # the rest of the segment
                    z = min(rsi - b, 64 - b % 64)
                elif z > 5:
                    z -= 1
                m = min(z * J - ref, n - i)
                y = _aec_y(xh if pp else 0, widen, r0, f2, f10)
                for t in range(m):
                    G._put(row, dest, usedest, i + t, y)
                i += m
                b += z
                continue
            kk = 0                          # a split-sample block's low bits, read below
            if opt == 0:
                pos = _se_block(buf, pos, tmp, ref, J)
            elif opt == uncomp:
                pos = _raw_block(buf, pos, tmp, J - ref, nbits)
            else:
                pos = _split_block(buf, pos, tmp, J - ref)
                kk = opt - 1
            if pos < 0:
                return i
            lo = pos                        # where they start
            pos += (J - ref) * kk
            m = min(J - ref, n - i)
            if pp:                          # mapped residuals from the sample before
                for t in range(m):
                    d = (tmp[t] << kk) | _get(buf, lo + t * kk, kk)
                    # within reach of the prediction (nearly always): +d/2
                    # for even d, -(d + 1)/2 for odd -- the parity without a
                    # branch (one taken at random half the time held each
                    # sample on a misprediction: 5.8 -> 4.3 ns a value)
                    if d <= 2 * min(xh - xmin, xmax - xh):
                        xh = xh + ((d >> 1) ^ -(d & 1))
                    else:
                        xh = xmin + d if xh - xmin <= xmax - xh else xmax - d
                    G._put(row, dest, usedest, i + t, _aec_y(xh, widen, r0, f2, f10))
            else:
                for t in range(m):
                    d = (tmp[t] << kk) | _get(buf, lo + t * kk, kk)
                    x = d - full if signed and d & top else d
                    G._put(row, dest, usedest, i + t, _aec_y(x, widen, r0, f2, f10))
            i += m
            b += 1
        if flags & _AEC_PAD_RSI:
            pos = (pos + 7) & ~7
    return n


@njit(cache=True, nogil=True)
def _split_block(buf, pos, tmp, m):
    """The FS codes of a split-sample block's m samples into tmp (_aec
    reads their low bits as it takes them); the position after, -1 if the
    stream ends first. The
    codes come from one 8-byte load while they fit in it, a leading-zero
    count each, not a load and a bit loop each."""
    t = 0
    while t < m:
        q = pos >> 3
        if q + 8 > buf.size:                    # the stream's last bytes: one at a time
            v, used = _fs(buf, pos)
            if v < 0:
                return -1
            pos += used
            tmp[t] = v
            t += 1
            continue
        w = G._be64(buf, q) << np.uint64(pos & 7)   # 57 bits or more of the stream
        used = 0
        while t < m:
            lz = _clz64(w)
            if lz + 1 + used > 56:              # not wholly within the bits loaded
                break
            tmp[t] = lz
            t += 1
            used += lz + 1
            w <<= np.uint64(lz + 1)
        if used == 0:                           # a code of 56 zeros or more
            v, used = _fs(buf, pos)
            if v < 0:
                return -1
            tmp[t] = v
            t += 1
        pos += used
    return pos


@njit(cache=True, nogil=True)
def _raw_block(buf, pos, tmp, m, nbits):
    for t in range(m):
        tmp[t] = _get(buf, pos + t * nbits, nbits)
    return pos + m * nbits


@njit(cache=True, nogil=True)
def _se_block(buf, pos, tmp, ref, J):
    """A second-extension block into tmp: FS codes of pairs (the first
    value of the first pair left out after a reference sample)."""
    k = ref
    i = 0
    while k < J:
        mm, used = _fs(buf, pos)
        if mm < 0:
            return -1
        pos += used
        s = 0
        while (s + 1) * (s + 2) // 2 <= mm:
            s += 1
        d1 = mm - s * (s + 1) // 2
        if k % 2 == 0:
            tmp[i] = s - d1
            i += 1
            k += 1
        tmp[i] = d1
        i += 1
        k += 1
    return pos
