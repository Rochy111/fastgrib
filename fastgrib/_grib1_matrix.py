"""GRIB1's matrix of values at each grid point (BDS octet 14, bit 2: WMO's
2-D wave spectra and the like): each matrix element a field of its own,
the message turned into one GRIB2 message of NR x NC fields -- their
bitmaps the primary bitmap's points and the element's bit in each
point's secondary bitmap, their packed values the element's, as packed.
Secondary bitmaps as ECMWF (GRIBEX, ecCodes) counts them: octets 12-13
the number of them (WMO's octet where the data begin is read too)."""
import struct

import numpy as np

from ._octets import ibm, ints, u

_TABLE13 = {1: 'direction', 2: 'frequency', 3: 'radial number'}


def header(b, q, stop):
    """The matrix's description from its BDS at q: (NR, NC, the
    coordinate values of each row and column -- None where code table 12
    gives no rule -- what each dimension is, the secondary bitmaps'
    start, the octet the data begin at by WMO's count)."""
    nr, nc = u(b, q + 14, 2), u(b, q + 16, 2)
    f1, n1, f2, n2, p1, p2 = (b[q + k] for k in range(18, 24))
    at = q + 24
    if nr == 0 or nc == 0 or at + 4 * (n1 + n2) > stop:
        raise ValueError('a matrix of %d x %d values, cut short' % (nr, nc))
    c1 = [ibm(b, at + 4 * k) for k in range(n1)]
    c2 = [ibm(b, at + 4 * (n1 + k)) for k in range(n2)]
    return (nr, nc, _coords(f1, c1, nr), _coords(f2, c2, nc), p1, p2, at + 4 * (n1 + n2),
            q + u(b, q + 11, 2) - 1)


def _coords(flag, c, n):
    """Code table 12: 0 the values given, 1 linear (f(1) = C1, f(n) =
    f(n-1) + C2), 11 geometric (f(n) = C2 f(n-1))."""
    if flag == 0 and len(c) >= n:
        return c[:n]
    if flag == 1 and len(c) >= 2:
        return [c[0] + c[1] * k for k in range(n)]
    if flag == 11 and len(c) >= 2:
        return [c[0] * c[1] ** k for k in range(n)]
    return None


def fields(b, q, stop, npts, bitmap, nbits, ext):
    """Each element's (bitmap over the grid's points: bytes, or None when
    none is missing; its packed values: bytes; how many) of the matrix
    message whose BDS is at q."""
    nr, nc, _, _, _, _, at, wmo_data = header(b, q, stop)
    m = nr * nc
    if bitmap is not None:
        pts = np.flatnonzero(np.unpackbits(np.frombuffer(bitmap, np.uint8))[:npts])
    else:
        pts = np.arange(npts)
    if ext & 0x20:                              # secondary bitmaps: one a point present
        count = u(b, q + 11, 2)
        if count != pts.size:                   # (not ECMWF's count: WMO's data octet)
            count = pts.size
        size = (count * m + 7) // 8
        sec = np.unpackbits(np.frombuffer(bytes(b[at:at + size]), np.uint8))[:count * m]
        if sec.size < count * m:
            raise ValueError('its secondary bitmaps are cut short')
        sec = sec.reshape(count, m).astype(bool)
        data = at + size if count == u(b, q + 11, 2) else max(at + size, wmo_data)
    else:
        sec = np.ones((pts.size, m), bool)
        data = at if wmo_data <= at else wmo_data
    n = int(sec.sum())
    if data + (n * nbits + 7) // 8 > stop:
        raise ValueError('its matrix values are cut short')
    raw = np.frombuffer(bytes(b[data:stop]), np.uint8)
    x = ints(raw, 0, nbits, n) if nbits else np.zeros(n, np.int64)
    where = (np.cumsum(sec.ravel()) - 1).reshape(sec.shape)  # each element's place in x
    from ._encode import _put_fixed
    out = []
    for k in range(m):
        have = sec[:, k]
        xk = np.ascontiguousarray(x[where[have, k]])
        packed = np.zeros((xk.size * nbits + 7) // 8, np.uint8)
        if nbits:
            _put_fixed(xk, nbits, packed, 0)
        if have.all() and pts.size == npts:
            bm = None
        else:
            full = np.zeros(npts, np.uint8)
            full[pts[have]] = 1
            bm = np.packbits(full).tobytes()
        out.append((bm, packed.tobytes(), xk.size))
    return out


def words(b, q, stop):
    """Each element's wording: 'matrix i,j' and what its coordinates are."""
    nr, nc, r, c, p1, p2 = header(b, q, stop)[:6]
    out = []
    for i in range(nr):
        for j in range(nc):
            w = 'matrix %d,%d' % (i + 1, j + 1)
            for name, vals, k in ((_TABLE13.get(p1), r, i), (_TABLE13.get(p2), c, j)):
                if name and vals is not None:
                    w += ' %s=%.6g' % (name, vals[k])
            out.append(w)
    return out


def message(disc, head, s5parts, elems):
    """The GRIB2 message of the matrix's fields: sections 1 and 3 once (in
    head with the first 4), then 4 to 7 a field."""
    s5prefix, nbits, suffix = s5parts
    s134 = head
    p = 0                                       # head: sections 1, 3, 4 (each as _sec made it)
    secs = []
    while p < len(s134):
        ln = struct.unpack_from('>I', s134, p)[0]
        secs.append(s134[p:p + ln])
        p += ln
    s1, s3, s4 = secs
    body = [s1, s3]
    for bm, packed, n in elems:
        s5 = struct.pack('>I', n) + s5prefix[4:] + bytes([nbits]) + suffix
        s6 = struct.pack('>IBB', 6, 6, 255) if bm is None else \
            struct.pack('>IBB', 6 + len(bm), 6, 0) + bm
        body += [s4, struct.pack('>IB', 5 + len(s5), 5) + s5, s6,
                 struct.pack('>IB', 5 + len(packed), 7) + packed]
    payload = b''.join(body)
    return b'GRIB' + bytes([0, 0, disc, 2]) + struct.pack('>Q', 16 + len(payload) + 4) + \
        payload + b'7777'
