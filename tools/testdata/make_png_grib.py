"""Rewrite a simple-packed GRIB2 (template 5.0) as PNG-packed (5.41): the
same packed integers, each message's as a PNG image (a grid row a PNG row
where the grid has no bitmap), every PNG filter kind used in turn. For
tests: wgrib2 reads the result through libpng.

    python tools/testdata/make_png_grib.py IN.grib2 OUT.grib2
"""
import struct
import sys
import zlib


def ints(data, nbits, n):
    out, have, acc = [], 0, 0
    it = iter(data)
    for _ in range(n):
        while have < nbits:
            acc = (acc << 8) | next(it)
            have += 8
        have -= nbits
        out.append((acc >> have) & ((1 << nbits) - 1))
        acc &= (1 << have) - 1
    return out


def png(samples, width, depth):
    nb = depth // 8
    chans, ctype, d = {8: (1, 0, 8), 16: (1, 0, 16), 24: (3, 2, 8), 32: (4, 6, 8)}[depth]
    rows, raw = [samples[i:i + width] for i in range(0, len(samples), width)], b''
    prev = bytes(width * nb)
    for y, r in enumerate(rows):
        line = b''.join(x.to_bytes(nb, 'big') for x in r) + bytes((width - len(r)) * nb)
        kind = y % 5
        bpp = nb
        out = bytearray()
        for i, x in enumerate(line):
            a = line[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            if kind == 0:
                p = 0
            elif kind == 1:
                p = a
            elif kind == 2:
                p = b
            elif kind == 3:
                p = (a + b) // 2
            else:
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                p = a if pa <= pb and pa <= pc else b if pb <= pc else c
            out.append((x - p) & 0xFF)
        raw += bytes([kind]) + bytes(out)
        prev = line

    def chunk(k, body):
        return struct.pack('>I', len(body)) + k + body + struct.pack('>I', zlib.crc32(k + body))
    ihdr = struct.pack('>IIBBBBB', width, len(rows), d, ctype, 0, 0, 0)
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', ihdr) + chunk(b'IDAT', zlib.compress(raw, 6))
            + chunk(b'IEND', b''))


def convert(msg):
    """A simple-packed message as PNG-packed; a constant one (no bits) as it is."""
    out, pos, s3ni = [msg[:16]], 16, None
    n = nbits = None
    while msg[pos:pos + 4] != b'7777':
        ln, num = struct.unpack('>IB', msg[pos:pos + 5])
        sec = bytearray(msg[pos:pos + ln])
        if num == 3:
            s3ni = struct.unpack('>I', sec[30:34])[0]
        if num == 5:
            assert struct.unpack('>H', sec[9:11])[0] == 0, 'not simple packing'
            n, nbits = struct.unpack('>I', sec[5:9])[0], sec[19]
            if nbits == 0:
                return msg
            depth = 8 if nbits <= 8 else 16 if nbits <= 16 else 24 if nbits <= 24 else 32
            sec[9:11] = struct.pack('>H', 41)
            sec[19] = depth
        if num == 7 and nbits:
            vals = ints(sec[5:], nbits, n)
            width = s3ni if s3ni and n % s3ni == 0 else n
            body = png(vals, width, depth)
            sec = bytearray(struct.pack('>IB', 5 + len(body), 7) + body)
        out.append(bytes(sec))
        pos += ln
    out.append(b'7777')
    m = b''.join(out)
    return m[:8] + struct.pack('>Q', len(m)) + m[16:]


def main(src, dst):
    b = open(src, 'rb').read()
    pos, msgs = 0, []
    while True:
        pos = b.find(b'GRIB', pos)
        if pos < 0:
            break
        total = struct.unpack('>Q', b[pos + 8:pos + 16])[0]
        msgs.append(convert(b[pos:pos + total]))
        pos += total
    open(dst, 'wb').write(b''.join(msgs))
    print('%d messages -> %s' % (len(msgs), dst))


if __name__ == '__main__':
    main(*sys.argv[1:3])
