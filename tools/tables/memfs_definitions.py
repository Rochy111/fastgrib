"""Write out the definition files an ecCodes build holds in memory (conda-forge's
Windows builds compile them into eccodes_memfs.dll, and ship no files), for
make_eccodes.py and make_params1_ecmwf.py to read.

    python tools/tables/memfs_definitions.py ECCODES_MEMFS_DLL OUT_DIR [PREFIX ...]

PREFIX: the paths wanted under definitions/ (default grib1/ and grib2/).
The library keeps a table of (path, content, length) entries; each path's
entry is found by its address and the content read through the PE's
sections. ecCodes is (C) ECMWF, Apache License 2.0 (see fastgrib/NOTICE).
"""
import os
import struct
import sys

ROOT = b'/MEMFS/definitions/'


def main(dll, outdir, prefixes):
    b = open(dll, 'rb').read()
    pe = struct.unpack_from('<I', b, 0x3C)[0]
    nsec = struct.unpack_from('<H', b, pe + 6)[0]
    opt = pe + 24
    optsz = struct.unpack_from('<H', b, pe + 20)[0]
    base = struct.unpack_from('<Q', b, opt + 24)[0]            # (PE32+: a 64-bit build)
    secs = []
    for i in range(nsec):
        vsz, va, rsz, raw = struct.unpack_from('<IIII', b, opt + optsz + 40 * i + 8)
        secs.append((va, max(vsz, rsz), raw))

    def to_va(o):
        return next(base + va + o - raw for va, sz, raw in secs if raw <= o < raw + sz)

    def to_off(v):
        return next(raw + v - base - va for va, sz, raw in secs if va <= v - base < va + sz)
    n, at = 0, 0
    while True:
        i = b.find(ROOT, at)
        if i < 0:
            break
        at = b.index(b'\0', i)
        rel = b[i + len(ROOT):at].decode()
        if not rel.startswith(tuple(prefixes)):
            continue
        k = b.find(struct.pack('<Q', to_va(i)))
        if k < 0 or k % 8:
            raise SystemExit('no table entry for %s' % rel)
        content, length = struct.unpack_from('<QQ', b, k + 8)
        out = os.path.join(outdir, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'wb') as f:
            f.write(b[to_off(content):to_off(content) + length])
        n += 1
    print('%d files -> %s' % (n, outdir))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or ['grib1/', 'grib2/'])
