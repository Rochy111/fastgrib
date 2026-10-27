"""Build fastgrib/tables/ncep_grids.py: NCEP's predefined GRIB1 grids (ON388
Table B: a message's PDS octet 7 with no GDS) as GRIB1 grid descriptions,
from the specifications on NCEP's Table B page (a US government work,
public domain). The grids it describes in full -- lat-lon, Mercator,
polar stereographic, Lambert conformal -- are kept; the staggered E
grids and those described only in words are left out.

    python tools/tables/make_ncep_grids.py TABLEB_HTML

TABLEB_HTML: https://www.nco.ncep.noaa.gov/pmb/docs/on388/tableb.html
"""
import html
import os
import re
import sys

NUM = r'(-?\d+(?:\.\d+)?)'


def _deg(text, key):
    """A latitude or longitude after `key` (N/E positive), None if absent."""
    m = re.search(r'\b%s\s+%s\s*([NSEW])' % (key, NUM), text)
    if not m:
        return None
    v = float(m.group(1))
    return -v if m.group(2) in 'SW' else v


def _val(text, key, unit=''):
    m = re.search(r'\b%s\s+%s\s*%s' % (key, NUM, unit), text)
    return float(m.group(1)) if m else None


def _int(text, key):
    m = re.search(r'\b%s\s+(\d+)\b' % key, text)
    return int(m.group(1)) if m else None


def _bits(text, key, n):
    m = re.search(re.escape(key) + r'[^01]*' + r'\s+'.join(['([01])'] * n), text)
    return int(''.join(m.groups()), 2) if m else None


def parse(page):
    t = html.unescape(re.sub(r'<[^>]+>', ' ', open(page, encoding='latin-1').read()))
    t = re.sub(r'\s+', ' ', t.replace('\xa0', ' '))
    out = {}
    for chunk in t.split('Top of Page'):
        m = re.search(r'(?:GRID DESCRIPTIONS )?\s*(\d{1,3})\s*(?:\(\w\))?\s+[^\d]', chunk)
        if not m or not re.search(r'\b(Nx|Ni)\s+\d', chunk):
            continue
        gid = int(m.group(1))
        desc = chunk[m.end() - 1:m.end() + 160].lower()
        ni = _int(chunk, 'Nx') or _int(chunk, 'Ni')
        nj = _int(chunk, 'Ny') or _int(chunk, 'Nj')
        la1, lo1 = _deg(chunk, 'La1'), _deg(chunk, 'Lo1')
        res = _bits(chunk, 'Res. & Comp. Flag', 8)
        scan = _bits(chunk, 'Scanning Mode (bits 1 2 3)', 3)
        if None in (ni, nj, la1, lo1, scan):
            continue
        scan <<= 5
        res = 0x88 if res is None else res
        if 'lambert' in desc:
            lov, dx = _deg(chunk, 'Lov') or _val(chunk, 'Lov'), _val(chunk, 'Dx', 'km')
            dy = _val(chunk, 'Dy', 'km') or dx
            l1, l2 = _deg(chunk, 'Latin1'), _deg(chunk, 'Latin2')
            if l1 is None:                      # a tangent cone said in words: 'at 25 deg north'
                m = re.search(r'\(at %s deg(?:rees)? north\)' % NUM, chunk)
                l1 = float(m.group(1)) if m else None
            if None in (lov, dx, l1):
                continue
            proj = _int(chunk, r'Projection Flag \(bit 1\)') or 0
            out[gid] = ('lambert', ni, nj, la1, lo1, res, lov, dx, dy, proj << 7, scan, l1,
                        l2 if l2 is not None else l1)
        elif 'polar stereographic' in desc:
            lov, dx = _deg(chunk, 'Lov') or _val(chunk, 'Lov'), _val(chunk, 'Dx', 'km')
            dy = _val(chunk, 'Dy', 'km') or dx
            if None in (lov, dx):
                continue
            proj = _int(chunk, r'Projection Flag \(bit 1\)') or 0
            out[gid] = ('polar', ni, nj, la1, lo1, res, lov, dx, dy, proj << 7, scan)
        elif 'mercator' in desc:
            la2, lo2 = _deg(chunk, 'La2'), _deg(chunk, 'Lo2')
            latin = _val(chunk, 'Latin')
            di, dj = _val(chunk, 'Di', 'km'), _val(chunk, 'Dj', 'km')
            if None in (la2, lo2, latin, di):
                continue
            out[gid] = ('mercator', ni, nj, la1, lo1, res, la2, lo2, latin, scan, di, dj or di)
        elif re.search(r'lat(itude)?[/ -]?lon|lon(gitude)?[/ -]?lat', desc):
            la2, lo2 = _deg(chunk, 'La2'), _deg(chunk, 'Lo2')
            di = _val(chunk, 'Di') or _val(chunk, 'Dx')
            dj = _val(chunk, 'Dj') or _val(chunk, 'Dy')
            if None in (la2, lo2, di, dj):
                continue
            out[gid] = ('latlon', ni, nj, la1, lo1, res, la2, lo2, di, dj, scan)
    # the global lat-lon grids described in a line: 'N-point (NIxNJ) global
    # longitude-latitude grid. (1,1) at (0E, 90N)' -- pole to pole, 0E eastward
    for m in re.finditer(r' (\d{1,3}) \d+-point \((\d+)x(\d+)\) global (?:longitude-latitude|'
                         r'lon/lat|longitude/latitude) grid\. \(1,1\) at \(0E, 90N\)', t, re.I):
        gid, ni, nj = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if gid not in out:
            di, dj = 360.0 / ni, 180.0 / (nj - 1)
            out[gid] = ('latlon', ni, nj, 90.0, 0.0, 0x80, -90.0, 360.0 - di, di, dj, 0)
    return out


def main(page):
    grids = parse(page)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'fastgrib', 'tables',
                       'ncep_grids.py')
    with open(out, 'w', encoding='utf-8', newline='\n') as f:
        f.write('"""NCEP\'s predefined GRIB1 grids (ON388 Table B): grid number ->\n'
                '(kind, Ni, Nj, La1, Lo1, resolution and component flags, ...): lat-lon\n'
                '(La2, Lo2, Di, Dj, scanning), polar (LoV, Dx, Dy km, projection centre,\n'
                'scanning), Lambert (the same, Latin1, Latin2), Mercator (La2, Lo2, Latin,\n'
                'scanning, Di, Dj km); degrees, north and east positive. Generated by\n'
                'tools/tables/make_ncep_grids.py from NCEP\'s Table B page (public domain); do\n'
                'not edit."""\n\n')
        f.write('GRIDS = {\n')
        for k in sorted(grids):
            f.write('    %d: %r,\n' % (k, grids[k]))
        f.write('}\n')
    kinds = {}
    for v in grids.values():
        kinds[v[0]] = kinds.get(v[0], 0) + 1
    print('%d grids %s -> %s' % (len(grids), kinds, os.path.normpath(out)))


if __name__ == '__main__':
    main(sys.argv[1])
