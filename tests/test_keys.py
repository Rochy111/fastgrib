"""keys(): every key as ecCodes names and gives it."""
import os

import fastgrib

from helpers import DATA, grib1, path


def test_keys_are_eccodes():
    """keys(): every key ecCodes gives for the test files' fields, as it
    gives them (tools/testdata/eccodes_keys.py) -- but a negative PV level's
    scaled value, which ecCodes reads unsigned (WMO: sign and magnitude),
    and a field of several's length, its message as fastgrib cuts it."""
    import gzip
    import json
    ec = json.loads(gzip.decompress(open(path('eccodes_keys.json', '.gz'), 'rb').read()))
    agree = 0
    for name, msgs in ec.items():
        f = fastgrib.open(os.path.join(DATA, name))
        assert len(f.records) == len(msgs)
        for r, want in zip(f.records, msgs):
            got = f.keys(r)
            for k, v in want.items():
                if k == 'totalLength' and r.sub or k in ('scaledValueOfFirstFixedSurface',
                                                          'level') and 'PV=-' in r.level:
                    continue
                if k in ('referenceValue', 'binaryScaleFactor', 'decimalScaleFactor',
                         'bitsPerValue') and got['packingType'] == 'grid_ieee':
                    continue                    # (ecCodes' zeros: none in the message)
                assert k in got, (name, r, k)
                g = got[k]
                if isinstance(v, float):
                    assert abs(g - v) <= 1e-6 * max(1.0, abs(v)), (name, r, k, g, v)
                else:
                    assert str(g) == str(v), (name, r, k, g, v)
                agree += 1
    assert agree > 17000


def test_keys_sections_grib1_remote(server):
    """keys() of GRIB1 fields (as the GRIB2 they read as, edition 1) and of
    remote ones; sections() gives each section whole."""
    k = grib1('g1_ec').keys(9)
    assert k['editionNumber'] == 1 and k['perturbationNumber'] == 5 and k['shortName'] == '2t'
    assert k['typeOfFirstFixedSurface'] == 103 and k['level'] == 2
    loc, rem = fastgrib.open(path('gfs_uv')), fastgrib.open(server + 'gfs_uv.grib2')
    for i in (0, 3):
        assert rem.keys(i) == loc.keys(i)
    s = loc.sections(3)
    assert sorted(s) == [1, 3, 4, 5, 6, 7]
    assert all(int.from_bytes(v[:4], 'big') == len(v) and v[4] == n for n, v in s.items())
