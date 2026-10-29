"""Test data for complex packing under missing-value management (5.3 with
octet 23 set to 1 or 2, every NBM field's): NBM QMD messages from the small
Hawaii and Puerto Rico grids, copied as they are published.

    python3 tools/testdata/make_nbm_qmd.py OUT_DIR

nbm_qmd_hi.grib2 holds a field with primary and secondary missing values (2)
and no point missing; nbm_qmd_pr.grib2 one with primary missing values (1)
and some points missing, its other groups as any of management 1's. make_test_data.sh's last loop then adds
wgrib2's inventory and values beside them (.inv, .bin).
"""
import os
import sys

import numpy as np

import fastgrib

URL = 'https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.20261028/12/qmd/blend.t12z.qmd.f120.%s.grib2'
PICKS = {
    'hi': [':APTMP:2 m above ground:'],
    'pr': [':APCP:surface:'],
}
WANT = {'hi': [(2, False)], 'pr': [(1, True)]}   # (management, points missing)


def main(out):
    for grid, patterns in PICKS.items():
        f = fastgrib.open(URL % grid)
        chosen = []
        for pattern, (mgmt, missing) in zip(patterns, WANT[grid]):
            for r in f.find(pattern):
                s5 = f.sections(r)[5]
                if int.from_bytes(s5[9:11], 'big') != 3 or s5[22] != mgmt:
                    continue
                if bool(np.isnan(f.read(r)).any()) == missing:
                    chosen.append(r)
                    break
        f.write(os.path.join(out, 'nbm_qmd_%s.grib2' % grid), chosen)
        print(grid, [(r.name, r.level) for r in chosen])


if __name__ == '__main__':
    main(sys.argv[1])
