"""Test data: ecCodes' own shortName and typeOfLevel of each field of GRIB2
files, one 'shortName typeOfLevel' line each, as NAME.ecc beside them.

    python tools/testdata/eccodes_names.py FILE.grib2 ...

Needs ecCodes' Python bindings (pip install eccodes eccodeslib).
"""
import os
import sys

import eccodes

eccodes.codes_grib_multi_support_on()           # each field of a multi-field message
for path in sys.argv[1:]:
    lines = []
    with open(path, 'rb') as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            lines.append('%s %s' % (eccodes.codes_get(h, 'shortName'),
                                    eccodes.codes_get(h, 'typeOfLevel')))
            eccodes.codes_release(h)
    out = path[:-6] + '.ecc'
    open(out, 'w').write('\n'.join(lines) + '\n')
    print('%s: %d fields' % (os.path.basename(out), len(lines)))
