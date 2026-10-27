"""ecCodes' keys of every field of some test files, for keys()' tests:
tests/data/eccodes_keys.json.gz ({file: [{key: value}, ...]}; code
tables' keys as their numbers).

    python3 tools/testdata/eccodes_keys.py DATA_DIR FILE...

Needs ecCodes' Python bindings (pip install eccodes eccodeslib).
"""
import gzip
import json
import os
import sys

import eccodes

D = sys.argv[1]
KEYS = ['editionNumber', 'discipline', 'totalLength', 'centre', 'subCentre', 'tablesVersion',
        'localTablesVersion', 'significanceOfReferenceTime', 'dataDate', 'dataTime',
        'productionStatusOfProcessedData', 'typeOfProcessedData', 'gridDefinitionTemplateNumber',
        'numberOfDataPoints', 'shapeOfTheEarth', 'Ni', 'Nj', 'Nx', 'Ny', 'N',
        'latitudeOfFirstGridPointInDegrees', 'longitudeOfFirstGridPointInDegrees',
        'latitudeOfLastGridPointInDegrees', 'longitudeOfLastGridPointInDegrees',
        'iDirectionIncrementInDegrees', 'jDirectionIncrementInDegrees', 'scanningMode',
        'resolutionAndComponentFlags', 'LaDInDegrees', 'LoVInDegrees', 'DxInMetres', 'DyInMetres',
        'DiInMetres', 'DjInMetres', 'projectionCentreFlag', 'Latin1InDegrees', 'Latin2InDegrees',
        'latitudeOfSouthernPoleInDegrees', 'longitudeOfSouthernPoleInDegrees',
        'angleOfRotationInDegrees', 'gridType', 'productDefinitionTemplateNumber',
        'parameterCategory', 'parameterNumber', 'typeOfGeneratingProcess',
        'generatingProcessIdentifier', 'indicatorOfUnitOfTimeRange', 'forecastTime',
        'typeOfFirstFixedSurface', 'scaleFactorOfFirstFixedSurface',
        'scaledValueOfFirstFixedSurface', 'typeOfSecondFixedSurface',
        'scaleFactorOfSecondFixedSurface', 'scaledValueOfSecondFixedSurface',
        'typeOfEnsembleForecast', 'perturbationNumber', 'numberOfForecastsInEnsemble',
        'numberOfTimeRange', 'typeOfStatisticalProcessing', 'indicatorOfUnitForTimeRange',
        'lengthOfTimeRange', 'stepType', 'stepRange', 'numberOfValues',
        'dataRepresentationTemplateNumber', 'packingType', 'referenceValue', 'binaryScaleFactor',
        'decimalScaleFactor', 'bitsPerValue', 'bitmapPresent', 'numberOfMissing', 'shortName',
        'typeOfLevel', 'level', 'uvRelativeToGrid', 'validityDate', 'validityTime']
eccodes.codes_grib_multi_support_on()
out = {}
for name in sys.argv[2:]:
    msgs = []
    with open(os.path.join(D, name), 'rb') as f:
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            d = {}
            for k in KEYS:
                try:
                    if not eccodes.codes_is_defined(h, k) or eccodes.codes_is_missing(h, k):
                        continue
                    v = eccodes.codes_get(h, k)
                    if isinstance(v, str) and k not in ('gridType', 'packingType', 'stepType',
                                                        'stepRange', 'shortName', 'typeOfLevel'):
                        v = eccodes.codes_get(h, k, int)       # a code table's number
                    d[k] = v if isinstance(v, (int, float, str)) else str(v)
                except Exception:
                    pass
            msgs.append(d)
            eccodes.codes_release(h)
    out[name] = msgs
text = json.dumps(out, separators=(',', ':'), sort_keys=True).encode()
open(os.path.join(D, 'eccodes_keys.json.gz'), 'wb').write(gzip.compress(text, 9, mtime=0))
print({k: len(v) for k, v in out.items()})
