"""Test data for GRIB1 reading, with ecCodes' reading of each message as
the reference (its latitudes, longitudes, values in the message's order and
a few keys, NAME_ec.npz beside NAME.grb):

- g1_ec.grb: ecCodes' GRIB1 samples -- regular lat-lon (both row orders,
  one with a bitmap), Gaussian, reduced Gaussian, rotated lat-lon, rotated
  and reduced rotated Gaussian, north and south polar stereographic -- each
  a smooth function of its points' true latitude and longitude, with an
  ECMWF ensemble member, accumulations (one as ECMWF codes them: a step,
  accumulated from the start), a maximum and an NCEP-table parameter among
  them, and last a spherical-harmonics message (its coefficients);
- g1_narr.grb (given NARR_GRB): a few messages of a NARR analysis --
  Lambert conformal, NCEP's table 131, bitmaps, a constant field, layers,
  winds along the grid, averages and accumulations.

    python3 tools/testdata/make_grib1_eccodes.py OUT_DIR [NARR_GRB [NARR_INV]]

NARR_GRB: https://www.ncei.noaa.gov/data/north-american-regional-reanalysis/
access/3-hourly/200709/20070901/narr-a_221_20070901_0000_000.grb, NARR_INV
its wgrib inventory (.inv beside it): g1_narr.inv, its lines for the
messages kept, numbered and placed as in g1_narr.grb.
Needs ecCodes' Python bindings (pip install eccodes eccodeslib).
"""
import os
import sys

import numpy as np
import eccodes

KEYS = ('shortName', 'typeOfLevel', 'level', 'stepRange', 'stepType', 'number', 'dataDate',
        'dataTime', 'gridType', 'centre', 'table2Version', 'indicatorOfParameter')

_LL = {'Ni': 41, 'Nj': 21, 'iDirectionIncrementInDegrees': 0.5,
       'jDirectionIncrementInDegrees': 0.5, 'longitudeOfFirstGridPointInDegrees': -10.0,
       'longitudeOfLastGridPointInDegrees': 10.0}
_ROT = {'latitudeOfSouthernPoleInDegrees': -40.0, 'longitudeOfSouthernPoleInDegrees': 10.0}
_PS = {'Ni': 25, 'Nj': 19, 'DxInMetres': 60000, 'DyInMetres': 60000,
       'orientationOfTheGridInDegrees': -105.0}
MESSAGES = [
    ('regular_ll_sfc_grib1', dict(_LL, latitudeOfFirstGridPointInDegrees=60.0,
                                  latitudeOfLastGridPointInDegrees=50.0, jScansPositively=0,
                                  indicatorOfParameter=167), None),
    ('regular_ll_sfc_grib1', dict(_LL, latitudeOfFirstGridPointInDegrees=-5.0,
                                  latitudeOfLastGridPointInDegrees=5.0, jScansPositively=1,
                                  indicatorOfParameter=167), 'bitmap'),
    ('regular_gg_sfc_grib1', {'indicatorOfParameter': 167}, None),
    ('reduced_gg_pl_32_grib1', {'indicatorOfParameter': 167}, None),
    ('rotated_ll_sfc_grib1', dict(_LL, latitudeOfFirstGridPointInDegrees=-5.0,
                                  latitudeOfLastGridPointInDegrees=5.0, jScansPositively=1,
                                  **_ROT), None),
    ('rotated_gg_sfc_grib1', dict(_ROT), None),
    ('reduced_rotated_gg_sfc_grib1', dict(_ROT), None),
    ('polar_stereographic_sfc_grib1', dict(_PS, latitudeOfFirstGridPointInDegrees=20.0,
                                           longitudeOfFirstGridPointInDegrees=-130.0,
                                           projectionCentreFlag=0), None),
    ('polar_stereographic_sfc_grib1', dict(_PS, latitudeOfFirstGridPointInDegrees=-20.0,
                                           longitudeOfFirstGridPointInDegrees=100.0,
                                           projectionCentreFlag=128), None),
    ('regular_ll_sfc_grib1', dict(_LL, latitudeOfFirstGridPointInDegrees=60.0,
                                  latitudeOfLastGridPointInDegrees=50.0, jScansPositively=0,
                                  indicatorOfParameter=167), 'member'),
    ('regular_ll_sfc_grib1', dict(_LL, latitudeOfFirstGridPointInDegrees=60.0,
                                  latitudeOfLastGridPointInDegrees=50.0, jScansPositively=0,
                                  indicatorOfParameter=228), 'accum'),
    ('regular_ll_sfc_grib1', dict(_LL, latitudeOfFirstGridPointInDegrees=60.0,
                                  latitudeOfLastGridPointInDegrees=50.0, jScansPositively=0,
                                  indicatorOfParameter=228, timeRangeIndicator=0, P1=6), None),
    ('regular_ll_sfc_grib1', dict(_LL, latitudeOfFirstGridPointInDegrees=60.0,
                                  latitudeOfLastGridPointInDegrees=50.0, jScansPositively=0,
                                  indicatorOfParameter=201, timeRangeIndicator=2, P1=0, P2=6),
     None),
    ('regular_ll_pl_grib1', dict(_LL, latitudeOfFirstGridPointInDegrees=60.0,
                                 latitudeOfLastGridPointInDegrees=50.0, jScansPositively=0,
                                 centre=7, table2Version=2, indicatorOfParameter=11,
                                 level=850), None),
]
# NARR: PRMSL, CLWMR 100 mb (constant, bitmap), TSOIL 0-10 cm, UGRD/VGRD 10 m,
# LFTX 500-1000 mb, 4LFTX 180-0 mb above ground, APCP 0-3 h, LHTFL 0-3 h ave
NARR = (2, 23, 269, 293, 294, 314, 374, 380, 390)


def field(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return 280 + 25 * np.cos(la) + 6 * np.sin(2 * lo) * np.cos(la) + 0.01 * lon


def make(sample, keys, kind):
    h = eccodes.codes_grib_new_from_samples(sample)
    for k, v in keys.items():
        eccodes.codes_set(h, k, v)
    eccodes.codes_set(h, 'dataDate', 20240115)
    eccodes.codes_set(h, 'dataTime', 1200)
    if kind == 'member':                        # ECMWF's local definition 1: a perturbed forecast
        eccodes.codes_set(h, 'setLocalDefinition', 1)
        eccodes.codes_set(h, 'localDefinitionNumber', 1)
        eccodes.codes_set(h, 'marsType', 'pf')
        eccodes.codes_set(h, 'number', 5)
        eccodes.codes_set(h, 'numberOfForecastsInEnsemble', 51)
    if kind == 'accum':
        eccodes.codes_set(h, 'stepType', 'accum')
        eccodes.codes_set(h, 'startStep', 0)
        eccodes.codes_set(h, 'endStep', 6)
    if 'Ni' in keys or eccodes.codes_get_size(h, 'values') == 0:   # sized to the grid first
        eccodes.codes_set_values(h, np.zeros(eccodes.codes_get(h, 'numberOfDataPoints')))
    lat = eccodes.codes_get_array(h, 'latitudes')
    lon = eccodes.codes_get_array(h, 'longitudes')
    v = field(lat, lon)
    if kind == 'bitmap':
        eccodes.codes_set(h, 'bitmapPresent', 1)
        eccodes.codes_set(h, 'missingValue', 9999.0)
        v = np.where((lat > 1) & (lon > 2), 9999.0, v)
    eccodes.codes_set_values(h, v)
    eccodes.codes_set(h, 'bitsPerValue', 16)    # (before the values, it empties a Gaussian grid)
    eccodes.codes_set_values(h, v)
    return h


def reference(path, stride=1):
    """ecCodes' reading of every message: one npz beside the file --
    values as float32 (what fastgrib gives), every stride-th point's
    latitude and longitude, kept once for messages on the same grid."""
    out = {}
    with open(path, 'rb') as f:
        k, last = 0, None
        while True:
            h = eccodes.codes_grib_new_from_file(f)
            if h is None:
                break
            if eccodes.codes_get(h, 'gridType') == 'sh':           # coefficients: no points
                out['v%d' % k] = eccodes.codes_get_values(h).astype(np.float32)
            else:
                lat = eccodes.codes_get_array(h, 'latitudes')[::stride]
                lon = eccodes.codes_get_array(h, 'longitudes')[::stride]
                if not (k and np.array_equal(lat, last[0]) and np.array_equal(lon, last[1])):
                    out['lat%d' % k], out['lon%d' % k] = last = lat, lon
                eccodes.codes_set(h, 'missingValue', 9.999e20)
                v = eccodes.codes_get_values(h)
                out['v%d' % k] = np.where(v == 9.999e20, np.nan, v).astype(np.float32)
            out['keys%d' % k] = np.array([str(eccodes.codes_get(h, x))
                                          if eccodes.codes_is_defined(h, x) else '-'
                                          for x in KEYS])
            eccodes.codes_release(h)
            k += 1
    np.savez_compressed(path[:-4] + '_ec.npz', keys=np.array(KEYS), stride=stride, **out)
    print(os.path.basename(path), k, 'messages')


def main(out, narr=None, inv=None):
    path = os.path.join(out, 'g1_ec.grb')
    with open(path, 'wb') as f:
        for sample, keys, kind in MESSAGES:
            h = make(sample, keys, kind)
            eccodes.codes_write(h, f)
            eccodes.codes_release(h)
        h = eccodes.codes_grib_new_from_samples('sh_sfc_grib1')
        eccodes.codes_write(h, f)
        eccodes.codes_release(h)
    reference(path)
    if narr:
        path = os.path.join(out, 'g1_narr.grb')
        at = {}
        with open(narr, 'rb') as f, open(path, 'wb') as g:
            k = 0
            while True:
                h = eccodes.codes_grib_new_from_file(f)
                if h is None:
                    break
                k += 1
                if k in NARR:
                    at[k] = g.tell()
                    g.write(eccodes.codes_get_message(h))
                eccodes.codes_release(h)
        reference(path, stride=17)
        if inv:
            lines = {int(x.split(':')[0]): x for x in open(inv).read().splitlines() if x}
            with open(path[:-4] + '.inv', 'w') as g:
                for n, k in enumerate(NARR, 1):
                    rest = lines[k].split(':', 2)[2]
                    g.write('%d:%d:%s' % (n, at[k], rest) + chr(10))


if __name__ == '__main__':
    main(*sys.argv[1:4])
