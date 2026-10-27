"""Test data for grids beyond the projections: tests/data/sv.grib2, space
views (3.90) of six geometries -- a sub-satellite longitude east and west,
sectors of a full disk (Xo, Yo), both row orders, one past the Earth's
edge -- each value its point's index, with ecCodes' points beside it
(sv_ec.npz: each value's latitude and longitude, (0, 0) off the Earth);
space views no other reader places (sv_more.grib2), a GRIB1 one (sv1.grb);
points listed one by one (3.130: irr.grib2);
and tests/data/ug.grb2, a field on an unstructured grid (3.101, ICON's:
its points in a grid file, named by number and UUID).

    python3 tools/testdata/make_other_grids.py OUT_DIR

Needs ecCodes' Python bindings (pip install eccodes eccodeslib).
"""
import os
import sys

import numpy as np
import eccodes

CASES = [  # Nx, Ny, sub-satellite longitude, dx, dy, Xp, Yp (1e-3 grid lengths), Xo, Yo, j+
    (60, 50, 0.0, 600, 600, 30000, 25000, 0, 0, 1),
    (60, 50, -75.0, 600, 600, 30000, 25000, 0, 0, 0),
    (80, 40, 140.7, 1800, 1800, 900000, 900000, 860, 500, 0),
    (64, 64, 0.0, 3622, 3622, 1856000, 1856000, 1800, 1000, 0),
    (40, 30, 9.5, 300, 300, 10000, 20000, 0, 0, 1),
    (100, 100, 0.0, 3622, 3622, 1856000, 1856000, 0, 0, 0),     # the full disk's corner
]


def space_views(out):
    msgs, ref = [], {}
    for k, (nx, ny, lop, dx, dy, xp, yp, xo, yo, jpos) in enumerate(CASES):
        h = eccodes.codes_grib_new_from_samples('GRIB2')
        eccodes.codes_set(h, 'gridDefinitionTemplateNumber', 90)
        for key, v in (('shapeOfTheEarth', 6), ('Nx', nx), ('Ny', ny),
                       ('latitudeOfSubSatellitePointInDegrees', 0.0),
                       ('longitudeOfSubSatellitePointInDegrees', lop), ('dx', dx), ('dy', dy),
                       ('Xp', xp), ('Yp', yp), ('Nr', 6610839), ('orientationOfTheGridInDegrees', 0),
                       ('Xo', xo), ('Yo', yo), ('iScansNegatively', 0), ('jScansPositively', jpos)):
            eccodes.codes_set(h, key, v)
        eccodes.codes_set_values(h, np.arange(nx * ny, dtype=float))
        ref['llv%d' % k] = eccodes.codes_get_array(h, 'latitudeLongitudeValues').reshape(-1, 3)
        msgs.append(eccodes.codes_get_message(h))
        eccodes.codes_release(h)
    open(os.path.join(out, 'sv.grib2'), 'wb').write(b''.join(msgs))
    np.savez_compressed(os.path.join(out, 'sv_ec.npz'), **ref)


MORE = [  # sub-satellite latitude, longitude, orientation, Nr (None: orthographic), j+
    (35.0, -100.0, 30.0, 6610839, 1),
    (90.0, 0.0, 0.0, 6610839, 1),
    (10.0, 140.0, 180.0, 6610839, 0),
    (40.0, 10.0, 0.0, None, 1),
]


def space_views_more(out):
    """sv_more.grib2: views neither ecCodes nor wgrib2 places -- off the
    equator, over a pole, turned, orthographic (Nr missing); and
    sv1.grb, a GRIB1 space view with ecCodes' points (sv1_ec.npz)."""
    msgs = []
    for lap, lop, ori, nr, jpos in MORE:
        h = eccodes.codes_grib_new_from_samples('GRIB2')
        eccodes.codes_set(h, 'gridDefinitionTemplateNumber', 90)
        for key, v in (('shapeOfTheEarth', 6), ('Nx', 81), ('Ny', 61),
                       ('latitudeOfSubSatellitePoint', int(lap * 1e6)),
                       ('longitudeOfSubSatellitePoint', int(lop * 1e6)), ('dx', 400), ('dy', 400),
                       ('Xp', 40000), ('Yp', 30000), ('orientationOfTheGrid', int(ori * 1e6)),
                       ('Xo', 0), ('Yo', 0), ('iScansNegatively', 0), ('jScansPositively', jpos)):
            eccodes.codes_set(h, key, v)
        if nr is None:
            eccodes.codes_set_missing(h, 'Nr')
        else:
            eccodes.codes_set(h, 'Nr', nr)
        eccodes.codes_set_values(h, np.arange(81 * 61, dtype=float))
        msgs.append(eccodes.codes_get_message(h))
        eccodes.codes_release(h)
    open(os.path.join(out, 'sv_more.grib2'), 'wb').write(b''.join(msgs))
    h = eccodes.codes_grib_new_from_samples('GRIB1')
    eccodes.codes_set(h, 'dataRepresentationType', 90)
    for key, v in (('Nx', 60), ('Ny', 50), ('latitudeOfSubSatellitePoint', 0),
                   ('longitudeOfSubSatellitePoint', -75000), ('dx', 600), ('dy', 600),
                   ('XpInGridLengths', 30), ('YpInGridLengths', 25), ('Nr', 6610839),
                   ('orientationOfTheGrid', 0), ('Xo', 0), ('Yo', 0), ('iScansNegatively', 0),
                   ('jScansPositively', 0)):
        eccodes.codes_set(h, key, v)
    eccodes.codes_set_values(h, np.arange(3000, dtype=float))
    np.savez_compressed(os.path.join(out, 'sv1_ec.npz'), llv=eccodes.codes_get_array(
        h, 'latitudeLongitudeValues').reshape(-1, 3))
    open(os.path.join(out, 'sv1.grb'), 'wb').write(eccodes.codes_get_message(h))
    eccodes.codes_release(h)


def irregular(out):
    """irr.grib2: 500 points listed one by one (3.130), each value its
    index, and its points as ecCodes reads them (irr_ec.npz: ecCodes places
    none)."""
    rng = np.random.default_rng(130)
    n = 500
    lat = np.round(rng.uniform(-89, 89, n), 6)
    lon = np.round(rng.uniform(0, 360, n), 6)
    h = eccodes.codes_grib_new_from_samples('GRIB2')
    eccodes.codes_set(h, 'gridDefinitionTemplateNumber', 130)
    eccodes.codes_set(h, 'numberOfDataPoints', n)
    eccodes.codes_set_array(h, 'latitude', np.round(lat * 1e6).astype(np.int64))   # (micro-
    eccodes.codes_set_array(h, 'longitude', np.round(lon * 1e6).astype(np.int64))  # degrees)
    eccodes.codes_set_values(h, np.arange(n, dtype=float))
    np.savez_compressed(os.path.join(out, 'irr_ec.npz'),
                        lat=eccodes.codes_get_array(h, 'latitude') / 1e6,
                        lon=eccodes.codes_get_array(h, 'longitude') / 1e6)
    open(os.path.join(out, 'irr.grib2'), 'wb').write(eccodes.codes_get_message(h))
    eccodes.codes_release(h)


def unstructured(out):
    h = eccodes.codes_grib_new_from_samples('GRIB2')
    eccodes.codes_set(h, 'gridDefinitionTemplateNumber', 101)
    eccodes.codes_set(h, 'numberOfGridUsed', 26)
    eccodes.codes_set(h, 'numberOfGridInReference', 1)
    eccodes.codes_set(h, 'uuidOfHGrid', 'a27b8de6-18c4-11e4-820a-b5b098c6a5c0'.replace('-', ''))
    n = 20480                                   # ICON R2B4's cells
    eccodes.codes_set(h, 'numberOfDataPoints', n)
    eccodes.codes_set_values(h, 250 + np.arange(n) * 0.001)
    open(os.path.join(out, 'ug.grb2'), 'wb').write(eccodes.codes_get_message(h))
    eccodes.codes_release(h)


if __name__ == '__main__':
    space_views(sys.argv[1])
    space_views_more(sys.argv[1])
    irregular(sys.argv[1])
    unstructured(sys.argv[1])
    print('sv.grib2, sv_more.grib2, sv1.grb, irr.grib2, ug.grb2')
