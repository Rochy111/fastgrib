"""Test data: grids no wgrib2 -new_grid makes, built with ecCodes from its
samples -- rotated lat-lon (3.1), reduced (thinned) Gaussian (3.40 with a
points-per-row list), rotated and reduced rotated Gaussian (3.41) -- each
field a smooth function of every point's true latitude and longitude (a
misplaced point shows), simple packing. Beside each NAME.grib2,
NAME_ec.npz: ecCodes' own latitudes, longitudes and values of its points,
in the message's order (ecCodes does not unrotate a regular rotated
Gaussian grid's: rot_gg's are its rotated coordinates).

    python3 tools/testdata/make_grids_eccodes.py OUT_DIR

Needs ecCodes' Python bindings (pip install eccodes eccodeslib).
"""
import os
import sys

import numpy as np
import eccodes

GRIDS = {
    'rot_ll': ('rotated_ll_sfc_grib2', {'jScansPositively': 1,
        'Ni': 61, 'Nj': 41, 'latitudeOfFirstGridPointInDegrees': -10.0,
        'longitudeOfFirstGridPointInDegrees': -15.0, 'latitudeOfLastGridPointInDegrees': 10.0,
        'longitudeOfLastGridPointInDegrees': 15.0, 'iDirectionIncrementInDegrees': 0.5,
        'jDirectionIncrementInDegrees': 0.5, 'latitudeOfSouthernPoleInDegrees': -40.0,
        'longitudeOfSouthernPoleInDegrees': 10.0, 'angleOfRotationInDegrees': 0.0}),
    'rot_ll_ns': ('rotated_ll_sfc_grib2', {
        'Ni': 31, 'Nj': 21, 'latitudeOfFirstGridPointInDegrees': 5.0,
        'longitudeOfFirstGridPointInDegrees': -7.5, 'latitudeOfLastGridPointInDegrees': -5.0,
        'longitudeOfLastGridPointInDegrees': 7.5, 'iDirectionIncrementInDegrees': 0.5,
        'jDirectionIncrementInDegrees': 0.5, 'jScansPositively': 0,
        'latitudeOfSouthernPoleInDegrees': -32.5, 'longitudeOfSouthernPoleInDegrees': -170.0,
        'angleOfRotationInDegrees': 0.0}),
    'red_gg': ('reduced_gg_pl_32_grib2', {}),
    'rot_gg': ('rotated_gg_sfc_grib2', {'latitudeOfSouthernPoleInDegrees': -40.0,
                                         'longitudeOfSouthernPoleInDegrees': 10.0}),
    'red_rot_gg': ('reduced_rotated_gg_pl_32_grib2', {'latitudeOfSouthernPoleInDegrees': -40.0,
                                                      'longitudeOfSouthernPoleInDegrees': 10.0}),
}


def field(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return 280 + 25 * np.cos(la) + 6 * np.sin(2 * lo) * np.cos(la) + 0.01 * lon


def main(out):
    for name, (sample, keys) in GRIDS.items():
        h = eccodes.codes_grib_new_from_samples(sample)
        for k, v in keys.items():
            eccodes.codes_set(h, k, v)
        eccodes.codes_set(h, 'packingType', 'grid_simple')
        eccodes.codes_set(h, 'bitsPerValue', 16)
        if 'Ni' in keys:                        # the values sized to the new grid first
            eccodes.codes_set_values(h, np.zeros(keys['Ni'] * keys['Nj']))
        lat = eccodes.codes_get_array(h, 'latitudes')
        lon = eccodes.codes_get_array(h, 'longitudes')
        eccodes.codes_set_values(h, field(lat, lon))
        with open(os.path.join(out, name + '.grib2'), 'wb') as f:
            eccodes.codes_write(h, f)
        np.savez_compressed(os.path.join(out, name + '_ec.npz'), lat=lat, lon=lon,
                            values=eccodes.codes_get_values(h))
        print(name, eccodes.codes_get(h, 'gridType'), lat.size, 'points')
        eccodes.codes_release(h)


if __name__ == '__main__':
    main(sys.argv[1])
