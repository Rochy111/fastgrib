"""Test data for spherical harmonics: tests/data/sh.grb2 (ecCodes' T63
samples, a surface and a model-level field, each packed complex 5.51 and
simple 5.50) and g1_sh.grb (GRIB1: complex, as GRIBEX writes it, and
simple), with sh_ec.npz: ecCodes' coefficients of each message, and
pyshtools' synthesis of the GRIB2 complex ones at fixed points -- an
independent check of fastgrib's (4pi-normalised real harmonics, no
Condon-Shortley phase, which ECMWF's complex coefficients map onto).

    python3 tools/testdata/make_spectral.py OUT_DIR

Needs ecCodes' Python bindings and pyshtools (pip install eccodes
eccodeslib pyshtools).
"""
import os
import sys

import numpy as np
import eccodes
import pyshtools

POINTS = 500


def points():
    rng = np.random.default_rng(7)
    lat = np.concatenate([rng.uniform(-90, 90, POINTS), [90, -90, 0, 89.99, -45.5]])
    lon = np.concatenate([rng.uniform(-180, 540, POINTS), [0, 123, 0, 10, 359.9]])
    return lat, lon


def shtools(c, J):
    cilm = np.zeros((2, J + 1, J + 1))
    k = 0
    for m in range(J + 1):
        for n in range(m, J + 1):
            re, im = c[k], c[k + 1]
            k += 2
            if m == 0:
                cilm[0, n, 0] = re
            else:
                cilm[0, n, m] = np.sqrt(2) * re
                cilm[1, n, m] = -np.sqrt(2) * im
    lat, lon = points()
    x = pyshtools.SHCoeffs.from_array(cilm, normalization='4pi', csphase=1)
    return np.asarray(x.expand(lat=lat, lon=lon), np.float64)


def main(out):
    ref = {}
    for name, samples in (('sh.grb2', ('sh_sfc_grib2', 'sh_ml_grib2')),
                          ('g1_sh.grb', ('sh_sfc_grib1', 'sh_ml_grib1'))):
        msgs = []
        for s in samples:
            for packing in ('spectral_complex', 'spectral_simple'):
                h = eccodes.codes_grib_new_from_samples(s)
                if packing == 'spectral_simple':    # (GRIB2's: repacked only when set again)
                    c = eccodes.codes_get_values(h)
                    eccodes.codes_set(h, 'packingType', packing)
                    eccodes.codes_set_values(h, c)
                msgs.append(eccodes.codes_get_message(h))
                k = '%s%d' % (name.split('.')[0], len(msgs) - 1)
                c = eccodes.codes_get_values(h)
                ref['v_' + k] = c
                if name == 'sh.grb2' and packing == 'spectral_complex':
                    ref['synth_' + k] = shtools(c, eccodes.codes_get(h, 'J'))
                eccodes.codes_release(h)
        open(os.path.join(out, name), 'wb').write(b''.join(msgs))
    lat, lon = points()
    rotated_stretched(out, ref)
    np.savez_compressed(os.path.join(out, 'sh_ec.npz'), lat=lat, lon=lon, **ref)
    print(sorted(ref))


# rotated (3.51 / GRIB1 60), stretched (52 / 70), both (53 / 80): the southern pole and
# angle; the pole of stretching and factor -- ARPEGE's kind: the rotation bringing the
# stretching pole to the frame's north pole
MORE = ((-40.0, 10.0, 25.0, None), (None, None, None, (90.0, 0.0, 2.4)),
        (-46.5, 182.5, 0.0, (46.5, 2.5, 2.4)))


def rotated_stretched(out, ref):
    """sh_more.grb2 and g1_sh_more.grb: the T63 sample's coefficients on
    rotated and stretched grids, ecCodes' coefficients of each (v_shm*,
    v_g1_shm*)."""
    for name, sample, tmpls, key in (('sh_more.grb2', 'sh_ml_grib2', (51, 52, 53),
                                      'gridDefinitionTemplateNumber'),
                                     ('g1_sh_more.grb', 'sh_ml_grib1', (60, 70, 80),
                                      'dataRepresentationType')):
        msgs = []
        for t, (splat, splon, angle, stretch) in zip(tmpls, MORE):
            h = eccodes.codes_grib_new_from_samples(sample)
            c = eccodes.codes_get_values(h)
            eccodes.codes_set(h, key, t)
            if splat is not None:
                eccodes.codes_set(h, 'latitudeOfSouthernPoleInDegrees', splat)
                eccodes.codes_set(h, 'longitudeOfSouthernPoleInDegrees', splon)
                eccodes.codes_set(h, 'angleOfRotationInDegrees', angle)
            if stretch is not None:
                eccodes.codes_set(h, 'latitudeOfStretchingPoleInDegrees', stretch[0])
                eccodes.codes_set(h, 'longitudeOfStretchingPoleInDegrees', stretch[1])
                eccodes.codes_set(h, 'stretchingFactor', stretch[2])
            eccodes.codes_set_values(h, c)
            msgs.append(eccodes.codes_get_message(h))
            ref['v_%s%d' % ('shm' if t < 60 else 'g1_shm', len(msgs) - 1)] = \
                eccodes.codes_get_values(h)
            eccodes.codes_release(h)
        open(os.path.join(out, name), 'wb').write(b''.join(msgs))


if __name__ == '__main__':
    main(sys.argv[1])
