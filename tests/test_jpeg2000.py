"""fastgrib's own JPEG2000: OpenJPEG's codestreams of every coding Part 1
allows decoded as OpenJPEG decodes them, and its encoder's codestreams
exact to 31 bits a sample (OpenJPEG, under ecCodes and wgrib2, loses
samples past 25 bits)."""
import numpy as np
import pytest

import fastgrib
from fastgrib import _jpeg2000 as J

from helpers import path


def test_jpeg2000_decodes_as_openjpeg():
    """tools/testdata/make_jpeg2000.py's codestreams (code-block styles,
    progressions, tiles, precincts, layers, 1 to 25 bits, odd shapes):
    every sample as OpenJPEG gives it -- the 9/7 wavelet's to its float32
    rounding -- lossless ones the image itself."""
    z = np.load(path('j2k', '.npz'))
    names = [k[3:] for k in z.files if k.startswith('cs_')]
    assert len(names) == 22
    for name in names:
        got = J.decode(z['cs_' + name].tobytes())
        ref = z['ref_' + name].astype(np.int64)
        if name == 'irreversible':
            assert np.abs(got - ref).max() <= 1, name
        else:
            assert np.array_equal(got, ref), name
        if name not in ('irreversible', 'layers_lossy', 'rlcp', 'rpcl', 'pcrl', 'cprl'):
            assert np.array_equal(got, z['in_' + name]), name
        assert np.array_equal(J.decode(z['cs_' + name].tobytes(), parallel=True), got)


@pytest.mark.parametrize('bits', [1, 8, 16, 25, 26, 31])
def test_jpeg2000_encodes_exactly(bits):
    """encode() of any depth to 31 bits decodes back exactly (odd shapes
    too); a GRIB field packed 'jpeg' at 30 bits reads back to its own
    precision."""
    rng = np.random.default_rng(bits)
    for w, h in ((1, 1), (17, 1), (1, 9), (70, 45)):
        x = rng.integers(0, 1 << bits, w * h)
        assert np.array_equal(J.decode(J.encode(x, w, h, bits)), x)
    if bits >= 26:
        f = fastgrib.open(path('gfs'))
        v = f.read(0)
        g = fastgrib.open(f.encode(v, 0, packing='jpeg', bits=30))
        assert g.keys(0)['packingType'] == 'grid_jpeg' and g.keys(0)['bitsPerValue'] == 30
        assert np.allclose(g.read(0), v, rtol=0, atol=1e-6 * np.abs(v).max())


def test_jpeg2000_damaged():
    """A damaged codestream raises a ValueError, never a crash or a wrong
    field."""
    cs = np.load(path('j2k', '.npz'))['cs_plain'].tobytes()
    for bad in (cs[:10], cs[:len(cs) // 2], b'\xff\x4f\xff\x51' + cs[4:40], b'not jpeg2000'):
        with pytest.raises(ValueError):
            J.decode(bad)
