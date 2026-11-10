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
    for w, h in ((1, 1), (17, 1), (1, 9), (70, 45), (33000, 2)):
        # (wider than 2^15: precincts past the first, each its own packet --
        # a field under a bitmap is one row of its values, a million or more)
        x = rng.integers(0, 1 << bits, w * h)
        assert np.array_equal(J.decode(J.encode(x, w, h, bits)), x)
    if bits >= 26:
        f = fastgrib.open(path('gfs'))
        v = f.read(0)
        g = fastgrib.open(f.encode(v, 0, packing='jpeg', bits=30))
        assert g.keys(0)['packingType'] == 'grid_jpeg' and g.keys(0)['bitsPerValue'] == 30
        assert np.allclose(g.read(0), v, rtol=0, atol=1e-6 * np.abs(v).max())


def _region_streams():
    """OpenJPEG's small codestreams, fastgrib's of a 900 x 700 image, and
    (where Pillow has OpenJPEG) OpenJPEG's of it: the 9/7 wavelet, tiles
    off the origin, precincts, small code-blocks, quality layers."""
    z = np.load(path('j2k', '.npz'))
    out = [z[k].tobytes() for k in z.files if k.startswith('cs_')]
    yy, xx = np.mgrid[0:700, 0:900]
    rng = np.random.default_rng(5)
    img = (30000 + 12000 * np.sin(xx / 37.0) * np.cos(yy / 53.0) +
           rng.normal(0, 300, xx.shape)).astype(np.uint16)
    out.append(J.encode(img.ravel().astype(np.int64), 900, 700, 16))
    try:
        import io
        from PIL import Image, features
        if not features.check('jpg_2000'):
            raise ImportError
    except ImportError:
        return out
    for cfg in (dict(irreversible=True, num_resolutions=6),
                dict(irreversible=True, tile_offset=(13, 7), offset=(13, 7),
                     tile_size=(300, 300), num_resolutions=5),
                dict(irreversible=True, precinct_size=(128, 128), codeblock_size=(32, 16),
                     progression='RPCL', num_resolutions=5),
                dict(irreversible=True, quality_mode='rates', quality_layers=[40, 10],
                     num_resolutions=7)):
        b = io.BytesIO()
        Image.fromarray(img).save(b, 'JPEG2000', no_jp2=True, **cfg)
        out.append(b.getvalue())
    return out


@pytest.mark.parametrize('windows', [False, True])
def test_jpeg2000_region(windows, monkeypatch):
    """decode(want=...): the samples wanted as a whole decode gives them
    -- points anywhere, boxes, corners -- from only the code-blocks
    reaching them, the inverse wavelet over windows around them (forced
    here even where it would not be taken) or over the whole."""
    whole = J._windows
    monkeypatch.setattr(J, '_windows', lambda p, nl, a: whole(p, nl, a)[:4] + (
        nl > 0 if windows else False,))
    rng = np.random.default_rng(7)
    for cs in _region_streams():
        full = J.decode(cs)
        n = full.size
        w = 900 if n == 630000 else max(1, int(np.sqrt(n)))
        for t in range(12):
            if t % 3 == 0:
                want = rng.integers(0, n, int(rng.integers(1, 40)))
            elif t % 3 == 1:
                a = int(rng.integers(0, n))
                want = np.unique(np.minimum(n - 1, a + (np.arange(30)[:, None] * w +
                                                        np.arange(30)).ravel()))
            else:
                want = np.unique([0, n - 1, min(w, n) - 1, max(0, n - w)])
            assert np.array_equal(J.decode(cs, want=want)[want], full[want])


def test_jpeg2000_wanted_remote():
    """wanted(): a remote read's walk -- each header asked for as it is
    reached (fetched here a few bytes at a time, so a header runs past what
    is in hand again and again), then the bodies of the code-blocks
    decode(want=) takes. From those bytes alone, the rest zeros, the
    samples wanted decode as from the whole."""
    rng = np.random.default_rng(11)
    for cs in _region_streams():
        full = J.decode(cs)
        whole = np.frombuffer(cs, np.uint8)
        for want in (rng.integers(0, full.size, 3), np.array([full.size - 1])):
            held, data = [], np.zeros_like(whole)
            last, step = -1, 0
            for _ in range(10000):
                try:
                    spans = J.wanted(data, held or [(0, 0)], want)
                    break
                except J._Need as n:
                    a = n.args[0]
                    step = 2 * step if a == last else 37        # (on while a header runs on)
                    last = a
                    b = min(whole.size, a + step)
                    data[a:b] = whole[a:b]
                    merged = []                 # (in order, apart: as _jpeg2000_part keeps them)
                    for x in sorted(held + [[a, b]]):
                        if merged and x[0] <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], x[1])
                        else:
                            merged.append(list(x))
                    held = merged
            for a, b in spans:
                data[a:b] = whole[a:b]
            assert sum(b - a for a, b in spans) < whole.size
            assert np.array_equal(J.decode(data.tobytes(), want=want)[want], full[want])
    with pytest.raises(J._Need):                # nothing in hand: its first bytes
        J.wanted(np.zeros(100, np.uint8), [(0, 0)], [0])


def test_jpeg2000_damaged():
    """A damaged codestream raises a ValueError, never a crash or a wrong
    field."""
    cs = np.load(path('j2k', '.npz'))['cs_plain'].tobytes()
    for bad in (cs[:10], cs[:len(cs) // 2], b'\xff\x4f\xff\x51' + cs[4:40], b'not jpeg2000'):
        with pytest.raises(ValueError):
            J.decode(bad)
