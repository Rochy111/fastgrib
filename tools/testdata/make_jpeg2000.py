"""Test data for fastgrib's JPEG2000 codec: tests/data/j2k.npz, OpenJPEG's
codestreams of small images coded every way Part 1 allows (each
code-block style alone and together, the five progression orders, tiles,
precincts, quality layers -- truncated, so lossy -- the 9/7 wavelet,
1 to 25 bits, odd shapes) with OpenJPEG's own decoding of each.

    gcc -O2 -o opj_tool tools/testdata/opj_tool.c -I/usr/include/openjpeg-2.4 -lopenjp2
    python3 tools/testdata/make_jpeg2000.py ./opj_tool OUT_DIR

(libopenjp2's development files: apt install libopenjp2-7-dev.)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

# name: w, h, bits, resolutions, code-block w, h, style, progression, tile, layers,
# irreversible, precinct exponent, rates
CASES = {
    'plain': (61, 47, 12, 6, 64, 64, 0, 0, 0, 1, 0, 0, ()),
    'one': (1, 1, 7, 1, 64, 64, 0, 0, 0, 1, 0, 0, ()),
    'row': (37, 1, 9, 1, 64, 64, 0, 0, 0, 1, 0, 0, ()),
    'column': (1, 29, 9, 1, 64, 64, 0, 0, 0, 1, 0, 0, ()),
    'bits1': (33, 21, 1, 4, 64, 64, 0, 0, 0, 1, 0, 0, ()),
    'bits16': (40, 30, 16, 5, 16, 16, 0, 0, 0, 1, 0, 0, ()),
    'bits25': (40, 30, 25, 5, 32, 8, 0, 0, 0, 1, 0, 0, ()),
    'levels0': (45, 33, 14, 1, 8, 8, 0, 0, 0, 1, 0, 0, ()),
    'bypass': (52, 44, 16, 5, 16, 16, 1, 0, 0, 1, 0, 0, ()),
    'reset': (52, 44, 16, 5, 16, 16, 2, 0, 0, 1, 0, 0, ()),
    'termall': (52, 44, 16, 5, 16, 16, 4, 0, 0, 1, 0, 0, ()),
    'causal': (52, 44, 16, 5, 16, 16, 8, 0, 0, 1, 0, 0, ()),
    'pterm': (52, 44, 16, 5, 16, 16, 16, 0, 0, 1, 0, 0, ()),
    'segsym': (52, 44, 16, 5, 16, 16, 32, 0, 0, 1, 0, 0, ()),
    'allstyles': (52, 44, 20, 5, 8, 32, 63, 0, 0, 1, 0, 0, ()),
    'rlcp': (70, 54, 12, 4, 8, 8, 0, 1, 0, 3, 0, 4, (30, 10, 0)),
    'rpcl': (70, 54, 12, 4, 8, 8, 0, 2, 0, 3, 0, 4, (30, 10, 0)),
    'pcrl': (70, 54, 12, 4, 8, 8, 0, 3, 32, 2, 0, 3, (20, 0)),
    'cprl': (70, 54, 12, 4, 8, 8, 0, 4, 32, 2, 0, 3, (20, 0)),
    'tiles': (77, 59, 16, 3, 16, 16, 0, 0, 24, 1, 0, 0, ()),
    'layers_lossy': (64, 50, 14, 5, 16, 16, 63, 0, 0, 4, 0, 0, (80, 40, 20, 10)),
    'irreversible': (64, 50, 12, 5, 32, 32, 0, 0, 0, 2, 1, 0, (30, 0)),
}


def image(w, h, bits, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    f = np.sin(x / 5.0) * np.cos(y / 7.0) + 0.3 * rng.standard_normal((h, w))
    f = (f - f.min()) / (f.max() - f.min() + 1e-12)
    return np.round(f * ((1 << bits) - 1)).astype(np.int64)


def main(tool, out):
    tmp = tempfile.mkdtemp()
    data = {}
    for k, (name, c) in enumerate(CASES.items()):
        w, h, bits = c[:3]
        x = image(w, h, bits, k)
        x.astype('<i4').tofile(os.path.join(tmp, 'in.i32'))
        args = [str(v) for v in (w, h, bits, 0) + c[3:12]] + [str(r) for r in c[12]]
        subprocess.run([tool, 'enc', os.path.join(tmp, 'in.i32'), os.path.join(tmp, 'x.j2k')] + args,
                       check=True)
        subprocess.run([tool, 'dec', os.path.join(tmp, 'x.j2k'), os.path.join(tmp, 'x.i32')],
                       check=True, stdout=subprocess.DEVNULL)
        data['cs_' + name] = np.fromfile(os.path.join(tmp, 'x.j2k'), np.uint8)
        data['ref_' + name] = np.fromfile(os.path.join(tmp, 'x.i32'), '<i4')
        data['in_' + name] = x.ravel().astype('<i4')
    np.savez_compressed(os.path.join(out, 'j2k.npz'), **data)
    print('j2k.npz:', len(CASES), 'codestreams')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
