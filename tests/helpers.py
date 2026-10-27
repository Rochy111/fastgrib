"""What the tests share: where the test data are, wgrib2's reading of a
file, bit-for-bit comparison, the GPU check, a GRIB1 file opened quietly,
an HTTP handler that serves byte ranges."""
import glob
import http.server
import os

import numpy as np

import fastgrib
from fastgrib import _file

DATA = os.path.join(os.path.dirname(__file__), 'data')


FILES = sorted(os.path.basename(p)[:-6] for p in glob.glob(os.path.join(DATA, '*.grib2')))


UNDEF = np.float32(9.999e20)


def path(name, ext='.grib2'):
    return os.path.join(DATA, name + ext)


def reference(name):
    """wgrib2's values, a float32 array per field."""
    f = fastgrib.open(path(name))
    sizes = [r.grid.ni * r.grid.nj for r in f.records]
    if os.path.exists(path(name, '.sha256')):
        return f, None
    flat = np.fromfile(path(name, '.bin'), '<f4')
    out, at = [], 0
    for r, n in zip(f.records, sizes):
        out.append(flat[at:at + n].reshape(r.grid.shape))
        at += n
    return f, out


def same(ours, wgrib2):
    ours = np.asarray(ours)
    miss = wgrib2 == UNDEF
    return (np.array_equal(np.isnan(ours), miss)
            and np.array_equal(ours[~miss].view(np.uint32), wgrib2[~miss].view(np.uint32)))


def as_wgrib2(v):
    return np.where(np.isnan(v), UNDEF, v).astype('<f4')


def gpu_ok():
    try:
        return _file._try_gpu()
    except Exception:
        return False


def grib1(name):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fastgrib.open(path(name, '.grb'))


class Ranges(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.translate_path(self.path)
        if not os.path.isfile(p):
            self.send_error(404)
            return
        size = os.path.getsize(p)
        rng = self.headers.get('Range')
        a, b = 0, size - 1
        if rng:
            a, b = rng.split('=')[1].split('-')
            a, b = int(a), min(int(b) if b else size - 1, size - 1)
        with open(p, 'rb') as f:                # only the range: files may be big
            f.seek(a)
            body = f.read(b - a + 1)
        self.send_response(206 if rng else 200)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
