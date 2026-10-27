"""Time fastgrib on a GRIB2 file: every field through fields() on the CPU,
on the GPU into numpy and on the card, and one field at a time; ms a field
and CPU ms a field.

    python tools/bench.py FILE [--repeats 3]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastgrib                                             # noqa: E402
from fastgrib import _file                                 # noqa: E402


def best(fn, repeats):
    out = (float('inf'), 0.0)
    for _ in range(repeats):
        w, c = time.perf_counter(), time.process_time()
        fn()
        out = min(out, (time.perf_counter() - w, time.process_time() - c))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('file')
    ap.add_argument('--repeats', type=int, default=3)
    a = ap.parse_args()
    f = fastgrib.open(a.file)
    n = len(f)
    g = f.grid()
    print('%s: %d fields, %s %s' % (os.path.basename(a.file), n, g, g.shape))
    gpu = _file._try_gpu()
    rows = [('CPU, fields()', 'cpu', False)]
    if gpu:
        rows += [('GPU -> numpy, fields()', 'gpu', False), ('GPU on the card, fields()', 'gpu', True)]
    for name, dev, card in rows:
        fastgrib.set_device(dev)
        sync = (lambda: __import__('cupy').cuda.get_current_stream().synchronize()) if card \
            else (lambda: None)
        for _ in f.fields(f.records[:8], gpu=card):
            pass
        w, c = best(lambda: ([None for _ in f.fields(gpu=card)], sync()), a.repeats)
        print('  %-28s %7.3f ms a field  (CPU %6.2f ms)' % (name, w / n * 1e3, c / n * 1e3))
    import numpy as np
    out = np.empty(g.shape, np.float32)
    singles = [('CPU, read() a field', 'cpu', False), ('CPU, read(out=) a field', 'cpu', True)]
    if gpu:
        singles += [('GPU -> numpy, read()', 'gpu', False), ('GPU -> numpy, read(out=)', 'gpu', True)]
    picks = [r for r in f.records[::max(1, n // 40)] if r.grid == g]
    for name, dev, reuse in singles:
        fastgrib.set_device(dev)
        o = out if reuse else None
        f.read(picks[0], out=o)
        w, c = best(lambda: [f.read(r, out=o) for r in picks], a.repeats)
        print('  %-28s %7.3f ms a field  (CPU %6.2f ms)' % (name, w / len(picks) * 1e3,
                                                            c / len(picks) * 1e3))
    fastgrib.set_device('auto')
    return 0


if __name__ == '__main__':
    sys.exit(main())
