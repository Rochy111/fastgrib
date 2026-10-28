"""Time fastgrib decoding every message of GRIB2 files, outside any other
code: one thread and all threads, with the machine's details, so a slow
pipeline can be told apart from slow decoding.

    python tools/time_files.py FILE [FILE ...]
"""
import os
import platform
import sys
import time

import numba
import numpy as np

import fastgrib
from fastgrib import _cpu


def main(paths):
    _cpu._fast_thread()
    fastgrib.warm()
    print('fastgrib %s, numba %s, numpy %s, Python %s, %s, %s logical CPUs' % (
        fastgrib.__version__, numba.__version__, np.__version__, platform.python_version(),
        platform.platform(), os.cpu_count()))
    print(platform.processor())
    for p in paths:
        f = fastgrib.open(p)
        packs = sorted({f.keys(i)['packingType'] for i in range(len(f))})
        print('\n%s: %d messages, %s' % (os.path.basename(p), len(f), ', '.join(packs)))
        for n in (1, None):
            fastgrib.set_threads(n)
            ms = []
            for i in range(len(f)):
                t = time.perf_counter()
                f.read(i)
                ms.append((time.perf_counter() - t) * 1e3)
            ms = np.array(ms)
            print('  %2d threads: mean %6.1f  median %6.1f  max %6.1f ms per message' % (
                _cpu.threads(), ms.mean(), np.median(ms), ms.max()))
        f.close()


if __name__ == '__main__':
    main(sys.argv[1:])
