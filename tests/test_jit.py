"""The deferred numba import: once the first kernel has loaded numba,
every fastgrib module -- one imported later too -- sees numba's own prange.
(A stand-in left behind fails only when its kernel compiles from scratch,
on a fresh install with no cache to hide it.) In a fresh interpreter, so
the import order is the one under test."""
import subprocess
import sys

from helpers import path


def test_prange_replaced_everywhere_after_load():
    code = '''
import sys
import numba
import fastgrib
from fastgrib import _jit
fastgrib.open(sys.argv[1]).read(0)                  # the first kernel: numba loads
assert _jit.prange is numba.prange, _jit.prange
from fastgrib._jit import prange                    # as a module imported later would
assert prange is numba.prange, prange
assert all(getattr(m, 'prange', numba.prange) is numba.prange
           for n, m in list(sys.modules.items()) if n.startswith('fastgrib') and m is not None)
print('ok')
'''
    out = subprocess.run([sys.executable, '-c', code, path('gfs')], capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith('ok'), out.stderr[-2000:]
