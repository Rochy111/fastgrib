"""The kernel image (_image.py): kernels run with no numba at all and read
as numba does, and whatever an image cannot do -- no image for this machine,
a damaged one, a kernel or argument types it does not hold -- goes to numba.
In fresh interpreters, each with images of its own (FASTGRIB_CACHE_DIR): an
image is looked for once a process."""
import json
import marshal
import os
import shutil
import struct
import subprocess
import sys
import time

import numpy as np
import pytest

from fastgrib import _image

from helpers import path

pytestmark = pytest.mark.skipif(_image.cpu_id() is None, reason='no kernel image on this machine')

# a field of every packing (simple, complex in runs, PNG, JPEG2000, CCSDS,
# IEEE, run-length, spherical harmonics, GRIB1 second-order)
FILES = [path(n) for n in ('gfs_simple', 'gfs', 'nbm', 'nbm_qmd_hi', 'gfs_png', 'gfs_jpeg', 'gfs_aec', 'gfs_ieee',
                           'gfs_rle')] + [path('sh', '.grb2'), path('g1_2nd', '.grb')]

READ = '''
import hashlib, json, sys, warnings
warnings.simplefilter('ignore')
import fastgrib
from fastgrib import _jit
values = []
for name in sys.argv[1:]:
    f = fastgrib.open(name)
    values += [hashlib.sha256(f.read(i).tobytes()).hexdigest() for i in range(len(f.records))]
    values += [hashlib.sha256(v.tobytes()).hexdigest() for _, v in f.fields()]
print(json.dumps(dict(values=values, image=_jit.image() is not None, numba='numba' in sys.modules)))
'''


def run(code, *args, cache=None, **env):
    """What a fresh interpreter printed last, as JSON: with an image unless
    `env` says otherwise, whatever the tests themselves run with."""
    e = {k: v for k, v in os.environ.items() if k not in ('FASTGRIB_IMAGE', 'FASTGRIB_CACHE_DIR')}
    e.update(FASTGRIB_AUTOWARM='0', FASTGRIB_IMAGE_BUILD='0', FASTGRIB_DEVICE='cpu')
    e.update(env)
    if cache is not None:
        e['FASTGRIB_CACHE_DIR'] = str(cache)
    out = subprocess.run([sys.executable, '-c', code] + [str(a) for a in args], capture_output=True, text=True,
                         timeout=600, env=e)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.fixture(scope='module')
def images(tmp_path_factory):
    """A directory with this machine's image, built as warm() builds one."""
    d = tmp_path_factory.mktemp('images')
    env = dict(os.environ, FASTGRIB_CACHE_DIR=str(d), FASTGRIB_AUTOWARM='0', FASTGRIB_IMAGE_BUILD='1',
               FASTGRIB_DEVICE='cpu')
    env.pop('FASTGRIB_IMAGE', None)
    out = subprocess.run([sys.executable, '-c', 'import fastgrib; fastgrib.warm()'], capture_output=True,
                         text=True, timeout=1800, env=env)
    assert out.returncode == 0, out.stderr[-3000:]
    assert os.path.exists(d / _image._name()) and not os.path.exists(d / (_image._name() + '.failed'))
    return d


@pytest.fixture(scope='module')
def by_numba():
    return run(READ, *FILES, FASTGRIB_IMAGE='0')['values']


def copy(images, tmp_path):
    d = tmp_path / 'images'
    shutil.copytree(images, d)
    return d


def test_reads_without_numba_as_numba_does(images, by_numba):
    got = run(READ, *FILES, cache=images)
    assert got['image'] and not got['numba']
    assert got['values'] == by_numba


def test_points_regrids_winds_and_packers_as_numba_does(images):
    code = '''
import hashlib, json, sys, warnings
warnings.simplefilter('ignore')
import numpy as np
import fastgrib
from fastgrib import _grids, _jit, _warm, _winds


def h(a):
    a = np.ascontiguousarray(getattr(a, 'values', a)) if not isinstance(a, (bytes, bytearray)) else a
    return hashlib.sha256(bytes(a) if isinstance(a, (bytes, bytearray)) else a.tobytes()).hexdigest()


f = fastgrib.open(_warm.SAMPLE)
out = []
for r in {_grids.layout(r.grid): r for r in f.records}.values():
    if r.grid.template not in _grids._SPECTRAL:
        out += [h(x) for x in f.latlon(r)]
    for method in (None, 'nearest'):
        out.append(h(f.regrid((np.array([30.0, 40.0]), np.array([250.0, 260.0])), r, method=method)))
u, v = f.winds(name='UGRD')
g = _grids.layout(f.find(':UGRD:')[0].grid)
out += [h(x) for x in _winds.to_grid(u, v, g) + _winds.to_earth(u.astype(np.float64), v.astype(np.float64), g)]
k = next(k for k, r in enumerate(f.records) if r.packing == 0 and r.bitmap < 0)
for packing in ('simple', 'complex', 'ieee', 'png', 'jpeg', 'ccsds'):
    out.append(h(f.encode(f.read(k), k, packing=packing)))
k = next(k for k, r in enumerate(f.records) if r.packing == 51)
out.append(h(f.encode(f.read(k), k, packing='simple')))
print(json.dumps(dict(values=out, image=_jit.image() is not None, numba='numba' in sys.modules)))
'''
    got = run(code, cache=images)
    assert got['image'] and not got['numba']
    assert got['values'] == run(code, FASTGRIB_IMAGE='0')['values']


def test_first_kernels_from_many_threads_at_once(images):
    code = '''
import hashlib, json, sys
from concurrent.futures import ThreadPoolExecutor
import fastgrib
from fastgrib import _jit
f = fastgrib.open(sys.argv[1])
n = len(f.records)
with ThreadPoolExecutor(16) as ex:
    got = list(ex.map(lambda i: hashlib.sha256(f.read(i % n).tobytes()).hexdigest(), range(4 * n)))
print(json.dumps(dict(values=got[:n], same=got == got[:n] * 4, image=_jit.image() is not None,
                      numba='numba' in sys.modules)))
'''
    got = run(code, FILES[1], cache=images)
    assert got['image'] and not got['numba'] and got['same']
    assert got['values'] == run(READ, FILES[1], FASTGRIB_IMAGE='0')['values'][:len(got['values'])]


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='no fork here')
def test_workers_forked_after_the_image_was_used(images):
    """Units the parent mapped come along; those a child maps it reads for
    itself, at its own place in the file the parent opened."""
    code = '''
import hashlib, json, multiprocessing, sys
import fastgrib
from fastgrib import _jit


def read(name):
    f = fastgrib.open(name)
    return [hashlib.sha256(f.read(i).tobytes()).hexdigest() for i in range(len(f.records))], 'numba' in sys.modules


fastgrib.open(sys.argv[1]).read(0)
with multiprocessing.get_context('fork').Pool(2) as pool:
    got = pool.map(read, sys.argv[1:] * 2)
print(json.dumps(dict(values=[v for v, _ in got[:len(sys.argv) - 1]], numba=any(n for _, n in got),
                      same=got[:len(sys.argv) - 1] == got[len(sys.argv) - 1:], image=_jit.image() is not None)))
'''
    names = [FILES[0], path('gfs_jpeg'), path('gfs_png')]
    got = run(code, *names, cache=images)
    assert got['image'] and not got['numba'] and got['same']
    want = [run(READ.replace('values += [hashlib.sha256(v.tobytes()).hexdigest() for _, v in f.fields()]', ''),
                n, FASTGRIB_IMAGE='0')['values'] for n in names]
    assert got['values'] == want


def test_an_array_of_any_class_goes_by_address(images, tmp_path):
    """Never as a word: a one-element array passes int(), and its value would
    be taken for an address."""
    raw = tmp_path / 'one'
    raw.write_bytes(b'\xff')
    code = '''
import json, sys
import numpy as np
from fastgrib import _jit, _octets


class Sub(np.ndarray):
    pass


m = np.memmap(sys.argv[1], np.uint8, 'r', shape=(1,))
s = np.full(1, 0x0f, np.uint8).view(Sub)
s.flags.writeable = False
got = [_octets.ones(a, 8) for a in (m, m, s, s)]          # the second of each: the direct way in
words = {t[0].__name__: w for t, (u, w) in _octets.ones.fast.items()}
print(json.dumps(dict(got=got, words=words, image=_jit.image() is not None, numba='numba' in sys.modules)))
'''
    got = run(code, raw, cache=images)
    assert got['image'] and not got['numba']
    assert got['got'] == [8, 8, 4, 4]
    assert got['words'] == {'memmap': [1], 'Sub': [1]}


def test_an_error_raised_as_numba_raises_it(images):
    code = '''
import json, sys
import numpy as np
from fastgrib import _cpu, _jit
try:
    _cpu._turn_rows(np.zeros(4, np.float32), 0)
    got = None
except Exception as e:
    got = [type(e).__name__, str(e)]
print(json.dumps(dict(got=got, image=_jit.image() is not None, numba='numba' in sys.modules)))
'''
    got = run(code, cache=images)
    assert got['image'] and not got['numba']
    assert got['got'] == run(code, FASTGRIB_IMAGE='0')['got'] == ['ZeroDivisionError', 'integer division by zero']


def test_what_an_image_does_not_hold_goes_to_numba(images, by_numba):
    code = '''
import hashlib, json, sys
import fastgrib
from fastgrib import _image, _jit
_image.open_image().kernels.pop('fastgrib._cpu._decode_one')
f = fastgrib.open(sys.argv[1])
got = [hashlib.sha256(f.read(i).tobytes()).hexdigest() for i in range(len(f.records))]
print(json.dumps(dict(values=got, image=_jit.image() is not None, numba='numba' in sys.modules)))
'''
    got = run(code, FILES[0], cache=images)
    assert got['image'] and got['numba']
    assert got['values'] == by_numba[:len(got['values'])]


def _damage(d, how):
    pointer = d / _image._name()
    image = d / pointer.read_text().strip()
    b = bytearray(image.read_bytes())
    _, _, _, size, at, n = _image.HEAD.unpack_from(b)
    if how == 'unit':                           # every unit of the kernel a read needs
        for r in marshal.loads(bytes(b[at:at + n]))['kernels']['fastgrib._cpu._decode_one'].values():
            b[r['at'] + r['stored'] // 2] ^= 0xFF
    elif how == 'index':
        b[at + n // 2] ^= 0xFF
    elif how == 'cut':
        b = b[:len(b) // 2]
    elif how == 'head':
        b[:8] = b'NOTANIMG'
    if how == 'gone':
        image.unlink()
    elif how == 'pointer':
        pointer.write_bytes(os.urandom(40))
    else:
        image.write_bytes(bytes(b))


@pytest.mark.parametrize('how', ['unit', 'index', 'cut', 'head', 'gone', 'pointer'])
def test_a_damaged_image_goes_to_numba(images, by_numba, tmp_path, how):
    d = copy(images, tmp_path)
    _damage(d, how)
    got = run(READ, FILES[0], cache=d)
    assert got['numba'] and got['image'] == (how == 'unit')     # (only a unit's own CRC is seen late)
    assert got['values'] == by_numba[:len(got['values'])]


def test_no_image_goes_to_numba(by_numba, tmp_path):
    got = run(READ, FILES[0], cache=tmp_path)
    assert not got['image'] and got['numba']
    assert got['values'] == by_numba[:len(got['values'])]


def test_turned_off_never_looks(images):
    code = '''
import json, sys
import fastgrib
fastgrib.open(sys.argv[1]).read(0)
print(json.dumps(dict(looked='fastgrib._image' in sys.modules, numba='numba' in sys.modules)))
'''
    assert run(code, FILES[0], cache=images, FASTGRIB_IMAGE='0') == dict(looked=False, numba=True)


def test_an_image_answers_for_its_machine_and_sources_only():
    name = _image._name()
    assert name != _image._name('00000000')
    assert _image._warm._sources()[:16] in name
    assert ('py%d.%d' % sys.version_info[:2]) in name and ('np' + np.__version__.split('.')[0]) in name


def test_a_build_that_fails_its_trial_is_never_seen(tmp_path):
    code = '''
import json, os
from fastgrib import _image
_image._tried_in_process = lambda *a: False
_image.build_once()
p = _image.path()
print(json.dumps(dict(current=_image.current(), failed=os.path.exists(p + '.failed'), files=os.listdir(os.path.dirname(p)))))
'''
    got = run(code, cache=tmp_path)
    assert got['current'] is None and got['failed']
    assert all(n.endswith('.failed') for n in got['files'])


def test_publish_tidies_only_its_own(tmp_path):
    name = _image._name()
    head = name.rpartition('-')[0]
    for n in ('notes.py', head + '-0123456789abcdef.img', head + '-0123456789abcdef.img.1.2', 'kernels-other.img',
              name + '.lock'):
        (tmp_path / n).write_text('x')
    new = tmp_path / (name + '.3.4')
    new.write_text('x')
    _image._publish(str(new), str(tmp_path / name))
    assert sorted(os.listdir(tmp_path)) == sorted(['notes.py', 'kernels-other.img', name, name + '.3.4',
                                                   name + '.lock'])
    assert (tmp_path / name).read_text() == name + '.3.4'


def test_only_what_numba_would_take_goes_in():
    slots, key = [], []
    assert _image._flatten((np.zeros(3), 7, 2.5, np.float32(1), None, True), slots, key)
    assert key[1:] == ['i8', 'f8', 'f4', 'n', '?']
    assert struct.unpack('<d', struct.pack('<Q', slots[-4]))[0] == 2.5
    odd = np.zeros(17, np.uint8)[1:].view(np.float64)
    assert not odd.flags.aligned
    for a in (np.zeros(3, '>f4'), np.ma.masked_array(np.zeros(3)), odd):
        assert not _image._flatten((a,), [], [])


def test_build_later_once_a_process_and_never_with_autowarm_off(monkeypatch):
    started = []
    monkeypatch.delenv('FASTGRIB_IMAGE', raising=False)
    monkeypatch.setattr(_image, '_spawned', [False])
    monkeypatch.setattr(_image, '_spawn', lambda: started.append(1))
    monkeypatch.setattr(_image, '_can_spawn', lambda: True)
    monkeypatch.setenv('FASTGRIB_AUTOWARM', '0')
    _image.build_later()
    assert not started and not _image._spawned[0]
    monkeypatch.setenv('FASTGRIB_AUTOWARM', '1')
    _image.build_later()
    _image.build_later()
    for _ in range(300):                        # (on a thread of its own)
        if started:
            break
        time.sleep(0.01)
    time.sleep(0.05)
    assert started == [1]
