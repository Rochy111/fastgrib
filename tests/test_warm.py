"""fastgrib.warm() and `fastgrib -warm`: the sample shipped in the package
(a field of every packing and grid) goes down every path; and what starts
warm() when installing did not."""
import os
import subprocess
import sys

import pytest

import fastgrib
from fastgrib import _warm
from fastgrib.__main__ import main

from helpers import path


def test_warm():
    assert fastgrib.warm() >= 0
    assert not _warm.stale()


def test_warm_on_the_command_line(capsys):
    assert main(['-warm']) == 0
    assert 'decoders compiled' in capsys.readouterr().out


def test_command_line_needs_a_file():
    with pytest.raises(SystemExit):
        main([])


class _Kernel:
    """_warm._kernel()'s stand-in: a cache whose index holds `entries`."""

    def __init__(self, entries):
        self.entries, self._cache, self._cache_file = entries, self, self

    def _load_index(self):
        return self.entries


def _done_here(tmp_path, monkeypatch, entries=('some kernel',)):
    """warm()'s file in tmp_path, and keep_up() on, starting nothing."""
    done = str(tmp_path / 'warmed.x')
    started = []
    monkeypatch.delenv('FASTGRIB_AUTOWARM')
    monkeypatch.setattr(_warm, '_warming', False)
    monkeypatch.setattr(_warm, '_done_file', lambda: done)
    monkeypatch.setattr(_warm, '_kernel', lambda: _Kernel(entries))
    monkeypatch.setattr(_warm, '_start', started.append)
    return done, started


def _warmed(done):
    with open(done, 'w') as f:
        f.write(_warm._sources())


def test_stale_until_this_source_was_warmed(tmp_path, monkeypatch):
    done, _ = _done_here(tmp_path, monkeypatch)
    assert _warm.stale()                                # never warmed
    with open(done, 'w') as f:
        f.write('another version of the sources')
    assert _warm.stale()
    _warmed(done)
    assert not _warm.stale()


def test_stale_if_numba_will_not_have_the_cache(tmp_path, monkeypatch):
    """Before numba 0.68 a stamp is the sources' time, and pip's copy has a new one."""
    done, _ = _done_here(tmp_path, monkeypatch, entries=())
    _warmed(done)
    assert _warm.stale()


def test_keep_up_starts_one_warm(tmp_path, monkeypatch):
    done, started = _done_here(tmp_path, monkeypatch)
    _warm.keep_up()
    assert started == [done + '.lock']


def test_keep_up_nothing_to_do(tmp_path, monkeypatch):
    done, started = _done_here(tmp_path, monkeypatch)
    _warmed(done)
    _warm.keep_up()
    assert not started


def test_keep_up_turned_off(tmp_path, monkeypatch):
    _, started = _done_here(tmp_path, monkeypatch)
    monkeypatch.setenv('FASTGRIB_AUTOWARM', '0')
    _warm.keep_up()
    assert not started


def test_keep_up_leaves_warm_to_itself(tmp_path, monkeypatch):
    """warm()'s own first kernel brings numba up: not a second warm()."""
    _, started = _done_here(tmp_path, monkeypatch)
    monkeypatch.setattr(_warm, '_warming', True)
    _warm.keep_up()
    assert not started


def test_keep_up_leaves_a_warm_running(tmp_path, monkeypatch):
    done, started = _done_here(tmp_path, monkeypatch)
    lock = done + '.lock'
    open(lock, 'w').close()
    _warm.keep_up()
    assert not started
    os.utime(lock, (0, 0))                              # one that died long ago
    _warm.keep_up()
    assert started == [lock] and not os.path.exists(lock)


def test_keep_up_never_raises(tmp_path, monkeypatch):
    _done_here(tmp_path, monkeypatch)

    def fail(*_):
        raise OSError('no')
    monkeypatch.setattr(_warm, '_start', fail)
    _warm.keep_up()
    monkeypatch.setattr(_warm, '_done_file', fail)
    _warm.keep_up()


def test_start_keeps_its_process_from_starting_another(monkeypatch):
    seen = {}

    def popen(cmd, **how):
        seen.update(how, cmd=cmd)
    monkeypatch.setattr(subprocess, 'Popen', popen)
    _warm._start('warmed.x.lock')
    assert seen['env']['FASTGRIB_AUTOWARM'] == '0'
    assert seen['cmd'][0] == sys.executable and "background('warmed.x.lock')" in seen['cmd'][-1]
    assert os.path.isdir(os.path.join(seen['cwd'], 'fastgrib'))     # this fastgrib, not another


def test_background_runs_warm_once(tmp_path, monkeypatch):
    lock = str(tmp_path / 'warmed.x.lock')
    ran = []
    monkeypatch.setattr(_warm, 'warm', lambda: ran.append(os.path.exists(lock)))
    monkeypatch.setattr(os, 'nice', lambda n: 0, raising=False)
    _warm.background(lock)
    assert ran == [True] and not os.path.exists(lock)   # held meanwhile, then let go
    open(lock, 'w').close()
    _warm.background(lock)
    assert ran == [True]                                # held by another: left to it


def test_background_failure_keeps_the_lock(tmp_path, monkeypatch):
    lock = str(tmp_path / 'warmed.x.lock')

    def fail():
        raise RuntimeError('no')
    monkeypatch.setattr(_warm, 'warm', fail)
    monkeypatch.setattr(os, 'nice', lambda n: 0, raising=False)
    with pytest.raises(RuntimeError):
        _warm.background(lock)
    assert os.path.exists(lock)                         # so not every program tries again


def test_the_first_kernel_looks_once_after_numba_is_up():
    """Not before (a program that only lists a file has no numba), and only once."""
    code = '''
import sys
import fastgrib
from fastgrib import _warm
looks = []
_warm.keep_up = lambda: looks.append('numba' in sys.modules)
f = fastgrib.open(sys.argv[1])
list(f.records)
assert not looks and 'numba' not in sys.modules
f.read(0)
f.read(1)
print(looks)
'''
    out = subprocess.run([sys.executable, '-c', code, path('gfs')], capture_output=True,
                         text=True, timeout=600)
    assert out.stdout.strip() == '[True]', out.stderr[-2000:]


def test_a_failed_look_does_not_stop_a_decode():
    code = '''
import sys
import fastgrib
from fastgrib import _warm


def fail(*_):
    raise RuntimeError('no')


_warm.stale = fail
print(fastgrib.open(sys.argv[1]).read(0).size)
'''
    env = {k: v for k, v in os.environ.items() if k != 'FASTGRIB_AUTOWARM'}
    out = subprocess.run([sys.executable, '-c', code, path('gfs')], env=env, capture_output=True,
                         text=True, timeout=600)
    assert out.returncode == 0 and int(out.stdout) > 0, out.stderr[-2000:]


@pytest.mark.skipif(not os.environ.get('FASTGRIB_BIG_TEST'),
                    reason='compiles everything from nothing (~1 min): set FASTGRIB_BIG_TEST=1')
def test_warm_reaches_every_cpu_kernel(tmp_path):
    """A kernel it misses is compiled by a user's first call of it. GPU
    kernels are left out, _cpu._groups_ok with them: they need the card."""
    code = '''
import importlib
import numba
import fastgrib
fastgrib.warm()
missed = []
for m in ('_cpu', '_encode', '_grib1_2nd', '_grids', '_images', '_jpeg2000', '_octets', '_regrid',
          '_spectral', '_winds'):
    mod = importlib.import_module('fastgrib.' + m)
    for name, k in vars(mod).items():
        if isinstance(k, numba.core.registry.CPUDispatcher) and k.__module__ == mod.__name__ \\
                and name != '_groups_ok' and not k._cache._cache_file._load_index():
            missed.append(m + '.' + name)
print(missed)
'''
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path))
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True,
                         timeout=1200)
    assert out.stdout.strip() == '[]', out.stdout + out.stderr[-2000:]
