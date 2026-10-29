"""Many threads at once: one file shared, files of their own, CPU and GPU,
whole reads, fields() and picks -- each result wgrib2's, no deadlock; and
forked worker processes of a parent that had decoded."""
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import fastgrib
from fastgrib import _cpu
from helpers import gpu_ok, path, reference, same

THREADS = 8
ROUNDS = 6


def _work(f, ref, k, gpu):
    """Round k of one thread: a whole read, a pick, then fields()."""
    recs = f.records
    vals = f.read(gpu=gpu)
    vals = vals.get() if gpu else vals
    ok = all(same(v, w) for v, w in zip(vals, ref))
    pick = recs[k % len(recs)::3]
    vals = f.read(pick, gpu=gpu)
    vals = vals.get() if gpu else vals
    ok &= all(same(v, ref[r.index - 1]) for r, v in zip(pick, vals))
    for r, v in f.fields(recs[::2], gpu=gpu):
        ok &= same(v.get() if gpu else v, ref[r.index - 1])
    return ok


@pytest.mark.parametrize('device', ['cpu', 'gpu'])
@pytest.mark.parametrize('shared', [True, False])
def test_threads(device, shared):
    if device == 'gpu' and not gpu_ok():
        pytest.skip('no usable GPU')
    gpu = device == 'gpu'
    names = ['gfs', 'nbm', 'gfs_aec', 'gfs_jpeg']
    refs = {n: reference(n) for n in names}
    one = {n: fastgrib.open(path(n)) for n in names}
    start = threading.Barrier(THREADS)

    def run(t):
        name = names[t % len(names)]
        f = one[name] if shared else fastgrib.open(path(name))
        start.wait()                            # all at once
        try:
            return all(_work(f, refs[name][1], k + t, gpu) for k in range(ROUNDS))
        finally:
            if not shared:
                f.close()

    with ThreadPoolExecutor(THREADS) as ex:
        results = list(ex.map(run, range(THREADS), timeout=300))
    assert all(results)
    for f in one.values():
        f.close()


def _read_in_child(name):
    fastgrib.warm()                             # (a field split across threads)
    f, ref = reference(name)
    return all(same(v, w) for v, w in zip(f.read(), ref))


@pytest.mark.skipif('fork' not in multiprocessing.get_all_start_methods(), reason='no fork here')
def test_forked_workers():
    """A worker forked after its parent decoded makes its own threads: the
    parent's pool, inherited without them, left it waiting for ever."""
    fastgrib.warm()
    assert _cpu._POOL is not None
    with multiprocessing.get_context('fork').Pool(2) as pool:
        assert all(pool.map_async(_read_in_child, ['gfs', 'nbm', 'gfs_aec', 'gfs_jpeg']).get(120))


@pytest.mark.parametrize('name', ['nbm_qmd_hi', 'nbm_qmd_pr', 'written'])
def test_split_runs_are_wgrib2s(name):
    """A complex-packed field split into many runs (each after the first
    decoded from a zero state and corrected): wgrib2's values to the bit --
    NBM's under missing-value management too, whose runs take their own
    loop."""
    from fastgrib import _cpu, _file
    f, ref = reference(name)
    if ref is None:
        pytest.skip('no stored values')
    checked = 0
    for r, want in zip(f.records, ref):
        buf, real = f._bytes([r], False)
        if real[0].packing not in (2, 3):
            continue
        cols = f._grid_cols(real[0].grid)
        got = _cpu.decode(buf, real, cols, undef=_file.NAN, _min_run=64)[0]
        assert same(got.reshape(want.shape), want), repr(r)
        checked += 1
    assert checked
