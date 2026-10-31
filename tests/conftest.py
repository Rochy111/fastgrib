"""The tests' fixtures: each numpy read on the GPU then the CPU (`route`),
and a local HTTP server of some test files (`server`)."""
import http.server
import os
import threading

import pytest

os.environ['FASTGRIB_AUTOWARM'] = '0'   # no background compiling behind the tests' backs

import fastgrib                         # noqa: E402

from helpers import Ranges, gpu_ok, path    # noqa: E402


@pytest.fixture(params=['gpu', 'cpu'], ids=['gpu-route', 'cpu-route'])
def route(request):
    """Every numpy read on the GPU, then on the CPU."""
    if request.param == 'gpu' and not gpu_ok():
        pytest.skip('no usable GPU')
    fastgrib.set_device(request.param)
    yield request.param
    fastgrib.set_device('auto')


@pytest.fixture(scope='module')
def server(tmp_path_factory):
    d = tmp_path_factory.mktemp('www')
    for name in ('gfs', 'nbm', 'gfs_uv', 'gefs', 'gfs_bm254'):
        data = open(path(name), 'rb').read()
        open(d / (name + '.grib2'), 'wb').write(data)
        open(d / (name + '.grib2.idx'), 'w').write(open(path(name, '.inv')).read())
    for ext in ('.grb', '.inv'):                # GRIB1, wgrib's inventory beside it
        open(d / ('g1_narr' + ext), 'wb').write(open(path('g1_narr', ext), 'rb').read())
    handler = lambda *a, **k: Ranges(*a, directory=str(d), **k)      # noqa: E731
    srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield 'http://127.0.0.1:%d/' % srv.server_address[1]
    srv.shutdown()
