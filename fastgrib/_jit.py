"""numba, imported when the first kernel runs rather than with fastgrib:
`njit`, `intrinsic` and `prange` as the modules use them, standing in
until then. Listing a file, picking fields, reading an index -- the work
that runs no kernel -- then never pays for numba (importing it took 240
of a 640 ms `fastgrib FILE`).

The first call of any kernel imports numba, makes every kernel known so
far (each from its cache as before: the functions are the same), and puts
them in place of their stand-ins in every fastgrib module, so kernels
calling kernels see numba's dispatchers. Kernels defined after that are
numba's from the start, and so is `prange` imported from here. A stand-in
called from Python works throughout.

That first call is also what notices when the cache was not filled for
this machine (_warm.keep_up): nothing else in a process that never decodes
has numba to ask.

Before any of that a first call looks for this machine's kernel image
(_image.py), which runs kernels with no numba at all: a kernel or argument
types it does not hold, and no image to be had, go on to numba as above.
"""
import os
import sys
import threading

_PENDING = []                   # stand-ins not yet made
_MADE = []                      # every stand-in
_LOCK = threading.RLock()
_LOOK = threading.Lock()
_numba = None                   # the numba module, once imported
_IMAGE = None                   # _image.Image the stand-ins call through, if any
_NO = None                      # _image.NO
_looked = False                 # this process has looked for an image


class _Kernel:
    """A stand-in for numba.njit(**opts)(fn) or numba.extending.intrinsic(fn)."""

    def __init__(self, fn, opts, kind):
        self.fn, self.opts, self.kind, self.real = fn, opts, kind, None
        self.fast = {}                          # (the image) Python types of the arguments -> unit
        self.__wrapped__, self.__name__ = fn, fn.__name__
        self.__doc__ = fn.__doc__

    def make(self, numba):
        if self.kind == 'intrinsic':
            from numba.extending import intrinsic as make
            return make(self.fn)
        return numba.njit(**self.opts)(self.fn)

    def __call__(self, *args, **kwargs):
        if self.real is None:
            if not _looked:
                _look()
            if _IMAGE is not None and not kwargs:
                r = _IMAGE.call(self, args)
                if r is not _NO:
                    return r
            load()
            r = self.real(*args, **kwargs)
            if not _image_off():
                from . import _image                    # (numba's cache may now hold more)
                _image.build_later()
            return r
        return self.real(*args, **kwargs)

    def __getattr__(self, name):                # (signatures, inspect_asm, py_func ...)
        if name == 'real':
            raise AttributeError(name)
        if self.real is None:
            load()
        return getattr(self.real, name)

    def __repr__(self):
        return '<kernel %s%s>' % (self.__name__, '' if self.real is not None
                                  else ' (numba not loaded yet)')


class _Prange:
    """numba.prange's stand-in (a plain range from Python)."""

    def __call__(self, *args):
        return range(*args)


_STAND_IN = _Prange()
prange = _STAND_IN


def njit(*args, **opts):
    """numba.njit, deferred (as @njit(cache=True, nogil=True), or bare)."""
    if args and callable(args[0]):
        return _defer(args[0], {}, 'njit')
    return lambda fn: _defer(fn, opts, 'njit')


def intrinsic(fn):
    """numba.extending.intrinsic, deferred."""
    return _defer(fn, {}, 'intrinsic')


def _defer(fn, opts, kind):
    with _LOCK:
        k = _Kernel(fn, opts, kind)
        if _numba is not None:                  # numba up already: made at once
            return k.make(_numba)
        _PENDING.append(k)
        _MADE.append(k)
        return k


def _look():
    """The image looked for, once a process: every thread here first waits for
    that, then each goes by what it found."""
    global _looked
    with _LOOK:
        if not _looked:
            if not _image_off():
                from . import _image
                _image.open_image()
            _looked = True


def _image_off():
    return os.environ.get('FASTGRIB_IMAGE', '').strip() == '0'


def image():
    return _IMAGE


def use_image(im):
    """Kernels not yet made go through im (an _image.Image) where it can."""
    global _IMAGE, _NO
    from . import _image
    _NO = _image.NO
    _IMAGE = im


def every_kernel():
    """Every stand-in made (each keeps its numba kernel in .real once made)."""
    return list(_MADE)


def load():
    """numba imported, every pending kernel made, and each fastgrib module's
    stand-ins replaced by them. The first time, a look at whether the
    cache suits this machine (a few file reads, outside the lock)."""
    with _LOCK:
        first = _numba is None
        _make()
    if first:
        from . import _warm
        _warm.keep_up()


def _make():
    global _numba
    if _numba is None:
        import numba
        for k in _PENDING:
            k.real = k.make(numba)
        _numba = numba
    if not _PENDING:
        return
    # the kernels' own modules (whatever sys.modules says now), and every
    # fastgrib module (those holding a kernel imported from another)
    pkg = __name__.rpartition('.')[0]
    spaces = {id(k.fn.__globals__): k.fn.__globals__ for k in _PENDING}
    for name, mod in list(sys.modules.items()):
        if mod is not None and (name == pkg or name.startswith(pkg + '.')):
            spaces[id(vars(mod))] = vars(mod)
    # (the sentinel held in a local: replacing this module's own
    # _STAND_IN global mid-loop would hide every stand-in after it --
    # this module's `prange` among them, which a module imported later
    # then picks up)
    stand_in = _STAND_IN
    for d in spaces.values():
        for key, val in list(d.items()):
            if isinstance(val, _Kernel):
                d[key] = val.real
            elif val is stand_in and not (d is globals() and key == '_STAND_IN'):
                d[key] = _numba.prange
    _PENDING.clear()
