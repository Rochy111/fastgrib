"""The kernels without numba: their machine code in one file, loaded in
microseconds. Taken from fastnc.

Bringing numba up takes three quarters of a second of a fresh process (of a
first field's 745 ms: 320 importing it, 380 its typing tables on first use, 36
making the kernels), and all but 9 ms of that is numba getting ready: the
kernels are a few MB of finished x86-64. So once numba has compiled them (warm,
_warm.py), the code is taken out of numba's cache and linked into an image
(`build`, in a process of its own), and later processes load what they call from
it with nothing but numpy and ctypes (`open_image`): no numba, no llvmlite, no
compiler. A read then runs at full speed from the first call of a process.

An image holds one unit per kernel and argument types: the kernel's object
code (with every kernel it calls) laid out in memory order, plus a small entry
function made at build time, `entry(slots, ret, exc)`, which takes the
arguments out of an array of 64-bit words and makes numba's native call. A
unit is mapped when first called. numba's objects are ELF even on Windows
and use the large code model, so relocating one comes down to adding the
unit's base address at known places, in one numpy operation.

The image answers only for this machine and these sources: its name holds a
hash of what CPUID reports (never run code built for another CPU: a home
directory can be shared by different nodes), of fastgrib's sources and the
Python and numpy versions. Each unit carries a CRC, and so does the index.
`build` tries an image in a separate process, on the warm sample against what
numba made of it, before it publishes it, so one that crashes or differs is
never seen. Anything an image cannot do falls back to what ran before it: a
kernel or argument types it does not hold load numba, a machine it cannot serve
(not x86-64 ELF, no executable memory) simply has none, and `FASTGRIB_IMAGE=0`
turns it off. `FASTGRIB_IMAGE_BUILD=0` leaves building it to warm(), and
FASTGRIB_CACHE_DIR says where images live: else beside the package's own
__pycache__, or in the user's cache where that cannot be written.
"""
import ctypes
import marshal
import os
import struct
import sys
import threading
import time
import zlib
from array import array

import numpy as np

from . import _jit, _warm

FORMAT = 1
MAGIC = b'FGRIBIMG'
HEAD = struct.Struct('<8sIIQQQ')    # magic, format, the index's CRC, file size, index at, its length
PAGE = 1 << 16                      # what a unit's read-only and writable parts are apart
RET_WORDS = 32                      # room after the slots for what a kernel returns
EXC_WORDS = 8                       # ... and for its exception pointer
EXTRA = array('Q', bytes(8 * (RET_WORDS + EXC_WORDS)))
MASK = (1 << 64) - 1
TURNED_DOWN = 77777                 # a unit's direct entry: not arguments for this unit
ADDRESSED = (np.ndarray, tuple)     # what goes to a direct entry as its address, a subclass too
FLOATS = (float, np.floating)       # flattened: an int() of one would lose it
NO = object()                       # call(): no such kernel or argument types here

_lock = threading.Lock()
_looked = [False]                   # this process has looked for an image
_spawned = [False]                  # a process was started to build one
_cpu = []
_stamp = []
_free = []


# ------------------------------------------------------- executable memory

if sys.platform == 'win32':
    def _system():
        k = ctypes.WinDLL('kernel32', use_last_error=True)
        k.VirtualAlloc.restype = ctypes.c_size_t
        k.VirtualAlloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
        k.VirtualProtect.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_ulong,
                                     ctypes.POINTER(ctypes.c_ulong)]
        return k

    _SYS = None

    def _alloc(n):
        global _SYS
        if _SYS is None:
            _SYS = _system()
        a = _SYS.VirtualAlloc(0, n, 0x3000, 0x04)       # commit + reserve, read-write
        if not a:
            raise MemoryError(n)
        return a

    def _executable(a, n):
        old = ctypes.c_ulong()
        if not _SYS.VirtualProtect(a, n, 0x20, ctypes.byref(old)):  # execute + read
            raise OSError('VirtualProtect failed')
else:
    def _system():
        c = ctypes.CDLL(None, use_errno=True)
        c.mmap.restype = ctypes.c_size_t
        c.mmap.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                           ctypes.c_int, ctypes.c_long]
        c.mprotect.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
        return c

    _SYS = None

    def _alloc(n):
        global _SYS
        if _SYS is None:
            _SYS = _system()
        a = _SYS.mmap(0, n, 3, 0x22, -1, 0)             # read-write, private + anonymous (Linux)
        if a == ctypes.c_size_t(-1).value:
            raise MemoryError(n)
        return a

    def _executable(a, n):
        if _SYS.mprotect(a, n, 5):                      # read + execute
            raise OSError('mprotect failed')


# ----------------------------------------------------------- which machine

# cpuid(leaf, subleaf, out) in each calling convention, and xgetbv(0)
_CPUID = {'win': bytes.fromhex('53 89C8 89D1 4D89C1 0FA2 418901 41895904 41894908 4189510C 5B C3'),
          'sysv': bytes.fromhex('53 89F8 89F1 4989D1 0FA2 418901 41895904 41894908 4189510C 5B C3')}
_XGETBV = bytes.fromhex('31C9 0F01D0 48C1E220 4809D0 C3')


def _x86_64():
    if sys.platform == 'win32':
        return os.environ.get('PROCESSOR_ARCHITECTURE', '').upper() == 'AMD64'
    return sys.platform.startswith('linux') and os.uname().machine == 'x86_64'


def cpu_id():
    """A hash of what this CPU can run, from CPUID itself and what the OS has
    enabled (the OS's name for a CPU can hide a hypervisor's masking); None
    where there is no image to be had."""
    if not _cpu:
        try:
            _cpu.append(_cpu_id())
        except (OSError, MemoryError):                  # (no executable memory to be had)
            _cpu.append(None)
    return _cpu[0]


def _cpu_id():
    if not _x86_64():
        return None
    code = _CPUID['win' if sys.platform == 'win32' else 'sysv']
    n = 1 << 12
    a = _alloc(1 << 16)
    ctypes.memmove(a, code, len(code))
    ctypes.memmove(a + 128, _XGETBV, len(_XGETBV))
    _executable(a, n)
    fn = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)(a)
    out = (ctypes.c_uint32 * 4)()

    def regs(leaf, sub=0):
        fn(leaf, sub, ctypes.addressof(out))
        return list(out)
    r0 = regs(0)
    r1 = regs(1)
    # (stepping and the core's own numbers in EBX left out: they can differ
    # from core to core; a kernel cannot depend on them)
    words = r0 + [r1[0] & 0x0FFF0FF0, r1[2], r1[3]]
    if r0[0] >= 7:
        r7 = regs(7)
        words += r7[1:]
        if r7[0] >= 1:
            words += regs(7, 1)[:1]
    if regs(0x80000000)[0] >= 0x80000001:
        words += regs(0x80000001)[2:]
    if r1[2] >> 27 & 1:                                 # OSXSAVE: what the OS saves
        words.append(ctypes.CFUNCTYPE(ctypes.c_uint64)(a + 128)() & 0xFFFFFFFF)
    return '%08x' % zlib.crc32(struct.pack('<%dI' % len(words), *words))


def _user_dir():
    if sys.platform == 'win32':
        root = os.environ.get('LOCALAPPDATA') or os.path.expanduser('~\\AppData\\Local')
    elif sys.platform == 'darwin':
        root = os.path.expanduser('~/Library/Caches')
    else:
        root = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
    return os.path.join(root, 'fastgrib')


def _dirs():
    """Where images may be, best first: beside the package's own cache
    (which a wheel carries, and pip's uninstall takes along), else the user's."""
    d = os.environ.get('FASTGRIB_CACHE_DIR')
    return [d] if d else [os.path.join(_warm.HERE, '__pycache__'), _user_dir()]


def _name(cpu=None):
    """The file that names this machine's image for these sources, or None
    if it can have none. An image is never replaced, it may be open in any
    number of processes (Windows would refuse): a new one is a new file, and
    this small one says which is current."""
    cpu = cpu or cpu_id()
    if cpu is None:
        return None
    if not _stamp:
        _stamp.append(_warm._sources()[:16])
    return 'kernels-%s-py%d.%d-np%s-%s.img' % (cpu, sys.version_info[0], sys.version_info[1],
                                               np.__version__.split('.')[0], _stamp[0])


def _writable(d):
    try:
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, 'kernels-%d.tmp' % os.getpid())
        with open(p, 'w'):
            pass
        os.remove(p)
        return True
    except OSError:
        return False


def path(cpu=None):
    """Where this machine's image goes for these sources, or None: where it
    is, if anywhere, else the first place that can be written."""
    name = _name(cpu)
    if name is None:
        return None
    places = [os.path.join(d, name) for d in _dirs()]
    for p in places:
        if os.path.exists(p):
            return p
    for p in places:
        if _writable(os.path.dirname(p)):
            return p
    return None


def current():
    """This machine's image file, or None."""
    name = _name()
    if name is None:
        return None
    for d in _dirs():
        try:
            with open(os.path.join(d, name)) as f:
                return os.path.join(d, f.read().strip())
        except (OSError, ValueError):
            pass
    return None


# ---------------------------------------------------------- calling a unit

class _Unit:
    __slots__ = ('rec', 'fn', 'enter', 'nslots', 'ret')

    def __init__(self, rec):
        self.rec, self.fn, self.enter = rec, None, None
        self.nslots, self.ret = rec['nslots'], rec['ret']


class Image:
    """An image file open: units mapped as they are first called."""

    def __init__(self, name):
        self.file = open(name, 'rb')
        try:
            head = self.file.read(HEAD.size)
            magic, fmt, crc, size, at, n = HEAD.unpack(head)
            if magic != MAGIC or fmt != FORMAT or size != os.fstat(self.file.fileno()).st_size:
                raise ValueError('not an image of this format')
            self.file.seek(at)
            raw = self.file.read(n)
            if zlib.crc32(raw) != crc:
                raise ValueError('an image index is damaged')
            index = marshal.loads(raw)
        except Exception:
            self.file.close()
            raise
        self.stamp = index['stamp']
        self.left = index.get('left', 0)    # units that could not be linked
        self.externals = index['externals']
        self.addresses = {}
        self.kernels = index['kernels']     # kernel's name -> {argument types: unit's record}
        self.fits = {}                      # (kernel, types as called) -> _Unit
        self.units = {}
        self.direct = _direct_works()       # units may read arrays for themselves

    def _external(self, name):
        a = self.addresses.get(name)
        if a is None:
            a = self.addresses[name] = _symbol(name)
        return a

    def _read(self, at, n):
        """Bytes of the file. (A process forked from this one shares its
        position: never moved where it need not be.)"""
        if hasattr(os, 'pread'):
            return os.pread(self.file.fileno(), n, at)
        self.file.seek(at)
        return self.file.read(n)

    def _map(self, u):
        """A unit put in executable memory, its addresses fixed."""
        with _lock:
            if u.fn is not None:
                return
            r = u.rec
            raw = self._read(r['at'], r['stored'])
            if zlib.crc32(raw) != r['crc']:
                raise ValueError('an image unit is damaged')
            size, nrel = r['size'], r['nrel']
            base = _alloc(size)
            mem = (ctypes.c_char * size).from_address(base)
            ctypes.memmove(base, raw, r['stored'] - nrel * 8)
            rel = np.frombuffer(raw, '<i8', nrel, r['stored'] - nrel * 8)
            for k in range(8):                  # (places are not aligned: code holds them)
                at = rel[rel % 8 == k]
                if at.size:
                    v = np.frombuffer(mem, '<i8', (size - k) // 8, k)
                    v[(at - k) // 8] += base
            for at, which, add in r['ext']:
                struct.pack_into('<q', mem, at, self._external(self.externals[which]) + add)
            _executable(base, r['rodata'])
            if r['fast'] is not None and self.direct:
                u.enter = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_size_t)(base + r['fast'])
            u.fn = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_size_t, ctypes.c_size_t,
                                    ctypes.c_size_t)(base + r['entry'])

    def _fit(self, kernel, key):
        """The unit for arguments of these types: their own, or else one that
        takes them as they are (numba's safe conversions: a writable array for
        a read-only one, a contiguous one where any layout will do, a narrower
        integer)."""
        name = kernel.fn.__module__ + '.' + kernel.fn.__qualname__
        units = self.kernels.get(name)
        if units is None:
            return None
        rec = units.get(key)
        if rec is None:
            for k, r in units.items():
                if len(k) == len(key) and all(_takes(p, a) for p, a in zip(k, key)):
                    rec = r
                    break
            if rec is None:
                return None
        u = self.units.get(id(rec))
        if u is None:
            u = self.units.setdefault(id(rec), _Unit(rec))
        return u

    def _unit(self, kernel, slots, key):
        """The unit for arguments flattened to slots, mapped; None if none."""
        key = tuple(key)
        fits = self.fits
        u = fits.get((id(kernel), key))
        if u is None:
            u = self._fit(kernel, key)
            if u is None:
                return None
            fits[id(kernel), key] = u
        if u.fn is None:
            try:
                self._map(u)
            except (OSError, ValueError, MemoryError):  # (damaged; memory not executable here)
                return None
        return u if len(slots) == u.nslots else None

    def call(self, kernel, args):
        """kernel(*args), or NO if there is no unit for those arguments.

        The way in that costs next to nothing under the GIL, where a thousand
        small calls from 24 threads queue: each array goes by its address,
        and the unit's own code reads it and checks it fits. The unit is the
        last that took arguments of these Python types (what an integer means
        cannot be told from its value), kept with which of them are words: an
        array of any class never is, for a one-element one would pass int().
        Anything else, the first call and whatever the unit turns down, goes
        through _flatten."""
        kinds = tuple(map(type, args))
        known = kernel.fast.get(kinds)
        if known is not None:
            u, words = known
            try:
                slots = list(map(id, args))
                for i in words:
                    slots[i] = int(args[i]) & MASK
            except TypeError:                           # (None: flattened as a word)
                pass
            else:
                buf = array('Q', slots)
                buf += EXTRA
                rc = u.enter(buf.buffer_info()[0])
                if rc != TURNED_DOWN:
                    return _returned(u, buf, kernel, rc, len(args))
        slots = []
        key = []
        if not _flatten(args, slots, key):
            return NO
        u = self._unit(kernel, slots, key)
        if u is None:
            return NO
        if u.enter is not None and not any(issubclass(c, FLOATS) for c in kinds):
            kernel.fast[kinds] = (u, [i for i, c in enumerate(kinds) if not issubclass(c, ADDRESSED)])
        buf = array('Q', slots)
        buf += EXTRA
        n = u.nslots
        a = buf.buffer_info()[0]
        ret = a + n * 8
        return _returned(u, buf, kernel, u.fn(a, ret, ret + RET_WORDS * 8), n)


def _direct_works():
    """Whether arrays here are laid out as a unit's own code reads them:
    numpy's compiled extensions depend on this, but so little else is known
    about a future numpy or a free-threaded Python."""
    a = np.zeros((2, 3), np.int16)
    a.flags.writeable = False
    p = id(a)

    def word(at, kind=ctypes.c_int64):
        return kind.from_address(at).value
    try:
        dims, strides, descr = word(p + 32), word(p + 40), word(p + 56)
        return (word(p + 16) == a.ctypes.data and word(p + 24, ctypes.c_int32) == 2
                and (word(dims), word(dims + 8)) == (2, 3) and (word(strides), word(strides + 8)) == (6, 2)
                and word(p + 64, ctypes.c_int32) & (0x400 | 1 | 0x100) == (1 | 0x100)
                and word(descr + 28, ctypes.c_int32) == a.dtype.num and id(a.dtype) == descr)
    except (OSError, ValueError):
        return False


def _returned(u, buf, kernel, rc, at):
    """What a unit's call returned (it put it at word `at` of buf)."""
    if rc and rc != -2:                         # (numba: -2 is "returned None")
        raise _raised(kernel, rc, buf, at)
    r = u.ret
    if r is None or rc:
        return None
    if r[0] == 'arrays':
        return _arrays(r, buf, at)
    got = struct.unpack_from(r[0], buf, at * 8)
    return got[0] if r[1] else got


def _raised(kernel, rc, buf, at):
    """What a kernel raised, as numba would: it left the address of a pickled
    (class, arguments, place) in the exception word (a kernel raises nothing
    of itself here: dividing by zero, say, on a damaged field)."""
    import pickle
    try:
        p = buf[at + RET_WORDS]
        if rc > 0 and p:
            data, size = struct.unpack('<QI', ctypes.string_at(p, 12))
            cls, args = pickle.loads(ctypes.string_at(data, size))[:2]
            return cls(*args)
    except Exception:                                   # noqa: BLE001 -- an error about an error
        pass
    return RuntimeError('kernel %s raised (%d)' % (kernel.__name__, rc))


def _let_go(meminfo):
    """A meminfo a kernel handed over, released as numba's runtime would:
    at no references left, its destructor (a growing buffer's), then itself."""
    if not _free:
        _free.append(ctypes.CFUNCTYPE(None, ctypes.c_size_t)(_symbol('free')))
    refs = ctypes.c_size_t.from_address(meminfo)        # (views of one array share it)
    refs.value -= 1
    if not refs.value:
        dtor, info, data, size = (ctypes.c_size_t * 4).from_address(meminfo + 8)
        if dtor:
            ctypes.CFUNCTYPE(None, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t)(dtor)(data, size, info)
        _free[0](meminfo)


def _arrays(r, buf, at):
    """Arrays a kernel made, as numba returns them (words: meminfo, parent,
    items, item size, data, shape, strides), copied out and let go, with any
    numbers beside them."""
    out = []
    for item in r[1]:
        if isinstance(item, str):
            out.append(struct.unpack_from('<' + item, buf, at * 8)[0])
            at += 1
            continue
        kind, nd = item
        w = struct.unpack_from('<%dq' % (5 + 2 * nd), buf, at * 8)
        meminfo, n, size, data = w[0], w[2], w[3], w[4]
        shape = w[5:5 + nd]
        if n:
            a = np.frombuffer((ctypes.c_char * (n * size)).from_address(data), kind).reshape(shape).copy()
        else:
            a = np.empty(shape, kind)
        if meminfo:
            _let_go(meminfo)
        out.append(a)
        at += 5 + 2 * nd
    return out[0] if r[2] else tuple(out)


def _flatten(args, slots, key):
    """Arguments added to slots, their descriptions to key; False for one
    no unit could take."""
    for a in args:
        t = type(a)
        if t is int:
            slots.append(a & MASK)
            key.append('i8')
        elif t is bool or t is np.bool_:
            slots.append(1 if a else 0)
            key.append('?')
        elif t is float or t is np.float64:
            slots.append(struct.unpack('<Q', struct.pack('<d', a))[0])
            key.append('f8')
        elif t is np.float32:
            slots.append(struct.unpack('<I', struct.pack('<f', a))[0])
            key.append('f4')
        elif isinstance(a, np.ndarray):
            if not _plain(a):
                return False
            key.append(_array(a, slots))
        elif a is None:
            slots.append(0)
            key.append('n')
        elif isinstance(a, np.integer):
            slots.append(int(a) & MASK)
            key.append(a.dtype.str[1:])
        elif t is tuple and a and all(type(x) is int for x in a):
            slots += [x & MASK for x in a]
            key.append(('t', len(a)))
        elif t is tuple and a and all(isinstance(x, np.ndarray) and _plain(x) for x in a):
            kinds = {_array(x, slots) for x in a}
            if len(kinds) != 1:
                return False
            key.append(('T', len(a)) + kinds.pop())
        else:
            return False
    return True


def _plain(a):
    """Whether numba would take an array as the kernels were compiled to:
    aligned, in this machine's byte order, its mask (if any) not ignored.
    (numpy.ma not imported, no array is masked: np.ma would import it, 14 ms.)"""
    ma = sys.modules.get('numpy.ma')
    return (a.flags.aligned and a.dtype.byteorder != '>'
            and not (ma is not None and isinstance(a, ma.MaskedArray)))


def _array(a, slots):
    """numba's flat form of an array added to slots (meminfo, parent, items,
    item size, data, shape, strides), and its description."""
    i = a.__array_interface__
    ptr, ro = i['data']
    shape, st, ts = i['shape'], i['strides'], i['typestr']
    es = int(ts[2:])
    slots += (0, 0, a.size, es, ptr)
    slots += shape
    if st is None:
        layout = 'C'
        n = len(shape)
        s = [es] * n
        for d in range(n - 2, -1, -1):
            s[d] = s[d + 1] * shape[d + 1]
        slots += s
    else:
        layout = 'F' if a.flags.f_contiguous else 'A'
        slots += [x & MASK for x in st]
    return ts[1:], len(shape), layout, ro


def _takes(param, arg):
    """Whether a parameter of the first description takes an argument of the
    second as it is."""
    if param == arg:
        return True
    if isinstance(param, tuple) and isinstance(arg, tuple):
        if len(param) == 6 and len(arg) == 6:
            return param[:2] == arg[:2] and _takes(param[2:], arg[2:])
        return (len(param) == 4 and len(arg) == 4 and param[0] == arg[0] and param[1] == arg[1]
                and (param[2] == arg[2] or param[2] == 'A' or (arg[1] <= 1 and arg[2] != 'A'))
                and (param[3] or not arg[3]))
    if isinstance(param, str) and isinstance(arg, str) and param[0] in 'iu' and arg[0] in 'iu':
        pk, pn, ak, an = param[0], int(param[1:]), arg[0], int(arg[1:])
        return (pk == ak and pn >= an) or (pk == 'i' and ak == 'u' and pn > an)
    return False


def _symbol(name):
    """An address from the C library. On Windows that is ucrtbase, where
    numba's own math goes too (through its _helperlib): msvcrt's cos, sin, atan,
    log10, pow and atan2 differ from it in the last bit, which a value here
    must not. numba's JIT takes memset and memcpy from msvcrt, which is slower
    (a big complex-packed field: 3.2 ms against 1.3)."""
    if sys.platform == 'win32':
        libs = ('ucrtbase', 'ntdll', 'kernel32')
    else:
        libs = (None, 'libm.so.6')
    for lib in libs:
        try:
            return ctypes.cast(getattr(ctypes.CDLL(lib), name), ctypes.c_void_p).value
        except (AttributeError, OSError):
            pass
    raise OSError('no symbol ' + name)


def _off():
    return os.environ.get('FASTGRIB_IMAGE', '').strip() == '0'


def _open(name):
    """An image file ready for kernels to be called through, or None."""
    try:
        im = Image(name)
    except (OSError, ValueError, EOFError, TypeError, MemoryError, struct.error):
        return None
    _jit.use_image(im)
    return im


def open_image():
    """This machine's image, ready for kernels to be called through (the
    stand-ins in _jit then use it), or None. Looked for once a process, from
    the first kernel; any thread that gets here meanwhile waits for that."""
    with _lock:
        if _looked[0]:
            return _jit.image()
        _looked[0] = True
    if _off():
        return None
    p = current()
    return None if p is None else _open(p)


# ---------------------------------------------------------------- building

class _Unlinkable(Exception):
    pass


class _Elf:
    """A relocatable ELF64 x86-64 object: sections, symbols, relocations."""

    def __init__(self, b):
        if b[:6] != b'\x7fELF\x02\x01' or struct.unpack_from('<H', b, 18)[0] != 62:
            raise _Unlinkable('not ELF64 little-endian x86-64')
        self.b = b
        shoff = struct.unpack_from('<Q', b, 0x28)[0]
        entsize, shnum, strndx = struct.unpack_from('<HHH', b, 0x3A)
        self.sec = []
        for i in range(shnum):
            name, ty, fl, _, off, size, link, info, align, ent = struct.unpack_from(
                '<IIQQQQIIQQ', b, shoff + i * entsize)
            self.sec.append(dict(n=name, type=ty, flags=fl, off=off, size=size, link=link,
                                 info=info, align=max(align, 1), ent=ent, at=None))
        st = self.sec[strndx]
        for s in self.sec:
            s['name'] = self._str(st, s['n'])
        self.sym = []
        for s in self.sec:
            if s['type'] == 2:                          # SHT_SYMTAB
                strs = self.sec[s['link']]
                for k in range(s['size'] // 24):
                    name, info, _, shndx, value, size = struct.unpack_from(
                        '<IBBHQQ', b, s['off'] + k * 24)
                    self.sym.append((self._str(strs, name), info >> 4, info & 15, shndx, value, size))

    def _str(self, sec, off):
        b = self.b
        a = sec['off'] + off
        return b[a:b.index(b'\0', a)].decode('latin1')

    def relocations(self):
        """(section it applies to, offset, type, symbol, addend)."""
        for s in self.sec:
            if s['type'] == 4 and self.sec[s['info']]['at'] is not None:        # SHT_RELA
                for k in range(s['size'] // 24):
                    off, info, add = struct.unpack_from('<QQq', self.b, s['off'] + k * 24)
                    yield self.sec[s['info']], off, info & 0xFFFFFFFF, info >> 32, add


def _align(n, a):
    return (n + a - 1) // a * a


R_64, R_PC32, R_PLT32, R_PC64 = 1, 2, 4, 24


def _link(blobs):
    """Objects laid out in one piece, read-only sections first: (bytes,
    places where the base address is added, external references as (place,
    name, addend), where the writable part starts, {defined name: offset})."""
    elves = [_Elf(b) for b in blobs]
    keep = []
    for e in elves:
        for s in e.sec:
            if s['flags'] & 2 and s['type'] in (1, 8) and s['size'] and s['name'] != '.eh_frame':
                if s['flags'] & 0x400:                  # SHF_TLS
                    raise _Unlinkable('thread-local data')
                keep.append(s)
            elif s['type'] in (14, 15, 16):             # (constructors)
                raise _Unlinkable('initializers')
    pos = 0
    for s in keep:
        if not s['flags'] & 1:                          # read-only: code and constants
            pos = _align(pos, s['align'])
            s['at'] = pos
            pos += s['size']
    rodata = pos = _align(pos, PAGE)
    for s in keep:
        if s['flags'] & 1:
            pos = _align(pos, s['align'])
            s['at'] = pos
            pos += s['size']
    common = {}
    defined = {}
    for e in elves:
        for name, bind, kind, shndx, value, size in e.sym:
            if shndx == 0xFFF2 and name not in common:  # SHN_COMMON
                pos = _align(pos, max(value, 1))
                common[name] = pos
                defined[name] = pos
                pos += size
            elif shndx and shndx < 0xFF00 and bind and name:    # global or weak
                s = e.sec[shndx]
                if s['at'] is not None and (bind == 1 or name not in defined):
                    defined[name] = s['at'] + value
    total = _align(pos + 1, PAGE)
    out = bytearray(total)
    for e in elves:
        for s in e.sec:
            if s['at'] is not None and s['type'] == 1:
                out[s['at']:s['at'] + s['size']] = e.b[s['off']:s['off'] + s['size']]
    rel, ext = [], []
    for e in elves:
        for s, off, kind, si, add in e.relocations():
            name, bind, ty, shndx, value, _ = e.sym[si]
            p = s['at'] + off
            if shndx == 0 or shndx == 0xFFF2 or (bind and name in defined):
                target = defined.get(name)
            elif ty == 3 or shndx < 0xFF00:             # a section, or a symbol in one
                at = e.sec[shndx]['at']
                if at is None:
                    raise _Unlinkable('a reference to a section left out: ' + e.sec[shndx]['name'])
                target = at + value
            else:
                raise _Unlinkable('symbol %r in section %#x' % (name, shndx))
            if kind == R_64:
                if target is None:
                    ext.append((p, name, add))
                else:
                    struct.pack_into('<q', out, p, target + add)
                    rel.append(p)
            elif target is None:
                raise _Unlinkable('a relative reference to %s' % name)
            elif kind == R_PC64:
                struct.pack_into('<q', out, p, target + add - p)
            elif kind in (R_PC32, R_PLT32):
                struct.pack_into('<i', out, p, target + add - p)
            else:
                raise _Unlinkable('relocation type %d' % kind)
    return bytes(out), rel, ext, rodata, defined


def _runtime_object(codegen):
    """What numba's runtime does for the kernels, in a few lines of LLVM, as
    its nrt.cpp does it: memory for a kernel (a failed malloc returned, as
    numba's callers expect), arrays made inside one (one malloc, data after
    the header), buffers that grow (data apart, freed by the meminfo's
    destructor), each meminfo freed with what it holds. numba's own needs
    numba loaded."""
    from llvmlite import ir
    import llvmlite.binding as ll
    i8p = ir.IntType(8).as_pointer()
    i64 = ir.IntType(64)
    i32 = ir.IntType(32)
    void = ir.VoidType()
    null = ir.Constant(i8p, None)
    mod = ir.Module('fastgrib_runtime')
    mod.triple = codegen._tm.triple
    malloc = ir.Function(mod, ir.FunctionType(i8p, [i64]), 'malloc')
    realloc = ir.Function(mod, ir.FunctionType(i8p, [i8p, i64]), 'realloc')
    free = ir.Function(mod, ir.FunctionType(void, [i8p]), 'free')
    dtor_type = ir.FunctionType(void, [i8p, i64, i8p])
    head = 48                       # NRT_MemInfo: refct, dtor, dtor_info, data, size, allocator

    def body(name, fnty):
        f = ir.Function(mod, fnty, name)
        return f, ir.IRBuilder(f.append_basic_block('e'))

    def field(b, mi, k):
        return b.gep(b.bitcast(mi, i64.as_pointer()), [ir.Constant(i64, k)])

    def as_word(b, p):
        return b.ptrtoint(p, i64)

    # a growing buffer's destructor: the elements' own (if any), then the data
    vd, b = body('fastgrib_varsize_dtor', dtor_type)
    with b.if_then(b.icmp_unsigned('!=', vd.args[2], null)):
        b.call(b.bitcast(vd.args[2], ir.FunctionType(void, [i8p]).as_pointer()), [vd.args[0]])
    b.call(free, [vd.args[0]])
    b.ret_void()

    new, b = body('NRT_MemInfo_new', ir.FunctionType(i8p, [i8p, i64, dtor_type.as_pointer(), i8p]))
    mi = b.call(malloc, [ir.Constant(i64, head)])
    with b.if_then(b.icmp_unsigned('!=', mi, null)):
        for k, v in enumerate((ir.Constant(i64, 1), as_word(b, new.args[2]), as_word(b, new.args[3]),
                               as_word(b, new.args[0]), new.args[1], ir.Constant(i64, 0))):
            b.store(v, field(b, mi, k))
    b.ret(mi)

    nv, b = body('NRT_MemInfo_new_varsize', ir.FunctionType(i8p, [i64]))
    data = b.call(malloc, [nv.args[0]])
    with b.if_then(b.icmp_unsigned('==', data, null)):
        b.ret(null)
    mi = b.call(new, [data, nv.args[0], vd, null])
    with b.if_then(b.icmp_unsigned('==', mi, null)):
        b.call(free, [data])
    b.ret(mi)

    f, b = body('NRT_MemInfo_new_varsize_dtor', ir.FunctionType(i8p, [i64, i8p]))
    mi = b.call(nv, [f.args[0]])
    with b.if_then(b.icmp_unsigned('!=', mi, null)):
        b.store(as_word(b, f.args[1]), field(b, mi, 2))
    b.ret(mi)

    for name, grow in (('NRT_MemInfo_varsize_alloc', False), ('NRT_MemInfo_varsize_realloc', True)):
        f, b = body(name, ir.FunctionType(i8p, [i8p, i64]))
        at = field(b, f.args[0], 3)
        data = (b.call(realloc, [b.inttoptr(b.load(at), i8p), f.args[1]]) if grow
                else b.call(malloc, [f.args[1]]))
        b.store(as_word(b, data), at)
        with b.if_then(b.icmp_unsigned('!=', data, null)):
            b.store(f.args[1], field(b, f.args[0], 4))
        b.ret(data)

    f, b = body('NRT_MemInfo_varsize_free', ir.FunctionType(void, [i8p, i8p]))
    b.call(free, [f.args[1]])
    at = field(b, f.args[0], 3)
    with b.if_then(b.icmp_unsigned('==', b.load(at), as_word(b, f.args[1]))):
        b.store(ir.Constant(i64, 0), at)
    b.ret_void()

    f, b = body('NRT_Reallocate', ir.FunctionType(i8p, [i8p, i64]))
    b.ret(b.call(realloc, [f.args[0], f.args[1]]))

    f, b = body('NRT_MemInfo_call_dtor', ir.FunctionType(void, [i8p]))
    mi = f.args[0]
    dtor = b.load(field(b, mi, 1))
    with b.if_then(b.icmp_unsigned('!=', dtor, ir.Constant(i64, 0))):
        b.call(b.inttoptr(dtor, dtor_type.as_pointer()),
               [b.inttoptr(b.load(field(b, mi, 3)), i8p), b.load(field(b, mi, 4)),
                b.inttoptr(b.load(field(b, mi, 2)), i8p)])
    b.call(free, [mi])
    b.ret_void()

    def make(name, args):
        f, b = body(name, ir.FunctionType(i8p, args))
        size = f.args[0]
        align = b.zext(f.args[1], i64) if len(args) > 1 else ir.Constant(i64, 32)
        raw = b.call(malloc, [b.add(size, b.add(ir.Constant(i64, head), b.mul(align, ir.Constant(i64, 2))))])
        none = f.append_basic_block('none')
        ir.IRBuilder(none).ret(raw)
        got = f.append_basic_block('got')
        b.cbranch(b.icmp_unsigned('==', raw, ir.Constant(i8p, None)), none, got)
        b.position_at_end(got)
        words = b.bitcast(raw, i64.as_pointer())
        at = b.add(b.ptrtoint(raw, i64), ir.Constant(i64, head))
        data = b.and_(b.add(at, b.sub(align, ir.Constant(i64, 1))), b.neg(align))
        for k, v in enumerate((ir.Constant(i64, 1), ir.Constant(i64, 0), ir.Constant(i64, 0), data, size,
                               ir.Constant(i64, 0))):
            b.store(v, b.gep(words, [ir.Constant(i64, k)]))
        b.ret(raw)
    make('NRT_MemInfo_alloc_aligned', [i64, i32])
    make('NRT_MemInfo_alloc_safe_aligned', [i64, i32])
    make('NRT_MemInfo_alloc', [i64])
    make('NRT_MemInfo_alloc_safe', [i64])
    f, b = body('NRT_Allocate', ir.FunctionType(i8p, [i64]))
    b.ret(b.call(malloc, [f.args[0]]))
    for name in ('NRT_MemInfo_destroy', 'NRT_Free'):
        f, b = body(name, ir.FunctionType(ir.VoidType(), [i8p]))
        b.call(free, [f.args[0]])
        b.ret_void()
    m = ll.parse_assembly(str(mod))
    m.verify()
    return codegen._tm.emit_object(m)


def _describe(t, ctx):
    """An argument type as call() describes it, and how many slots it takes;
    None for what a unit cannot take."""
    from numba import types
    if isinstance(t, types.Array):
        if t.dtype not in (types.int8, types.int16, types.int32, types.int64, types.uint8, types.uint16,
                           types.uint32, types.uint64, types.float32, types.float64, types.boolean):
            return None
        if t.layout not in 'CFA':
            return None
        d = np.dtype(str(t.dtype) if t.dtype != types.boolean else 'bool').str[1:]
        return (d, t.ndim, t.layout, not t.mutable), 5 + 2 * t.ndim
    if isinstance(t, types.Literal):
        return None                                     # (a call from a kernel, not from Python)
    if t == types.boolean:
        return '?', 1
    if isinstance(t, types.Integer):
        return ('i' if t.signed else 'u') + str(t.bitwidth // 8), 1
    if isinstance(t, types.Float):
        return 'f%d' % (t.bitwidth // 8), 1
    if t == types.none:
        return 'n', 1
    if isinstance(t, types.UniTuple) and t.dtype == types.int64:
        return ('t', t.count), t.count
    if isinstance(t, types.UniTuple) and isinstance(t.dtype, types.Array):
        one = _describe(t.dtype, ctx)
        return None if one is None else (('T', t.count) + one[0], t.count * one[1])
    return None


def _return_format(t):
    """(a struct format for what a kernel returns, whether it is one value), or
    ('arrays', what _arrays reads, whether it is one value), or None for no
    value; False for what cannot be taken back."""
    from numba import types

    def item(x):
        if isinstance(x, types.Array) and x.layout == 'C':
            return np.dtype(str(x.dtype) if x.dtype != types.boolean else 'bool').str, x.ndim
        return 'q' if x == types.int64 else 'd' if x == types.float64 else None
    if isinstance(t, types.Array):
        return ('arrays', [item(t)], True) if item(t) else False
    if isinstance(t, types.BaseTuple) and any(isinstance(x, types.Array) for x in t.types):
        items = [item(x) for x in t.types]
        return ('arrays', items, False) if all(items) else False
    if t == types.none:
        return None
    if t == types.boolean:
        return ('@?', True)
    if isinstance(t, types.Integer):
        return ('@' + {8: 'b', 16: 'h', 32: 'i', 64: 'q'}[t.bitwidth].upper()
                if not t.signed else '@' + {8: 'b', 16: 'h', 32: 'i', 64: 'q'}[t.bitwidth], True)
    if isinstance(t, types.Float):
        return ('@' + {32: 'f', 64: 'd'}[t.bitwidth], True)
    if isinstance(t, types.BaseTuple):
        got = ''
        for x in t.types:
            f = _return_format(x)
            if not f or not f[1]:
                return False
            got += f[0][1:]
        return ('@' + got, False)
    return False


def _type_numbers(kind):
    """numpy's numbers for a type: int64 is two of them on some systems."""
    return sorted({np.dtype(c).num for c in 'bBhHiIlLqQfd?' if np.dtype(c).str[1:] == kind})


def _tuple_items():
    """Where a tuple's items start, in this Python (it has moved before)."""
    a = object()
    t = (a,)
    for at in (24, 32, 40):
        if ctypes.c_size_t.from_address(id(t) + at).value == id(a):
            return at
    return None


def _direct_entry(mod, desc, call):
    """fastgrib_fast(args): one word per argument, an array's being its address
    (a tuple's, for a tuple of arrays). Each array read for itself and
    checked to be what this unit takes (dimensions, flags, type, byte
    order), else TURNED_DOWN and nothing has run. Field offsets: PyArrayObject
    and PyArray_Descr as numpy's compiled extensions rely on them, checked
    against the running numpy by _direct_works."""
    from llvmlite import ir
    i8, i32, i64 = ir.IntType(8), ir.IntType(32), ir.IntType(64)
    items_at = _tuple_items()
    if any(isinstance(d, tuple) and (d[0] == 't' or (d[0] == 'T' and items_at is None)) for d, _ in desc):
        return
    f = ir.Function(mod, ir.FunctionType(i32, [i64.as_pointer()]), 'fastgrib_fast')
    b = ir.IRBuilder(f.append_basic_block('e'))
    no = f.append_basic_block('no')
    ir.IRBuilder(no).ret(ir.Constant(i32, TURNED_DOWN))

    def word(a, at=0, ty=i64):
        return b.load(b.inttoptr(b.add(a, ir.Constant(i64, at)), ty.as_pointer()))

    def must(ok):
        go = f.append_basic_block()
        b.cbranch(ok, go, no)
        b.position_at_end(go)

    def array(raw, d):
        kind, nd, layout, readonly = d
        must(b.icmp_unsigned('==', word(raw, 24, i32), ir.Constant(i32, nd)))
        want = 0x100 | (0 if readonly else 0x400) | {'C': 1, 'F': 2, 'A': 0}[layout]    # aligned, writable
        must(b.icmp_unsigned('==', b.and_(word(raw, 64, i32), ir.Constant(i32, want)), ir.Constant(i32, want)))
        descr = word(raw, 56)
        num = word(descr, 28, i32)
        ok = ir.Constant(ir.IntType(1), 0)
        for n in _type_numbers(kind):
            ok = b.or_(ok, b.icmp_unsigned('==', num, ir.Constant(i32, n)))
        must(ok)
        must(b.icmp_unsigned('!=', word(descr, 26, i8), ir.Constant(i8, ord('>'))))
        dims, strides = word(raw, 32), word(raw, 40)
        shape = [word(dims, 8 * k) for k in range(nd)]
        count = ir.Constant(i64, 1)
        for s in shape:
            count = b.mul(count, s)
        zero = ir.Constant(i64, 0)
        return [zero, zero, count, ir.Constant(i64, int(kind[1:])), word(raw, 16)] + shape + [
            word(strides, 8 * k) for k in range(nd)]
    slots = []
    for k, (d, _) in enumerate(desc):
        raw = b.load(b.gep(f.args[0], [ir.Constant(i64, k)]))
        if isinstance(d, tuple) and d[0] == 'T':
            must(b.icmp_unsigned('==', word(raw, 16), ir.Constant(i64, d[1])))
            for i in range(d[1]):
                slots += array(word(raw, items_at + 8 * i), d[2:])
        elif isinstance(d, tuple):
            slots += array(raw, d)
        else:
            slots.append(ir.Constant(i64, 0) if d == 'n' else raw)
    ret = b.gep(f.args[0], [ir.Constant(i64, len(desc))])
    exc = b.gep(f.args[0], [ir.Constant(i64, len(desc) + RET_WORDS)])
    b.ret(call(b, slots.__getitem__, ret, exc))


def _entry_object(ov, ctx, codegen, slots_of, desc):
    """The functions that make numba's native call from an array of words:
    fastgrib_entry, with arguments flattened by _flatten, and fastgrib_fast."""
    from llvmlite import ir
    import llvmlite.binding as ll
    fnty = ctx.call_conv.get_function_type(ov.signature.return_type, ov.signature.args)
    i8p = ir.IntType(8).as_pointer()
    i64 = ir.IntType(64)
    mod = ir.Module('fastgrib_entry')
    mod.triple = codegen._tm.triple
    target = ir.Function(mod, fnty, ov.fndesc.llvm_func_name)
    params = list(fnty.args[2:])
    if len(params) != len(slots_of):
        raise _Unlinkable('%d parameters, %d slots' % (len(params), len(slots_of)))

    def call(b, value, ret, exc):
        args = [b.bitcast(ret, fnty.args[0]), b.bitcast(exc, fnty.args[1])]
        for t, at in zip(params, slots_of):
            v = value(at)
            if isinstance(t, ir.PointerType):
                v = b.inttoptr(v, t)
            elif isinstance(t, ir.IntType):
                if t.width < 64:
                    v = b.trunc(v, t)
            elif isinstance(t, ir.DoubleType):
                v = b.bitcast(v, t)
            elif isinstance(t, ir.FloatType):
                v = b.bitcast(b.trunc(v, ir.IntType(32)), t)
            else:
                raise _Unlinkable('parameter type %s' % t)
            args.append(v)
        return b.call(target, args)
    f = ir.Function(mod, ir.FunctionType(ir.IntType(32), [i64.as_pointer(), i8p, i8p]), 'fastgrib_entry')
    b = ir.IRBuilder(f.append_basic_block('e'))
    b.ret(call(b, lambda at: b.load(b.gep(f.args[0], [ir.Constant(i64, at)])), f.args[1], f.args[2]))
    _direct_entry(mod, desc, call)
    m = ll.parse_assembly(str(mod))
    m.verify()
    return codegen._tm.emit_object(m)


def _modules():
    import importlib
    import pkgutil
    pkg = __name__.rpartition('.')[0]
    for m in pkgutil.iter_modules(sys.modules[pkg].__path__):
        if m.name not in ('_image', '__main__'):
            try:
                importlib.import_module(pkg + '.' + m.name)
            except ImportError:                         # (xarray, say: not installed)
                pass


_objects = {}                   # library name -> object code of each overload, as numba loads them
_prepared = []


def _prepare():
    """Every kernel made, numba handing over object code as it loads it (it
    keeps none in memory afterwards). Before anything loads a kernel."""
    if _prepared:
        return
    from numba.core import codegen as cg
    real = cg.CPUCodeLibrary._unserialize.__func__

    def keep(cls, codegen, state):
        name, kind, data = state
        if kind == 'object':                            # (a library is named after its kernel: all
            _objects.setdefault(name, []).append(data[0])   # the overloads of one have that name)
        return real(cls, codegen, state)
    cg.CPUCodeLibrary._unserialize = classmethod(keep)
    _modules()
    _jit.load()
    _prepared.append(True)


def _sample():
    """A hash of every field of the warm sample as the kernels in use read it."""
    import hashlib
    import warnings
    from ._cpu import decode
    from ._file import GribFile, NAN
    h = hashlib.sha256()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f = GribFile(_warm.SAMPLE)
        for r in f.records:
            buf, real = f._bytes([r], False)
            h.update(decode(buf, real, f._grid_cols(real[0].grid), undef=NAN).tobytes())
        f.close()
    return h.hexdigest()


def build(target=None, verbose=False):
    """Link what numba's cache holds for this machine into an image, and put
    it in place once it has run in a process of its own and read the warm
    sample as numba does. The image's path, or None."""
    _prepare()
    objects = _objects
    target = target or path()
    if target is None:
        return None
    from numba import types
    ctx = codegen = runtime = None
    externals = {}
    index = {}
    blobs = []
    at = HEAD.size
    done = dropped = 0
    for k in _jit.every_kernel():
        if k.kind != 'njit':
            continue
        d = k.real
        if ctx is None:
            ctx = d.targetctx
            codegen = ctx.codegen()
            runtime = _runtime_object(codegen)
        try:
            cached = d._cache._cache_file._load_index()
        except Exception:                               # noqa: BLE001 -- numba's insides
            continue
        magic = codegen.magic_tuple()
        for key in cached:
            sig = key[0]
            if key[1] != magic:
                continue
            desc = [_describe(a, ctx) for a in sig]
            if any(x is None for x in desc):
                continue
            try:
                d.compile(sig)
                ov = d.overloads[sig]
                ret = _return_format(ov.signature.return_type)
                if ret is False:
                    continue
                own = ov.fndesc.mangled_name.encode() + b'\0'
                obj = next((o for o in objects.get(ov.library.name, ()) if own in o), None)
                if obj is None:
                    obj = ov.library.serialize_using_object_code()[2][0]
                # which slot each native parameter comes from (None: numba
                # passes nothing for it; the slot stays unused)
                slots_of = []
                pos = 0
                for a, (_, n) in zip(sig, desc):
                    if a == types.none:
                        if len(ctx.get_arg_packer([a]).argument_types):
                            slots_of.append(pos)
                    else:
                        slots_of += range(pos, pos + n)
                    pos += n
                entry = _entry_object(ov, ctx, codegen, slots_of, desc)
                code, rel, ext, rodata, defined = _link([obj, entry, runtime])
                ext = list(_usable(ext))
                for _, n, _ in ext:
                    if not _found(n):                       # (a kernel no library holds; a helper of numba's)
                        raise _Unlinkable('%s is not to be had' % n)
                stored = code + np.array(rel, '<i8').tobytes()
                rec = dict(at=at, stored=len(stored), size=len(code), nrel=len(rel),
                           crc=zlib.crc32(stored), rodata=rodata, entry=defined['fastgrib_entry'],
                           fast=defined.get('fastgrib_fast'), nslots=pos, ret=ret,
                           ext=[(p, externals.setdefault(n, len(externals)), a) for p, n, a in ext])
                name = k.fn.__module__ + '.' + k.fn.__qualname__
                index.setdefault(name, {})[tuple(x[0] for x in desc)] = rec
                blobs.append(stored)
                at += len(stored)
                done += 1
            except _Unlinkable as e:
                dropped += 1
                if verbose:
                    print('not linked: %s %s: %s' % (k.__name__, sig, e))
    if not done:
        return None
    stamp = _cache_stamp()
    tail = marshal.dumps(dict(stamp=stamp, externals=list(externals), kernels=index, left=dropped))
    name = '%s.%08x.%d' % (target, stamp, os.getpid())
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(name, 'wb') as f:
        f.write(HEAD.pack(MAGIC, FORMAT, zlib.crc32(tail), at + len(tail), at, len(tail)))
        for b in blobs:
            f.write(b)
        f.write(tail)
    if not _tried_in_process(name, _sample()):
        os.remove(name)
        return None
    _publish(name, target)
    if verbose:
        print('image: %d units (%d left out), %.1f MB: %s' % (done, dropped, at / 1e6, target))
    return target


def _found(name):
    try:
        return bool(_symbol(name))
    except OSError:
        return False


# numba's helpers for the wrapper around each kernel: nothing here calls that
_WRAPPER = ('numba_do_raise', 'numba_gil_ensure', 'numba_gil_release', 'numba_runtime_build_excinfo_struct',
            'numba_unpickle')


def _usable(ext):
    """External references, those a kernel's own code cannot reach dropped:
    CPython's API and numba's helpers for its wrapper around each kernel,
    which nothing here calls (the place stays 0: a crash, not something
    working by chance). Any other helper of numba's stays, and so no unit is
    made of a kernel that needs it."""
    for p, name, add in ext:
        if not name.startswith(('Py', '_Py', 'NRT_adapt', 'NRT_decref', 'NRT_incref')) and name not in _WRAPPER:
            yield p, name, add


def _cache_stamp():
    """What numba's cache held when an image was built: which kernels and
    argument types, not when it was written (installing changes that)."""
    n = 0
    try:
        for e in os.scandir(_warm._kernel()._cache._cache_path):
            if e.name.endswith('.nbi'):
                n ^= zlib.crc32(('%s %d' % (e.name, e.stat().st_size)).encode())
    except (OSError, AttributeError):                   # (another numba's insides)
        pass
    return n


def _publish(name, target):
    """Image file `name` made the current one for `target` (one write, seen
    whole or not at all), and those no longer wanted removed: an older one
    still open somewhere stays until the next build."""
    tmp = '%s.%d.tmp' % (target, os.getpid())
    with open(tmp, 'w') as f:
        f.write(os.path.basename(name))
    for _ in range(100):
        try:
            os.replace(tmp, target)
            break
        except PermissionError:                         # (Windows: being read this instant)
            time.sleep(0.01)
    d = os.path.dirname(target)
    head = os.path.basename(target).rpartition('-')[0]
    if not head.startswith('kernels-'):                 # (somewhere else: not ours to tidy)
        return
    keep = (os.path.basename(name), os.path.basename(target))
    for n in os.listdir(d):
        if n.startswith(head) and n not in keep and not n.endswith(('.lock', '.failed')):
            try:
                os.remove(os.path.join(d, n))
            except OSError:
                pass


SELF_TEST = """
import sys
from fastgrib import _image
_image.self_test(sys.argv[1], sys.argv[2])
"""


def _home():
    return os.path.dirname(_warm.HERE)


def _tried_in_process(name, want):
    """Whether an image runs, and reads as numba did, in a process of its own:
    a bad one must take down nothing but that."""
    import subprocess
    env = dict(os.environ, FASTGRIB_IMAGE='1', FASTGRIB_AUTOWARM='0')
    try:
        r = subprocess.run([sys.executable, '-c', SELF_TEST, name, want], capture_output=True, timeout=120,
                           env=env, cwd=_home(), **_quiet())
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and r.stdout.strip().endswith(b'image ok')


def self_test(name, want):
    """Every packing of the warm sample through an image, compared with what
    numba read, without numba."""
    _looked[0] = True
    if _open(name) is None:
        raise SystemExit('no image')
    if _sample() != want:
        raise SystemExit('the image reads differently')
    if 'numba' in sys.modules:
        raise SystemExit('numba was loaded')
    print('image ok')


# what a process leaves for the next one

NO_WINDOW = 0x08000000          # CREATE_NO_WINDOW: a console nobody sees, which children share
BELOW_NORMAL = 0x00004000       # (not DETACHED_PROCESS: with no console at all, each console
                                # program the build starts would open a window of its own)


def _quiet():
    """How to start a process that nobody should see."""
    return {'creationflags': NO_WINDOW} if sys.platform == 'win32' else {}


def _can_spawn():
    if os.environ.get('FASTGRIB_IMAGE_BUILD', '').strip() == '0' or getattr(sys, 'frozen', False):
        return False
    exe = os.path.basename(sys.executable or '').lower()
    return exe.startswith(('python', 'pypy'))


def _stale():
    """Whether numba's cache holds more than this machine's image has."""
    if _name() is None:
        return False
    name = current()
    if name is None:
        return True
    try:
        im = Image(name)
    except (OSError, ValueError, EOFError, TypeError, struct.error):
        return True
    im.file.close()
    return im.stamp != _cache_stamp()


def _failed(p):
    """Whether building failed here lately. Not worth starting again from every
    process, but whatever was in the way may be gone by tomorrow."""
    try:
        return time.time() - os.path.getmtime(p + '.failed') < 86400
    except OSError:
        return False


def build_later():
    """After numba ran a kernel: an image for later processes, built by a
    process of its own at low priority unless it would add nothing, on a
    thread of its own so that what called the kernel does not wait for that.
    Once a process, numba up. Compiling comes first: a warm() that is due
    (keep_up saw to it) builds one itself."""
    if _spawned[0] or _off() or not _can_spawn() or os.environ.get('FASTGRIB_AUTOWARM', '1') == '0':
        return
    _spawned[0] = True
    threading.Thread(target=_spawn, daemon=True).start()


def _spawn():
    try:
        p = path()
        if p is None or _warm.stale() or not _stale() or os.path.exists(p + '.lock') or _failed(p):
            return                                      # (a pool's workers: one build between them)
        import subprocess
        code = 'from fastgrib import _image; _image.build_once()'
        kw = {}
        if sys.platform == 'win32':
            kw['creationflags'] = BELOW_NORMAL | NO_WINDOW
        else:
            kw['start_new_session'] = True
        env = dict(os.environ, FASTGRIB_AUTOWARM='0')
        subprocess.Popen([sys.executable, '-c', code], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True, env=env, cwd=_home(), **kw)
    except Exception:                                   # noqa: BLE001 -- a head start only
        pass


def build_now():
    """An image built before this returns (in a process of its own: numba
    hands out a kernel's object code only as it loads it). warm()'s last step."""
    if _off() or not _can_spawn() or not _stale():
        return
    import subprocess
    env = dict(os.environ, FASTGRIB_AUTOWARM='0')
    subprocess.run([sys.executable, '-c', 'from fastgrib import _image; _image.build_once()'],
                   stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   env=env, cwd=_home(), **_quiet())


def build_once():
    """build(), unless another process is at it. A failure is noted: this
    machine then goes by numba."""
    if sys.platform != 'win32':
        try:
            os.nice(10)
        except OSError:
            pass
    p = path()
    if p is None:
        return
    lock = p + '.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            if time.time() - os.path.getmtime(lock) < 600:
                return
            os.remove(lock)
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            return
    except OSError:
        return
    ok = False
    try:
        ok = build(p) is not None
    finally:
        os.close(fd)
        try:
            if not ok:
                with open(p + '.failed', 'w'):
                    pass
            elif os.path.exists(p + '.failed'):
                os.remove(p + '.failed')
            os.remove(lock)
        except OSError:
            pass
