"""Octets and bits, the layer every other module reads through: GRIB's
integers (unsigned, and signed as sign and magnitude), IBM floats (GRIB1),
wgrib2's integer powers, and numba bit readers -- `bits`, branch-free,
for the decoders' hot loops; `ints`, a run of equal-width integers."""
import numpy as np
from ._jit import intrinsic, njit


# --- octets (Python)

def u(b, at, n):
    """The unsigned big-endian integer of b[at:at + n]."""
    return int.from_bytes(bytes(b[at:at + n]), 'big')


def sm(raw, nbytes):
    """A sign-magnitude integer of `nbytes` bytes (GRIB's signed ints)."""
    top = 1 << (8 * nbytes - 1)
    return -(raw & (top - 1)) if raw & top else raw


def s(b, at, n):
    """The signed (sign and magnitude) integer of b[at:at + n]."""
    return sm(u(b, at, n), n)


def sm_bytes(v, nbytes):
    """v as `nbytes` octets of sign and magnitude."""
    v = int(round(v))
    top = 1 << (8 * nbytes - 1)
    return ((top | -v) if v < 0 else v).to_bytes(nbytes, 'big')


def ibm(b, at):
    """The IBM single-precision float at b[at] (sign, base-16 exponent,
    24 bits): exactly a float64."""
    v = u(b, at, 4)
    if v & 0x7FFFFFFF == 0:
        return 0.0
    m = (v & 0xFFFFFF) * 16.0 ** (((v >> 24) & 0x7F) - 70)
    return -m if v & 0x80000000 else m


def ibm_array(raw):
    """IBM floats (bytes) as float64, exactly."""
    v = np.frombuffer(bytes(raw), '>u4').astype(np.int64)
    m = (v & 0xFFFFFF).astype(np.float64) * 16.0 ** (((v >> 24) & 0x7F) - 70)
    return np.where(v & 0x80000000, -m, m)


def int_power(x, y):
    """wgrib2's Int_Power: x**y by squaring, 1/x first for y < 0."""
    if y < 0:
        y, x = -y, 1.0 / x
    v = 1.0
    while y:
        if y & 1:
            v *= x
        x *= x
        y >>= 1
    return v


# --- bits (numba)
#
# The small helpers here and in the decoders are plain @njit, not
# inline='always': LLVM inlines them anyway, while numba's own IR inlining
# of a helper taking arrays left reference counting in the hot loops --
# complex packing 1.6 -> 1.3 ns a value under numba 0.59, 15.6 -> 1.2 under
# numba 0.63 (Linux), where those loops had become ten times slower.

@njit(cache=True, nogil=True)
def be64(buf, q):
    """The 8 bytes from byte q, big-endian. Unsigned indices, so numba adds
    no negative-index fix-up and LLVM makes it one load and a byte swap."""
    u = np.uint64(q)
    return ((np.uint64(buf[u]) << np.uint64(56))
            | (np.uint64(buf[u + np.uint64(1)]) << np.uint64(48))
            | (np.uint64(buf[u + np.uint64(2)]) << np.uint64(40))
            | (np.uint64(buf[u + np.uint64(3)]) << np.uint64(32))
            | (np.uint64(buf[u + np.uint64(4)]) << np.uint64(24))
            | (np.uint64(buf[u + np.uint64(5)]) << np.uint64(16))
            | (np.uint64(buf[u + np.uint64(6)]) << np.uint64(8))
            | np.uint64(buf[u + np.uint64(7)]))


@njit(cache=True, nogil=True)
def rd(buf, bit, n):
    """n (0..32) bits, big-endian, from absolute bit offset `bit`; zeros
    past the end."""
    if n == 0:
        return 0
    b = bit >> 3
    s = bit & 7
    nb = (s + n + 7) >> 3
    v = 0
    for k in range(nb):
        v = (v << 8) | (np.int64(buf[b + k]) if b + k < buf.size else 0)
    return (v >> (nb * 8 - s - n)) & ((1 << n) - 1)


@njit(cache=True, nogil=True)
def bits(buf, bit, n):
    """n (0..56) bits, as rd, without a branch: one 8-byte load and shifts
    (0 bits shift out to 0); rd near the buffer's end."""
    if (bit >> 3) + 8 > buf.size:
        return rd(buf, bit, n)
    x = (be64(buf, bit >> 3) << np.uint64(bit & 7)) >> np.uint64(1)
    return np.int64(x >> np.uint64(63 - n))


@intrinsic
def clz64(typingctx, x):
    """The leading zero bits of a 64-bit integer (64 for 0): LLVM's ctlz,
    one instruction (lzcnt) where a loop shifting bit by bit took one
    iteration a zero."""
    from llvmlite import ir
    from numba import types
    from numba.core import cgutils
    def codegen(context, builder, sig, args):
        i64 = ir.IntType(64)
        fn = cgutils.get_or_insert_function(
            builder.module, ir.FunctionType(i64, [i64, ir.IntType(1)]), 'llvm.ctlz.i64')
        return builder.call(fn, [args[0], ir.Constant(ir.IntType(1), 0)])
    return types.int64(x), codegen


@intrinsic
def _ctpop64(typingctx, x):
    """The set bits of a 64-bit integer: LLVM's ctpop (popcnt)."""
    from llvmlite import ir
    from numba import types
    from numba.core import cgutils
    def codegen(context, builder, sig, args):
        i64 = ir.IntType(64)
        fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(i64, [i64]),
                                            'llvm.ctpop.i64')
        return builder.call(fn, [args[0]])
    return types.int64(x), codegen


@intrinsic
def take(typingctx, ctr, i):
    """ctr[i]'s value, then one more there: an atomic fetch-and-add, so
    each of several threads calling it takes a different value (work
    shared out by a counter: _cpu.share)."""
    from numba import types
    from numba.core import cgutils
    def codegen(context, builder, sig, args):
        ary = context.make_array(sig.args[0])(context, builder, args[0])
        ptr = cgutils.get_item_pointer(context, builder, sig.args[0], ary, [args[1]])
        return builder.atomic_rmw('add', ptr, context.get_constant(types.int64, 1),
                                  'acq_rel')
    return types.int64(ctr, i), codegen


@njit(cache=True, nogil=True)
def ones(buf, n):
    """How many of buf's first n bits (a bitmap) are set -- bits past its
    end count as clear; 8 bytes a popcount, nothing unpacked (unpacking a
    3.7M-point bitmap to count it took 0.5 ms, most of it fresh pages)."""
    n = min(n, buf.size * 8)
    full = n >> 3
    t = 0
    k = 0
    while k + 8 <= full:
        t += _ctpop64(be64(buf, k))
        k += 8
    while k < full:
        t += _ctpop64(np.uint64(buf[k]))
        k += 1
    if n & 7:
        t += _ctpop64(np.uint64(buf[full] >> (8 - (n & 7))))
    return t


@njit(cache=True, nogil=True)
def ints(buf, bit, w, n):
    """n integers of w bits each from bit `bit` on (int64)."""
    out = np.empty(n, np.int64)
    for i in range(n):
        out[i] = bits(buf, bit + i * w, w)
    return out
