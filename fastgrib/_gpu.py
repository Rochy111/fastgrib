"""The GPU decoder (cupy): _cpu.decode's arguments and its values, to the
bit, into a cupy array or a numpy one (fastgrib's front end decides when
to use it).

    recs = _cpu.scan(buf)
    cols = _cpu.defined(buf, recs[0])
    out = decode(buf, recs, cols)                     # cupy (len(recs), len(cols))
    decode(buf, recs, cols, out=host_array)           # numpy, DMA'd in
    for i, block in decode_iter(buf, recs, cols):     # a file, batched
        ...

Complex packing (5.2, 5.3), the bulk of NCEP's GRIB2, decodes on the card:
  1. group headers, and a block scan of their lengths and bit counts;
  2. per chunk of CH values, its effect on the spatial differencing state
     (last value L, last difference d) as the affine map
     (L, d) -> (L + m d + b, d + a); a scan of those maps gives every chunk
     its exact starting state, in integers, so parallel order changes
     nothing. Groups of one difference (width 0, or NBM's 1-bit zeros)
     sum in closed form;
  3. the field's range of integers, sampled at the chunk starts and
     widened by a margin, and wgrib2's float of each, (R + X 2^E) 10^-D in
     double, once a distinct value: consumer cards run FP64 at 1/32-1/128
     of FP32, which per point would be the whole cost;
  4. every point's value, a table lookup (the FP64 expression for an
     integer outside the table), straight into its column.
Other packings (simple, PNG, JPEG2000, CCSDS), constant fields and records
with fewer defined values than their differencing order decode on the CPU
instead: the same values.

Records go in sub-batches sized to the card (2^19 values an SM, half for
host results), each sub-batch's kernels one CUDA graph launch, its upload
on a stream of its own. Host results are DMA'd from the card straight into
the caller's array (page-locked for it), queued behind the next upload:
GeForce cards under WDDM have one copy engine, which takes copies in order.

NBM CONUS, 3.7M-point fields, all 769 of a file (RTX 5070 Ti, PCIe 5 x16):
                    1 a call   decode_iter
  on the card        0.21       0.10 ms a field   (the kernels: ~0.10)
  into a numpy out   0.64       0.36              (the copy engine: ~0.34)
  _cpu.decode        1.4        0.55              (on every core)

Kept between calls, given back by release(): device -- a kernel workspace
(~17 MB a field of a sub-batch), two byte slots, and for host results two
scratch buffers; page-locked host memory -- two staging buffers, the GRIB
buffer while it is reused (at most SRC_MAX and an eighth of RAM; uploads
then skip the staging copy), and up to NREG numpy results.

Needs cupy and a GPU of compute capability 5.0 or later. Compiled without
FMA contraction, so the double arithmetic rounds as the host's does.
"""
import os
import sys
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ._jit import njit

from . import _cpu as G

CH = 32                    # values a chunk
TPB = 256                  # threads a block
IPT = 2                    # groups a thread in the header scan
TABMAX = 1 << 18           # integers a record's float table covers
SM_VALUES = 1 << 19        # values a sub-batch per SM, device results (half
                           # for host results); 70 SMs: 37M and 18M
MAX_SUB = 64               # records a sub-batch at most
CPU_BELOW = 1 << 17        # numpy results of smaller fields: grib2 decodes them
                           # (GFS 1 deg, 65k points: 0.18 ms one at a time against
                           # 0.33, 0.036 against 0.042 at 64; 0.5 deg the GPU wins)
NREG = 2                   # numpy results kept page-locked for DMA
SRC_MAX = 2 << 30          # largest GRIB buffer kept page-locked
RING, NRING = 4 << 20, 4   # copy-back ring when a result cannot be locked
NCOPY = 4                  # host copy threads (staging fill, ring)

_FI = ('ng', 'n', 'nch', 'bref', 'bw', 'bl', 'data', 'nbits', 'refw', 'bitsw',
       'refl', 'incl', 'bitsl', 'lastl', 'mgmt', 'order', 'minv', 'e0', 'e1',
       'ks0', 'ks1', 'nbg', 'nbc', 'tabmax', 'nbytes')
_FF = ('r0', 'f2', 'f10', 'miss')
_FP = ('buf', 'gref', 'gw', 'gl', 'gk', 'gb', 'gta', 'gtb', 'cm', 'ca', 'cb',
       'ctm', 'cta', 'ctb', 'cg', 'out', 'dest', 'st', 'tab')
PAR = np.dtype([(k, 'i8') for k in _FI] + [(k, 'f8') for k in _FF]
               + [(k, 'u8') for k in _FP])

SRC = r'''
typedef long long i64;
typedef unsigned long long u64;
typedef unsigned char u8;
#define CH @CH@
#define TPB @TPB@
#define IPT @IPT@
#define BLK (TPB * IPT)
#define FULL 0xffffffffu
#define I64MAX 9223372036854775807ll
#define I64MIN (-I64MAX - 1)

struct Par {
    i64 ng, n, nch, bref, bw, bl, data, nbits, refw, bitsw, refl, incl, bitsl, lastl,
        mgmt, order, minv, e0, e1, ks0, ks1, nbg, nbc, tabmax, nbytes;
    double r0, f2, f10, miss;
    const u8* buf; i64* gref; int* gw; i64* gl; i64* gk; i64* gb; i64* gta; i64* gtb;
    i64* cm; i64* ca; i64* cb; i64* ctm; i64* cta; i64* ctb; i64* cg;
    float* out; const int* dest; i64* st; float* tab;
};

// n (0..32) bits big-endian from bit `bit`: two aligned 32-bit loads
// (b is 4-byte aligned, and padded 8 bytes past its data)
__device__ __forceinline__ i64 rd(const u8* __restrict__ b, i64 bit, int n, i64 nbytes) {
    // never past the record's bytes (+8 spare), whatever corrupt headers say:
    // the host's check reports those
    i64 a = min((bit >> 5) << 2, (nbytes + 3) & ~3ll);
    int s = (int)(bit - a * 8);
    const unsigned* w = (const unsigned*)(b + a);
    // the 32 bits from bit s of the big-endian pair: one funnel shift
    // (a 64-bit shift is several 32-bit instructions)
    unsigned x = __funnelshift_l(__byte_perm(w[1], 0, 0x0123), __byte_perm(w[0], 0, 0x0123), s);
    return n ? (i64)(x >> (32 - n)) : 0;        // a select, not a branch
}

// (L, d) -> (L + m d + b, d + a); comp(x, y) is x then y
struct A { i64 m, a, b; };
__device__ __forceinline__ A comp(A x, A y) {
    A r; r.m = x.m + y.m; r.a = x.a + y.a; r.b = x.b + y.b + y.m * x.a; return r;
}
__device__ __forceinline__ A shup(A x, int o) {
    A r; r.m = __shfl_up_sync(FULL, x.m, o); r.a = __shfl_up_sync(FULL, x.a, o);
    r.b = __shfl_up_sync(FULL, x.b, o); return r;
}

// block-wide exclusive scan of each thread's t; *tot the block's total
__device__ __forceinline__ A block_excl(A t, A* tot) {
    __shared__ A ws[TPB / 32];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    A inc = t;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) { A y = shup(inc, o); if (lane >= o) inc = comp(y, inc); }
    if (lane == 31) ws[wid] = inc;
    __syncthreads();
    if (wid == 0) {
        A w = {0, 0, 0}; if (lane < TPB / 32) w = ws[lane];
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) { A y = shup(w, o); if (lane >= o) w = comp(y, w); }
        if (lane < TPB / 32) ws[lane] = w;
    }
    __syncthreads();
    A ex = shup(inc, 1); if (lane == 0) { ex.m = 0; ex.a = 0; ex.b = 0; }
    if (wid > 0) ex = comp(ws[wid - 1], ex);
    *tot = ws[TPB / 32 - 1];
    __syncthreads();
    return ex;
}

__device__ __forceinline__ i64 wmin(i64 x) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = min(x, __shfl_xor_sync(FULL, x, o));
    return x;
}
__device__ __forceinline__ i64 wmax(i64 x) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = max(x, __shfl_xor_sync(FULL, x, o));
    return x;
}

__device__ __forceinline__ i64 gk0(const Par& P, i64 g) { return P.gta[g / BLK] + P.gk[g]; }
__device__ __forceinline__ i64 gbit0(const Par& P, i64 g) { return P.gtb[g / BLK] + P.gb[g]; }

__device__ __forceinline__ bool miss_of(const Par& P, i64 r, int w, i64 raw) {
    if (P.mgmt == 0) return false;
    if (w == 0) {
        i64 allone = (1ll << P.nbits) - 1;
        return r == allone || (P.mgmt == 2 && r == allone - 1);
    }
    i64 mw = (1ll << w) - 1;
    return raw == mw || (P.mgmt == 2 && raw == mw - 1);
}

// the values [bit, bit + cnt w) of a group (cnt <= CH): one difference
// throughout (width 0, or width 1 all zeros -- NBM's packer never writes
// width 0, so most of its values come as 1-bit zeros), all missing, or
// neither. Under secondary missing values a 1-bit 0 is missing.
#define RUN_PLAIN 0
#define RUN_CONST 1
#define RUN_MISSING 2
__device__ __forceinline__ int run_kind(const Par& P, i64 r, int w, i64 bit, i64 cnt) {
    if (w == 0) return miss_of(P, r, 0, 0) ? RUN_MISSING : RUN_CONST;
    if (w == 1 && cnt > 0 && rd(P.buf, bit, (int)cnt, P.nbytes) == 0)
        return P.mgmt == 2 ? RUN_MISSING : RUN_CONST;
    return RUN_PLAIN;
}

__device__ __forceinline__ i64 kseed_of(const Par& P) {
    return P.order == 2 ? P.ks1 : (P.order == 1 ? P.ks0 : -1);
}

// 1: group headers + block-local scan of (length, bits)
extern "C" __global__ void grp(const Par* __restrict__ PP) {
    const Par& P = PP[blockIdx.y];
    // the range of integers the table covers starts empty (set before
    // any block reads it: rng and the order-0 bound below come later
    // or by atomics)
    if (blockIdx.x == 0 && threadIdx.x == 0 && P.order > 0) { P.st[0] = I64MAX; P.st[1] = I64MIN; }
    if ((i64)blockIdx.x >= P.nbg) return;
    i64 base = (i64)blockIdx.x * BLK + threadIdx.x * IPT;
    i64 L[IPT], B[IPT];
    i64 glo = I64MAX, ghi = I64MIN;
    A t = {0, 0, 0};
    #pragma unroll
    for (int k = 0; k < IPT; ++k) {
        i64 g = base + k; L[k] = 0; B[k] = 0;
        if (g < P.ng) {
            i64 r = rd(P.buf, P.bref + g * P.nbits, (int)P.nbits, P.nbytes);
            int w = (int)(P.refw + rd(P.buf, P.bw + g * P.bitsw, (int)P.bitsw, P.nbytes));
            i64 l = (g == P.ng - 1) ? P.lastl
                  : P.refl + rd(P.buf, P.bl + g * P.bitsl, (int)P.bitsl, P.nbytes) * P.incl;
            P.gref[g] = r; P.gw[g] = w; P.gl[g] = l;
            L[k] = l; B[k] = (i64)w * l;
            if (P.order == 0) { glo = min(glo, r); ghi = max(ghi, r + ((1ll << w) - 1)); }
        }
        t.a += L[k]; t.b += B[k];
    }
    if (P.order == 0) {                          // values are raw + reference
        glo = wmin(glo); ghi = wmax(ghi);
        if ((threadIdx.x & 31) == 0 && glo <= ghi) {
            atomicMin((long long*)&P.st[0], glo); atomicMax((long long*)&P.st[1], ghi);
        }
    }
    A tot; A ex = block_excl(t, &tot);
    i64 ka = ex.a, kb = ex.b;
    #pragma unroll
    for (int k = 0; k < IPT; ++k) {
        i64 g = base + k;
        if (g < P.ng) { P.gk[g] = ka; P.gb[g] = kb; }
        ka += L[k]; kb += B[k];
    }
    if (threadIdx.x == 0) { P.gta[blockIdx.x] = tot.a; P.gtb[blockIdx.x] = tot.b; }
}

// 2, 4: exclusive scan of a record's block totals, in place
extern "C" __global__ void top(const Par* __restrict__ PP, int which) {
    const Par& P = PP[blockIdx.y];
    i64 nb = which ? P.nbc : P.nbg;
    i64* tm = which ? P.ctm : (i64*)0;
    i64* ta = which ? P.cta : P.gta;
    i64* tb = which ? P.ctb : P.gtb;
    __shared__ A carry;
    if (threadIdx.x == 0) { carry.m = 0; carry.a = 0; carry.b = 0; }
    __syncthreads();
    for (i64 base = 0; base < nb; base += TPB) {
        i64 i = base + threadIdx.x;
        A x = {0, 0, 0};
        if (i < nb) { x.m = tm ? tm[i] : 0; x.a = ta[i]; x.b = tb[i]; }
        A tot; A ex = block_excl(x, &tot);
        A c = carry;
        A g = comp(c, ex);
        __syncthreads();
        if (i < nb) { if (tm) tm[i] = g.m; ta[i] = g.a; tb[i] = g.b; }
        if (threadIdx.x == 0) carry = comp(c, tot);
        __syncthreads();
    }
}

__device__ __forceinline__ i64 group_of(const Par& P, i64 k) {
    i64 lo = 0, hi = P.ng;
    while (hi - lo > 1) { i64 mid = (lo + hi) >> 1; if (gk0(P, mid) <= k) lo = mid; else hi = mid; }
    while (lo < P.ng - 1 && gk0(P, lo) + P.gl[lo] <= k) ++lo;
    return lo;
}

// a chunk's map: the seeds' chunk as a map from (0, 0), chunks before it
// the identity
__device__ A chunk_map(const Par& P, i64 c) {
    A out = {0, 0, 0};
    i64 k = c * CH, k1 = k + CH; if (k1 > P.n) k1 = P.n;
    i64 g = group_of(P, k);
    P.cg[c] = g;
    i64 kseed = kseed_of(P);
    if (P.order == 0 || k1 - 1 < kseed) return out;
    bool fromzero = k <= kseed;
    i64 L = 0, d = 0, cnt = 0;
    if (fromzero) { L = P.order == 2 ? P.e1 : P.e0; d = P.order == 2 ? P.e1 - P.e0 : 0; k = kseed + 1; }
    if (k >= k1) { out.a = d; out.b = L; return out; }
    while (g < P.ng - 1 && gk0(P, g) + P.gl[g] <= k) ++g;
    if (P.order == 2 && !fromzero) {
        // a value a step whatever the group, as in chunk_values: the
        // warp's chunks keep in step
        i64 r = P.gref[g]; int w = P.gw[g]; i64 e = gk0(P, g) + P.gl[g];
        i64 base = r + P.minv, bit = P.data + gbit0(P, g) + (k - gk0(P, g)) * w;
        for (; k < k1; ++k, bit += w) {
            while (k >= e && g < P.ng - 1) {
                ++g; r = P.gref[g]; w = P.gw[g]; base = r + P.minv;
                bit = P.data + gbit0(P, g); e += P.gl[g];
            }
            i64 raw = rd(P.buf, bit, w, P.nbytes);
            if (miss_of(P, r, w, raw)) continue;
            d += raw + base; L += d; ++cnt;
        }
        out.m = cnt; out.a = d; out.b = L;
        return out;
    }
    while (k < k1) {
        i64 r = P.gref[g]; int w = P.gw[g]; i64 g0 = gk0(P, g);
        i64 e = g0 + P.gl[g]; if (e > k1) e = k1;
        i64 base = r + P.minv, bit = P.data + gbit0(P, g) + (k - g0) * w;
        int kind = run_kind(P, r, w, bit, e - k);
        if (kind == RUN_MISSING) { k = e; ++g; continue; }
        if (kind == RUN_CONST) {                  // closed form
            i64 c_ = e - k;
            if (P.order == 2) { L += c_ * d + base * (c_ * (c_ + 1) / 2); d += c_ * base; }
            else L += c_ * base;
            cnt += c_; k = e; ++g; continue;
        }
        for (; k < e; ++k, bit += w) {
            i64 raw = rd(P.buf, bit, w, P.nbytes);
            if (miss_of(P, r, w, raw)) continue;
            if (P.order == 2) { d += raw + base; L += d; } else { L += raw + base; }
            ++cnt;
        }
        ++g;
    }
    if (fromzero) { out.a = d; out.b = L; }
    else if (P.order == 2) { out.m = cnt; out.a = d; out.b = L; }
    else { out.b = L; }
    return out;
}

// 3: chunk maps + block-local scan, a chunk a thread
extern "C" __global__ void __launch_bounds__(TPB, 4) chk(const Par* __restrict__ PP) {
    const Par& P = PP[blockIdx.y];
    if ((i64)blockIdx.x >= P.nbc) return;
    i64 c = (i64)blockIdx.x * TPB + threadIdx.x;
    A x = {0, 0, 0};
    if (c < P.nch) x = chunk_map(P, c);
    A tot; A ex = block_excl(x, &tot);
    if (c < P.nch) { P.cm[c] = ex.m; P.ca[c] = ex.a; P.cb[c] = ex.b; }
    if (threadIdx.x == 0) { P.ctm[blockIdx.x] = tot.m; P.cta[blockIdx.x] = tot.a; P.ctb[blockIdx.x] = tot.b; }
}

// a chunk's values from its starting state, handed one by one to `emit`
template <class F>
__device__ __forceinline__ void chunk_values(const Par& P, i64 c, F emit) {
    i64 k0 = c * CH, k = k0, k1 = k + CH; if (k1 > P.n) k1 = P.n;
    i64 L = 0, d = 0;
    if (P.order > 0) {
        i64 cb_ = c / TPB;
        A pre = {P.ctm[cb_], P.cta[cb_], P.ctb[cb_]}, loc = {P.cm[c], P.ca[c], P.cb[c]};
        A s = comp(pre, loc); L = s.b; d = s.a;
    }
    i64 g = P.cg[c];
    bool plain = k0 > kseed_of(P);             // no seed in this chunk
    if (plain && P.order == 2) {
        // NCEP's common case, a value a step whatever the group: constant
        // groups read 0 bits (or 1-bit zeros) and take the same steps, so
        // a warp's 32 chunks keep in step -- branching by kind of group
        // left 11 of 32 lanes working
        i64 r = P.gref[g]; int w = P.gw[g]; i64 e = gk0(P, g) + P.gl[g];
        i64 base = r + P.minv, bit = P.data + gbit0(P, g) + (k - gk0(P, g)) * w;
        for (; k < k1; ++k, bit += w) {
            while (k >= e && g < P.ng - 1) {
                ++g; r = P.gref[g]; w = P.gw[g]; base = r + P.minv;
                bit = P.data + gbit0(P, g); e += P.gl[g];
            }
            i64 raw = rd(P.buf, bit, w, P.nbytes);
            if (miss_of(P, r, w, raw)) { emit(k, k0, false, 0); continue; }
            d += raw + base; L += d;
            emit(k, k0, true, L);
        }
        return;
    }
    while (k < k1) {
        i64 r = P.gref[g]; int w = P.gw[g]; i64 g0 = gk0(P, g);
        i64 e = g0 + P.gl[g]; if (e > k1) e = k1;
        i64 base = r + P.minv, bit = P.data + gbit0(P, g) + (k - g0) * w;
        if (plain && P.order == 2) {             // NCEP's common case
            int kind = run_kind(P, r, w, bit, e - k);
            if (kind == RUN_MISSING) {
                for (; k < e; ++k) emit(k, k0, false, 0);
                ++g; continue;
            }
            if (kind == RUN_CONST) {             // one difference: no bits to read
                for (; k < e; ++k) { d += base; L += d; emit(k, k0, true, L); }
                ++g; continue;
            }
            for (; k < e; ++k, bit += w) {
                i64 raw = rd(P.buf, bit, w, P.nbytes);
                if (miss_of(P, r, w, raw)) { emit(k, k0, false, 0); continue; }
                d += raw + base; L += d;
                emit(k, k0, true, L);
            }
        } else {
            for (; k < e; ++k, bit += w) {
                i64 raw = rd(P.buf, bit, w, P.nbytes);
                if (miss_of(P, r, w, raw)) { emit(k, k0, false, 0); continue; }
                i64 v;
                if (P.order == 0) v = raw + r;
                else if (k == P.ks0) { v = P.e0; L = v; }
                else if (P.order == 2 && k == P.ks1) { v = P.e1; d = P.e1 - P.e0; L = v; }
                else if (P.order == 2) { d += raw + base; L += d; v = L; }
                else { L += raw + base; v = L; }
                emit(k, k0, true, v);
            }
        }
        ++g;
    }
}

// The float table's integers: the sampled range [st0, st1] widened by a
// margin (values between two samples can stray past both), within
// tabmax; size 0 when there is no table
__device__ __forceinline__ void tab_range(const Par& P, i64* lo, i64* size) {
    i64 a = P.st[0], b = P.st[1];
    *lo = 0; *size = 0;
    if (a > b || P.tabmax <= 0) return;
    i64 span = b - a + 1;
    if (span > P.tabmax) return;
    i64 m = P.order == 0 ? 0 : span / 4 + 64;
    if (span + 2 * m > P.tabmax) m = (P.tabmax - span) / 2;
    *lo = a - m; *size = span + 2 * m;
}

// wgrib2's float of an integer: (R + X 2^E) 10^-D in double, rounded once
__device__ __forceinline__ float fval(const Par& P, i64 v) {
    return (float)((P.r0 + (double)v * P.f2) * P.f10);
}
// an integer outside the table (rare: none in NBM's 769 fields), its
// float straight into its column -- out of line, as inline FP64 beside the
// lookup the compiler predicates it in and every warp pays the slow pipe
__device__ __noinline__ void put_slow(const Par& P, i64 k, i64 v) {
    i64 j = P.dest ? (i64)P.dest[k] : k;
    if (j >= 0) P.out[j] = fval(P, v);
}

// 5: the sampled range: every chunk's starting value (the last defined
// value before it, a value of the field every CH points) once the seeds
// are in
extern "C" __global__ void rng(const Par* __restrict__ PP) {
    const Par& P = PP[blockIdx.y];
    if (P.order == 0 || (i64)blockIdx.x >= P.nbc) return;
    i64 c = (i64)blockIdx.x * TPB + threadIdx.x;
    i64 lo = I64MAX, hi = I64MIN;
    if (c < P.nch && c * CH > kseed_of(P)) {
        i64 b = c / TPB;
        A pre = {P.ctm[b], P.cta[b], P.ctb[b]}, loc = {P.cm[c], P.ca[c], P.cb[c]};
        lo = hi = comp(pre, loc).b;
    }
    lo = wmin(lo); hi = wmax(hi);
    if ((threadIdx.x & 31) == 0 && lo <= hi) {
        atomicMin((long long*)&P.st[0], lo); atomicMax((long long*)&P.st[1], hi);
    }
}

// 6: the table: wgrib2's float of each integer in the range, once each
extern "C" __global__ void tab(const Par* __restrict__ PP) {
    const Par& P = PP[blockIdx.y];
    i64 lo, size; tab_range(P, &lo, &size);
    i64 t = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < size) P.tab[t] = fval(P, lo + t);
}

// 7: the values, as floats, straight into their columns. The serial
// chunk loop only puts each point's table offset in shared memory (-1
// missing, -2 written already); the lane-parallel write-out looks them up,
// a store being 32 consecutive floats (or 32 consecutive points' columns)
#define T_MISS (-1)
#define T_DONE (-2)
// occupancy: 3 blocks an SM (80 registers), measured ~7% faster than 90
extern "C" __global__ void __launch_bounds__(TPB, 3) put(const Par* __restrict__ PP) {
    const Par& P = PP[blockIdx.y];
    __shared__ int tile[TPB / 32][32][CH + 1];
    i64 c0 = (i64)blockIdx.x * TPB;
    if (c0 >= P.nch) return;
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    i64 c = c0 + threadIdx.x;
    i64 lo, size; tab_range(P, &lo, &size);
    if (c < P.nch) {
        chunk_values(P, c, [&](i64 k, i64 k0, bool def, i64 v) {
            int x = T_MISS;
            if (def) {
                i64 t = v - lo;
                if (t >= 0 && t < size) x = (int)t;
                else { put_slow(P, k, v); x = T_DONE; }
            }
            tile[wid][lane][k - k0] = x;
        });
    }
    __syncwarp();
    i64 w0 = (c - lane) * CH;
    const int* __restrict__ dst = P.dest;
    const float* __restrict__ T = P.tab;
    float* __restrict__ fo = P.out;
    #pragma unroll 4
    for (int t = lane; t < 32 * CH; t += 32) {
        i64 idx = w0 + t;
        if (idx < P.n) {
            int x = tile[wid][t / CH][t % CH];
            if (x == T_DONE) continue;
            float y = x == T_MISS ? (float)P.miss : __ldg(T + x);
            if (!dst) fo[idx] = y; else { int j = dst[idx]; if (j >= 0) fo[j] = y; }
        }
    }
}

// 8: a north-up field (_cpu._FLIP) put in file order: its rows of ni
// turned round, in place -- a pass over the field, where a column map
// was one more int a point to build on the host and read here
extern "C" __global__ void turn(float* __restrict__ f, i64 ni, i64 nj) {
    i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ni) return;
    for (i64 j = blockIdx.y; j < nj / 2; j += gridDim.y) {
        float* a = f + j * ni + i;
        float* b = f + (nj - 1 - j) * ni + i;
        float y = *a; *a = *b; *b = y;
    }
}
'''.replace('@CH@', str(CH)).replace('@TPB@', str(TPB)).replace('@IPT@', str(IPT))

_KERNELS = ('grp', 'top', 'chk', 'rng', 'tab', 'put', 'turn')
_NO_TABLE = False          # tests: every point through the FP64 expression
_NO_CPU = False            # tests: numpy results of small fields on the card too


# --- public -----------------------------------------------------------------

def decode(buf, recs, cols, out=None, undef=G.UNDEF):
    """grib2.decode on the current CUDA device: (len(recs), len(cols))
    float32, row i for recs[i], `undef` where a record has no value.

    out None or a cupy array: the result on the card, queued on cupy's
    current stream; returns without waiting. out a numpy array: the values
    DMA'd into it (the array it belongs to stays page-locked for later
    calls, NREG at most); returns when they are in. Fields under
    CPU_BELOW points go to grib2.decode instead, faster for them."""
    if isinstance(out, np.ndarray) and np.size(cols) < CPU_BELOW and not _NO_CPU:
        return G.decode(buf, recs, cols, out, undef)
    cp, dev = _device()
    job = _Job(cp, dev, buf, recs, cols, out, undef)
    with dev.lock:
        if job.host:
            pipe = _HostPipe(cp, dev, job.ncols, dev.host_scratch)
            pipe.push_all(job, job.out)
            done = pipe.flush()
        elif job.gpu:
            _pipeline(job, _sub_size(job.n, dev.sub_values[0], len(job.gpu)))
    job.decode_cpu()
    if job.host:
        done.synchronize()
    job.check()
    return job.out


def decode_iter(buf, recs, cols, budget=None, host=True, undef=G.UNDEF):
    """decode() over any number of records, batched as grib2.batch_size()
    sizes them (budget: its cap, bytes of result): yields (i, block),
    block's rows for recs[i:i + len(block)]. A block stays valid until the
    next one is asked for.

    host=True: numpy blocks from two buffers sharing the budget (page-
    locked once), filled by one pipeline that runs on through the batches
    -- while you work on a block the next is decoding and being DMA'd in.
    host=False: cupy blocks from one buffer, queued on the current
    stream."""
    if host and np.size(cols) < CPU_BELOW and not _NO_CPU:
        yield from G.decode_iter(buf, recs, cols, budget, undef)
        return
    cp, dev = _device()
    cols = np.ascontiguousarray(cols, dtype=np.int64)
    budget = G.default_budget() if budget is None else budget
    if not host:
        n = G.batch_size(cols.size, len(recs), budget)
        res = cp.empty((n, cols.size), dtype=cp.float32)
        for i in range(0, len(recs), n):
            part = recs[i:i + n]
            yield i, decode(buf, part, cols, res[:len(part)], undef)
        return
    n = G.batch_size(cols.size, len(recs), budget // 2, G.BATCH_BYTES // 2)
    res = [np.empty((n, cols.size), dtype=np.float32) for _ in range(2)]
    pipe = _HostPipe(cp, dev, cols.size, [None, None])   # its own scratch
    prev = None                              # (i, block, done) a batch behind
    try:
        for t, i in enumerate(range(0, len(recs), n)):
            block = res[t % 2][:len(recs[i:i + n])]
            job = _Job(cp, dev, buf, recs[i:i + n], cols, block, undef)
            done = []                        # the batch's last copy, once queued
            with dev.lock:
                pipe.push_all(job, block, done)
            job.decode_cpu()
            if prev is not None:
                yield _settle(dev, pipe, prev)
            prev = (i, block, done, job)
        if prev is not None:
            yield _settle(dev, pipe, prev)
    finally:                                 # done, or dropped: its buffers' locks go
        for r in res:
            unlock(r)


def _settle(dev, pipe, batch):
    """(i, block) of a batch once its values are in: its last copy is
    queued by the next batch's first sub-batch, or here."""
    i, block, done, job = batch
    job.check()
    if not done:
        with dev.lock:
            ev = pipe.flush()
        if ev is not None:
            done.append(ev)
    if done:
        done[0].synchronize()
    return i, block


def unlock(arr):
    """Let go of the page lock decode() keeps on a numpy result's array
    (NREG at most, for reuse) or on a GRIB buffer, for one that leaves for
    good -- as the front end's read() hands its results over and close()
    lets a file go."""
    import cupy as cp
    ptrs = {_root(arr).ctypes.data, arr.ctypes.data}
    with _devices_lock:
        devs = list(_devices.values())
    for dev in devs:
        with dev.lock:
            for key in [k for k in dev.reg if k[0] in ptrs]:
                dev.xs.synchronize()
                cp.cuda.runtime.hostUnregister(key[0])
                del dev.reg[key]
            if dev.src is not None and dev.src[0][0] in ptrs:     # a GRIB buffer
                dev.ups.synchronize()
                cp.cuda.runtime.hostUnregister(dev.src[0][0])
                dev.src = None


def release():
    """Give back everything decode() keeps between calls, on every device;
    the next decode() makes it again."""
    import cupy as cp
    with _devices_lock:
        for dev in _devices.values():
            with dev.lock:
                dev.release(cp)


# --- the device -------------------------------------------------------------

_devices = {}
_devices_lock = threading.Lock()


def _device():
    import cupy as cp
    i = cp.cuda.Device().id
    with _devices_lock:
        if i not in _devices:
            _devices[i] = _Device(cp)
        return cp, _devices[i]


class _Device:
    """A device's kernels, streams and kept buffers. One caller at a time
    (`lock`)."""

    def __init__(self, cp):
        _nvrtc_path()
        # met-cu replaces cupy's RawModule with one keyed on kernel names,
        # stale after a source change; cupy's own, if it has
        ptx = sys.modules.get('metcu.ptx_runtime')
        raw = getattr(ptx, '_REAL_RawModule', None) or cp.RawModule
        mod = raw(code=SRC, options=('--fmad=false',))
        self.k = {name: mod.get_function(name) for name in _KERNELS}
        self.lock = threading.Lock()
        self.ups = cp.cuda.Stream(non_blocking=True)      # uploads
        self.xs = cp.cuda.Stream(non_blocking=True)       # host results' copies
        self.cap = cp.cuda.Stream(non_blocking=True)      # graph capture
        self.copiers = None
        # sub-batches to fill the card: values by SM, within free memory
        sm = cp.cuda.Device().attributes['MultiProcessorCount']
        v = int(min(sm * SM_VALUES, cp.cuda.runtime.memGetInfo()[0] // 64))
        self.sub_values = (v, v // 2)
        self._reset()

    def _reset(self):
        self.ws = self.ws_ev = self.ws_stream = None      # kernel workspace
        self.slot = [None, None]          # device bytes + parameters
        self.slot_free = [None, None]     # the kernels last reading each
        self.pin = [None, None]           # page-locked upload staging
        self.pin_ev = [None, None]        # the upload last reading each
        self.host_scratch = [None, None]  # decode()'s host results on their way out
        self.graphs = {}
        self.reg = OrderedDict()          # page-locked result arrays
        self.src = None                   # (key, buffer) page-locked GRIB bytes
        self.maps = OrderedDict()         # id -> (grib2's kept column map, on the card)
        self.ring = None

    def release(self, cp):
        for s in (self.ups, self.xs, self.cap):
            s.synchronize()
        for e in [self.ws_ev] + self.slot_free + self.pin_ev:
            if e is not None:
                e.synchronize()
        for ptr, _ in self.reg:
            cp.cuda.runtime.hostUnregister(ptr)
        if self.src is not None:
            cp.cuda.runtime.hostUnregister(self.src[0][0])
        if self.copiers is not None:
            self.copiers.shutdown()
            self.copiers = None
        self._reset()

    # -- buffers: grown to the need, outside cupy's pool (which kept every
    #    size outgrown: 1.2 GB held for a 141 MB workspace over NBM's file)

    def workspace(self, cp, nbytes, st):
        """The kernels' workspace, free for stream st: on st itself work
        just queues behind its last use; on another stream it waits."""
        if self.ws_ev is not None and (self.ws_stream != st.ptr or self.ws.nbytes < nbytes):
            self.ws_ev.synchronize()
        if self.ws is None or self.ws.nbytes < nbytes:
            self.ws = None
            self.ws = _raw(cp, nbytes)
        return self.ws

    def column_map(self, cp, D):
        """D on the card. grib2 keeps its whole-grid maps, so those stay
        here too (as many as it keeps), not uploaded every call."""
        hit = self.maps.get(id(D))
        if hit is not None and hit[0] is D:
            self.maps.move_to_end(id(D))
            return hit[1]
        dD = cp.asarray(D)
        with G._MAPS_LOCK:
            kept = any(m is D for m in G._MAPS.values())
        if kept:
            if len(self.maps) >= G._MAPS_KEEP:
                cp.cuda.runtime.deviceSynchronize()    # the oldest may be in use
                self.maps.popitem(last=False)
            self.maps[id(D)] = (D, dD)
        return dD

    def ws_used(self, st):
        self.ws_ev, self.ws_stream = st.record(), st.ptr

    def device_slot(self, cp, j, nbytes):
        """Bytes slot j, its last readers waited for on the upload stream."""
        if self.slot[j] is None or self.slot[j].nbytes < nbytes:
            if self.slot_free[j] is not None:
                self.slot_free[j].synchronize()
            self.slot[j] = self.slot_free[j] = None
            self.graphs = {key: g for key, g in self.graphs.items() if key[0] != j}
            self.slot[j] = _raw(cp, nbytes)
        if self.slot_free[j] is not None:
            self.ups.wait_event(self.slot_free[j])
        return self.slot[j]

    def pinned(self, cp, j, nbytes):
        """Page-locked staging j, once its last upload is out."""
        if self.pin_ev[j] is not None:
            self.pin_ev[j].synchronize()
        if self.pin[j] is None or self.pin[j].size < nbytes:
            size = -(-max(nbytes, 1 << 16) // (1 << 20)) * (1 << 20)
            self.pin[j] = np.frombuffer(cp.cuda.alloc_pinned_memory(size), np.uint8, size)
        return self.pin[j]

    def copy_pool(self):
        if self.copiers is None:
            self.copiers = ThreadPoolExecutor(NCOPY, thread_name_prefix='fastgrib_gpu')
        return self.copiers

    # -- page-locking the caller's memory, for DMA straight to or from it

    def lock_source(self, cp, arr):
        """The GRIB buffer, kept locked while reused, so uploads skip the
        staging copy (the host's biggest cost a record). Not a buffer over
        SRC_MAX or an eighth of RAM -- never a big memory-mapped file."""
        root = _root(arr)
        if not root.flags.c_contiguous:
            return False
        key = (root.ctypes.data, root.nbytes)
        if self.src is not None and self.src[0] == key:
            return True
        if key[1] > min(SRC_MAX, G.memory()[0] // 8):
            return False
        if self.src is not None:
            self.ups.synchronize()
            cp.cuda.runtime.hostUnregister(self.src[0][0])
            self.src = None
        try:
            cp.cuda.runtime.hostRegister(key[0], key[1], 0)
        except Exception:
            return False
        self.src = (key, root)                  # held alive while locked
        return True

    def lock_result(self, cp, out):
        """The array a numpy result belongs to, kept locked for later calls
        (a warm 15 MB row locks in ~0.05 ms), NREG at most, least recently
        used let go first. Not over SRC_MAX or an eighth of RAM: a bigger
        one (or out alone, when that is what is too big) goes through the
        copy ring instead."""
        root = _root(out) if _root(out).flags.c_contiguous else out
        cap = min(SRC_MAX, G.memory()[0] // 8)
        if root.nbytes > cap:
            root = out
        key = (root.ctypes.data, root.nbytes)
        if key not in self.reg and key[1] > cap:
            return False
        if key in self.reg:
            self.reg.move_to_end(key)
            return True
        try:
            cp.cuda.runtime.hostRegister(key[0], key[1], 0)
        except Exception:
            return False
        self.reg[key] = root
        while len(self.reg) > NREG:
            (ptr, _), _ = self.reg.popitem(last=False)
            self.xs.synchronize()
            cp.cuda.runtime.hostUnregister(ptr)
        return True

    # -- launching

    def run_graph(self, cp, st, j, slot, R, nbg, nbc):
        """A sub-batch's kernels as one CUDA graph (seven launches from
        Python were most of a single field's host time): captured once a
        slot and shape -- the parameters sit at the slot's start, the grids
        rounded up to powers of two, surplus blocks returning at once."""
        gb = 1 << max(int(nbg) - 1, 0).bit_length()
        gc = 1 << max(int(nbc) - 1, 0).bit_length()
        # grow-only: a graph whose grids cover these is used as it is
        # (records of mixed sizes recaptured one call in eight, ~150 us)
        key = (j, slot.data.ptr, R)
        hit = self.graphs.get(key)
        g = hit[0] if hit is not None and hit[1] >= gb and hit[2] >= gc else None
        if g is None:
            if hit is not None:
                gb, gc = max(gb, hit[1]), max(gc, hit[2])
            if len(self.graphs) > 64:
                self.graphs.clear()
            k, cs = self.k, self.cap
            cs.begin_capture()
            k['grp']((gb, R), (TPB,), (slot,), stream=cs)
            k['top']((1, R), (TPB,), (slot, np.int32(0)), stream=cs)
            k['chk']((gc, R), (TPB,), (slot,), stream=cs)
            k['top']((1, R), (TPB,), (slot, np.int32(1)), stream=cs)
            k['rng']((gc, R), (TPB,), (slot,), stream=cs)
            k['tab']((TABMAX // 256, R), (256,), (slot,), stream=cs)
            k['put']((gc, R), (TPB,), (slot,), stream=cs)
            g = cs.end_capture()
            self.graphs[key] = (g, gb, gc)
        g.launch(st)

    def copy_back(self, cp, ev, scr, grows, out):
        """Device rows -> out's rows through the page-locked ring, for an
        out that would not lock: the card fills slots, threads empty them."""
        if self.ring is None:
            self.ring = [np.frombuffer(cp.cuda.alloc_pinned_memory(RING), np.uint8, RING)
                         for _ in range(NRING)]
        self.xs.wait_event(ev)
        n = out.shape[1] * 4
        ob = out.reshape(-1).view(np.uint8)
        busy = [None] * NRING
        s = 0
        for j, gi in enumerate(grows):
            for off in range(0, n, RING):
                m = min(RING, n - off)
                if busy[s] is not None:
                    busy[s].result()
                cp.cuda.runtime.memcpyAsync(self.ring[s].ctypes.data, scr.data.ptr + j * n + off,
                                            m, cp.cuda.runtime.memcpyDeviceToHost, self.xs.ptr)
                busy[s] = self.copy_pool().submit(_land, self.xs.record(), self.ring[s],
                                                  ob, gi * n + off, m)
                s = (s + 1) % NRING
        for f in busy:
            if f is not None:
                f.result()


# --- records ----------------------------------------------------------------

@njit(cache=True, nogil=True)
def _seeds(buf, p):
    """Value indices of a spatially differenced record's first `order`
    defined values, -1 where there are fewer."""
    order = p[G._ORDER] if p[G._DRT] == 3 else 0
    ks0 = ks1 = -1
    if order == 0:
        return ks0, ks1
    bref, bw, bl, q = G._layout(buf, p)[:4]
    mgmt = p[G._MGMT]
    allone = (1 << p[G._NBITS]) - 1
    k = 0
    bit = q * 8
    for g in range(p[G._NG]):
        gref, gw, gl = G._group(buf, p, bref, bw, bl, g)
        mw = (1 << gw) - 1
        for j in range(gl):
            if gw == 0:
                miss = mgmt > 0 and (gref == allone or (mgmt == 2 and gref == allone - 1))
            else:
                raw = G._rd(buf, bit + j * gw, gw)
                miss = mgmt > 0 and (raw == mw or (mgmt == 2 and raw == mw - 1))
            if not miss:
                if ks0 < 0:
                    ks0 = k + j
                    if order == 1:
                        return ks0, ks1
                else:
                    return ks0, k + j
        k += gl
        bit += gw * gl
    return ks0, ks1


def _row(arr, rec, p, f, row):
    """Fill a record's PAR row (but its pointers): its byte span, or None
    when it decodes on the host."""
    if p[G._DRT] not in (2, 3) or p[G._NG] == 0:
        return None
    order = int(p[G._ORDER]) if p[G._DRT] == 3 else 0
    ks0, ks1 = _seeds(arr, p)
    if order and (ks0 < 0 or (order == 2 and ks1 < 0)):
        return None
    bref, bw, bl, q, e0, e1, minv = G._layout(arr, p)
    base = rec.s7 + 5                            # the first byte uploaded
    ng, n = int(p[G._NG]), int(p[G._NVAL])
    nch = -(-n // CH)
    row['ng'], row['n'], row['nch'] = ng, n, nch
    row['bref'], row['bw'], row['bl'] = bref - base * 8, bw - base * 8, bl - base * 8
    row['data'] = (q - base) * 8
    row['nbits'], row['refw'], row['bitsw'] = p[G._NBITS], p[G._REFW], p[G._BITSW]
    row['refl'], row['incl'] = p[G._REFL], p[G._INCL]
    row['bitsl'], row['lastl'] = p[G._BITSL], p[G._LASTL]
    row['mgmt'], row['order'], row['minv'] = p[G._MGMT], order, minv
    row['e0'], row['e1'], row['ks0'], row['ks1'] = e0, e1, ks0, ks1
    row['nbg'], row['nbc'] = -(-ng // (TPB * IPT)), -(-nch // TPB)
    row['tabmax'] = 0 if _NO_TABLE else TABMAX
    row['r0'], row['f2'], row['f10'], row['miss'] = f
    row['nbytes'] = rec.s7end + 4 - base        # reads stay inside these
    return base, rec.s7end + 4                  # through the '7777' (rd reads ahead)


def _sub_size(n, values, count):
    """Records a sub-batch of `count` n-value records: about `values`
    values (1 to MAX_SUB), evened out so the last is not left short. A
    call of half a sub-batch or more goes in two at least, one uploading
    while the other decodes (NBM, 8 a call: 0.138 ms a field against
    0.179); smaller ones, in one launch (GFS 0.5 deg: 0.073 against 0.095)."""
    sub = int(max(1, min(MAX_SUB, values // max(n, 1))))
    parts = max(1, -(-count // sub))
    if parts == 1 and count >= 2 and count * n >= values // 2:
        parts = 2
    return max(1, -(-count // parts))


# --- the pipeline -----------------------------------------------------------

class _Job:
    """One call's records, prepared: where each goes, which decode on the
    card (gpu: [(index, byte span)]) and which on the host (cpu)."""

    def __init__(self, cp, dev, buf, recs, cols, out, undef):
        self.cp, self.dev, self.recs, self.undef = cp, dev, recs, undef
        self.arr = G._u8(buf)
        self.cols = np.ascontiguousarray(cols, dtype=np.int64)
        self.ncols = self.cols.size
        self.host = isinstance(out, np.ndarray)
        if self.host and (out.shape != (len(recs), self.ncols) or out.dtype != np.float32
                          or not out.flags.c_contiguous):
            raise ValueError('out: a C-contiguous float32 (%d, %d) array'
                             % (len(recs), self.ncols))
        self.out = cp.empty((len(recs), self.ncols), dtype=cp.float32) if out is None else out
        self.st = cp.cuda.get_current_stream()
        self.gpu, self.cpu, self.checks = [], [], []
        if not recs:
            return
        self.P, F, D = G._place(self.arr, recs, self.cols, undef)
        self.dD = dev.column_map(cp, D) if (self.P[:, G._DEST] >= 0).any() else None
        self.rows = np.zeros(len(recs), PAR)
        self.rows64 = self.rows.view(np.int64).reshape(len(recs), len(PAR.names))
        self.fills = bool(((self.P[:, G._DEST] >= 0) & (self.P[:, G._FILL] != 0)).any())
        for i, r in enumerate(recs):
            span = _row(self.arr, r, self.P[i], F[i], self.rows[i])
            (self.gpu if span is not None else self.cpu).append((i, span))
        self.n = int(self.rows['n'][self.gpu[0][0]]) if self.gpu else 0
        self.direct = None                   # GRIB buffer locked? (in _sub, under the lock)
        # the card's records' group headers checked on the CPU's threads
        # meanwhile (the kernels' reads stay in bounds whatever they say)
        pool = G._pool()
        self.checks = [(i, pool.submit(G._groups_ok, self.arr, self.P[i])) for i, _ in self.gpu]

    def check(self):
        """Raise for a record whose packed data do not add up."""
        for i, fut in self.checks:
            if not fut.result():
                raise ValueError('%r: the packed data do not add up (damaged)' % self.recs[i])

    def decode_cpu(self):
        """The records the card does not take, decoded here."""
        if self.cpu:
            idx = [i for i, _ in self.cpu]
            vals = G.decode(self.arr, [self.recs[i] for i in idx], self.cols,
                            undef=self.undef)
            self.out[idx] = vals if self.host else self.cp.asarray(vals)


def _pipeline(job, sub):
    """Device results: the GPU records in sub-batches of `sub`, each into
    its own rows of job.out."""
    for k, s0 in enumerate(range(0, len(job.gpu), sub)):
        part = job.gpu[s0:s0 + sub]
        _sub(job, part, job.out, [i for i, _ in part], k % 2, None)


class _HostPipe:
    """Host results: sub-batches decode into a pair of device scratch
    buffers by turns, and each one's rows go to their numpy destination --
    DMA'd straight in when it locks, through the ring otherwise -- once the
    next sub-batch's upload is queued. GeForce cards under WDDM have one
    copy engine, taking copies in the order queued: an upload behind a
    sub-batch's downloads held the next decode up (0.54 ms a field against
    0.39), and between calls the same (decode_iter keeps one pipe going)."""

    def __init__(self, cp, dev, n, scratch):
        self.cp, self.dev, self.n = cp, dev, n
        self.scratch = scratch               # [buf, buf], grown here
        self.free = [None, None]             # each one's last copy out
        self.turn = 0
        self.pending = None

    def push_all(self, job, out, done=None):
        """job's GPU records in sub-batches into out; done gets an event
        once the last one's copy is queued."""
        sub = _sub_size(job.n, self.dev.sub_values[1], len(job.gpu))
        need = min(sub, len(job.gpu)) * self.n * 4
        if job.gpu and (self.scratch[0] is None or self.scratch[0].nbytes < need):
            # the held sub-batch's rows are in the old scratch: their copy
            # goes first (sent later, it read the new one -- a batch's last
            # rows wrong, unless the new buffer happened to land at the
            # old address)
            if self.pending is not None:
                self._send(*self.pending)
                self.pending = None
            self.dev.xs.synchronize()
            self.scratch[0] = self.scratch[1] = None
            self.scratch[0], self.scratch[1] = _raw(self.cp, need), _raw(self.cp, need)
            self.free = [None, None]
        direct = self.dev.lock_result(self.cp, out)
        for s0 in range(0, len(job.gpu), sub):
            part = job.gpu[s0:s0 + sub]
            last = s0 + sub >= len(job.gpu)
            self._push(job, part, out, direct, done if last else None)

    def _push(self, job, part, out, direct, done):
        w = self.turn
        self.turn ^= 1
        n = self.n
        rows = self.scratch[w][:len(part) * n * 4].view(self.cp.float32).reshape(len(part), n)
        _sub(job, part, rows, range(len(part)), w, self.free[w] if direct else None)
        queued = (w, [i for i, _ in part], out, direct, job.st.record(), done)
        if self.pending is not None:
            self._send(*self.pending)
        self.pending = queued

    def flush(self):
        """Queue the held copy; an event for when every copy is in."""
        if self.pending is not None:
            self._send(*self.pending)
            self.pending = None
        return self.dev.xs.record()

    def _send(self, w, grows, out, direct, ev, done):
        dev, n, rt = self.dev, self.n, self.cp.cuda.runtime
        if direct:
            dev.xs.wait_event(ev)
            for j, gi in enumerate(grows):
                rt.memcpyAsync(out[gi].ctypes.data, self.scratch[w].data.ptr + j * n * 4,
                               n * 4, rt.memcpyDeviceToHost, dev.xs.ptr)
            self.free[w] = dev.xs.record()
        else:
            dev.copy_back(self.cp, ev, self.scratch[w], grows, out)
        if done is not None:
            done.append(dev.xs.record())


def _sub(job, part, out, orow, j, wait):
    """One sub-batch: parameters and bytes uploaded into device slot j on
    the upload stream, then its kernels on the caller's stream; part[q]
    decodes into row orow[q] of out."""
    cp, dev, st, arr = job.cp, job.dev, job.st, job.arr
    R = len(part)
    idx = np.array([i for i, _ in part], np.int64)
    spans = np.array([sp_ for _, sp_ in part], np.int64).reshape(R, 2)
    boff = np.empty(R + 1, np.int64)
    lay = np.empty(8, np.int64)
    _plan(job.rows64, idx, spans, boff, lay)
    total, wbytes, pofs = int(lay[0]), int(lay[1]), int(lay[2])
    wp = dev.workspace(cp, wbytes, st).data.ptr
    slot = dev.device_slot(cp, j, total)
    sp = slot.data.ptr
    # the bytes: straight from the locked GRIB buffer, or copied into
    # staging by the copy threads while the parameters are filled in
    if job.direct is None:
        job.direct = dev.lock_source(cp, arr)
    if job.direct:
        pin = dev.pinned(cp, j, pofs)
        fills = []
    else:
        pin = dev.pinned(cp, j, total)
        fills = [dev.copy_pool().submit(_fill, pin, int(boff[q]), arr, a, b)
                 for q, (_, (a, b)) in enumerate(part)]
    par = pin[:R * PAR.itemsize].view(np.int64).reshape(R, len(PAR.names))
    _point(job.rows64, idx, boff, lay, wp, sp, out.data.ptr,
           np.asarray(orow, np.int64), job.ncols * 4,
           job.dD.data.ptr if job.dD is not None else 0, job.P[:, G._DEST], par)
    rt, up = cp.cuda.runtime, dev.ups
    if job.direct:
        for q, (_, (a, b)) in enumerate(part):
            rt.memcpyAsync(sp + int(boff[q]), arr.ctypes.data + a, b - a,
                           rt.memcpyHostToDevice, up.ptr)
        rt.memcpyAsync(sp, pin.ctypes.data, R * PAR.itemsize, rt.memcpyHostToDevice, up.ptr)
    else:
        for f in fills:
            f.result()
        rt.memcpyAsync(sp, pin.ctypes.data, total, rt.memcpyHostToDevice, up.ptr)
    dev.pin_ev[j] = up.record()
    st.wait_event(dev.pin_ev[j])
    if wait is not None:                        # the rows' last copy out is done
        st.wait_event(wait)
    if job.fills:                               # after that wait: before it, a
        for q, i in enumerate(idx):             # fill hit rows still being copied out
            if job.P[i, G._DEST] >= 0 and job.P[i, G._FILL]:
                out[orow[q]].fill(job.undef)    # columns these do not reach
    dev.run_graph(cp, st, j, slot, R, int(lay[6]), int(lay[7]))
    for q, i in enumerate(idx):                 # north-up rows, put in file order
        if job.P[i, G._DEST] == G._FLIP:
            ni = job.recs[i].grid.ni
            nj = job.ncols // ni
            if nj > 1:
                dev.k['turn']((-(-ni // 256), min(nj // 2, 65535)), (256,),
                              (out[orow[q]], np.int64(ni), np.int64(nj)), stream=st)
    dev.slot_free[j] = st.record()
    dev.ws_used(st)


# PAR's columns, for _plan and _point (PAR is all 8-byte fields: a row of
# int64 to numba)
_C = {k: i for i, k in enumerate(PAR.names)}
(_C_NG, _C_NCH, _C_NBG, _C_NBC, _C_BUF, _C_GW, _C_OUT, _C_DEST, _C_ST,
 _C_TAB) = (_C[k] for k in ('ng', 'nch', 'nbg', 'nbc', 'buf', 'gw', 'out',
                            'dest', 'st', 'tab'))
# the workspace's i64 arrays a record, in order, and their lengths
# (0 groups, 1 chunks, 2 group blocks, 3 chunk blocks)
_WS_COLS = np.array([_C[k] for k in ('gref', 'gl', 'gk', 'gb', 'gta', 'gtb', 'cm',
                                     'ca', 'cb', 'cg', 'ctm', 'cta', 'ctb')], np.int64)
_WS_LEN = np.array([0, 0, 0, 0, 2, 2, 1, 1, 1, 1, 3, 3, 3], np.int64)
_PAR_BYTES = PAR.itemsize


@njit(cache=True, nogil=True)
def _plan(rows, idx, spans, boff, lay):
    """A sub-batch's layout. Slot: the parameters first (one address a
    slot, so the graph replays), then each record's bytes with 8 spare
    after them, boff[q] each one's offset. Workspace: i64 arrays a record
    (gref gl gk gb | gta gtb | cm ca cb cg | ctm cta ctb), then the group
    widths, the ranges and the tables. lay = (slot bytes, workspace bytes,
    parameter bytes, widths, ranges, tables, most group blocks, most chunk
    blocks)."""
    R = idx.size
    pofs = -(-R * _PAR_BYTES // 256) * 256
    boff[0] = pofs
    wl = gt = mbg = mbc = 0
    for q in range(R):
        nb = spans[q, 1] - spans[q, 0]
        boff[q + 1] = boff[q] + (nb + 7) // 8 * 8 + 8
        r = rows[idx[q]]
        wl += 4 * r[_C_NG] + 2 * r[_C_NBG] + 4 * r[_C_NCH] + 3 * r[_C_NBC]
        gt += r[_C_NG]
        mbg = max(mbg, r[_C_NBG])
        mbc = max(mbc, r[_C_NBC])
    o_g = -(-wl * 8 // 256) * 256
    o_s = o_g + -(-gt * 4 // 256) * 256
    o_t = o_s + -(-R * 16 // 256) * 256
    lay[0], lay[1], lay[2] = boff[R], o_t + R * TABMAX * 4, pofs
    lay[3], lay[4], lay[5], lay[6], lay[7] = o_g, o_s, o_t, mbg, mbc


@njit(cache=True, nogil=True)
def _point(rows, idx, boff, lay, wp, sp, outp, orow, rowbytes, dptr, dst, par):
    """The sub-batch's parameter rows, pointers filled in, into par."""
    o = wp
    g = wp + lay[3]
    for q in range(idx.size):
        i = idx[q]
        par[q, :] = rows[i, :]
        r = par[q]
        n4 = (r[_C_NG], r[_C_NCH], r[_C_NBG], r[_C_NBC])
        for c in range(_WS_COLS.size):
            r[_WS_COLS[c]] = o
            o += n4[_WS_LEN[c]] * 8
        r[_C_BUF] = sp + boff[q]
        r[_C_GW] = g
        g += r[_C_NG] * 4
        r[_C_OUT] = outp + orow[q] * rowbytes
        r[_C_DEST] = dptr + dst[i] * 4 if dst[i] >= 0 else 0
        r[_C_ST] = wp + lay[4] + q * 16
        r[_C_TAB] = wp + lay[5] + q * TABMAX * 4


# --- helpers ----------------------------------------------------------------

def _raw(cp, nbytes):
    """A device buffer outside cupy's pool, nbytes rounded up to 16 MB."""
    size = -(-nbytes // (16 << 20)) * (16 << 20)
    return cp.ndarray((size,), cp.uint8, cp.cuda.MemoryPointer(cp.cuda.Memory(size), 0))


def _root(a):
    while isinstance(a.base, np.ndarray):
        a = a.base
    return a


def _fill(pin, o, arr, a, b):
    pin[o:o + b - a] = arr[a:b]


def _land(ev, slot, ob, dst, m):
    ev.synchronize()
    ob[dst:dst + m] = slot[:m]


def _nvrtc_path():
    """cupy's Windows wheels ship nvrtc's builtins DLL beside
    nvidia/cuda_nvrtc, which nvrtc only finds on PATH."""
    if sys.platform != 'win32':
        return
    try:
        import nvidia
    except ImportError:
        return
    for root in nvidia.__path__:
        b = os.path.join(root, 'cuda_nvrtc', 'bin')
        if os.path.isdir(b) and b not in os.environ.get('PATH', ''):
            os.environ['PATH'] = b + os.pathsep + os.environ.get('PATH', '')
