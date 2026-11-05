"""Tensor-core split-K decode attention for Turing (sm75) via NVRTC (torch.cuda._compile_kernel).

Why not Triton: Triton 3.x emits no tensor-core MMA on sm75 (tl.dot lowers to FMA), which makes the int8 and head_dim-64
paths compute-bound at ~170-290 GB/s. This kernel uses `mma.sync.m16n8k8` (fp16 in, fp32 accumulate, available on
sm75) and `movmatrix` (sm75 register transpose), so no shared memory is used at all.

Formulation (one warp per (row b, kv head h, split), 16 keys per step, G <= 8 query heads):
  S^T (16 keys x 8 q)   = K (16 keys x hd) . Q^T          -> hd/8 mma per step, no M padding
  O^T (hd x 8 q)       += V^T (hd x 8 keys) . P^T (x2)    -> hd/8 mma per step
K and V rows are loaded straight into mma A-fragments: lane (r = lane/4, c = lane%4) loads 16-byte pieces
(j*4 + c) of key rows r and r+8 (fully coalesced), which fixes a permutation of head dims that Q uses too (QK^T is
permutation invariant) and that is undone when O is stored. V^T fragments and P^T come from movmatrix.trans.
int8: pieces are converted with the 0x6400 magic-number trick; K scales multiply S columns, V scales multiply P.
Softmax state is per (lane, query) with the row sum reduced across lanes once at the end.
"""
from __future__ import annotations

import torch
from torch import Tensor

_SRC = r"""
typedef unsigned int u32;
typedef unsigned short u16;
typedef long long i64;

#define CH_PIECE (16 / ELT)        // elements per 16-byte piece
#define WP (CH_PIECE / 2)          // half2 words per piece after conversion
#define NV (HD * ELT / 64)         // 16-byte pieces per row per lane
#define NS (HD / 8)                // half2 words per row per lane (= mma k-steps)
#define NM (HD / 16)               // O^T m-tiles

__device__ __forceinline__ void mma(float* d, u32 a0, u32 a1, u32 b0) {
  asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ u32 trans(u32 x) {
  u32 y;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
  return y;
}
__device__ __forceinline__ u32 pack_h2(float lo, float hi) {
  u16 a, b;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(a) : "f"(lo));
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(b) : "f"(hi));
  return (u32)a | ((u32)b << 16);
}
__device__ __forceinline__ u16 f2h(float x) {
  u16 a;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(a) : "f"(x));
  return a;
}
__device__ __forceinline__ float h2f(u16 x) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(x));
  return f;
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
// 4 signed int8 (one word) -> 2 half2 words, exact: (x ^ 0x80) in the mantissa of 1024.0, minus 1152.
__device__ __forceinline__ void i8x4_to_h2x2(u32 w, u32& lo, u32& hi) {
  const u32 m = 0x64806480u;  // half2(1152, 1152)
  u32 x = w ^ 0x80808080u;
  u32 a = __byte_perm(x, 0x64646464u, 0x4140);
  u32 b = __byte_perm(x, 0x64646464u, 0x4342);
  asm("sub.f16x2 %0, %1, %2;" : "=r"(lo) : "r"(a), "r"(m));
  asm("sub.f16x2 %0, %1, %2;" : "=r"(hi) : "r"(b), "r"(m));
}

struct Raw { uint4 p[NV]; };

__device__ __forceinline__ void load_raw(Raw& R, const char* row, int c) {
#pragma unroll
  for (int j = 0; j < NV; ++j) R.p[j] = *reinterpret_cast<const uint4*>(row + (j * 4 + c) * 16);
}
__device__ __forceinline__ void to_words(const Raw& R, u32* W) {
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    u32 v[4] = {R.p[j].x, R.p[j].y, R.p[j].z, R.p[j].w};
#if INT8
#pragma unroll
    for (int i = 0; i < 4; ++i) i8x4_to_h2x2(v[i], W[j * 8 + 2 * i], W[j * 8 + 2 * i + 1]);
#else
#pragma unroll
    for (int i = 0; i < 4; ++i) W[j * 4 + i] = v[i];
#endif
  }
}
// head dim held in half2 word s (low half) by lane-in-quad c
__device__ __forceinline__ int dim_of(int c, int s) { return ((s / WP) * 4 + c) * CH_PIECE + 2 * (s % WP); }

extern "C" __global__ void __launch_bounds__(128) KNAME(
    const u16* __restrict__ Q, const char* __restrict__ Kc, const char* __restrict__ Vc,
    const u16* __restrict__ Ks, const u16* __restrict__ Vs, const i64* __restrict__ Pos,
    const i64* __restrict__ Start, u16* __restrict__ Out, float* __restrict__ Ob, float* __restrict__ Mb,
    float* __restrict__ Lb, int n_items, int KH, int S, int window, double scale_log2_d,
    int sqb, int sqh, int skb, int skh, int sks, int svb, int svh, int svs, int ssb, int ssh, int stb, int sth,
    int sob, int soh) {
  const int lane = threadIdx.x & 31;
  const int item = blockIdx.x * 4 + (threadIdx.x >> 5);
  if (item >= n_items) return;
  const int r = lane >> 2, c = lane & 3;
  const int bk = item / SPLITS, sid = item % SPLITS;
  const int b = bk / KH, h = bk % KH;
  const float scale_log2 = (float)scale_log2_d;
  const i64 p = Pos[b];
  i64 lo = Start[b];
  if (lo < 0) lo = 0;
  if (window > 0 && p - window + 1 > lo) lo = p - window + 1;
  const i64 n = p + 1 - lo;
  const i64 per = (n + SPLITS - 1) / SPLITS;
  const i64 chunk = (per + 15) / 16 * 16;
  const i64 s0 = lo + (i64)sid * chunk;
  const i64 s1 = (s0 + chunk < p + 1) ? s0 + chunk : p + 1;

  // Q^T B-fragments: query r (zero if r >= G), dims dim_of(c, s)
  u32 qf[NS];
  {
    const u16* qrow = Q + (i64)b * sqb + (i64)(h * G + (r < G ? r : 0)) * sqh;
#pragma unroll
    for (int s = 0; s < NS; ++s)
      qf[s] = (r < G) ? *reinterpret_cast<const u32*>(qrow + dim_of(c, s)) : 0u;
  }
  float acc[NM][4];
#pragma unroll
  for (int i = 0; i < NM; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
  float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
  const char* kbase = Kc + ((i64)b * skb + (i64)h * skh) * ELT;
  const char* vbase = Vc + ((i64)b * svb + (i64)h * svh) * ELT;
#if INT8
  const u16* ksb = Ks + (i64)b * ssb + (i64)h * ssh;
  const u16* vsb = Vs + (i64)b * stb + (i64)h * sth;
#endif

  for (i64 t0 = s0; t0 < s1; t0 += 16) {
    const i64 ta = t0 + r, tb = t0 + r + 8;
    const bool va = ta < s1, vb = tb < s1;
    i64 sa = va ? ta : s0, sb = vb ? tb : s0;
#if RING
    sa %= S; sb %= S;
#endif
    Raw KA, KB, VA, VB;
    load_raw(KA, kbase + sa * sks * ELT, c);
    load_raw(KB, kbase + sb * sks * ELT, c);
    load_raw(VA, vbase + sa * svs * ELT, c);
    load_raw(VB, vbase + sb * svs * ELT, c);
#if INT8
    const float ksa = h2f(ksb[sa]), kscb = h2f(ksb[sb]);
    const float vsa = h2f(vsb[sa]), vscb = h2f(vsb[sb]);
#endif
    float sc[4] = {0.f, 0.f, 0.f, 0.f};
    {
      u32 ka[NS], kb[NS];
      to_words(KA, ka);
      to_words(KB, kb);
#pragma unroll
      for (int s = 0; s < NS; ++s) mma(sc, ka[s], kb[s], qf[s]);
    }
#if INT8
    sc[0] *= ksa; sc[1] *= ksa; sc[2] *= kscb; sc[3] *= kscb;
#endif
    sc[0] = va ? sc[0] * scale_log2 : -INFINITY;
    sc[1] = va ? sc[1] * scale_log2 : -INFINITY;
    sc[2] = vb ? sc[2] * scale_log2 : -INFINITY;
    sc[3] = vb ? sc[3] * scale_log2 : -INFINITY;
    float mx0 = fmaxf(sc[0], sc[2]), mx1 = fmaxf(sc[1], sc[3]);
#pragma unroll
    for (int o = 4; o < 32; o <<= 1) {
      mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, o));
      mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, o));
    }
    const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
    const float al0 = ex2(m0 - mn0), al1 = ex2(m1 - mn1);
    float p00 = ex2(sc[0] - mn0), p01 = ex2(sc[1] - mn1), p10 = ex2(sc[2] - mn0), p11 = ex2(sc[3] - mn1);
    l0 = l0 * al0 + p00 + p10;
    l1 = l1 * al1 + p01 + p11;
    m0 = mn0; m1 = mn1;
#if INT8
    p00 *= vsa; p01 *= vsa; p10 *= vscb; p11 *= vscb;
#endif
#pragma unroll
    for (int i = 0; i < NM; ++i) { acc[i][0] *= al0; acc[i][1] *= al1; acc[i][2] *= al0; acc[i][3] *= al1; }
    const u32 pa = trans(pack_h2(p00, p01)), pb = trans(pack_h2(p10, p11));
    {
      u32 w[NS];
      to_words(VA, w);
#pragma unroll
      for (int i = 0; i < NM; ++i) mma(acc[i], trans(w[2 * i]), trans(w[2 * i + 1]), pa);
      to_words(VB, w);
#pragma unroll
      for (int i = 0; i < NM; ++i) mma(acc[i], trans(w[2 * i]), trans(w[2 * i + 1]), pb);
    }
  }
#pragma unroll
  for (int o = 4; o < 32; o <<= 1) {
    l0 += __shfl_xor_sync(0xffffffffu, l0, o);
    l1 += __shfl_xor_sync(0xffffffffu, l1, o);
  }
  // acc[i][e]: query 2c + (e & 1), head dim dim_of(r >> 1, 2i + (e >> 1)) + (r & 1)
  const int q0 = 2 * c, q1 = 2 * c + 1;
#if SPLITS == 1
  const float i0 = 1.f / l0, i1 = 1.f / l1;
  u16* o0 = Out + (i64)b * sob + (i64)(h * G + q0) * soh;
  u16* o1 = Out + (i64)b * sob + (i64)(h * G + q1) * soh;
#pragma unroll
  for (int i = 0; i < NM; ++i) {
    const int d0 = dim_of(r >> 1, 2 * i) + (r & 1), d1 = dim_of(r >> 1, 2 * i + 1) + (r & 1);
    if (q0 < G) { o0[d0] = f2h(acc[i][0] * i0); o0[d1] = f2h(acc[i][2] * i0); }
    if (q1 < G) { o1[d0] = f2h(acc[i][1] * i1); o1[d1] = f2h(acc[i][3] * i1); }
  }
#else
  float* ob0 = Ob + ((i64)item * G + q0) * HD;
  float* ob1 = Ob + ((i64)item * G + q1) * HD;
#pragma unroll
  for (int i = 0; i < NM; ++i) {
    const int d0 = dim_of(r >> 1, 2 * i) + (r & 1), d1 = dim_of(r >> 1, 2 * i + 1) + (r & 1);
    if (q0 < G) { ob0[d0] = acc[i][0]; ob0[d1] = acc[i][2]; }
    if (q1 < G) { ob1[d0] = acc[i][1]; ob1[d1] = acc[i][3]; }
  }
  if (r == 0) {
    if (q0 < G) { Mb[(i64)item * G + q0] = m0; Lb[(i64)item * G + q0] = l0; }
    if (q1 < G) { Mb[(i64)item * G + q1] = m1; Lb[(i64)item * G + q1] = l1; }
  }
#endif
}
"""

_CACHE: dict = {}


def _kernel(hd: int, G: int, int8: bool, ring: bool, splits: int, device_index: int):
    key = (hd, G, int8, ring, splits, device_index)
    k = _CACHE.get(key)
    if k is None:
        name = f"mathk_decode_hd{hd}_g{G}_{'i8' if int8 else 'f16'}_{'ring' if ring else 'lin'}_s{splits}"
        defs = (f"#define HD {hd}\n#define G {G}\n#define INT8 {int(int8)}\n#define ELT {1 if int8 else 2}\n"
                f"#define RING {int(ring)}\n#define SPLITS {splits}\n#define KNAME {name}\n"
                "#define INFINITY __int_as_float(0x7f800000)\n")
        with torch.cuda.device(device_index):
            k = _compile(defs + _SRC, name)
        _CACHE[key] = k
    return k


def _compile(src: str, name: str):
    """torch.cuda._compile_kernel. Older torch (e.g. 2.9) prepends `extern "C"` to the
    source unless it already starts with it, which breaks our leading #defines / typedefs ("identifier u32 is
    undefined"); those versions take a `header_code` argument, so the preamble goes there and the source starts at
    the kernel's own `extern "C"`. Newer torch (e.g. 2.13) has no header_code and
    compiles the source verbatim: unchanged call.
    torch 2.9's launcher also passes Python floats as C float (2.13: double), so the `double scale_log2_d`
    parameter would read garbage; it becomes `float` there (the kernel casts it to float anyway: same value)."""
    import inspect  # noqa: PLC0415
    from torch.cuda import _utils  # noqa: PLC0415
    if "ctypes.c_float(arg)" in inspect.getsource(_utils):
        src = src.replace("double scale_log2_d", "float scale_log2_d")
    if "header_code" in inspect.signature(torch.cuda._compile_kernel).parameters:
        i = src.index('extern "C"')
        return torch.cuda._compile_kernel(src[i:], name, header_code=src[:i], nvcc_options=["-std=c++17"])
    return torch.cuda._compile_kernel(src, name, nvcc_options=["-std=c++17"])


def supported(q: Tensor, k_cache: Tensor) -> bool:
    if not q.is_cuda or torch.cuda.get_device_capability(q.device) < (7, 5):
        return False
    H, hd = q.shape[-2], q.shape[-1]
    G = H // k_cache.shape[1]
    return hd in (64, 128) and 1 <= G <= 8 and q.dtype == torch.float16


def launch(q, k_cache, v_cache, pos, starts, scale, window, ring, k_scale, v_scale, num_splits, out):
    """Same contract as decode_attn._launch (q (B, H, hd) fp16; caches (B, K, S, hd); out (B, H, hd) fp16)."""
    from .decode_attn import LOG2E, _combine_kernel  # noqa: PLC0415
    import triton  # noqa: PLC0415

    B, H, hd = q.shape
    KH, S = k_cache.shape[1], k_cache.shape[2]
    G = H // KH
    int8 = k_cache.dtype == torch.int8
    elt = 1 if int8 else 2
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
    assert (k_cache.stride(2) * elt) % 16 == 0 and (v_cache.stride(2) * elt) % 16 == 0
    assert pos.dtype == torch.int64 and starts.dtype == torch.int64
    if int8:
        ks, vs = k_scale, v_scale
        ssb, ssh, stb, sth = ks.stride(0), ks.stride(1), vs.stride(0), vs.stride(1)
        assert ks.stride(2) == 1 and vs.stride(2) == 1
    else:
        ks, vs = q, q  # unused
        ssb = ssh = stb = sth = 0
    n_items = B * KH * num_splits
    if num_splits > 1:
        ob = torch.empty(n_items * G, hd, device=q.device, dtype=torch.float32)
        mb = torch.empty(n_items * G, device=q.device, dtype=torch.float32)
        lb = torch.empty_like(mb)
    else:
        ob = mb = lb = out
    kern = _kernel(hd, G, int8, bool(ring), int(num_splits), q.device.index or 0)
    kern(grid=((n_items + 3) // 4, 1, 1), block=(128, 1, 1),
         args=[q, k_cache, v_cache, ks, vs, pos, starts, out, ob, mb, lb, n_items, KH, S, int(window),
               float(scale * LOG2E), q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1),
               k_cache.stride(2), v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), ssb, ssh, stb, sth,
               out.stride(0), out.stride(1)])
    if num_splits > 1:
        _combine_kernel[(B * KH * G,)](out, ob, mb, lb, KH, out.stride(0), out.stride(1), G=G, HD=hd,
                                       SPLITS=num_splits, SP2=triton.next_power_of_2(num_splits), num_warps=4)
    return out
