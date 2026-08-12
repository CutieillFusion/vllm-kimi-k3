#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton kernels for the K3 dense (non-expert) weight formats.

These are a *different* quantization from the 1-bit expert store, on all three
axes, so none of kernels.py carries over:

  int4-g64   signed int4, group size 64, one f32 scale per group, stored
             offset-by-8 with two values per byte -- c/kimi_k3.c:1316-1328:
                 s = max|w| over group / 7
                 v = clamp(round(w/s), -8, 7)
                 byte[j] = (v_2j + 8) | ((v_2j+1 + 8) << 4)
             4 + 32/64 = 4.5 bits/weight.

  int8-row   signed int8, one f32 scale per output row -- kimi_k3.c:1306-1314:
                 s = max|w| over row / 127
             8 + 32/I bits/weight.

(The expert store is MXFP4-derived instead: e2m1-sign codes, group 32, and a
UE8M0 power-of-two exponent byte rather than an f32 scale.)

Which format a tensor gets follows the engine's own split: K3_BITS=4 for KDA,
latent and shared-expert projections; K3_MLA_BITS=8 for the MLA projections;
K3_HEAD_BITS=8 for lm_head.
"""

import torch

from vllm.triton_utils import tl, triton

I4_GROUP = 64


@triton.jit
def _k3_i4g_gemv_kernel(
    a_ptr,
    c_ptr,
    w_ptr,
    s_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_sn,
    stride_cm,
    stride_cn,
    GS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y[m, n] = sum_k x[m, k] * (nib[n, k] - 8) * s[n, k // GS].

    Rather than interleaving the two nibbles back into k order -- which Triton
    has no cheap primitive for -- this de-interleaves the activation instead:
    the low nibble of byte j is element 2j and the high nibble is 2j+1, so one
    strided pair of x loads pairs with them directly. GS is even, so both
    elements of a byte always share a scale.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    half: tl.constexpr = BLOCK_K // 2
    jj = tl.arange(0, half)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        ke = k0 + 2 * jj  # even element indices
        b = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + (k0 // 2 + jj)[None, :],
            mask=n_mask[:, None],
            other=0,
        ).to(tl.uint8)
        lo = (b & 0xF).to(tl.float32) - 8.0
        hi = ((b >> 4) & 0xF).to(tl.float32) - 8.0
        xe = tl.load(
            a_ptr + pid_m * stride_am + ke * stride_ak, mask=ke < K, other=0.0
        ).to(tl.float32)
        xo = tl.load(
            a_ptr + pid_m * stride_am + (ke + 1) * stride_ak, mask=ke + 1 < K, other=0.0
        ).to(tl.float32)
        sc = tl.load(
            s_ptr + offs_n[:, None] * stride_sn + (ke // GS)[None, :],
            mask=n_mask[:, None],
            other=0.0,
        )
        acc += tl.sum((xe[None, :] * lo + xo[None, :] * hi) * sc, axis=1)

    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(
        c_ptr + pid_m * stride_cm + offs_n * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=n_mask,
    )


@triton.jit
def _k3_i8_gemv_kernel(
    a_ptr,
    c_ptr,
    w_ptr,
    s_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """y[m, n] = s[n] * sum_k x[m, k] * q[n, k]; scale is per row, so it comes
    out of the reduction entirely and lands once at the end."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        q = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0,
        ).to(tl.float32)
        x = tl.load(
            a_ptr + pid_m * stride_am + offs_k * stride_ak, mask=k_mask, other=0.0
        ).to(tl.float32)
        acc += tl.sum(x[None, :] * q, axis=1)

    acc *= tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
    tl.store(
        c_ptr + pid_m * stride_cm + offs_n * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=n_mask,
    )


@triton.jit
def _k3_dense_gemm_kernel(
    a_ptr,
    c_ptr,
    w_ptr,
    s_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_sn,
    stride_cm,
    stride_cn,
    BITS: tl.constexpr,
    GS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Prefill path: dequantize a [BLOCK_N, BLOCK_K] tile, then tl.dot.

    Worth having despite the GEMV being better at M=1, because prefill reads
    each weight once for BLOCK_M tokens while the GEMV rereads it per token.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        if BITS == 4:
            # Same de-interleave as the GEMV, but the activation tile has to be
            # gathered in the matching order so tl.dot lines up.
            half: tl.constexpr = BLOCK_K // 2
            jj = tl.arange(0, half)
            ke = k0 + 2 * jj
            b = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + (k0 // 2 + jj)[None, :],
                mask=n_mask[:, None],
                other=0,
            ).to(tl.uint8)
            lo = (b & 0xF).to(tl.float32) - 8.0
            hi = ((b >> 4) & 0xF).to(tl.float32) - 8.0
            sc = tl.load(
                s_ptr + offs_n[:, None] * stride_sn + (ke // GS)[None, :],
                mask=n_mask[:, None],
                other=0.0,
            )
            ae = tl.load(
                a_ptr + offs_m[:, None] * stride_am + ke[None, :] * stride_ak,
                mask=m_mask[:, None] & (ke < K)[None, :],
                other=0.0,
            )
            ao = tl.load(
                a_ptr + offs_m[:, None] * stride_am + (ke + 1)[None, :] * stride_ak,
                mask=m_mask[:, None] & (ke + 1 < K)[None, :],
                other=0.0,
            )
            acc += tl.dot(ae.to(tl.bfloat16), tl.trans((lo * sc).to(tl.bfloat16)))
            acc += tl.dot(ao.to(tl.bfloat16), tl.trans((hi * sc).to(tl.bfloat16)))
        else:
            q = tl.load(
                w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                mask=n_mask[:, None] & k_mask[None, :],
                other=0,
            ).to(tl.float32)
            acc += tl.dot(a.to(tl.bfloat16), tl.trans(q.to(tl.bfloat16)))

    if BITS == 8:
        acc *= tl.load(s_ptr + offs_n, mask=n_mask, other=0.0)[None, :]
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0)
        acc += bias[None, :].to(tl.float32)
    tl.store(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


# Below this many tokens the GEMV wins; see bench_dense.py. Same reasoning as
# the MoE path -- tl.dot needs 16 rows and decode only ever has one.
GEMV_MAX_TOKENS = 8


def k3_dense_matmul(
    x,
    qweight,
    scales,
    bits,
    bias=None,
    group_size=I4_GROUP,
    block_n=64,
    block_k=128,
    num_warps=4,
):
    """y = x @ dequant(qweight)^T (+ bias). x is [..., K], result [..., N]."""
    orig = x.shape
    a = x.reshape(-1, orig[-1])
    M, K = a.shape
    N = qweight.shape[0]
    out = torch.empty((M, N), dtype=x.dtype, device=x.device)

    if M <= GEMV_MAX_TOKENS:
        grid: tuple[int, ...] = (M, triton.cdiv(N, block_n))
        if bits == 4:
            _k3_i4g_gemv_kernel[grid](
                a,
                out,
                qweight,
                scales,
                bias if bias is not None else a,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                qweight.stride(0),
                scales.stride(0),
                out.stride(0),
                out.stride(1),
                GS=group_size,
                HAS_BIAS=bias is not None,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
            )
        else:
            _k3_i8_gemv_kernel[grid](
                a,
                out,
                qweight,
                scales,
                bias if bias is not None else a,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                qweight.stride(0),
                out.stride(0),
                out.stride(1),
                HAS_BIAS=bias is not None,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_warps=num_warps,
            )
    else:
        bm = 64
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, block_n),)
        _k3_dense_gemm_kernel[grid](
            a,
            out,
            qweight,
            scales,
            bias if bias is not None else a,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            qweight.stride(0),
            scales.stride(0) if bits == 4 else 0,
            out.stride(0),
            out.stride(1),
            BITS=bits,
            GS=group_size,
            HAS_BIAS=bias is not None,
            BLOCK_M=bm,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
        )
    return out.reshape(orig[:-1] + (N,))


# ---------------------------------------------------------------------------
# Quantizers. These reproduce c/kimi_k3.c exactly, including the clamp
# asymmetry (int4 clamps to [-8, 7] but divides by 7, so the -8 code is only
# reachable by rounding) and the 1e-20 scale floor.


def quantize_i4g(w: torch.Tensor, group_size: int = I4_GROUP):
    """[O, I] float -> ([O, I/2] uint8, [O, I/gs] f32). kimi_k3.c:1316-1328."""
    output_size, input_size = w.shape
    if input_size % group_size:
        raise ValueError(
            f"int4-g{group_size} needs input size divisible by group size, "
            f"got {input_size}"
        )
    g = w.float().reshape(output_size, input_size // group_size, group_size)
    s = g.abs().amax(dim=2) / 7.0
    s = s.clamp_min(1e-20)
    v = torch.round(g / s.unsqueeze(2)).clamp_(-8, 7).to(torch.int32) + 8
    v = v.reshape(output_size, input_size)
    packed = (v[:, 0::2] | (v[:, 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), s.contiguous()


def quantize_i8(w: torch.Tensor):
    """[O, I] float -> ([O, I] int8, [O] f32). kimi_k3.c:1306-1314."""
    output_size, input_size = w.shape
    q = torch.empty(output_size, input_size, dtype=torch.int8, device=w.device)
    s = torch.empty(output_size, dtype=torch.float32, device=w.device)
    step = max(1, min(output_size, (1 << 24) // max(1, input_size)))
    for r0 in range(0, output_size, step):
        r1 = min(output_size, r0 + step)
        f = w[r0:r1].float()
        sc = (f.abs().amax(dim=1) / 127.0).clamp_min(1e-20)
        q[r0:r1] = torch.round(f / sc.unsqueeze(1)).clamp_(-127, 127).to(torch.int8)
        s[r0:r1] = sc
        del f, sc
    return q, s


def dequant_reference(qweight, scales, bits, group_size=I4_GROUP):
    """Slow inverse, for tests."""
    if bits == 8:
        return qweight.float() * scales.unsqueeze(1)
    output_size, half = qweight.shape
    v = torch.empty(output_size, half * 2, dtype=torch.int32, device=qweight.device)
    v[:, 0::2] = (qweight & 0xF).int() - 8
    v[:, 1::2] = ((qweight >> 4) & 0xF).int() - 8
    return v.float() * scales.repeat_interleave(group_size, dim=1)
