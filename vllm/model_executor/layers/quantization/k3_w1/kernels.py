#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton kernels for the K3 1-bit ("k3-w1") expert store.

Format, matching c/backend_cuda_k3.cu:435-545 exactly:

    W[o, i] = a * 2^(scale[o, i//32] - 127) * (bit[o, i] ? +1 : -1)

One sign bit per weight, LSB-first within each byte, in input-index order; one
UE8M0 exponent per group of 32 inputs, stored in memory as a four-bit offset
from 109; and a single global amplitude `a` that lives outside the checkpoint.
`a` folds into the group scale, so it is applied once per row rather than per
weight -- the same trick the CUDA kernel uses.

Triton rather than CUDA on purpose: AttnRes already falls back to Triton on
sm_121 (csrc gates it at 10.0f), so the runtime dependency is one we are
carrying regardless, and this needs no vLLM rebuild to iterate on.
"""

import torch

from vllm.triton_utils import tl, triton

from .loader import SCALE_NIBBLE_BASE

# Group size is fixed by the checkpoint: config.json declares group_size 32 and
# scale_dtype uint8. The loader losslessly packs the observed 109..124 range.
GROUP = 32
UE8M0_BIAS = 127


@triton.jit
def _unpack_signs(
    packed_ptr, row_off, k_off, stride_o, BLOCK_O: tl.constexpr, BLOCK_K: tl.constexpr
):
    """[BLOCK_O, BLOCK_K//8] packed bytes -> [BLOCK_O, BLOCK_K] of +/-1.0.

    Bit j of byte b is weight index 8*b + j (LSB first), which is the order
    pack_rows_1bit writes and warp_row_dot_w1 reads back.
    """
    byte_k: tl.constexpr = BLOCK_K // 8
    b_cols = tl.arange(0, byte_k)
    ptrs = packed_ptr + row_off[:, None] * stride_o + (k_off // 8 + b_cols)[None, :]
    b = tl.load(ptrs).to(tl.uint8)
    # [BO, BK/8] -> [BO, BK/8, 8] -> [BO, BK]
    shifts = tl.arange(0, 8)
    bits = (b[:, :, None] >> shifts[None, None, :]) & 1
    bits = tl.reshape(bits, (BLOCK_O, BLOCK_K))
    # bit 1 -> +1, bit 0 -> -1.  The CUDA kernel spells this as an XOR of the
    # sign bit; here the arithmetic form lets Triton keep it in the FMA.
    return bits.to(tl.float32) * 2.0 - 1.0


@triton.jit
def _load_scales(
    scale_ptr, row_off, k_off, stride_o, BLOCK_O: tl.constexpr, BLOCK_K: tl.constexpr
):
    """Packed UE8M0 nibbles -> float multipliers, expanded x32."""
    # Literal 32/127 rather than the module globals: Triton folds a literal
    # into a constexpr but not a captured Python global, and tl.arange /
    # broadcast_to both require constexpr extents.
    ng: tl.constexpr = BLOCK_K // 32
    g_cols = k_off // 32 + tl.arange(0, ng)
    ptrs = scale_ptr + row_off[:, None] * stride_o + (g_cols // 2)[None, :]
    byte = tl.load(ptrs).to(tl.uint8)
    e = ((byte >> ((g_cols & 1) * 4)[None, :]) & 15).to(tl.int32)
    e += 109
    s = tl.exp2((e - 127).to(tl.float32))  # [BO, ng]
    s = tl.broadcast_to(s[:, :, None], (BLOCK_O, ng, 32))
    return tl.reshape(s, (BLOCK_O, BLOCK_K))


@triton.jit
def _k3_w1_gemm_kernel(
    a_ptr,
    c_ptr,
    packed_ptr,
    scale_ptr,
    sorted_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_pad_ptr,
    topk_weights_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_pe,
    stride_pn,
    stride_se,
    stride_sn,
    stride_cm,
    stride_cn,
    top_k: tl.constexpr,
    amplitude,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Grouped 1-bit expert GEMM: C[m, n] = sum_k A[m, k] * W[expert(m), n, k].

    Same block-sorted layout every vLLM fused-MoE kernel uses -- tokens are
    grouped by expert with moe_align_block_size, so each program owns one
    (token-block, output-block) tile of a single expert and the weight tile is
    read once for BLOCK_M tokens rather than once per token.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_tokens_post_pad = tl.load(num_tokens_post_pad_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_pad:
        return

    expert = tl.load(expert_ids_ptr + pid_m)
    if expert == -1:  # block belongs to another EP rank
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_ids_ptr + offs_m)
    valid = tok < num_valid_tokens
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + (tok // top_k)[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=valid[:, None] & (offs_k[None, :] < K),
            other=0.0,
        ).to(tl.float32)

        sign = _unpack_signs(
            packed_ptr + expert * stride_pe,
            offs_n,
            k0,
            stride_pn,
            BLOCK_O=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        scl = _load_scales(
            scale_ptr + expert * stride_se,
            offs_n,
            k0,
            stride_sn,
            BLOCK_O=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        w = sign * scl  # [BN, BK]
        acc += tl.dot(a, tl.trans(w))

    acc *= amplitude
    if MUL_ROUTED_WEIGHT:
        acc *= tl.load(topk_weights_ptr + tok, mask=valid, other=0.0)[:, None]

    tl.store(
        c_ptr + tok[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=valid[:, None] & (offs_n[None, :] < N),
    )


@triton.jit
def _k3_w1_gemv_kernel(
    a_ptr,
    c_ptr,
    packed_ptr,
    scale_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    expert_map_ptr,
    N,
    K,
    stride_am,
    stride_ak,
    stride_pe,
    stride_pn,
    stride_se,
    stride_sn,
    stride_cm,
    stride_cn,
    top_k: tl.constexpr,
    amplitude,
    HAS_EXPERT_MAP: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One (token, routed-slot) pair per program-x, BLOCK_N output rows per y.

    The grouped GEMM above pads each expert's token list up to BLOCK_M and
    feeds tl.dot; at B=1 every expert holds exactly one token, so 15 of every
    16 rows are padding and the tensor cores are being fed nothing. This path
    drops tl.dot and reduces over K directly -- the same shape as
    warp_row_dot_w1 in c/backend_cuda_k3.cu, which is what actually reaches
    DRAM bandwidth on this weight layout. It also needs no block-sorting pass,
    since each pair reads its expert straight out of topk_ids.
    """
    pid_p = tl.program_id(0)
    pid_n = tl.program_id(1)

    e = tl.load(topk_ids_ptr + pid_p)
    if HAS_EXPERT_MAP:
        e = tl.load(expert_map_ptr + e)
        if e < 0:  # expert lives on another EP rank
            return

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    token = pid_p // top_k

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        xv = tl.load(
            a_ptr + token * stride_am + offs_k * stride_ak, mask=offs_k < K, other=0.0
        ).to(tl.float32)
        sign = _unpack_signs(
            packed_ptr + e * stride_pe,
            offs_n,
            k0,
            stride_pn,
            BLOCK_O=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        scl = _load_scales(
            scale_ptr + e * stride_se,
            offs_n,
            k0,
            stride_sn,
            BLOCK_O=BLOCK_N,
            BLOCK_K=BLOCK_K,
        )
        acc += tl.sum(xv[None, :] * sign * scl, axis=1)

    acc *= amplitude
    if MUL_ROUTED_WEIGHT:
        acc *= tl.load(topk_weights_ptr + pid_p)

    tl.store(
        c_ptr + pid_p * stride_cm + offs_n * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=n_mask,
    )


@triton.jit
def _situ_and_mul_kernel(out_ptr, x_ptr, d, beta, linear_beta, BLOCK: tl.constexpr):
    """beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta).

    Matches vllm SituAndMul.forward_native and our k3_w1_gate_up_fast; kept
    local so the gate/up halves can be consumed straight out of the fused
    w13 output without a second full-width temporary.
    """
    row = tl.program_id(0)
    for off in range(0, d, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        m = cols < d
        g = tl.load(x_ptr + row * 2 * d + cols, mask=m, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + row * 2 * d + d + cols, mask=m, other=0.0).to(tl.float32)
        # tanh(z) == 2*sigmoid(2z) - 1. tl.math.tanh is not present in every
        # Triton build; the identity is exact and always is.
        g = beta * (2.0 * tl.sigmoid(2.0 * g / beta) - 1.0) * tl.sigmoid(g)
        if linear_beta > 0.0:
            u = linear_beta * (2.0 * tl.sigmoid(2.0 * u / linear_beta) - 1.0)
        tl.store(out_ptr + row * d + cols, (g * u).to(out_ptr.dtype.element_ty), mask=m)


def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: float | None):
    d = x.shape[-1] // 2
    flat = x.reshape(-1, x.shape[-1])
    out = torch.empty((flat.shape[0], d), dtype=x.dtype, device=x.device)
    _situ_and_mul_kernel[(flat.shape[0],)](
        out,
        flat,
        d,
        float(beta),
        -1.0 if linear_beta is None else float(linear_beta),
        BLOCK=1024,
        num_warps=4,
    )
    return out.reshape(x.shape[:-1] + (d,))


def _blocks(M, N, BLOCK_M, BLOCK_N):
    return (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)


def _pick_block_k(K: int, want: int) -> int:
    """Largest power-of-2 <= `want` that divides K, floored at GROUP.

    The K loop is unmasked -- the packed and scale loads read whole tiles --
    so BLOCK_K has to divide K exactly rather than rely on a bounds mask.
    K3's real dims (3584 latent, 3072 intermediate) both take the full 256;
    this only steps down for smaller shapes, e.g. tests.
    """
    bk = want
    while bk > GROUP and K % bk:
        bk //= 2
    if K % bk:
        raise ValueError(f"K={K} is not a multiple of {GROUP}")
    return bk


def w1_grouped_gemm(
    a,
    packed,
    scale,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad,
    topk_weights,
    out,
    top_k,
    amplitude,
    mul_routed_weight,
    block_m=16,
    block_n=32,
    block_k=128,
):
    """One grouped 1-bit expert GEMM. `a` is [T, K] or [T*top_k, K]."""
    assert packed.dtype == torch.uint8 and scale.dtype == torch.uint8
    N, K = packed.shape[1], packed.shape[2] * 8
    block_k = _pick_block_k(K, block_k)
    EM = sorted_ids.numel()
    _k3_w1_gemm_kernel[_blocks(EM, N, block_m, block_n)](
        a,
        out,
        packed,
        scale,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        N,
        K,
        EM,
        a.shape[0] * top_k,
        a.stride(0),
        a.stride(1),
        packed.stride(0),
        packed.stride(1),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        top_k=top_k,
        amplitude=amplitude,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=3,
    )


def w1_gemv(
    a,
    packed,
    scale,
    topk_ids,
    topk_weights,
    expert_map,
    out,
    top_k,
    amplitude,
    mul_routed_weight,
    block_n=32,
    block_k=256,
    num_warps=2,
    num_stages=3,
):
    """Decode-shaped 1-bit expert GEMM: one program per (token, slot) pair.

    `a` is [T, K] with top_k>1 (gate/up, indexed by token) or [T*top_k, K]
    with top_k==1 (down, indexed by pair) -- same convention as the grouped
    path. `out` must be pre-zeroed: programs whose expert is remote return
    without writing.
    """
    assert packed.dtype == torch.uint8 and scale.dtype == torch.uint8
    N, K = packed.shape[1], packed.shape[2] * 8
    block_k = _pick_block_k(K, block_k)
    pairs = topk_ids.numel()
    _k3_w1_gemv_kernel[(pairs, triton.cdiv(N, block_n))](
        a,
        out,
        packed,
        scale,
        topk_ids,
        topk_weights,
        expert_map,
        N,
        K,
        a.stride(0),
        a.stride(1),
        packed.stride(0),
        packed.stride(1),
        scale.stride(0),
        scale.stride(1),
        out.stride(0),
        out.stride(1),
        top_k=top_k,
        amplitude=amplitude,
        HAS_EXPERT_MAP=expert_map is not None,
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def dequant_reference(
    packed: torch.Tensor, scale: torch.Tensor, amplitude: float
) -> torch.Tensor:
    """Slow, obviously-correct dequant. The kernels are checked against this.

    packed [E, N, K/8], scale [E, N, K/64] -> [E, N, K] float32.
    """
    E, N, kb = packed.shape
    K = kb * 8
    bits = (
        packed.unsqueeze(-1) >> torch.arange(8, device=packed.device, dtype=torch.uint8)
    ) & 1
    sign = bits.reshape(E, N, K).float() * 2.0 - 1.0
    lo = scale & 15
    hi = scale >> 4
    exponent = torch.stack((lo, hi), dim=-1).reshape(*scale.shape[:-1], -1)
    exponent = exponent[..., : K // GROUP].int() + SCALE_NIBBLE_BASE
    s = torch.exp2((exponent - UE8M0_BIAS).float())
    s = s.repeat_interleave(GROUP, dim=-1)
    return sign * s * amplitude
