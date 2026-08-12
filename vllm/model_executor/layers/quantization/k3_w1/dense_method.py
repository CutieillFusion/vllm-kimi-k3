#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM linear method for K3's dense (non-expert) weights.

Why this is needed at all: the 1-bit expert store fixes the 1446 GB of routed
experts, but the checkpoint's `ignore` list exempts self_attn, shared_experts,
the dense MLP and lm_head from quantization entirely, so those arrive as bf16
-- 114.4 GB across the model, roughly 38 GB per node at TP=4 once attention
and shared experts shard and the latent projections replicate. Against 121 GB
of unified memory with 95.7 GB of packed experts already resident, that does not
fit. Our own engine avoids it by quantizing at load time (kimi_k3.c:23), and
this is the same thing inside vLLM.

Quantization happens in process_weights_after_loading, not create_weights, so
the ordinary safetensors loader fills a bf16 parameter first and nothing about
weight loading has to change. Note the consequence: vLLM loads every weight
before running the post-load hooks, so peak memory is the full bf16 non-expert
set. That is fine on its own (~38 GB) but constrains load ordering against the
expert store.
"""

import regex as re
import torch

from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs

from .dense_kernels import I4_GROUP, k3_dense_matmul, quantize_i4g, quantize_i8

# The engine's split, kimi_k3.c:39-41 and :1543-1545.
#   K3_BITS=4       KDA / latent / shared / dense      (default 4)
#   K3_MLA_BITS=8   MLA projections                    (default 8)
#   K3_HEAD_BITS=8  lm_head                            (default 8)
# MLA and lm_head stay at 8 because they are the sensitive ones; the rest go
# to int4-g64. Matched here by module name -- MLA uses the low-rank q_a/q_b/
# kv_a/kv_b naming and KDA uses plain q/k/v/o, so the two are distinguishable
# without consulting the per-layer type map.
MLA_PROJ = re.compile(
    r"\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|kv_a_proj|kv_b_proj|"
    r"fused_qkv_a_proj)$"
)
HEAD_PROJ = re.compile(r"(^|\.)lm_head$")
# The MoE router gets int8, never int4: it picks 16 of 896 experts, and a
# coarse router changes which experts run rather than how well they run.
GATE_PROJ = re.compile(r"block_sparse_moe\.gate$")


def bits_for_prefix(prefix: str, dense_bits: int, mla_bits: int, head_bits: int) -> int:
    if HEAD_PROJ.search(prefix):
        return head_bits
    if GATE_PROJ.search(prefix):
        return 8
    if MLA_PROJ.search(prefix):
        return mla_bits
    return dense_bits


class KimiK3DenseLinearMethod(LinearMethodBase):
    """int4-g64 or int8-per-row, quantized from bf16 after loading."""

    def __init__(self, bits: int, group_size: int = I4_GROUP):
        if bits not in (4, 8):
            raise ValueError(f"k3 dense supports 4 or 8 bits, got {bits}")
        self.bits = bits
        self.group_size = group_size

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # A plain bf16 ModelWeightParameter, exactly as UnquantizedLinearMethod
        # would make: the checkpoint stores these unquantized, so the stock
        # loader (and every shard/merge rule that goes with it) still applies.
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)
        layer.k3_bits = self.bits
        layer.k3_group_size = self.group_size

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Idempotent: patch_loader runs this early, before the expert
        # parameters are materialized, and vLLM then calls it again for every
        # module. The second call must not try to requantize the zero-element
        # placeholder left behind by the first.
        if hasattr(layer, "k3_qweight"):
            return
        w = layer.weight.data
        bits = self.bits
        # int4-g64 needs I%64==0; the engine falls back to int8 rather than
        # padding (kimi_k3.c:1303), so do the same and stay bit-comparable.
        if bits == 4 and w.shape[1] % self.group_size:
            bits = 8
        if bits == 4:
            q, s = quantize_i4g(w, self.group_size)
        else:
            q, s = quantize_i8(w)
        layer.register_buffer("k3_qweight", q)
        layer.register_buffer("k3_scales", s)
        layer.k3_bits = bits
        # Drop the bf16 copy -- the entire point of the method -- but leave a
        # zero-element parameter behind rather than None. MLA reconstructs its
        # absorbed matrices from kv_b_proj via get_and_maybe_dequant_weights
        # (mla.py:362), whose generic path reads `weight.device` before
        # calling quant_method.apply(layer, eye) to recover the dequantized
        # weight. apply() handles that fine; only the `.device` probe needs a
        # real tensor to look at.
        del layer.weight
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False
            ),
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return k3_dense_matmul(
            x,
            layer.k3_qweight,
            layer.k3_scales,
            layer.k3_bits,
            bias=bias,
            group_size=layer.k3_group_size,
        )
