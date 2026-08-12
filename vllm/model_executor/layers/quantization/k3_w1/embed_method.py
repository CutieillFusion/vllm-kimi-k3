#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""int8 vocab embedding and lm_head.

These are the last bf16 tensors of any size. ParallelLMHead extends
VocabParallelEmbedding rather than LinearBase, so the quantization config is
never consulted for either -- both fall through to UnquantizedEmbeddingMethod
and stay bf16. Together that is 4.6 GB of checkpoint, TP-sharded to ~2.3 GB
per PP stage: embed on stage 0, lm_head on stage 1. At int8 per row it is
~1.15 GB, and it lands on both stages, which is what the KV shortfall needs
(-0.34 GiB on PP0_TP1, and PP1 has no other slack left).

int8 per row is the right granularity here for the same reason it is on the
MLA projections: a row is one vocabulary entry, quantized independently, so
nothing is shared across tokens and the error stays per-token rather than
compounding. Rows are also exactly what an embedding gather returns, so
dequantizing after the gather touches only the rows actually used.
"""

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.utils import set_weight_attrs

from .dense_kernels import k3_dense_matmul, quantize_i8


class KimiK3EmbeddingMethod(QuantizeMethodBase):
    """Backs both VocabParallelEmbedding and ParallelLMHead."""

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
        # bf16 first, so the stock loader's vocab sharding still applies;
        # quantized in the post-load hook like the dense path.
        weight = torch.nn.Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": 1, "output_dim": 0})
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(layer, "k3_qweight"):
            return  # idempotent, as dense
        w = layer.weight.data
        q, s = quantize_i8(w)
        layer.register_buffer("k3_qweight", q)
        layer.register_buffer("k3_scales", s)
        del layer.weight
        # Zero-element stand-in: tie_weights and various probes read
        # layer.weight without needing its contents.
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(0, dtype=w.dtype, device=w.device), requires_grad=False
            ),
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor):
        # Gather first, dequantize second: only the rows this batch touches
        # are ever materialized in bf16.
        rows = F.embedding(input_, layer.k3_qweight)
        scale = F.embedding(input_, layer.k3_scales.unsqueeze(1))
        return (rows.to(torch.float32) * scale).to(torch.bfloat16)

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        # lm_head's logits projection.
        return k3_dense_matmul(x, layer.k3_qweight, layer.k3_scales, 8, bias=bias)
