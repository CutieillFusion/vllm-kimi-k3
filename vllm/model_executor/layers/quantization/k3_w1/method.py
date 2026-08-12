#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""vLLM quantization method for the Kimi-K3 1-bit ("k3-w1") expert store.

Why this exists: K3's checkpoint ships MXFP4 experts at 17,547,264 B each,
which is 362 GB/node for our 224-expert shard against 121 GB of unified
memory. The raw 1-bit store is 5,160,960 B/expert; lossless nibble-packing of
its 109..124 scale exponents makes the resident form 4,644,864 B/expert, or
95.7 GB per rank. Everything else about the model already runs on sm_121.

Format is exactly what c/backend_cuda_k3.cu runs -- see kernels.py. The one
piece that is not in the checkpoint is the global amplitude `a`, which our
engine reads from K3_W1_A (default 1.69); it is carried here as a config
field so a store packed with a different `a` stays self-describing.
"""

import os
import time
from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.activation import (
    ApplyMoEActivationConfig,
    MoEActivation,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.utils import set_weight_attrs

from . import install_k3_w1 as _install_k3_w1
from .dense_method import KimiK3DenseLinearMethod, bits_for_prefix
from .embed_method import KimiK3EmbeddingMethod
from .kernels import GROUP, situ_and_mul, w1_gemv, w1_grouped_gemm
from .loader import pack_slot_scales
from .store import resolve_expert_store


def _defer_experts() -> bool:
    """Whether to allocate expert parameters empty and fill them later.

    On by default; K3_W1_EAGER_EXPERTS=1 restores the ordinary behaviour for
    a single-layer test where the peak does not matter.
    """
    import os

    return os.environ.get("K3_W1_EAGER_EXPERTS", "0") != "1"


DEFAULT_AMPLITUDE = 1.69
BLOCK_M = 16

# Below this many tokens the grouped GEMM is padding a handful of real rows up
# to BLOCK_M to feed tl.dot, and the GEMV path -- which needs no block-sorting
# pass at all -- wins. Measured per full MoE layer on one EP rank of four (224
# resident experts, 16-of-896 routing, H=3584 I=3072), bench_crossover.py:
#
#     T      1      2      4      8     16     64
#     gemv   0.371  0.617  0.891  1.480 3.019  12.365 ms
#     group  0.461  0.738  1.083  1.910 3.560   9.711 ms
#
# so the crossover sits between 16 and 64. Note this depends on the routing
# being spread: with few distinct local experts, grouped can serve several
# selections from one slot read and wins much earlier.
GEMV_MAX_TOKENS = 32


class KimiK3OneBitConfig(QuantizationConfig):
    """Two unrelated quantizations under one config, because K3 needs both.

    Routed experts: 1-bit sign-only with UE8M0 group-32 scales inherited from
    the checkpoint (kernels.py).

    Everything else: int4-g64 with f32 scales, or int8 per row for the MLA
    projections and lm_head (dense_kernels.py). The checkpoint's `ignore` list
    exempts these from its own MXFP4, so they arrive bf16 -- 114.4 GB across
    the model, which does not fit next to 95.7 GB of packed experts on a 121 GB
    node. They are quantized after loading rather than left alone.
    """

    def __init__(
        self,
        amplitude: float = DEFAULT_AMPLITUDE,
        group_size: int = GROUP,
        dense_bits: int = 4,
        mla_bits: int = 8,
        head_bits: int = 8,
        expert_store: str = "k3_w1",
        expert_tensor_parallel_size: int = 2,
        expert_pipeline_parallel_size: int = 2,
    ) -> None:
        super().__init__()
        if group_size != GROUP:
            raise ValueError(
                f"k3_w1 scales are inherited byte-identically from the MXFP4 "
                f"checkpoint, which is group-32; got {group_size}"
            )
        self.amplitude = float(amplitude)
        self.group_size = group_size
        # Dense weights are a different quantization entirely -- signed int4 on
        # group-64 with f32 scales, or int8 per row. Defaults mirror the
        # engine's K3_BITS / K3_MLA_BITS / K3_HEAD_BITS. 16 disables it, which
        # is only viable if the non-expert set fits in bf16 (it does not, at
        # 114.4 GB across the model).
        self.dense_bits = int(dense_bits)
        self.mla_bits = int(mla_bits)
        self.head_bits = int(head_bits)
        self.expert_store = expert_store
        self.expert_tensor_parallel_size = int(expert_tensor_parallel_size)
        self.expert_pipeline_parallel_size = int(expert_pipeline_parallel_size)
        self.model_dir: str | None = None

    def __repr__(self) -> str:
        return (
            f"KimiK3OneBitConfig(amplitude={self.amplitude}, "
            f"dense_bits={self.dense_bits}, mla_bits={self.mla_bits}, "
            f"head_bits={self.head_bits}, expert_store={self.expert_store!r})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "k3_w1"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        # Triton-only; no tensor-core or tcgen05 dependency. GB10 is 121.
        return 80

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "KimiK3OneBitConfig":
        # --quantization k3_w1 hands us an empty dict, so the env is the only
        # way to tune this from a launch script. Names mirror the engine's.
        import os

        def _e(k, d):
            return int(os.environ.get(k, config.get(k.lower().replace("k3_", ""), d)))

        return cls(
            amplitude=float(
                os.environ.get("K3_W1_A", config.get("amplitude", DEFAULT_AMPLITUDE))
            ),
            group_size=config.get("group_size", GROUP),
            dense_bits=_e("K3_BITS", 4),
            mla_bits=_e("K3_MLA_BITS", 8),
            head_bits=_e("K3_HEAD_BITS", 8),
            expert_store=config.get("expert_store", "k3_w1"),
            expert_tensor_parallel_size=config.get("expert_tensor_parallel_size", 2),
            expert_pipeline_parallel_size=config.get(
                "expert_pipeline_parallel_size", 2
            ),
        )

    def maybe_update_config(
        self,
        model_name: str,
        hf_config=None,
        revision: str | None = None,
    ) -> None:
        """Resolve the model snapshot that contains the expert stores."""
        if os.environ.get("K3_W1_DIR"):
            self.model_dir = (
                os.path.abspath(model_name) if os.path.isdir(model_name) else None
            )
            return
        if os.path.isdir(model_name):
            self.model_dir = os.path.abspath(model_name)
        else:
            import huggingface_hub
            import huggingface_hub.constants

            self.model_dir = huggingface_hub.snapshot_download(
                model_name,
                revision=revision,
                allow_patterns=[f"{self.expert_store}/**"],
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        root = os.path.join(self.model_dir, self.expert_store)
        if not os.path.isdir(root):
            raise ValueError(
                f"k3_w1 expert store not found at {root}. The converted model "
                "must include k3_w1/pp<pp>-tp<tp>/experts.w2."
            )

    def get_expert_store_dir(self) -> tuple[str, bool]:
        """Return this worker's expert directory and whether it is PP-local."""
        override = os.environ.get("K3_W1_DIR")
        if override:
            return override, os.environ.get("K3_W1_STAGE_LOCAL", "0") == "1"
        if self.model_dir is None:
            raise RuntimeError("k3_w1 model snapshot was not resolved")

        from vllm.distributed import get_ep_group, get_pp_group

        ep = get_ep_group()
        pp = get_pp_group()
        return resolve_expert_store(
            self.model_dir,
            self.expert_store,
            pp_rank=pp.rank_in_group,
            pp_size=pp.world_size,
            ep_rank=ep.rank_in_group,
            ep_size=ep.world_size,
            expected_pp_size=self.expert_pipeline_parallel_size,
            expected_ep_size=self.expert_tensor_parallel_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, RoutedExperts):
            return KimiK3OneBitMoEMethod(self, layer.moe_config)
        # ParallelLMHead extends VocabParallelEmbedding, not LinearBase, so
        # this has to come first -- otherwise both fall through to
        # UnquantizedEmbeddingMethod and stay bf16 (~2.3 GB per PP stage).
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            VocabParallelEmbedding,
        )

        if isinstance(layer, VocabParallelEmbedding):
            if os.environ.get("K3_QUANT_EMBED", "1") == "1":
                return KimiK3EmbeddingMethod()
            return None
        if isinstance(layer, LinearBase):
            # The vision tower is small (0.9 GB) and not what this is for;
            # leaving it in bf16 keeps it out of the numerics story.
            if "vision" in prefix or "mm_projector" in prefix:
                return UnquantizedLinearMethod()
            bits = bits_for_prefix(
                prefix, self.dense_bits, self.mla_bits, self.head_bits
            )
            if bits >= 16:
                return UnquantizedLinearMethod()
            # Group size is tunable because it is the broadest weight lever
            # left: g128 is 4.25 bits/weight against g64's 4.5, ~0.5 GB per
            # rank, and it applies to every int4 tensor at once.
            import os as _o

            return KimiK3DenseLinearMethod(bits, int(_o.environ.get("K3_GROUP", "64")))
        return None


def _local_stream_misses(layer, topk_ids):
    n_res = getattr(layer, "k3_resident", None)
    emap = getattr(layer, "expert_map", None)
    if n_res is None or emap is None or n_res >= layer.k3_num_experts:
        return torch.empty(0, dtype=torch.long, device=topk_ids.device)
    loc = emap[topk_ids.reshape(-1).long()]
    return torch.unique(loc[loc >= n_res])


def _load_stream_slots(layer, local_slots, active_only=False):
    """Load local expert slots and build the map for one execution pass."""
    n_res = layer.k3_resident
    n_str = layer.k3_stream_slots
    if len(local_slots) > n_str:
        raise RuntimeError(
            f"k3_w1 needs {len(local_slots)} streamed experts for one MoE "
            f"pass, but only {n_str} scratch slots are configured"
        )
    store, ordinal = layer.k3_store, layer.k3_moe_ordinal
    base_map = layer.expert_map
    emap = torch.full_like(base_map, -1) if active_only else base_map.clone()
    inter = layer.w13_qweight.shape[1] // 2
    started = time.perf_counter()
    for j, local_slot in enumerate(local_slots):
        dst = n_res + j
        s = pack_slot_scales(store.read_slot(ordinal, local_slot))
        layer.w13_qweight[dst][:inter].copy_(torch.from_numpy(s["w1p"]))
        layer.w13_qweight[dst][inter:].copy_(torch.from_numpy(s["w3p"]))
        layer.w13_scales[dst][:inter].copy_(torch.from_numpy(s["w1s"]))
        layer.w13_scales[dst][inter:].copy_(torch.from_numpy(s["w3s"]))
        layer.w2_qweight[dst].copy_(torch.from_numpy(s["w2p"]))
        layer.w2_scales[dst].copy_(torch.from_numpy(s["w2s"]))
        emap[base_map == local_slot] = dst
    elapsed = time.perf_counter() - started
    previous = getattr(layer, "k3_stream_loads", 0)
    layer.k3_stream_loads = previous + len(local_slots)
    layer.k3_stream_seconds = getattr(layer, "k3_stream_seconds", 0.0) + elapsed
    if previous == 0 or previous // 256 != layer.k3_stream_loads // 256:
        slot_bytes = sum(
            tensor[0].numel()
            for tensor in (
                layer.w13_qweight,
                layer.w13_scales,
                layer.w2_qweight,
                layer.w2_scales,
            )
        )
        from .patch_loader import trace

        trace(
            f"[stream] layer={ordinal} loads={layer.k3_stream_loads} "
            f"bytes={layer.k3_stream_loads * slot_bytes} "
            f"seconds={layer.k3_stream_seconds:.3f}"
        )
    return emap


def _stream_missing(layer, topk_ids):
    """Load all misses for a single-pass routed-expert execution."""
    missed = _local_stream_misses(layer, topk_ids)
    if missed.numel() == 0:
        return getattr(layer, "expert_map", None)
    return _load_stream_slots(layer, missed.tolist())


class KimiK3OneBitMoEMethod(FusedMoEMethodBase):
    """Sign-bit expert GEMMs, dequantized inside a Triton grouped GEMM.

    Deliberately not a modular kernel: `moe_kernel` stays None, so
    `is_monolithic` is False and RoutedExperts calls `apply` directly. The
    modular-kernel path exists to compose dispatch/combine backends for
    large-batch EP, none of which applies at B=1 on four nodes.
    """

    def __init__(self, quant_config: KimiK3OneBitConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.token_chunk = max(0, int(os.environ.get("K3_MOE_TOKEN_CHUNK", "0")))
        self.activation_config = ApplyMoEActivationConfig(
            activation_situ_beta=moe.activation_situ_beta,
            activation_situ_linear_beta=moe.activation_situ_linear_beta,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        if hidden_size % GROUP or intermediate_size_per_partition % GROUP:
            raise ValueError(
                f"k3_w1 needs both dims divisible by {GROUP}: "
                f"hidden={hidden_size} inter={intermediate_size_per_partition}"
            )
        w13_rows = self.moe.w13_num_shards * intermediate_size_per_partition

        # Allocated EMPTY and materialized later, by patch_loader, once the
        # dense weights have been quantized. vLLM allocates every parameter
        # before loading anything, so holding the real 95.7 GB of experts
        # here would sit alongside 38 GB of not-yet-quantized bf16 dense --
        # 144 GB against ~122 available. Deferring costs nothing: the store is
        # read after load_weights either way.
        # Tail streaming. K3_RESIDENT_EXPERTS caps how many of this rank's
        # experts stay in memory; the rest are read from experts.w2 on demand
        # into K3_STREAM_SLOTS scratch slots appended after the resident ones.
        # Below full residency, the tail is read on demand into stream slots.
        # Full residency fits because scales are packed losslessly in memory.
        import os as _os

        n_res = int(_os.environ.get("K3_RESIDENT_EXPERTS", "0")) or num_experts
        n_res = max(1, min(num_experts, n_res))
        n_str = (
            0
            if n_res >= num_experts
            else max(1, int(_os.environ.get("K3_STREAM_SLOTS", "16")))
        )
        layer.k3_resident = n_res
        layer.k3_stream_slots = n_str
        layer.k3_num_experts = num_experts
        slots = n_res + n_str
        layer.k3_expert_shapes = {
            "w13_qweight": (slots, w13_rows, hidden_size // 8),
            "w2_qweight": (slots, hidden_size, intermediate_size_per_partition // 8),
            "w13_scales": (slots, w13_rows, hidden_size // (2 * GROUP)),
            "w2_scales": (
                slots,
                hidden_size,
                intermediate_size_per_partition // (2 * GROUP),
            ),
        }
        defer = _defer_experts()

        def _mk(shape):
            return torch.nn.Parameter(
                torch.empty(0 if defer else shape, dtype=torch.uint8),
                requires_grad=False,
            )

        # Packed sign bits: 8 weights per byte, input-index order, LSB first.
        w13_qweight = _mk(layer.k3_expert_shapes["w13_qweight"])
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = _mk(layer.k3_expert_shapes["w2_qweight"])
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        # The stores use one UE8M0 byte per 32 inputs, but every real exponent
        # is in 109..124. Keep two lossless four-bit offsets per byte; the
        # kernel restores the base before exp2.
        w13_scales = _mk(layer.k3_expert_shapes["w13_scales"])
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = _mk(layer.k3_expert_shapes["w2_scales"])
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        layer.k3_amplitude = self.quant_config.amplitude
        from .patch_loader import trace

        trace(
            f"[create_weights] moe experts={num_experts} "
            f"deferred={defer} hidden={hidden_size}"
        )

    def get_fused_moe_quant_config(self, layer: RoutedExperts):
        # No modular kernel, so no FusedMoEQuantConfig to hand one.
        return None

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # Nothing to repack: the store's layout is already the kernel's layout.
        # That is the point of packing offline rather than at load time.
        return

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts=None,
        shared_experts_input=None,
    ) -> torch.Tensor:
        # Routed output only, never a (shared, routed) tuple. MoERunner
        # ._apply_quant_method already runs the shared experts around this call
        # -- SharedExpertsOrder.NO_OVERLAP before, MULTI_STREAM_OVERLAPPED
        # after -- and reads _shared_experts.output itself
        # (runner/moe_runner.py:587-623). The shared_experts argument is here
        # only so a modular kernel can overlap them internally; running them
        # here as well would double-apply them.
        return self._forward_routed(layer, x, topk_weights, topk_ids)

    def _forward_routed(self, layer, x, topk_weights, topk_ids) -> torch.Tensor:
        token_chunk = self.token_chunk
        fully_resident = layer.k3_resident >= layer.k3_num_experts
        if fully_resident and token_chunk and x.shape[0] > token_chunk:
            # Bound the T*top_k MoE temporaries without shrinking vLLM's
            # attention prefill chunk. Restrict this to full residency: token-
            # outer chunking with a streamed tail would reread that tail once
            # per chunk instead of once per layer invocation.
            output = torch.empty_like(x)
            for start in range(0, x.shape[0], token_chunk):
                end = min(start + token_chunk, x.shape[0])
                output[start:end].copy_(
                    self._forward_routed(
                        layer,
                        x[start:end],
                        topk_weights[start:end],
                        topk_ids[start:end],
                    )
                )
            return output
        if fully_resident:
            return self._forward_routed_chunk(
                layer,
                x,
                topk_weights,
                topk_ids,
                layer.expert_map,
            )

        missed = _local_stream_misses(layer, topk_ids)
        n_str = getattr(layer, "k3_stream_slots", 0)
        if missed.numel() <= n_str:
            emap = _stream_missing(layer, topk_ids)
            return self._forward_routed_chunk(layer, x, topk_weights, topk_ids, emap)

        base_map = layer.expert_map
        resident_map = base_map.clone()
        resident_map[resident_map >= layer.k3_resident] = -1
        local_slots = base_map[topk_ids.long()]
        has_resident = torch.any(
            (local_slots >= 0) & (local_slots < layer.k3_resident)
        ).item()
        if has_resident:
            output_pairs = self._forward_routed_chunk(
                layer,
                x,
                topk_weights,
                topk_ids,
                resident_map,
                reduce=False,
            )
        else:
            output_pairs = torch.zeros(
                (x.shape[0], topk_ids.shape[1], x.shape[1]),
                dtype=x.dtype,
                device=x.device,
            )

        missed_slots = missed.tolist()
        for start in range(0, len(missed_slots), n_str):
            active_map = _load_stream_slots(
                layer, missed_slots[start : start + n_str], active_only=True
            )
            self._forward_routed_chunk(
                layer,
                x,
                topk_weights,
                topk_ids,
                active_map,
                reduce=False,
                pairs_out=output_pairs,
            )
        return output_pairs.sum(dim=1)

    def _forward_routed_chunk(
        self,
        layer,
        x,
        topk_weights,
        topk_ids,
        emap,
        reduce=True,
        pairs_out=None,
    ) -> torch.Tensor:
        T, H = x.shape
        top_k = topk_ids.shape[1]
        intermediate_size = layer.w13_qweight.shape[1] // self.moe.w13_num_shards
        use_gemv = T <= GEMV_MAX_TOKENS
        if use_gemv:
            # No block-sorting pass at all: each (token, slot) program reads
            # its expert straight out of topk_ids.
            def gemm(a, packed, scale, out, tk, mul):
                w1_gemv(
                    a,
                    packed,
                    scale,
                    topk_ids,
                    topk_weights,
                    emap,
                    out,
                    tk,
                    layer.k3_amplitude,
                    mul,
                )
        else:
            sorted_ids, expert_ids, npad = moe_align_block_size(
                topk_ids, BLOCK_M, layer.global_num_experts, emap
            )

            def gemm(a, packed, scale, out, tk, mul):
                w1_grouped_gemm(
                    a,
                    packed,
                    scale,
                    sorted_ids,
                    expert_ids,
                    npad,
                    topk_weights,
                    out,
                    tk,
                    layer.k3_amplitude,
                    mul,
                    block_m=BLOCK_M,
                )

        # gate/up, then SITU, then down. The routing weight is applied on the
        # down GEMM (not the up) so it multiplies the post-activation value,
        # matching every other vLLM fused-MoE kernel.
        inter = torch.zeros(
            (T * top_k, layer.w13_qweight.shape[1]), dtype=x.dtype, device=x.device
        )
        gemm(x, layer.w13_qweight, layer.w13_scales, inter, top_k, False)

        act = layer.activation
        if act == MoEActivation.SITU and self.moe.activation_situ_beta is not None:
            h = torch.empty(
                (T * top_k, intermediate_size), dtype=x.dtype, device=x.device
            )
            apply_moe_activation(
                act,
                h,
                inter,
                activation_config=self.activation_config,
            )
        else:
            h = situ_and_mul(
                inter,
                self.moe.activation_situ_beta or 1.0,
                self.moe.activation_situ_linear_beta,
            )
        # At the deployment prefill size, `inter` is 1.50 GiB. It is dead
        # once SITU has produced `h`; releasing it here lets the CUDA allocator
        # reuse that block for the down projection instead of keeping all three
        # MoE temporaries live together.
        del inter

        # top_k=1 on the down GEMM: `h` already has one row per (token, slot),
        # so the row index is the pair index rather than the token index.
        if pairs_out is None:
            down = torch.zeros((T * top_k, H), dtype=x.dtype, device=x.device)
        else:
            if pairs_out.shape != (T, top_k, H):
                raise ValueError(
                    f"pairs_out has shape {tuple(pairs_out.shape)}, expected "
                    f"{(T, top_k, H)}"
                )
            # Resident and streamed maps cover disjoint pair positions. The
            # destination starts at zero, so each pass can write its positions
            # in place without a second full pair tensor or a BF16 add.
            down = pairs_out.view(T * top_k, H)
        gemm(h, layer.w2_qweight, layer.w2_scales, down, 1, True)
        del h
        pairs = down.view(T, top_k, H)
        return pairs.sum(dim=1) if reduce else pairs


_install_k3_w1()
