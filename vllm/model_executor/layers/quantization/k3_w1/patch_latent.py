#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Let the latent projections be quantized.

vLLM builds routed_expert_down_proj and routed_expert_up_proj as
ReplicatedLinear with an explicit `quant_config=None`
(models/kimi_k3/nvidia/model.py:562,579), so get_quant_method is never
consulted for them and they stay bf16 whatever we register. They are also
replicated rather than sharded, by design -- the runner's docstring says the
replicated up-proj is what makes routed+shared combine correctly at any TP
size.

Measured on the real checkpoint, that is 9.5 GB per node in bf16 against
2.7 GB at int4-g64, and it is the difference between the model fitting and
not:

    latent bf16   dense 20.6 GB + experts 106.4 = 127.0 GB   (have ~124)
    latent int4   dense 13.8 GB + experts 106.4 = 120.2 GB

Quantizing them is not a new risk: our own engine already runs the latent
projections at int4-g64 -- K3_BITS=4 covers "KDA/latent/shared/dense"
(c/kimi_k3.c:39) -- and that is the configuration measured at PCC 0.9996.

Implemented by substituting the ReplicatedLinear symbol the K3 model module
looks up, rather than editing vLLM, so this stays out-of-tree. It is still a
patch against module internals: if a future vLLM renames these projections or
stops using ReplicatedLinear, `verify()` reports that rather than silently
reverting to bf16.
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

TARGETS = ("routed_expert_down_proj", "routed_expert_up_proj")
# The MoE router is built the same way (quant_config=None) and is replicated on
# every rank: 1.18 GB of the checkpoint, 1.18 GB per node. Included only when
# K3_QUANT_GATE=1, because this is the routing decision -- int8 per row is a
# safe 2x, int4 would not be.
GATE = "block_sparse_moe.gate"
_applied = False


def apply():
    """Idempotent. Call before the model is constructed."""
    global _applied
    if _applied:
        return True
    import vllm.models.kimi_k3.nvidia.model as k3

    class _MaybeQuantReplicatedLinear(k3.ReplicatedLinear):
        def __init__(self, *args, **kwargs):
            prefix = kwargs.get("prefix", "")
            want = prefix.endswith(TARGETS) or (
                os.environ.get("K3_QUANT_GATE") == "1" and prefix.endswith(GATE)
            )
            if want and kwargs.get("quant_config") is None:
                from vllm.config import get_current_vllm_config

                cfg = get_current_vllm_config()
                qc = getattr(cfg, "quant_config", None)
                # Only take over when *our* method is the active one; leave
                # any other quantization scheme alone.
                if type(qc).__name__ == "KimiK3OneBitConfig":
                    kwargs["quant_config"] = qc
            super().__init__(*args, **kwargs)

    k3.ReplicatedLinear = _MaybeQuantReplicatedLinear  # type: ignore[misc]
    _applied = True
    logger.info("k3_w1: latent projections routed through the quant config")
    return True


def verify(model) -> tuple[int, int]:
    """(quantized, total) latent projections in a built model."""
    q = t = 0
    for name, mod in model.named_modules():
        if name.endswith(TARGETS):
            t += 1
            q += hasattr(mod, "k3_qweight")
    return q, t
