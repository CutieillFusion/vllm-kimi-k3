#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Load the flat k3-w1 expert store into vLLM's MoE parameters.

The store is not an HF checkpoint, so vLLM's safetensors loader cannot see it.
It is one headerless file, `<dir>/experts.w2`, addressed arithmetically -- see
c/kimi_k3.c:1455-1480:

    slot      = [ w1p | w1s | w2p | w2s | w3p | w3s ]
    slot_off  = (moe_layer_ordinal * experts_per_rank + expert_local) * slot_bytes

`moe_layer_ordinal` counts only sparse layers (0..nmoe-1), not the model's
layer index. With K3_W2_SHARD=r/N the file holds only experts where
`e % N == r`, packed contiguously, so `expert_local = e // N`.

vLLM wants gate and up fused as w13 = [gate; up], which is our [w1; w3] --
w1 is the half that takes tanh*sigmoid and w3 the linearly-clipped half, in
both c/backend_cuda_k3.cu:530-532 and vllm's SituAndMul.
"""

import os
from io import FileIO

import numpy as np
import torch

SCALE_NIBBLE_BASE = 109


def pack_scale_nibbles(scale: np.ndarray) -> np.ndarray:
    """Pack two UE8M0 exponents per byte without changing their values."""
    if scale.shape[-1] % 2:
        raise ValueError(f"scale group dimension must be even, got {scale.shape[-1]}")
    delta = scale.astype(np.int16) - SCALE_NIBBLE_BASE
    lo = int(delta.min())
    hi = int(delta.max())
    if lo < 0 or hi > 15:
        raise ValueError(
            f"UE8M0 exponent outside lossless nibble range "
            f"[{SCALE_NIBBLE_BASE}, {SCALE_NIBBLE_BASE + 15}]: "
            f"got [{lo + SCALE_NIBBLE_BASE}, {hi + SCALE_NIBBLE_BASE}]"
        )
    return (delta[..., 0::2] | (delta[..., 1::2] << 4)).astype(np.uint8)


def pack_slot_scales(slot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return a slot with its three scale tensors packed for the kernels."""
    return {
        **slot,
        "w1s": pack_scale_nibbles(slot["w1s"]),
        "w2s": pack_scale_nibbles(slot["w2s"]),
        "w3s": pack_scale_nibbles(slot["w3s"]),
    }


class K3W1Store:
    def __init__(self, path: str, hidden: int, inter: int, experts_per_rank: int):
        self.path = os.path.join(path, "experts.w2") if os.path.isdir(path) else path
        self.hidden = hidden  # latent dim, 3584
        self.inter = inter  # moe intermediate, 3072
        self.experts_per_rank = experts_per_rank
        self._fh: FileIO | None = None
        self._fadv = hasattr(os, "posix_fadvise")
        self._buf: bytearray | None = None

        self.e_w1p = inter * (hidden // 8)
        self.e_w1s = inter * (hidden // 32)
        self.e_w2p = hidden * (inter // 8)
        self.e_w2s = hidden * (inter // 32)
        self.slot = 2 * (self.e_w1p + self.e_w1s) + self.e_w2p + self.e_w2s

        # The engine refuses a store whose slot is not 4096-aligned because it
        # reads with O_DIRECT; keep the same check so a store that works here
        # is one the engine can also open.
        if self.slot % 4096:
            raise ValueError(f"slot {self.slot} is not 4096-aligned")

        size = os.path.getsize(self.path)
        if size % self.slot:
            raise ValueError(
                f"{self.path}: {size} bytes is not a whole number of "
                f"{self.slot}-byte slots"
            )
        self.n_slots = size // self.slot

    def __repr__(self) -> str:
        return (
            f"K3W1Store({self.path}, slot={self.slot / 2**20:.2f} MiB, "
            f"{self.n_slots} slots)"
        )

    def _offsets(self):
        o = [0]
        for n in (self.e_w1p, self.e_w1s, self.e_w2p, self.e_w2s, self.e_w1p):
            o.append(o[-1] + n)
        return o  # w1p w1s w2p w2s w3p w3s

    def read_slot(self, moe_layer: int, expert_local: int) -> dict[str, np.ndarray]:
        idx = moe_layer * self.experts_per_rank + expert_local
        if not 0 <= idx < self.n_slots:
            raise IndexError(
                f"slot {idx} (layer {moe_layer}, expert {expert_local}) outside "
                f"the {self.n_slots}-slot store"
            )
        # One handle for the whole load: a full rank reads 92 x 224 = 20,608
        # slots, and reopening per slot turns that into 20,608 opens.
        if self._fh is None:
            self._fh = FileIO(self.path, "rb")
        self._fh.seek(idx * self.slot)
        # One buffer reused for every slot. A rank reads 20,608 of them, and a
        # fresh 4.92 MiB bytearray each time is 101 GB of allocation churn that
        # the allocator holds onto rather than returning -- which is what
        # pushes the node over during expert fill. Callers must consume the
        # returned views before the next read_slot, which fill_layer does.
        if self._buf is None:
            self._buf = bytearray(self.slot)
        raw = self._buf
        got = self._fh.readinto(raw)
        if got != self.slot:
            raise OSError(f"short read at slot {idx}: {got} != {self.slot}")
        # Drop this slot from the page cache. A rank streams 106 GB through
        # here, and every byte of it would otherwise become cache competing
        # with the 119 GB of weights on a 121 GB box -- which is enough to
        # push the node into thrashing during load. The engine avoids the same
        # problem by opening the store O_DIRECT (kimi_k3.c:1447). Each slot is
        # read exactly once, so there is nothing to lose by evicting it.
        if self._fadv:
            try:
                os.posix_fadvise(
                    self._fh.fileno(),
                    idx * self.slot,
                    self.slot,
                    os.POSIX_FADV_DONTNEED,
                )
            except OSError:
                self._fadv = False
        buf = np.frombuffer(raw, dtype=np.uint8)
        o = self._offsets()
        hidden, inter = self.hidden, self.inter
        return {
            "w1p": buf[o[0] : o[1]].reshape(inter, hidden // 8),
            "w1s": buf[o[1] : o[2]].reshape(inter, hidden // 32),
            "w2p": buf[o[2] : o[3]].reshape(hidden, inter // 8),
            "w2s": buf[o[3] : o[4]].reshape(hidden, inter // 32),
            "w3p": buf[o[4] : o[5]].reshape(inter, hidden // 8),
            "w3s": buf[o[5] :].reshape(inter, hidden // 32),
        }

    def fill_layer(self, layer, moe_layer: int, expert_ids: list[int] | None = None):
        """Populate one RoutedExperts layer's parameters from the store.

        `expert_ids` are store-local indices; default is this rank's whole
        contiguous range, which is what a plain K3_W2_SHARD=r/N split gives.
        """
        n = layer.w13_qweight.shape[0]
        ids = list(range(n)) if expert_ids is None else expert_ids
        if len(ids) != n:
            raise ValueError(f"{len(ids)} expert ids for {n} local expert slots")

        inter = self.inter
        for slot_i, e in enumerate(ids):
            s = pack_slot_scales(self.read_slot(moe_layer, e))
            w13p = layer.w13_qweight[slot_i]
            w13s = layer.w13_scales[slot_i]
            w13p[:inter].copy_(torch.from_numpy(s["w1p"]))
            w13p[inter:].copy_(torch.from_numpy(s["w3p"]))
            w13s[:inter].copy_(torch.from_numpy(s["w1s"]))
            w13s[inter:].copy_(torch.from_numpy(s["w3s"]))
            layer.w2_qweight[slot_i].copy_(torch.from_numpy(s["w2p"]))
            layer.w2_scales[slot_i].copy_(torch.from_numpy(s["w2s"]))
