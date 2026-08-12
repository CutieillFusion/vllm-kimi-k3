# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Kimi-K3 1-bit expert quantization support."""

_installed = False


def install_k3_w1() -> None:
    """Install the Kimi-K3 hooks that lack native extension points."""
    global _installed
    if _installed:
        return
    from . import patch_latent, patch_loader, patch_rank_order

    patch_latent.apply()
    patch_rank_order.apply()
    patch_loader.apply()
    _installed = True


__all__ = ["install_k3_w1"]
