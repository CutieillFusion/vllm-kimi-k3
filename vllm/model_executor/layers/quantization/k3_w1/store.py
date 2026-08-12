# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-local expert-store discovery for k3_w1."""

from pathlib import Path


def resolve_expert_store(
    model_dir: str,
    expert_store: str,
    *,
    pp_rank: int,
    pp_size: int,
    ep_rank: int,
    ep_size: int,
    expected_pp_size: int,
    expected_ep_size: int,
) -> tuple[str, bool]:
    """Resolve the shard for one worker and validate its parallel topology."""
    if ep_size != expected_ep_size:
        raise ValueError(
            f"k3_w1 expert stores require expert parallel size "
            f"{expected_ep_size}, got {ep_size}"
        )
    if pp_size != expected_pp_size:
        raise ValueError(
            f"k3_w1 expert stores require pipeline parallel size "
            f"{expected_pp_size}, got {pp_size}"
        )

    root = Path(model_dir, expert_store)
    names = (f"pp{pp_rank}-tp{ep_rank}", f"pp{pp_rank}t{ep_rank}")
    for name in names:
        candidate = root / name
        if (candidate / "experts.w2").is_file():
            return str(candidate), pp_size > 1
    expected = root / names[0] / "experts.w2"
    raise FileNotFoundError(f"k3_w1 expert shard not found: {expected}")
