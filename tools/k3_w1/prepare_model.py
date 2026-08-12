# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Assemble a model directory for the native Kimi-K3 k3_w1 backend."""

import argparse
import json
import os
import shutil
from pathlib import Path


def _place(source: Path, target: Path, mode: str) -> None:
    if mode == "symlink":
        destination = os.path.relpath(source.resolve(), target.parent.resolve())
        if target.is_symlink():
            if os.readlink(target) == destination:
                return
            target.unlink()
        elif target.exists():
            return
        target.symlink_to(destination)
    elif target.exists():
        return
    elif mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def prepare_model(
    dense_model: Path,
    expert_store: Path,
    output: Path,
    pp_rank: int,
    tp_rank: int,
    pp_size: int = 2,
    tp_size: int = 2,
    mode: str = "symlink",
) -> Path:
    """Add one expert shard and k3_w1 metadata to a dense model directory."""
    config_path = dense_model / "config.json"
    expert_path = expert_store / "experts.w2"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not expert_path.is_file():
        raise FileNotFoundError(expert_path)
    if not 0 <= pp_rank < pp_size or not 0 <= tp_rank < tp_size:
        raise ValueError("PP and TP ranks must be within their configured sizes")

    output.mkdir(parents=True, exist_ok=True)
    for source in dense_model.iterdir():
        if source.name == "config.json" or source.name == "k3_w1":
            continue
        _place(source, output / source.name, mode)

    config = json.loads(config_path.read_text())
    quantization_config = {
        "quant_method": "k3_w1",
        "amplitude": 1.69,
        "group_size": 32,
        "dense_bits": 4,
        "mla_bits": 8,
        "head_bits": 8,
        "expert_store": "k3_w1",
        "expert_tensor_parallel_size": tp_size,
        "expert_pipeline_parallel_size": pp_size,
    }
    config["quantization_config"] = quantization_config
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        text_config["quantization_config"] = quantization_config.copy()
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    shard = output / "k3_w1" / f"pp{pp_rank}-tp{tp_rank}"
    shard.mkdir(parents=True, exist_ok=True)
    _place(expert_path, shard / "experts.w2", mode)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-model", type=Path, required=True)
    parser.add_argument("--expert-store", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pp-rank", type=int, required=True)
    parser.add_argument("--tp-rank", type=int, required=True)
    parser.add_argument("--pp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument(
        "--mode", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    args = parser.parse_args()
    prepare_model(**vars(args))


if __name__ == "__main__":
    main()
