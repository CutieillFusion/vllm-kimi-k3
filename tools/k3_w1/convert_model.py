# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert an official Kimi-K3 MXFP4 snapshot to the k3_w1 model layout."""

import argparse
import json
import os
import struct
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import regex as re

from tools.k3_w1.prepare_model import prepare_model

EXPERT_RE = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe"
    r"\.experts\.(\d+)\.(w1|w2|w3)\.weight_(packed|scale)$"
)


def pack_rows_1bit(packed: np.ndarray) -> np.ndarray:
    """Pack MXFP4 nibbles as one sign bit per weight, low nibble first."""
    rows, packed_columns = packed.shape
    signs = np.empty((rows, packed_columns * 2), dtype=np.uint8)
    signs[:, 0::2] = (packed & 0x0F) < 8
    signs[:, 1::2] = (packed >> 4) < 8
    return np.packbits(signs, axis=1, bitorder="little")


def _read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as file:
        header_size = struct.unpack("<Q", file.read(8))[0]
        return json.loads(file.read(header_size)), 8 + header_size


def _expert_layers(model: Path) -> tuple[list[int], dict[int, str]]:
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    shards: dict[int, set[str]] = {}
    for name, shard in index.items():
        match = EXPERT_RE.match(name)
        if match:
            shards.setdefault(int(match.group(1)), set()).add(shard)
    if not shards:
        raise ValueError(f"no Kimi-K3 expert tensors found in {index_path}")
    split_shards = {layer: names for layer, names in shards.items() if len(names) != 1}
    if split_shards:
        raise ValueError(f"expert layers span multiple files: {split_shards}")
    layers = sorted(shards)
    if len(layers) % 2:
        raise ValueError(
            f"PP=2 requires an even number of MoE layers, got {len(layers)}"
        )
    return layers, {layer: next(iter(shards[layer])) for layer in layers}


def _tensor_bytes(file_fd: int, base: int, metadata: dict) -> bytes:
    start, end = metadata["data_offsets"]
    data = os.pread(file_fd, end - start, base + start)
    if len(data) != end - start:
        raise OSError(f"short safetensors read: {len(data)} != {end - start}")
    return data


def _convert_layer(job: tuple) -> tuple[int, float]:
    (
        source,
        layer,
        local_layer,
        stage,
        n_experts,
        slot_bytes,
        output_paths,
    ) = job
    started = time.monotonic()
    source_path = Path(source)
    header, base = _read_header(source_path)
    tensors = {}
    for name, metadata in header.items():
        match = EXPERT_RE.match(name)
        if match and int(match.group(1)) == layer:
            key = (int(match.group(2)), match.group(3), match.group(4))
            tensors[key] = metadata

    input_fd = os.open(source_path, os.O_RDONLY)
    output_fds = [os.open(path, os.O_WRONLY) for path in output_paths]
    try:
        experts_per_rank = n_experts // 2
        for expert in range(n_experts):
            record = []
            for weight in ("w1", "w2", "w3"):
                packed_meta = tensors[(expert, weight, "packed")]
                scale_meta = tensors[(expert, weight, "scale")]
                packed = np.frombuffer(
                    _tensor_bytes(input_fd, base, packed_meta), dtype=np.uint8
                ).reshape(packed_meta["shape"])
                record.append(pack_rows_1bit(packed).tobytes())
                record.append(_tensor_bytes(input_fd, base, scale_meta))
            data = b"".join(record)
            if len(data) != slot_bytes:
                raise ValueError(
                    f"layer {layer} expert {expert}: slot is {len(data)} bytes, "
                    f"expected {slot_bytes}"
                )
            tp_rank = expert % 2
            local_expert = expert // 2
            offset = (local_layer * experts_per_rank + local_expert) * slot_bytes
            if os.pwrite(output_fds[stage * 2 + tp_rank], data, offset) != len(data):
                raise OSError(f"short expert-store write at layer {layer}")
    finally:
        os.close(input_fd)
        for file_fd in output_fds:
            os.close(file_fd)
    return layer, time.monotonic() - started


def convert_model(model: Path, output: Path, workers: int = 4) -> Path:
    """Build all four PP=2, EP=2 expert stores and a runnable model view."""
    model = model.resolve()
    output = output.resolve()
    if model == output:
        raise ValueError("output must differ from the source model directory")
    config = json.loads((model / "config.json").read_text())["text_config"]
    n_experts = int(config["num_experts"])
    if n_experts % 2:
        raise ValueError(f"EP=2 requires an even number of experts, got {n_experts}")
    hidden = int(config["routed_expert_hidden_size"])
    intermediate = int(config["moe_intermediate_size"])
    slot_bytes = (
        2 * (intermediate * (hidden // 8) + intermediate * (hidden // 32))
        + hidden * (intermediate // 8)
        + hidden * (intermediate // 32)
    )

    layers, shards = _expert_layers(model)
    layers_per_stage = len(layers) // 2
    store_root = output / "k3_w1"
    if store_root.exists():
        raise FileExistsError(f"refusing to overwrite {store_root}")
    output_paths = []
    store_bytes = layers_per_stage * (n_experts // 2) * slot_bytes
    for pp_rank in range(2):
        for tp_rank in range(2):
            path = store_root / f"pp{pp_rank}-tp{tp_rank}" / "experts.w2"
            path.parent.mkdir(parents=True)
            with path.open("xb") as file:
                file.truncate(store_bytes)
            output_paths.append(str(path))

    jobs = []
    for ordinal, layer in enumerate(layers):
        stage, local_layer = divmod(ordinal, layers_per_stage)
        jobs.append(
            (
                str(model / shards[layer]),
                layer,
                local_layer,
                stage,
                n_experts,
                slot_bytes,
                output_paths,
            )
        )

    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = executor.map(_convert_layer, jobs)
        for completed, (layer, elapsed) in enumerate(results, 1):
            print(
                f"[k3_w1] layer {layer}: {elapsed:.1f}s ({completed}/{len(jobs)})",
                flush=True,
            )

    for pp_rank in range(2):
        for tp_rank in range(2):
            prepare_model(
                model,
                store_root / f"pp{pp_rank}-tp{tp_rank}",
                output,
                pp_rank,
                tp_rank,
            )
    print(f"[k3_w1] converted model in {(time.monotonic() - started) / 60:.1f} min")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    convert_model(args.model, args.output, args.workers)


if __name__ == "__main__":
    main()
