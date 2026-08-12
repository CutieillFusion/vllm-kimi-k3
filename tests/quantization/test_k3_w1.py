# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import numpy as np
import pytest

from tools.k3_w1.convert_model import pack_rows_1bit
from tools.k3_w1.prepare_model import prepare_model
from vllm.model_executor.layers.quantization.k3_w1.store import (
    resolve_expert_store,
)


def test_prepare_model_declares_backend_and_places_shard(tmp_path: Path):
    dense = tmp_path / "dense"
    experts = tmp_path / "experts"
    output = tmp_path / "model"
    dense.mkdir()
    experts.mkdir()
    (dense / "config.json").write_text(
        json.dumps({"text_config": {}, "architectures": ["KimiK3"]})
    )
    (dense / "weights.safetensors").write_bytes(b"dense")
    (experts / "experts.w2").write_bytes(b"experts")

    prepare_model(dense, experts, output, pp_rank=1, tp_rank=0)

    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "k3_w1"
    assert config["text_config"]["quantization_config"]["quant_method"] == "k3_w1"
    assert (output / "k3_w1" / "pp1-tp0" / "experts.w2").read_bytes() == b"experts"


def test_pack_rows_1bit_preserves_mxfp4_sign_order():
    codes = np.arange(16, dtype=np.uint8)
    packed = (codes[0::2] | (codes[1::2] << 4)).reshape(1, -1)

    result = pack_rows_1bit(packed)

    assert result.tolist() == [[0xFF, 0x00]]


def test_resolve_expert_store_uses_runtime_ranks(tmp_path: Path):
    shard = tmp_path / "k3_w1" / "pp1-tp0"
    shard.mkdir(parents=True)
    (shard / "experts.w2").touch()

    path, stage_local = resolve_expert_store(
        str(tmp_path),
        "k3_w1",
        pp_rank=1,
        pp_size=2,
        ep_rank=0,
        ep_size=2,
        expected_pp_size=2,
        expected_ep_size=2,
    )

    assert path == str(shard)
    assert stage_local


def test_resolve_expert_store_rejects_wrong_topology(tmp_path: Path):
    with pytest.raises(ValueError, match="expert parallel size 2, got 1"):
        resolve_expert_store(
            str(tmp_path),
            "k3_w1",
            pp_rank=0,
            pp_size=2,
            ep_rank=0,
            ep_size=1,
            expected_pp_size=2,
            expected_ep_size=2,
        )
