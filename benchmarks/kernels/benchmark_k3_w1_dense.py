# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep decode GEMV configurations for the native Kimi-K3 dense kernel."""

import argparse
import itertools

import torch

from vllm.model_executor.layers.quantization.k3_w1.dense_kernels import (
    k3_dense_matmul,
)


def elapsed_ms(function, iterations: int) -> float:
    for _ in range(3):
        function()
    torch.accelerator.synchronize()
    start = torch.Event(enable_timing=True)
    end = torch.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-size", type=int, default=6144)
    parser.add_argument("--input-size", type=int, default=7168)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()

    generator = torch.Generator(device="cuda").manual_seed(0)
    inputs = torch.randn(
        1,
        args.input_size,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    if args.bits == 4:
        weights = torch.randint(
            0,
            256,
            (args.output_size, args.input_size // 2),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        scales = torch.rand(
            args.output_size,
            args.input_size // 64,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
    else:
        weights = torch.randint(
            -127,
            128,
            (args.output_size, args.input_size),
            dtype=torch.int8,
            device="cuda",
            generator=generator,
        )
        scales = torch.rand(
            args.output_size,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )

    reference = k3_dense_matmul(inputs, weights, scales, args.bits)
    weight_bytes = (
        weights.numel() * weights.element_size()
        + scales.numel() * scales.element_size()
    )
    print(f"{'BN':>4} {'BK':>4} {'W':>2} {'ms':>8} {'GB/s':>8} {'max diff':>10}")
    best = None
    for block_n, block_k, warps in itertools.product(
        (32, 64, 128), (64, 128, 256), (2, 4, 8)
    ):
        try:
            result = k3_dense_matmul(
                inputs,
                weights,
                scales,
                args.bits,
                block_n=block_n,
                block_k=block_k,
                num_warps=warps,
            )
            max_diff = (result.float() - reference.float()).abs().max().item()

            def run(
                block_n: int = block_n,
                block_k: int = block_k,
                warps: int = warps,
            ) -> None:
                k3_dense_matmul(
                    inputs,
                    weights,
                    scales,
                    args.bits,
                    block_n=block_n,
                    block_k=block_k,
                    num_warps=warps,
                )

            milliseconds = elapsed_ms(run, args.iterations)
        except Exception as error:
            print(f"{block_n:4d} {block_k:4d} {warps:2d} {str(error)[:40]}")
            continue
        bandwidth = weight_bytes / milliseconds / 1e6
        print(
            f"{block_n:4d} {block_k:4d} {warps:2d} {milliseconds:8.3f} "
            f"{bandwidth:8.1f} {max_diff:10.4f}"
        )
        if best is None or milliseconds < best[0]:
            best = (milliseconds, block_n, block_k, warps, bandwidth)

    if best is not None:
        milliseconds, block_n, block_k, warps, bandwidth = best
        print(
            f"best: BN={block_n} BK={block_k} warps={warps}: "
            f"{milliseconds:.3f} ms, {bandwidth:.1f} GB/s"
        )


if __name__ == "__main__":
    main()
