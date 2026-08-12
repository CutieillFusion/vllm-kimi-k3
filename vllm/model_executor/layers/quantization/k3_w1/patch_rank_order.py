# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional deterministic Ray rank placement for sharded local stores."""

import os


def _order_by_ip(items, desired_ips):
    by_ip = {item[2]: item for item in items}
    if len(by_ip) != len(items):
        raise RuntimeError(f"K3 rank placement has duplicate node IPs: {items}")
    missing = [ip for ip in desired_ips if ip not in by_ip]
    extra = [ip for ip in by_ip if ip not in desired_ips]
    if missing or extra or len(items) != len(desired_ips):
        raise RuntimeError(
            f"K3 rank placement mismatch: missing={missing}, extra={extra}, "
            f"actual={list(by_ip)}"
        )
    return [by_ip[ip] for ip in desired_ips]


def apply() -> None:
    """Order Ray placement bundles when K3_RANK_IPS is configured."""
    desired = os.environ.get("K3_RANK_IPS", "")
    if not desired:
        return
    desired_ips = [ip.strip() for ip in desired.split(",") if ip.strip()]

    from vllm.v1.executor import ray_executor_v2, ray_utils

    original = ray_utils.get_bundles_sorted_by_node
    if getattr(original, "_k3_rank_order", False):
        return

    def ordered(placement_group):
        return _order_by_ip(original(placement_group), desired_ips)

    ordered._k3_rank_order = True  # type: ignore[attr-defined]
    ray_utils.get_bundles_sorted_by_node = ordered
    ray_executor_v2.get_bundles_sorted_by_node = ordered
