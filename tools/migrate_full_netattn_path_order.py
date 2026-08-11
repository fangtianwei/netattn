"""Rewrite Full NetAttn display metadata to shallow-to-deep path order.

This migration deliberately leaves checkpoint indices, weights, coefficients,
losses, and accuracies untouched. It can be rerun safely.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.ResNet_NetAttn_CIFAR100 import subnet_index_to_bits


NUM_BLOCKS = 8
BLOCK_NAMES = (
    "L1.B0",
    "L1.B1",
    "L2.B0",
    "L2.B1",
    "L3.B0",
    "L3.B1",
    "L4.B0",
    "L4.B1",
)
TOPK_DIR = ROOT / "exp03_topk"


def bits(index: int) -> str:
    return subnet_index_to_bits(index, NUM_BLOCKS)


def decode_path(index: int) -> str:
    return " | ".join(
        f"{name}:{'Full' if bit == '1' else 'Skip'}"
        for name, bit in zip(BLOCK_NAMES, bits(index))
    )


def conv_macs(
    in_channels: int, out_channels: int, kernel: int, output_size: int
) -> int:
    return output_size**2 * out_channels * in_channels * kernel**2


def block_mac_weights() -> list[int]:
    specs = (
        (64, 64, 1, 32),
        (64, 64, 1, 32),
        (64, 128, 2, 32),
        (128, 128, 1, 16),
        (128, 256, 2, 16),
        (256, 256, 1, 8),
        (256, 512, 2, 8),
        (512, 512, 1, 4),
    )
    costs = []
    for in_channels, out_channels, stride, input_size in specs:
        output_size = input_size // stride
        cost = conv_macs(in_channels, out_channels, 3, output_size)
        cost += conv_macs(out_channels, out_channels, 3, output_size)
        if stride != 1 or in_channels != out_channels:
            cost += conv_macs(in_channels, out_channels, 1, output_size)
        costs.append(cost)
    return costs


def path_metrics(indices: list[int]) -> dict[str, Any]:
    strings = [bits(index) for index in indices]
    prefix_counts = [
        len({path[:length] for path in strings})
        for length in range(1, NUM_BLOCKS + 1)
    ]
    parent_counts = [1, *prefix_counts[:-1]]
    stem_cost = conv_macs(3, 64, 3, 32)
    classifier_cost = 512 * 100
    block_costs = block_mac_weights()
    baseline = stem_cost + sum(block_costs) + classifier_cost
    selected = (
        stem_cost
        + sum(count * cost for count, cost in zip(parent_counts, block_costs))
        + classifier_cost
    )
    return {
        "path_bits": strings,
        "prefix_counts": prefix_counts,
        "parent_counts": parent_counts,
        "active_state_ratio": sum(prefix_counts) / NUM_BLOCKS,
        "relative_cost_proxy": selected / baseline,
    }


def rewrite_attention_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "subnet_idx" not in fieldnames or "bits" not in fieldnames:
        raise ValueError(f"Unexpected attention CSV schema: {path}")
    for row in rows:
        index = int(row["subnet_idx"])
        row["bits"] = bits(index)
        if "path" in row:
            row["path"] = decode_path(index)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rewrite_results_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        raise ValueError(f"Unexpected results CSV schema: {path}")
    for row in rows:
        if not row.get("path_indices"):
            continue
        indices = [int(index) for index in json.loads(row["path_indices"])]
        metrics = path_metrics(indices)
        for key, value in metrics.items():
            row[key] = json.dumps(value) if isinstance(value, list) else str(value)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def rewrite_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["path_order"]["index_encoding"] = (
        "subnet_idx uses its least-significant bit for L1.B0; displayed bits "
        "reverse the integer representation so the leftmost bit is L1.B0 "
        "and the rightmost bit is L4.B1"
    )
    example = payload["path_order"]["validated_example"]
    example["bits"] = bits(int(example["subnet_idx"]))
    example["decoded"] = decode_path(int(example["subnet_idx"]))
    for row in payload["results"]:
        if not row.get("path_indices"):
            continue
        metrics = path_metrics([int(index) for index in row["path_indices"]])
        row.update(metrics)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    attention_files = sorted((ROOT / "exp02_full_netattn" / "runs").glob(
        "*/attention_*.csv"
    ))
    for path in attention_files:
        rewrite_attention_csv(path)

    results_csv = TOPK_DIR / "netattn_ranking_control_seed42_results.csv"
    results_json = TOPK_DIR / "netattn_ranking_control_seed42.json"
    rewrite_results_csv(results_csv)
    payload = rewrite_json(results_json)

    # Reuse the report generator so future evaluations and this migration emit
    # byte-for-byte compatible Markdown structure.
    from exp03_topk.run_netattn_ranking_control_seed42 import write_markdown

    write_markdown(payload)
    print(f"updated {len(attention_files)} attention CSV files")
    print("updated Top-K CSV, JSON, and Markdown artifacts")


if __name__ == "__main__":
    main()
