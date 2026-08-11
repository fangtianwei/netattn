"""Seed-42 controls for evaluating the Full NetAttn path ranking.

The script evaluates several coefficient vectors from one shared full-path
forward pass per CIFAR-100 test batch. This is mathematically equivalent to
rerunning the same checkpoint with fixed path coefficients, while avoiding
repeated enumeration of the 256 terminal path features.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.cifar100_da_dataset import get_dataloaders
from models.ResNet_NetAttn_Config_CIFAR100 import (
    BasicBlock,
    ResNet_NetAttn_Config_CIFAR100,
    subnet_index_to_bits,
)


RUN_DIR = (
    SCRIPT_DIR.parent
    / "exp02_full_netattn"
    / "runs"
    / "ResNet-18-CIFAR100-NetAttn-nozero-sum=1_2026-04-15_21-40-57"
)
CHECKPOINT_PATH = RUN_DIR / "best_model.pth"
ATTENTION_CSV_PATH = RUN_DIR / "attention_final.csv"
OUTPUT_CSV_PATH = SCRIPT_DIR / "netattn_ranking_control_seed42_results.csv"
OUTPUT_JSON_PATH = SCRIPT_DIR / "netattn_ranking_control_seed42.json"
OUTPUT_MD_PATH = SCRIPT_DIR / "netattn_ranking_control_seed42.md"

SEED = 42
RANDOM_SEED = 42007
NUM_RANDOM_SETS = 20
TOPK = 7
BATCH_SIZE = 128
# Use the same test transform and batch size as the notebook. A single-process
# loader avoids Windows worker-spawn restrictions in noninteractive runs.
NUM_WORKERS = 0
USE_AMP = True
NUM_BLOCKS = 8
NUM_PATHS = 2**NUM_BLOCKS
BLOCK_NAMES = [
    "L1.B0",
    "L1.B1",
    "L2.B0",
    "L2.B1",
    "L3.B0",
    "L3.B1",
    "L4.B0",
    "L4.B1",
]


@dataclass(frozen=True)
class Selection:
    name: str
    family: str
    paths: tuple[int, ...]
    coefficients: np.ndarray


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bits(index: int) -> str:
    """Return canonical path bits from shallowest to deepest block."""
    return subnet_index_to_bits(index, NUM_BLOCKS)


def decode_path(index: int) -> str:
    return " | ".join(
        f"{name}:{'Full' if bit == '1' else 'Skip'}"
        for name, bit in zip(BLOCK_NAMES, bits(index))
    )


def prefix_profile(path_indices: tuple[int, ...]) -> tuple[list[int], list[int]]:
    strings = [bits(index) for index in path_indices]
    prefix_counts = [
        len({path[:prefix_len] for path in strings})
        for prefix_len in range(1, NUM_BLOCKS + 1)
    ]
    parent_counts = [1] + prefix_counts[:-1]
    return prefix_counts, parent_counts


def conv_macs(
    in_channels: int,
    out_channels: int,
    kernel: int,
    output_size: int,
) -> int:
    return output_size * output_size * out_channels * in_channels * kernel * kernel


def block_mac_weights() -> list[int]:
    """Approximate convolution MACs for the eight CIFAR-style ResNet-18 blocks."""

    specs = [
        # in, out, stride, input spatial size
        (64, 64, 1, 32),
        (64, 64, 1, 32),
        (64, 128, 2, 32),
        (128, 128, 1, 16),
        (128, 256, 2, 16),
        (256, 256, 1, 8),
        (256, 512, 2, 8),
        (512, 512, 1, 4),
    ]
    costs: list[int] = []
    for in_channels, out_channels, stride, input_size in specs:
        output_size = input_size // stride
        cost = conv_macs(in_channels, out_channels, 3, output_size)
        cost += conv_macs(out_channels, out_channels, 3, output_size)
        if stride != 1 or in_channels != out_channels:
            cost += conv_macs(in_channels, out_channels, 1, output_size)
        costs.append(cost)
    return costs


def relative_cost(path_indices: tuple[int, ...]) -> dict[str, Any]:
    prefix_counts, parent_counts = prefix_profile(path_indices)
    block_costs = block_mac_weights()
    stem_cost = conv_macs(3, 64, 3, 32)
    classifier_cost = 512 * 100
    baseline = stem_cost + sum(block_costs) + classifier_cost
    selected = (
        stem_cost
        + sum(count * cost for count, cost in zip(parent_counts, block_costs))
        + classifier_cost
    )
    return {
        "prefix_counts": prefix_counts,
        "parent_counts": parent_counts,
        "active_state_ratio": sum(prefix_counts) / NUM_BLOCKS,
        "relative_cost": selected / baseline,
        "cost_definition": (
            "CIFAR-style ResNet-18 convolution-MAC proxy under prefix-DAG "
            "execution, including one shared stem and classifier"
        ),
    }


def extract_state_dict(loaded: Any) -> dict[str, torch.Tensor]:
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        loaded = loaded["model_state_dict"]
    if not isinstance(loaded, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(loaded)!r}")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in loaded.items()
    }


def load_attention_table() -> pd.DataFrame:
    frame = pd.read_csv(ATTENTION_CSV_PATH)
    required = {"subnet_idx", "bits", "weight", "path"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Attention CSV is missing columns: {sorted(missing)}")
    if len(frame) != NUM_PATHS:
        raise ValueError(f"Expected {NUM_PATHS} attention rows, found {len(frame)}")

    frame = frame.sort_values("subnet_idx").reset_index(drop=True)
    for row in frame.itertuples(index=False):
        expected_bits = bits(int(row.subnet_idx))
        if str(row.bits).zfill(NUM_BLOCKS) != expected_bits:
            raise AssertionError(
                f"CSV path-order mismatch at {row.subnet_idx}: "
                f"{row.bits!r} != {expected_bits!r}"
            )
        expected_path = decode_path(int(row.subnet_idx))
        if str(row.path) != expected_path:
            raise AssertionError(
                f"CSV path decoding mismatch at {row.subnet_idx}: "
                f"{row.path!r} != {expected_path!r}"
            )

    example = frame.loc[frame["subnet_idx"] == 63].iloc[0]
    if str(example["bits"]).zfill(NUM_BLOCKS) != "11111100":
        raise AssertionError("subnet_idx=63 is not encoded as 11111100")
    if not str(example["path"]).endswith("L4.B0:Skip | L4.B1:Skip"):
        raise AssertionError("11111100 does not skip the last two residual blocks")
    return frame


def make_structure_matched_random_sets(
    top_indices: tuple[int, ...],
) -> tuple[list[tuple[int, ...]], list[int]]:
    """Randomize path identities while preserving the exact prefix-tree cost.

    XORing every path by the same 8-bit mask is a bijection at every prefix
    depth. It therefore preserves all prefix and parent counts exactly.
    """

    generator = random.Random(RANDOM_SEED)
    masks = list(range(1, NUM_PATHS))
    generator.shuffle(masks)
    random_sets: list[tuple[int, ...]] = []
    used: set[tuple[int, ...]] = set()
    used_masks: list[int] = []
    for mask in masks:
        candidate = tuple(sorted(index ^ mask for index in top_indices))
        if candidate in used:
            continue
        used.add(candidate)
        random_sets.append(candidate)
        used_masks.append(mask)
        if len(random_sets) == NUM_RANDOM_SETS:
            break
    if len(random_sets) != NUM_RANDOM_SETS:
        raise RuntimeError(f"Could construct only {len(random_sets)} random sets")

    target = relative_cost(top_indices)
    for candidate in random_sets:
        candidate_cost = relative_cost(candidate)
        relative_gap = abs(
            candidate_cost["relative_cost"] - target["relative_cost"]
        ) / target["relative_cost"]
        if relative_gap > 0.05:
            raise AssertionError(
                f"Random set cost gap {relative_gap:.6f} exceeds 5%"
            )
    return random_sets, used_masks


def build_selections(
    checkpoint_weights: np.ndarray,
    csv_weights: np.ndarray,
) -> tuple[list[Selection], dict[str, Any]]:
    # Match the source notebook: ranking and retained Top-K values come from
    # attention_final.csv, while the Full NetAttn reference uses checkpoint
    # logits directly.
    order = np.argsort(-csv_weights, kind="stable")
    top_indices = tuple(sorted(int(index) for index in order[:TOPK]))
    bottom_indices = tuple(sorted(int(index) for index in order[-TOPK:]))
    top_mass = float(csv_weights[list(top_indices)].sum(dtype=np.float64))

    learned_top_coefficients = np.zeros(NUM_PATHS, dtype=np.float32)
    learned_top_coefficients[list(top_indices)] = csv_weights[list(top_indices)]

    equal_value = top_mass / TOPK

    def equal_coefficients(indices: tuple[int, ...]) -> np.ndarray:
        values = np.zeros(NUM_PATHS, dtype=np.float32)
        values[list(indices)] = equal_value
        return values

    selections = [
        Selection(
            name="Full NetAttn",
            family="full",
            paths=tuple(range(NUM_PATHS)),
            coefficients=checkpoint_weights.astype(np.float32),
        ),
        Selection(
            name="NetAttn Top-7",
            family="top7_learned",
            paths=top_indices,
            coefficients=learned_top_coefficients,
        ),
        Selection(
            name="Top-7 equal mass-matched",
            family="top7_equal",
            paths=top_indices,
            coefficients=equal_coefficients(top_indices),
        ),
    ]

    random_sets, xor_masks = make_structure_matched_random_sets(top_indices)
    for position, indices in enumerate(random_sets):
        selections.append(
            Selection(
                name=f"Random-7 equal mass-matched #{position:02d}",
                family="random7_equal",
                paths=indices,
                coefficients=equal_coefficients(indices),
            )
        )

    selections.append(
        Selection(
            name="Bottom-7 equal mass-matched",
            family="bottom7_equal",
            paths=bottom_indices,
            coefficients=equal_coefficients(bottom_indices),
        )
    )
    metadata = {
        "top_indices": list(top_indices),
        "top_paths_ranked": [int(index) for index in order[:TOPK]],
        "bottom_indices": list(bottom_indices),
        "bottom_paths_ranked": [int(index) for index in order[-TOPK:]],
        "top7_attention_sum": top_mass,
        "equal_coefficient": equal_value,
        "random_seed": RANDOM_SEED,
        "random_xor_masks": xor_masks,
        "random_sampling": (
            "Each Random-7 set is the Top-7 index set transformed by one "
            "fixed random 8-bit XOR mask. This randomizes path identities "
            "while preserving the exact prefix-tree structure and MAC proxy."
        ),
    }
    return selections, metadata


def evaluate(
    model: ResNet_NetAttn_Config_CIFAR100,
    testloader: Any,
    selections: list[Selection],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    coefficient_matrix = torch.from_numpy(
        np.stack([selection.coefficients for selection in selections])
    ).to(device=device, dtype=torch.float32)
    correct = torch.zeros(len(selections), dtype=torch.long, device=device)
    loss_sum = torch.zeros(len(selections), dtype=torch.float64, device=device)
    total = 0

    autocast_enabled = bool(USE_AMP and device.type == "cuda")
    with torch.inference_mode():
        for images, labels in tqdm(testloader, desc="seed-42 ranking controls"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                path_features = model._enumerate_subnetworks(images)
                num_paths, batch_size, channels, height, width = path_features.shape
                pooled = model.avgpool(
                    path_features.reshape(
                        num_paths * batch_size, channels, height, width
                    )
                )
                pooled = torch.flatten(pooled, 1).reshape(
                    num_paths, batch_size, channels
                )

            # Match the model/notebook reduction order exactly. A batched
            # einsum changes floating-point accumulation enough to flip an
            # occasional borderline prediction.
            aggregated = torch.stack(
                [
                    torch.sum(
                        pooled
                        * coefficient_matrix[selection_index].view(
                            num_paths, 1, 1
                        ),
                        dim=0,
                    )
                    for selection_index in range(len(selections))
                ],
                dim=0,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=autocast_enabled,
            ):
                # Keep the classifier GEMM shape identical to a standalone
                # notebook run for every control.
                logits = torch.stack(
                    [model.fc(features) for features in aggregated], dim=0
                )

            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels.unsqueeze(0)).sum(dim=1)
            repeated_labels = labels.unsqueeze(0).expand(len(selections), -1)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                repeated_labels.reshape(-1),
                reduction="none",
            ).reshape(len(selections), -1)
            loss_sum += losses.double().sum(dim=1)
            total += labels.numel()

    rows: list[dict[str, Any]] = []
    for index, selection in enumerate(selections):
        cost = relative_cost(selection.paths)
        rows.append(
            {
                "setting": selection.name,
                "family": selection.family,
                "num_paths": len(selection.paths),
                "coefficient_sum": float(selection.coefficients.sum()),
                "accuracy_percent": float(correct[index].item() * 100.0 / total),
                "test_loss": float(loss_sum[index].item() / total),
                "relative_cost_proxy": float(cost["relative_cost"]),
                "active_state_ratio": float(cost["active_state_ratio"]),
                "prefix_counts": cost["prefix_counts"],
                "parent_counts": cost["parent_counts"],
                "path_indices": list(selection.paths),
                "path_bits": [bits(path) for path in selection.paths],
                "path_coefficients": [
                    float(selection.coefficients[path]) for path in selection.paths
                ],
            }
        )
    return rows


def aggregate_random_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    random_rows = [row for row in rows if row["family"] == "random7_equal"]
    accuracies = np.asarray(
        [row["accuracy_percent"] for row in random_rows], dtype=np.float64
    )
    return {
        "count": len(random_rows),
        "accuracy_mean_percent": float(accuracies.mean()),
        "accuracy_sample_std_percent": float(accuracies.std(ddof=1)),
        "accuracy_min_percent": float(accuracies.min()),
        "accuracy_max_percent": float(accuracies.max()),
        "accuracy_values_percent": accuracies.tolist(),
    }


def write_csv(rows: list[dict[str, Any]], random_summary: dict[str, Any]) -> None:
    fields = [
        "setting",
        "family",
        "num_paths",
        "coefficient_sum",
        "accuracy_percent",
        "test_loss",
        "relative_cost_proxy",
        "active_state_ratio",
        "prefix_counts",
        "parent_counts",
        "path_indices",
        "path_bits",
        "path_coefficients",
    ]
    with OUTPUT_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            for key in (
                "prefix_counts",
                "parent_counts",
                "path_indices",
                "path_bits",
                "path_coefficients",
            ):
                serialized[key] = json.dumps(serialized[key], ensure_ascii=False)
            writer.writerow(serialized)
        writer.writerow(
            {
                "setting": "Random-7 aggregate",
                "family": "random7_aggregate",
                "num_paths": TOPK,
                "coefficient_sum": "",
                "accuracy_percent": random_summary["accuracy_mean_percent"],
                "test_loss": "",
                "relative_cost_proxy": "",
                "active_state_ratio": "",
                "prefix_counts": "",
                "parent_counts": "",
                "path_indices": "",
                "path_bits": "",
                "path_coefficients": "",
            }
        )


def result_by_family(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
    matches = [row for row in rows if row["family"] == family]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {family}, found {len(matches)}")
    return matches[0]


def write_markdown(payload: dict[str, Any]) -> None:
    rows = payload["results"]
    random_summary = payload["random_summary"]
    full = result_by_family(rows, "full")
    top = result_by_family(rows, "top7_learned")
    top_equal = result_by_family(rows, "top7_equal")
    bottom = result_by_family(rows, "bottom7_equal")
    top_paths = payload["selection_metadata"]["top_paths_ranked"]
    bottom_paths = payload["selection_metadata"]["bottom_paths_ranked"]

    lines = [
        "# Seed-42 NetAttn Ranking Control",
        "",
        "## Protocol",
        "",
        f"- Checkpoint: `{payload['checkpoint']['path']}`",
        f"- Checkpoint SHA-256: `{payload['checkpoint']['sha256']}`",
        "- Dataset: CIFAR-100 test split (10,000 images), using the same "
        "normalization as the source notebook and no test-time augmentation.",
        f"- Environment: `{payload['environment']['python']}`; "
        f"PyTorch `{payload['environment']['torch']}`; "
        f"device `{payload['environment']['device']}`.",
        "- Path order: the eight bits are ordered from shallow to deep as "
        "`L1.B0, L1.B1, L2.B0, L2.B1, L3.B0, L3.B1, L4.B0, L4.B1`; "
        "`11111100` therefore skips the last two residual transformations. "
        "The stable internal index retains its historical encoding for "
        "checkpoint compatibility.",
        "- Full NetAttn uses the checkpoint softmax coefficients, whose sum is "
        "one. Matching the source notebook, Top-7 is ranked and weighted by "
        "`attention_final.csv` and retains those seven learned coefficients "
        "without renormalization.",
        f"- Equal mass-matched controls assign each selected path "
        f"`M7/7 = {payload['selection_metadata']['equal_coefficient']:.9f}`, "
        f"where `M7 = {payload['selection_metadata']['top7_attention_sum']:.9f}`.",
        f"- Random control: {payload['selection_metadata']['random_sampling']} "
        f"The fixed RNG seed is {RANDOM_SEED}.",
        "- The active-state ratio is the sum of unique active states after the "
        "eight residual blocks divided by the eight states of ResNet-18. The "
        "relative MAC proxy additionally weights parent states by block "
        "convolution cost. Neither quantity is a wall-clock measurement. All "
        "random sets have exactly the same prefix profile and both cost proxies "
        "as Top-7.",
        "- All coefficient settings are evaluated from the same terminal path "
        "features in each batch. This avoids redundant full-family backbone "
        "passes and does not change the resulting weighted features or logits.",
        "",
        "## Results",
        "",
        "| Setting | Paths | Coeff. sum | Active states | MAC proxy | Accuracy (%) |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Full NetAttn | 256 | {full['coefficient_sum']:.6f} | "
        f"{full['active_state_ratio']:.3f}x | {full['relative_cost_proxy']:.3f}x | "
        f"{full['accuracy_percent']:.3f} |",
        f"| NetAttn Top-7, learned | 7 | {top['coefficient_sum']:.6f} | "
        f"{top['active_state_ratio']:.3f}x | {top['relative_cost_proxy']:.3f}x | "
        f"{top['accuracy_percent']:.3f} |",
        f"| Top-7, equal mass-matched | 7 | {top_equal['coefficient_sum']:.6f} | "
        f"{top_equal['active_state_ratio']:.3f}x | "
        f"{top_equal['relative_cost_proxy']:.3f}x | "
        f"{top_equal['accuracy_percent']:.3f} |",
        f"| Structure-matched Random-7, equal mass-matched | 7 | "
        f"{top_equal['coefficient_sum']:.6f} | "
        f"{top_equal['active_state_ratio']:.3f}x | "
        f"{top_equal['relative_cost_proxy']:.3f}x | "
        f"{random_summary['accuracy_mean_percent']:.3f} +/- "
        f"{random_summary['accuracy_sample_std_percent']:.3f} |",
        f"| Bottom-7, equal mass-matched | 7 | {bottom['coefficient_sum']:.6f} | "
        f"{bottom['active_state_ratio']:.3f}x | "
        f"{bottom['relative_cost_proxy']:.3f}x | "
        f"{bottom['accuracy_percent']:.3f} |",
        "",
        "The random value is the mean and sample standard deviation over "
        f"{random_summary['count']} structure-matched path sets. Its observed "
        f"range is {random_summary['accuracy_min_percent']:.3f}% to "
        f"{random_summary['accuracy_max_percent']:.3f}%.",
        "",
        "## Selected Paths",
        "",
        "### NetAttn Top-7 (rank order)",
        "",
    ]
    for rank, index in enumerate(top_paths, start=1):
        lines.append(
            f"{rank}. `{bits(index)}` (index {index}, learned coefficient "
            f"{payload['learned_weights'][index]:.9f}): {decode_path(index)}"
        )
    lines.extend(["", "### NetAttn Bottom-7 (rank order)", ""])
    for rank, index in enumerate(bottom_paths, start=1):
        lines.append(
            f"{rank}. `{bits(index)}` (index {index}, learned coefficient "
            f"{payload['learned_weights'][index]:.9f}): {decode_path(index)}"
        )
    lines.extend(
        [
            "",
            "The 20 Random-7 path sets, XOR masks, individual accuracies, and "
            "coefficient vectors are stored in the accompanying JSON and CSV.",
            "",
            "## Validation and Limitations",
            "",
            f"- Maximum absolute difference between checkpoint-derived and CSV "
            f"attention coefficients: "
            f"`{payload['validation']['checkpoint_csv_max_abs_difference']:.3e}`.",
            f"- Full-checkpoint accuracy reproduced here: "
            f"`{full['accuracy_percent']:.3f}%`.",
            "- This is a single-checkpoint control. The 20 random sets quantify "
            "subset-selection variability within seed 42, not training-seed "
            "variability.",
            "- The structure-matched random sets are conditional controls, not "
            "uniform samples from all possible 7-of-256 subsets. The XOR "
            "construction deliberately holds prefix sharing and the cost proxy "
            "constant.",
            "- Accuracy comparisons isolate path identity and coefficient "
            "assignment for this jointly trained mixture. They do not measure "
            "standalone subnetwork accuracy or prove globally optimal path "
            "selection.",
            "- The reported cost is analytic. No wall-clock latency or reduced "
            "CUDA-memory measurement was performed in this control.",
            "",
        ]
    )
    OUTPUT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    # The shared CIFAR loader resolves its server path relative to the notebook
    # directory, so preserve the notebook's working-directory convention.
    os.chdir(SCRIPT_DIR)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full 256-path evaluation")
    device = torch.device("cuda:0")

    attention_frame = load_attention_table()
    model = ResNet_NetAttn_Config_CIFAR100(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        in_channels=3,
        num_classes=100,
        zero_init_residual=False,
        attention_total_budget=1.0,
        enforce_attention_total_budget=True,
    )
    loaded = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    state_dict = extract_state_dict(loaded)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    learned_weights_tensor = torch.softmax(
        model.subnetwork_attention_logits.detach().float(), dim=0
    )
    learned_weights = learned_weights_tensor.cpu().numpy()
    csv_weights = attention_frame["weight"].to_numpy(dtype=np.float64)
    max_difference = float(
        np.max(np.abs(learned_weights.astype(np.float64) - csv_weights))
    )
    if max_difference > 1e-6:
        raise AssertionError(
            f"Checkpoint and CSV attention differ by {max_difference:.3e}"
        )

    _, testloader, _, _, _, _ = get_dataloaders(
        is_server=True,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        use_mixup=False,
        use_randaugment=False,
        use_random_erasing=False,
        mixup_alpha=0.75,
        cutmix_alpha=0.5,
        label_smoothing=0.0,
        randaugment_config="rand-m9-n1-mstd0.4-inc1",
        random_erasing_prob=0.0,
    )

    selections, selection_metadata = build_selections(learned_weights, csv_weights)
    rows = evaluate(model, testloader, selections, device)
    random_summary = aggregate_random_results(rows)

    payload = {
        "experiment": "seed-42 NetAttn ranking control",
        "checkpoint": {
            "path": str(CHECKPOINT_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(CHECKPOINT_PATH),
            "type": "best_model.pth",
            "training_seed": SEED,
        },
        "environment": {
            "python": sys.executable,
            "python_version": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "amp_eval": bool(USE_AMP),
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
        },
        "path_order": {
            "direction": "shallow-to-deep",
            "block_names": BLOCK_NAMES,
            "index_encoding": (
                "subnet_idx uses its least-significant bit for L1.B0; "
                "displayed bits reverse the integer representation so the "
                "leftmost bit is L1.B0 and the rightmost bit is L4.B1"
            ),
            "validated_example": {
                "subnet_idx": 63,
                "bits": "11111100",
                "decoded": decode_path(63),
            },
        },
        "selection_metadata": selection_metadata,
        "cost": {
            "active_state_definition": (
                "sum of unique active states after the eight residual blocks "
                "divided by eight"
            ),
            "definition": relative_cost(tuple(range(NUM_PATHS)))[
                "cost_definition"
            ],
            "cost_matching_tolerance": 0.05,
            "random_sets_exactly_structure_matched": True,
            "wall_clock_measured": False,
        },
        "validation": {
            "checkpoint_csv_max_abs_difference": max_difference,
            "checkpoint_attention_sum": float(learned_weights.sum()),
            "csv_attention_sum": float(csv_weights.sum()),
        },
        "learned_weights": learned_weights.astype(float).tolist(),
        "results": rows,
        "random_summary": random_summary,
        "limitations": [
            "Single trained checkpoint (seed 42).",
            (
                "Random controls use global XOR transforms of the Top-7 set "
                "to preserve its exact prefix-tree structure and cost; they "
                "are not uniform 7-of-256 samples."
            ),
            (
                "Relative cost is an analytic prefix-DAG convolution-MAC "
                "proxy, not measured latency."
            ),
            (
                "The experiment evaluates subsets inside a jointly trained "
                "mixture and does not establish standalone path accuracy or "
                "global optimality."
            ),
        ],
    }

    OUTPUT_JSON_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(rows, random_summary)
    write_markdown(payload)

    print(json.dumps(
        {
            "full_accuracy": result_by_family(rows, "full")["accuracy_percent"],
            "top7_learned_accuracy": result_by_family(
                rows, "top7_learned"
            )["accuracy_percent"],
            "top7_equal_accuracy": result_by_family(
                rows, "top7_equal"
            )["accuracy_percent"],
            "random7_mean_accuracy": random_summary["accuracy_mean_percent"],
            "random7_std_accuracy": random_summary[
                "accuracy_sample_std_percent"
            ],
            "bottom7_accuracy": result_by_family(
                rows, "bottom7_equal"
            )["accuracy_percent"],
            "top7_attention_sum": selection_metadata["top7_attention_sum"],
            "output_markdown": str(OUTPUT_MD_PATH),
            "output_csv": str(OUTPUT_CSV_PATH),
            "output_json": str(OUTPUT_JSON_PATH),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
