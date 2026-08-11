"""Convert retained experiment notebooks to an ANN-only execution path."""

from __future__ import annotations

import ast
import copy
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIRS = (
    ROOT / "exp01_baselines",
    ROOT / "exp02_full_netattn",
    ROOT / "exp04_path",
)


def _condition_text(node: ast.If, lines: list[str]) -> str:
    return "\n".join(lines[node.test.lineno - 1 : node.test.end_lineno])


def _body_text(nodes: list[ast.stmt], lines: list[str]) -> str:
    if not nodes:
        return ""
    start = nodes[0].lineno - 1
    end = nodes[-1].end_lineno
    body = lines[start:end]
    result: list[str] = []
    for line in body:
        if line.startswith("    "):
            result.append(line[4:])
        else:
            result.append(line)
    return "\n".join(result)


def _is_steps_assignment(node: ast.If) -> bool:
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Assign):
        return False
    return any(isinstance(target, ast.Name) and target.id == "steps_inputs" for target in node.body[0].targets)


def _else_calls_test(node: ast.If) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "test"
        for statement in node.orelse
        for child in ast.walk(statement)
    )


def _structural_edits(source: str) -> str:
    tree = ast.parse(source)
    lines = source.splitlines()
    edits: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        condition = _condition_text(node, lines)

        if isinstance(node.test, ast.Name) and node.test.id == "IS_ANN":
            replacement = "" if _is_steps_assignment(node) else _body_text(node.body, lines)
            edits.append((node.lineno - 1, node.end_lineno, replacement))
            continue

        if "IS_MONITOR" in condition or re.search(r"\bis_monitor\b", condition):
            replacement = _body_text(node.orelse, lines) if _else_calls_test(node) else ""
            edits.append((node.lineno - 1, node.end_lineno, replacement))
            continue

        if "FORCE_CPU" in condition:
            block = "\n".join(lines[node.lineno - 1 : node.end_lineno])
            if "args.backend" in block or "BACKEND" in block:
                edits.append((node.lineno - 1, node.end_lineno, ""))

    # An outer deleted/replaced block makes edits inside it redundant.
    selected: list[tuple[int, int, str]] = []
    for edit in sorted(edits, key=lambda item: (item[0], -item[1])):
        if any(start <= edit[0] and edit[1] <= end for start, end, _ in selected):
            continue
        selected.append(edit)

    for start, end, replacement in sorted(selected, reverse=True):
        replacement_lines = replacement.splitlines() if replacement else []
        lines[start:end] = replacement_lines
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")


def _clean_lines(source: str) -> str:
    source = source.replace("if IS_ANN and hasattr(", "if hasattr(")
    source = source.replace("net(steps_inputs).mean(0)", "net(inputs)")
    source = source.replace("net(steps_inputs)", "net(inputs)")
    source = source.replace("def test(epoch, is_monitor):", "def test(epoch):")
    source = re.sub(r"test\(([^,\n]+),\s*is_monitor\s*=\s*False\)", r"test(\1)", source)
    source = re.sub(r"test\(([^,\n]+),\s*False\)", r"test(\1)", source)
    source = source.replace("is_dynamic_dataset", "_")

    assignments = (
        "FORCE_TORCH_BACKEND =",
        "IS_MONITOR =",
        "IS_ANN =",
        "TIMESTEPS =",
        "timesteps =",
        "backend =",
    )
    comment_markers = (
        "snn",
        "sew",
        "spiking rate",
        "firing rate",
        "firing_rate",
        "发放率",
        "step_mode",
        "backend",
        "多步模式",
        "时间步",
    )
    code_markers = ("backend     :", "backend:", "firing rate", "firing_rate", "发放率")

    cleaned: list[str] = []
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        lowered = stripped.lower()
        if stripped in {"#     connect_f = 'ADD',", "# )"}:
            continue
        if any(stripped.startswith(prefix) for prefix in assignments):
            continue
        if any(marker in lowered for marker in code_markers):
            continue
        if stripped.startswith("#") and any(marker in lowered for marker in comment_markers):
            continue
        if "#" in line:
            code, comment = line.split("#", 1)
            if any(marker in comment.lower() for marker in comment_markers):
                line = code.rstrip() + ("\n" if line.endswith("\n") else "")
        line = line.rstrip(" \t\r\n") + ("\n" if line.endswith(("\n", "\r")) else "")
        if not line.strip() and cleaned and not cleaned[-1].strip():
            continue
        cleaned.append(line)

    result = "".join(cleaned)
    result = re.sub(r"\n{4,}", "\n\n\n", result)
    return result


def clean_code(source: str) -> str:
    source = _structural_edits(source)
    source = _clean_lines(source)
    ast.parse(source)
    return source


def main() -> None:
    paths = sorted(path for directory in EXPERIMENT_DIRS for path in directory.glob("*.ipynb"))
    changed: list[Path] = []

    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        if path.name == "Full-NetAttn-Uniform-Training-VRAM.ipynb":
            continue

        preserved = [
            (cell.get("execution_count"), copy.deepcopy(cell.get("outputs", [])))
            for cell in notebook.get("cells", [])
        ]
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            cleaned = clean_code(source)
            cell["source"] = cleaned.splitlines(keepends=True)

        current = [
            (cell.get("execution_count"), cell.get("outputs", []))
            for cell in notebook.get("cells", [])
        ]
        if current != preserved:
            raise AssertionError(f"outputs or execution counts changed: {path}")

        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        changed.append(path)
        print(path.relative_to(ROOT))

    if len(changed) != 9:
        raise AssertionError(f"expected 9 changed notebooks, got {len(changed)}")
    print(f"updated notebooks: {len(changed)}")


if __name__ == "__main__":
    main()
