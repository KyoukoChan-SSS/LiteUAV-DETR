#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "tag",
    "family",
    "backend",
    "precision",
    "scope",
    "count",
    "mean_ms",
    "std_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "fps",
    "result_dir",
    "note",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def add_torch_rows(rows: list[dict[str, Any]], result_dir: Path) -> None:
    # Covers both protocols: torch_fp32_bc.json (unfused, default) and
    # torch_fp32_fold_bc.json (--fold runs). The fold flag lands in `precision`
    # so the two never blur together in the summary.
    for path in sorted(result_dir.glob("*/torch_fp32*_bc.json")):
        data = read_json(path)
        tag = data.get("meta", {}).get("tag") or path.parent.name
        precision = data.get("meta", {}).get("precision", "fp32")
        if data.get("meta", {}).get("fold"):
            precision = f"{precision}+fold"
        for scope, block_name, stat_name in (
            ("B_inference", "B_inference", "inference_latency"),
            ("C_end2end", "C_end2end", "end2end_total_latency"),
        ):
            block = data.get(block_name, {})
            stat = block.get(stat_name, {})
            rows.append(
                {
                    "tag": tag,
                    "family": "torch_bc",
                    "backend": "torch",
                    "precision": precision,
                    "scope": scope,
                    "count": stat.get("count"),
                    "mean_ms": stat.get("mean_ms"),
                    "std_ms": stat.get("std_ms"),
                    "p50_ms": stat.get("p50_ms"),
                    "p90_ms": stat.get("p90_ms"),
                    "p95_ms": stat.get("p95_ms"),
                    "p99_ms": stat.get("p99_ms"),
                    "fps": stat.get("fps"),
                    "result_dir": str(result_dir),
                    "note": (
                        f"postprocess_errors={block.get('postprocess_errors')}"
                        if block.get("postprocess_errors")
                        else ""
                    ),
                }
            )


def add_artifact_rows(rows: list[dict[str, Any]], result_dir: Path) -> None:
    for path in sorted(result_dir.glob("*/*.json")):
        if path.name.endswith(".trtexec_times.json"):
            continue
        data = read_json(path)
        stat = data.get("latency", {})
        note = ""
        log_path = path.with_suffix(".log")
        if path.name.startswith("onnx") and log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            if "Failed to create CUDAExecutionProvider" in log_text or "libcudnn.so.9" in log_text:
                note = "ONNXRuntime CUDA provider unavailable; likely CPU fallback"
        rows.append(
            {
                "tag": path.parent.name,
                "family": "artifact",
                "backend": data.get("backend", path.stem),
                "precision": data.get("precision", ""),
                "scope": path.stem,
                "count": stat.get("count"),
                "mean_ms": stat.get("mean_ms"),
                "std_ms": stat.get("std_ms"),
                "p50_ms": stat.get("p50_ms"),
                "p90_ms": stat.get("p90_ms"),
                "p95_ms": stat.get("p95_ms"),
                "p99_ms": stat.get("p99_ms"),
                "fps": stat.get("fps"),
                "result_dir": str(result_dir),
                "note": note,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latency result JSON files into one CSV.")
    parser.add_argument("--torch-dir", action="append", default=[], type=Path)
    parser.add_argument("--artifact-dir", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for result_dir in args.torch_dir:
        if result_dir.exists():
            add_torch_rows(rows, result_dir)
    for result_dir in args.artifact_dir:
        if result_dir.exists():
            add_artifact_rows(rows, result_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output} rows={len(rows)}")


if __name__ == "__main__":
    main()
