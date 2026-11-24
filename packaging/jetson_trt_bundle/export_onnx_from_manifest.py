#!/usr/bin/env python3
"""Export missing LiteUAVDETR post-mode ONNX files listed in the manifest.

LiteUAVDETR model implementation is adapted from https://github.com/Peterande/D-FINE.
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


BUNDLE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = BUNDLE_DIR / "trt_manifest.json"
DEFAULT_REPO = BUNDLE_DIR.parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export manifest ONNX files before Jetson TensorRT build.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def resolve(repo: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else repo / p


def write_status(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["tag", "status", "onnx", "config", "weights", "note"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    repo = args.repo.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = set(args.only or [])
    rows: list[dict[str, Any]] = []

    for model in manifest.get("models", []):
        tag = model["tag"]
        if selected and tag not in selected:
            continue
        onnx = resolve(manifest_path.parent, model["onnx"])
        config = resolve(repo, model["config"])
        weights = resolve(repo, model["weights"])
        image_size = int(model.get("image_size", 640))
        row = {
            "tag": tag,
            "onnx": str(onnx),
            "config": str(config),
            "weights": str(weights),
            "status": "pending",
        }
        if onnx.exists() and not args.force:
            row.update(status="existing")
            rows.append(row)
            continue
        missing = [str(p) for p in (config, weights) if not p.exists()]
        cmd = [
            args.python,
            str(repo / "tools/deployment/export_liteuavdetr_onnx_precision.py"),
            "--config", str(config),
            "--resume", str(weights),
            "--output", str(onnx),
            "--precision", "fp32",
            "--image-size", str(image_size),
            "--batch-size", "1",
        ]
        line = " ".join(shlex.quote(x) for x in cmd)
        print(f"[export] {tag}: {line}")
        if args.dry_run:
            note = line if not missing else f"{line}; missing={';'.join(missing)}"
            row.update(status="dry_run", note=note)
            rows.append(row)
            continue
        if missing:
            row.update(status="missing_input", note=";".join(missing))
            rows.append(row)
            continue
        onnx.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        proc = subprocess.run(cmd, cwd=str(repo), text=True)
        row.update(
            status=proc.returncode,
            note=f"elapsed_s={time.time() - started:.3f}",
        )
        rows.append(row)

    status_path = manifest_path.parent / "onnx_export_status.csv"
    write_status(status_path, rows)
    print(f"[status] {status_path}")
    failures = [r for r in rows if r["status"] not in {0, "existing", "dry_run"}]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
