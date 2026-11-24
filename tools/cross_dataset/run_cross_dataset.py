#!/usr/bin/env python3
"""Run and summarize the VisDrone-to-OOD cross-dataset protocol.

The orchestrator is intentionally standard-library only.  Every target is
evaluated in a fresh worker process so CUDA/model state cannot leak between
datasets, and the existing training, single-dataset evaluation, and latency
entry points remain untouched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LITE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = LITE_ROOT / "configs/cross_dataset/visdrone_to_ood8.json"
WORKER = Path(__file__).with_name("evaluate_one.py")
MODEL_CONFIGS = {
    "ultrafast": LITE_ROOT
    / "configs/liteuavdetr/custom/liteuavdetr_ultrafast_visdrone_640.yml",
    "fast": LITE_ROOT / "configs/liteuavdetr/custom/liteuavdetr_fast_visdrone_640.yml",
}
METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)
SAFE_DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ProtocolError(RuntimeError):
    """Raised when the manifest or local dataset violates the protocol."""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    images: Path
    annotations: Path
    expected_images: int | None
    expected_annotations_sha256: str | None
    expected_annotation_category_ids: tuple[int, ...]


@dataclass(frozen=True)
class Protocol:
    name: str
    model_num_classes: int
    categories: dict[int, str]
    datasets: tuple[DatasetSpec, ...]


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise ProtocolError(f"Missing file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"Invalid JSON in {path}: {exc}") from exc


def _resolve_path(value: str, root: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute():
        expanded = root / expanded
    return expanded.resolve()


def load_protocol(
    manifest_path: Path,
    dataset_root: Path,
    selected_names: Iterable[str] | None = None,
) -> Protocol:
    raw = _load_json(manifest_path)
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ProtocolError("Manifest schema_version must be 1")

    protocol_name = raw.get("protocol")
    if not isinstance(protocol_name, str) or not protocol_name:
        raise ProtocolError("Manifest protocol must be a non-empty string")

    category_space = raw.get("category_space")
    if not isinstance(category_space, dict):
        raise ProtocolError("Manifest category_space must be an object")
    if category_space.get("remap_mscoco_category") is not False:
        raise ProtocolError("Cross-dataset category_space must disable MS COCO remapping")
    try:
        model_num_classes = int(category_space["model_num_classes"])
        categories = {
            int(item["id"]): str(item["name"])
            for item in category_space["categories"]
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"Invalid category_space: {exc}") from exc
    if not categories or len(categories) != len(category_space["categories"]):
        raise ProtocolError("Category IDs must be non-empty and unique")

    raw_datasets = raw.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ProtocolError("Manifest datasets must be a non-empty list")

    requested = set(selected_names or ())
    known_names: set[str] = set()
    datasets: list[DatasetSpec] = []
    for item in raw_datasets:
        if not isinstance(item, dict):
            raise ProtocolError("Each dataset entry must be an object")
        name = str(item.get("name", ""))
        if not SAFE_DATASET_NAME.fullmatch(name):
            raise ProtocolError(f"Unsafe or empty dataset name: {name!r}")
        if name in known_names:
            raise ProtocolError(f"Duplicate dataset name: {name}")
        known_names.add(name)
        if requested and name not in requested:
            continue
        try:
            expected_ids = tuple(sorted({int(v) for v in item["expected_annotation_category_ids"]}))
            images = _resolve_path(str(item["images"]), dataset_root)
            annotations = _resolve_path(str(item["annotations"]), dataset_root)
            expected_images = item.get("expected_images")
            expected_images = None if expected_images is None else int(expected_images)
            expected_hash = item.get("expected_annotations_sha256")
            expected_hash = None if expected_hash is None else str(expected_hash).lower()
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"Invalid dataset entry {name}: {exc}") from exc
        if not expected_ids or not set(expected_ids).issubset(categories):
            raise ProtocolError(
                f"Dataset {name} declares IDs outside the protocol category space: {expected_ids}"
            )
        if expected_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
            raise ProtocolError(f"Dataset {name} has an invalid annotation SHA-256")
        datasets.append(
            DatasetSpec(
                name=name,
                images=images,
                annotations=annotations,
                expected_images=expected_images,
                expected_annotations_sha256=expected_hash,
                expected_annotation_category_ids=expected_ids,
            )
        )

    unknown = requested - known_names
    if unknown:
        raise ProtocolError(f"Unknown dataset selection: {', '.join(sorted(unknown))}")
    if not datasets:
        raise ProtocolError("Dataset selection is empty")
    return Protocol(
        name=protocol_name,
        model_num_classes=model_num_classes,
        categories=categories,
        datasets=tuple(datasets),
    )


def validate_dataset(
    dataset: DatasetSpec,
    categories: dict[int, str],
    *,
    strict_counts: bool = True,
) -> dict[str, Any]:
    if not dataset.images.is_dir():
        raise ProtocolError(f"{dataset.name}: missing image directory: {dataset.images}")
    if not dataset.annotations.is_file():
        raise ProtocolError(f"{dataset.name}: missing annotation file: {dataset.annotations}")

    coco = _load_json(dataset.annotations)
    if not isinstance(coco, dict):
        raise ProtocolError(f"{dataset.name}: COCO annotation root must be an object")
    images = coco.get("images")
    annotations = coco.get("annotations")
    coco_categories = coco.get("categories")
    if not isinstance(images, list) or not images:
        raise ProtocolError(f"{dataset.name}: COCO images must be a non-empty list")
    if not isinstance(annotations, list):
        raise ProtocolError(f"{dataset.name}: COCO annotations must be a list")
    if not isinstance(coco_categories, list):
        raise ProtocolError(f"{dataset.name}: COCO categories must be a list")

    try:
        observed_categories = {
            int(item["id"]): str(item["name"])
            for item in coco_categories
        }
        annotation_ids = tuple(sorted({int(item["category_id"]) for item in annotations}))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolError(f"{dataset.name}: invalid COCO category data: {exc}") from exc

    if observed_categories != categories:
        raise ProtocolError(
            f"{dataset.name}: category table does not exactly match VisDrone IDs/names; "
            f"observed={observed_categories}"
        )
    if annotation_ids != dataset.expected_annotation_category_ids:
        raise ProtocolError(
            f"{dataset.name}: annotation category IDs {annotation_ids} do not match "
            f"manifest {dataset.expected_annotation_category_ids}"
        )
    if (
        strict_counts
        and dataset.expected_images is not None
        and len(images) != dataset.expected_images
    ):
        raise ProtocolError(
            f"{dataset.name}: expected {dataset.expected_images} images, found {len(images)}"
        )
    annotation_sha256 = sha256_file(dataset.annotations)
    if (
        dataset.expected_annotations_sha256 is not None
        and annotation_sha256 != dataset.expected_annotations_sha256
    ):
        raise ProtocolError(
            f"{dataset.name}: annotation SHA-256 mismatch; "
            f"expected {dataset.expected_annotations_sha256}, found {annotation_sha256}"
        )

    return {
        "name": dataset.name,
        "images_directory": str(dataset.images),
        "annotations_file": str(dataset.annotations),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "annotation_category_ids": list(annotation_ids),
        "annotations_sha256": annotation_sha256,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=LITE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_worker_command(
    *,
    python: str,
    protocol: Protocol,
    dataset: DatasetSpec,
    config: Path,
    checkpoint: Path,
    output_dir: Path,
    device: str,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> list[str]:
    dataset_dir = output_dir / dataset.name
    return [
        python,
        str(WORKER),
        "--dataset-name",
        dataset.name,
        "--protocol",
        protocol.name,
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--images",
        str(dataset.images),
        "--annotations",
        str(dataset.annotations),
        "--output",
        str(dataset_dir / "metrics.json"),
        "--work-dir",
        str(dataset_dir / "work"),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--expected-num-classes",
        str(protocol.model_num_classes),
        "--seed",
        str(seed),
    ]


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=LITE_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        return process.wait()


def validate_resumable_result(
    result_path: Path,
    *,
    protocol: Protocol,
    dataset: DatasetSpec,
    config: Path,
    checkpoint: Path,
) -> None:
    result = _load_json(result_path)
    expected = {
        "status": "ok",
        "protocol": protocol.name,
        "dataset": dataset.name,
        "config": str(config.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "images": str(dataset.images.resolve()),
        "annotations": str(dataset.annotations.resolve()),
    }
    mismatches = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if result.get(key) != value
    }
    metrics = result.get("metrics")
    if not isinstance(metrics, dict) or not set(METRIC_NAMES).issubset(metrics):
        mismatches["metrics"] = (metrics, f"all of {METRIC_NAMES}")
    if mismatches:
        details = "; ".join(
            f"{key}: found={found!r}, expected={wanted!r}"
            for key, (found, wanted) in mismatches.items()
        )
        raise ProtocolError(f"Cannot resume incompatible result {result_path}: {details}")


def _average(values: Iterable[float | None]) -> float | None:
    defined = [float(value) for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def write_summaries(
    output_dir: Path,
    protocol: Protocol,
    preflight: dict[str, dict[str, Any]],
    failures: dict[str, str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for dataset in protocol.datasets:
        metrics_path = output_dir / dataset.name / "metrics.json"
        metrics: dict[str, float | None] = {}
        status = "failed" if dataset.name in failures else "not_run"
        if metrics_path.is_file() and dataset.name not in failures:
            result = _load_json(metrics_path)
            metrics = result.get("metrics", {})
            status = str(result.get("status", "ok"))
        info = preflight.get(dataset.name, {})
        row: dict[str, Any] = {
            "dataset": dataset.name,
            "status": status,
            "image_count": info.get("image_count"),
            "annotation_count": info.get("annotation_count"),
        }
        row.update({name: metrics.get(name) for name in METRIC_NAMES})
        row["error"] = failures.get(dataset.name)
        rows.append(row)

    successful = [row for row in rows if row["status"] == "ok"]
    macro = {
        name: _average(row[name] for row in successful)
        for name in METRIC_NAMES
    }
    summary = {
        "schema_version": 1,
        "protocol": protocol.name,
        "successful_datasets": len(successful),
        "requested_datasets": len(protocol.datasets),
        "per_dataset": rows,
        "macro_average": macro,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fieldnames = [
        "dataset",
        "status",
        "image_count",
        "annotation_count",
        *METRIC_NAMES,
        "error",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def percentage(value: Any) -> str:
        return "-" if value is None else f"{float(value) * 100:.2f}"

    lines = [
        f"# {protocol.name}",
        "",
        "COCO AP/AR values below are percentages; undefined size bins are shown as `-`.",
        "",
        "| Dataset | Status | Images | AP | AP50 | AP75 | APs | APm | APl |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {status} | {images} | {ap} | {ap50} | {ap75} | {aps} | {apm} | {apl} |".format(
                dataset=row["dataset"],
                status=row["status"],
                images=row["image_count"] if row["image_count"] is not None else "-",
                ap=percentage(row["AP"]),
                ap50=percentage(row["AP50"]),
                ap75=percentage(row["AP75"]),
                aps=percentage(row["APs"]),
                apm=percentage(row["APm"]),
                apl=percentage(row["APl"]),
            )
        )
    lines.extend(
        [
            "",
            "| Macro average | Successful datasets | AP | AP50 | AP75 | APs | APm | APl |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
            "| Mean | {count} | {ap} | {ap50} | {ap75} | {aps} | {apm} | {apl} |".format(
                count=len(successful),
                ap=percentage(macro["AP"]),
                ap50=percentage(macro["AP50"]),
                ap75=percentage(macro["AP75"]),
                aps=percentage(macro["APs"]),
                apm=percentage(macro["APm"]),
                apl=percentage(macro["APl"]),
            ),
            "",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def _selected_names(value: str | None) -> list[str] | None:
    if value is None:
        return None
    names = [name.strip() for name in value.split(",") if name.strip()]
    if not names:
        raise ProtocolError("--datasets did not contain a dataset name")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a VisDrone-trained LiteUAVDETR release checkpoint on the "
            "strict eight-dataset zero-shot protocol."
        )
    )
    parser.add_argument("--model-variant", choices=sorted(MODEL_CONFIGS), default="fast")
    parser.add_argument("--config", type=Path, help="Override the matching VisDrone config")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=os.environ.get("LITEUAVDETR_DATASET_ROOT"),
        help="Root containing DroneVehicle, OOD, MAVREC, and VTUAV_v1.0",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--datasets", help="Comma-separated subset, e.g. DroneVehicle,UAVDT")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without loading torch/GPU")
    parser.add_argument(
        "--no-strict-counts",
        action="store_true",
        help="Allow alternate image counts while keeping category validation strict",
    )
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument("--resume", action="store_true", help="Reuse completed metrics.json files")
    behavior.add_argument("--overwrite", action="store_true", help="Replace existing per-dataset results")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later datasets if one worker fails",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dataset_root is None:
        raise ProtocolError(
            "Set --dataset-root or LITEUAVDETR_DATASET_ROOT to the directory containing the OOD datasets"
        )
    if args.batch_size < 1:
        raise ProtocolError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ProtocolError("--num-workers must be non-negative")

    manifest = _resolve_path(str(args.manifest), LITE_ROOT)
    dataset_root = _resolve_path(str(args.dataset_root), Path.cwd())
    config = _resolve_path(str(args.config or MODEL_CONFIGS[args.model_variant]), LITE_ROOT)
    checkpoint = _resolve_path(str(args.checkpoint), LITE_ROOT)
    for label, path in (("manifest", manifest), ("config", config), ("checkpoint", checkpoint)):
        if not path.is_file():
            raise ProtocolError(f"Missing {label}: {path}")

    protocol = load_protocol(manifest, dataset_root, _selected_names(args.datasets))
    preflight: dict[str, dict[str, Any]] = {}
    for dataset in protocol.datasets:
        info = validate_dataset(
            dataset,
            protocol.categories,
            strict_counts=not args.no_strict_counts,
        )
        preflight[dataset.name] = info
        print(
            f"[preflight] {dataset.name}: {info['image_count']} images, "
            f"{info['annotation_count']} annotations, IDs={info['annotation_category_ids']}"
        )

    if args.dry_run:
        print(f"Preflight passed: {len(protocol.datasets)} dataset(s); no model was loaded.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        _resolve_path(str(args.output_dir), LITE_ROOT)
        if args.output_dir
        else LITE_ROOT / "output/cross_dataset" / f"{args.model_variant}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "schema_version": 1,
        "protocol": protocol.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "model_variant": args.model_variant,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "dataset_root": str(dataset_root),
        "device": args.device,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "datasets": [dataset.name for dataset in protocol.datasets],
        "preflight": preflight,
    }
    (output_dir / "run.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    failures: dict[str, str] = {}
    for dataset in protocol.datasets:
        dataset_dir = output_dir / dataset.name
        result_path = dataset_dir / "metrics.json"
        if result_path.exists() and args.resume:
            try:
                validate_resumable_result(
                    result_path,
                    protocol=protocol,
                    dataset=dataset,
                    config=config,
                    checkpoint=checkpoint,
                )
            except ProtocolError as exc:
                failures[dataset.name] = str(exc)
                print(f"[failed] {dataset.name}: {exc}", file=sys.stderr)
                if not args.continue_on_error:
                    break
                continue
            print(f"[resume] {dataset.name}: verified {result_path}")
            continue
        if result_path.exists() and not args.overwrite:
            failures[dataset.name] = (
                f"Result exists: {result_path}; use --resume or --overwrite"
            )
            print(f"[failed] {dataset.name}: {failures[dataset.name]}", file=sys.stderr)
            if not args.continue_on_error:
                break
            continue
        if result_path.exists() and args.overwrite:
            result_path.unlink()

        command = build_worker_command(
            python=args.python,
            protocol=protocol,
            dataset=dataset,
            config=config,
            checkpoint=checkpoint,
            output_dir=output_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )
        print(f"[evaluate] {dataset.name}")
        returncode = run_logged(command, dataset_dir / "evaluate.log")
        if returncode != 0 or not result_path.is_file():
            failures[dataset.name] = f"worker exited with code {returncode}"
            print(f"[failed] {dataset.name}: {failures[dataset.name]}", file=sys.stderr)
            if not args.continue_on_error:
                break

    summary = write_summaries(output_dir, protocol, preflight, failures)
    print(f"Summary: {output_dir / 'summary.md'}")
    if failures or summary["successful_datasets"] != len(protocol.datasets):
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
