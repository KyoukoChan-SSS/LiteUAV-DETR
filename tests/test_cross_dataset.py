from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


LITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LITE_ROOT / "tools/cross_dataset"))

import run_cross_dataset as cross  # noqa: E402
import evaluate_one as worker  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class CrossDatasetProtocolTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path]:
        dataset_root = root / "datasets"
        images = dataset_root / "images"
        images.mkdir(parents=True)
        annotations = dataset_root / "annotations.json"
        categories = [
            {"id": 1, "name": "pedestrian"},
            {"id": 4, "name": "car"},
        ]
        write_json(
            annotations,
            {
                "images": [{"id": 10, "file_name": "one.jpg", "width": 8, "height": 8}],
                "annotations": [
                    {"id": 1, "image_id": 10, "category_id": 4, "bbox": [0, 0, 2, 2]}
                ],
                "categories": categories,
            },
        )
        manifest = root / "manifest.json"
        write_json(
            manifest,
            {
                "schema_version": 1,
                "protocol": "test_protocol",
                "category_space": {
                    "model_num_classes": 12,
                    "remap_mscoco_category": False,
                    "categories": categories,
                },
                "datasets": [
                    {
                        "name": "Target",
                        "images": "images",
                        "annotations": "annotations.json",
                        "expected_images": 1,
                        "expected_annotation_category_ids": [4],
                    }
                ],
            },
        )
        return dataset_root, manifest

    def test_load_and_validate_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root, manifest = self.make_fixture(root)
            protocol = cross.load_protocol(manifest, dataset_root)
            self.assertEqual(protocol.name, "test_protocol")
            self.assertEqual(protocol.model_num_classes, 12)
            self.assertEqual([item.name for item in protocol.datasets], ["Target"])

            info = cross.validate_dataset(protocol.datasets[0], protocol.categories)
            self.assertEqual(info["image_count"], 1)
            self.assertEqual(info["annotation_count"], 1)
            self.assertEqual(info["annotation_category_ids"], [4])

    def test_worker_maps_all_twelve_coco_metrics(self) -> None:
        values = [index / 100 for index in range(12)]
        values[5] = -1.0
        metrics = worker._metric_dict(values)
        self.assertEqual(tuple(metrics), worker.COCO_BBOX_METRICS)
        self.assertEqual(metrics["AP50"], 0.01)
        self.assertIsNone(metrics["APl"])

    def test_unknown_selection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root, manifest = self.make_fixture(root)
            with self.assertRaisesRegex(cross.ProtocolError, "Unknown dataset"):
                cross.load_protocol(manifest, dataset_root, ["Missing"])

    def test_wrong_annotation_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_root, manifest = self.make_fixture(root)
            annotation_path = dataset_root / "annotations.json"
            value = json.loads(annotation_path.read_text(encoding="utf-8"))
            value["annotations"][0]["category_id"] = 1
            write_json(annotation_path, value)
            protocol = cross.load_protocol(manifest, dataset_root)
            with self.assertRaisesRegex(cross.ProtocolError, "do not match manifest"):
                cross.validate_dataset(protocol.datasets[0], protocol.categories)

    def test_summary_excludes_undefined_metrics_from_macro(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            dataset = cross.DatasetSpec(
                name="Target",
                images=output / "images",
                annotations=output / "annotations.json",
                expected_images=1,
                expected_annotations_sha256=None,
                expected_annotation_category_ids=(4,),
            )
            protocol = cross.Protocol(
                name="test_protocol",
                model_num_classes=12,
                categories={4: "car"},
                datasets=(dataset,),
            )
            metrics = {name: 0.1 for name in cross.METRIC_NAMES}
            metrics["APl"] = None
            write_json(
                output / "Target/metrics.json",
                {"status": "ok", "metrics": metrics},
            )
            summary = cross.write_summaries(
                output,
                protocol,
                {"Target": {"image_count": 1, "annotation_count": 1}},
                {},
            )
            self.assertEqual(summary["successful_datasets"], 1)
            self.assertAlmostEqual(summary["macro_average"]["AP"], 0.1)
            self.assertIsNone(summary["macro_average"]["APl"])
            self.assertTrue((output / "summary.csv").is_file())
            self.assertIn("10.00", (output / "summary.md").read_text(encoding="utf-8"))

    def test_worker_command_keeps_dataset_and_output_isolated(self) -> None:
        dataset = cross.DatasetSpec(
            name="Target",
            images=Path("/datasets/images"),
            annotations=Path("/datasets/annotations.json"),
            expected_images=1,
            expected_annotations_sha256=None,
            expected_annotation_category_ids=(4,),
        )
        protocol = cross.Protocol(
            name="test_protocol",
            model_num_classes=12,
            categories={4: "car"},
            datasets=(dataset,),
        )
        command = cross.build_worker_command(
            python="python",
            protocol=protocol,
            dataset=dataset,
            config=Path("/repo/config.yml"),
            checkpoint=Path("/repo/model.pth"),
            output_dir=Path("/results"),
            device="cuda:0",
            batch_size=1,
            num_workers=2,
            seed=7,
        )
        self.assertIn("/results/Target/metrics.json", command)
        self.assertIn("/results/Target/work", command)
        self.assertIn("test_protocol", command)

    def test_resume_rejects_a_different_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "metrics.json"
            dataset = cross.DatasetSpec(
                name="Target",
                images=root / "images",
                annotations=root / "annotations.json",
                expected_images=1,
                expected_annotations_sha256=None,
                expected_annotation_category_ids=(4,),
            )
            protocol = cross.Protocol(
                name="test_protocol",
                model_num_classes=12,
                categories={4: "car"},
                datasets=(dataset,),
            )
            write_json(
                result_path,
                {
                    "status": "ok",
                    "protocol": "test_protocol",
                    "dataset": "Target",
                    "config": str((root / "config.yml").resolve()),
                    "checkpoint": str((root / "old.pth").resolve()),
                    "images": str(dataset.images.resolve()),
                    "annotations": str(dataset.annotations.resolve()),
                    "metrics": {name: 0.1 for name in cross.METRIC_NAMES},
                },
            )
            with self.assertRaisesRegex(cross.ProtocolError, "different_checkpoint.pth"):
                cross.validate_resumable_result(
                    result_path,
                    protocol=protocol,
                    dataset=dataset,
                    config=root / "config.yml",
                    checkpoint=root / "different_checkpoint.pth",
                )


if __name__ == "__main__":
    unittest.main()
