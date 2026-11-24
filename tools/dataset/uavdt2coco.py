"""Convert the official UAVDT detection train/test split to release COCO JSON.

The source parsing follows the original project converter. The release format
uses contiguous class IDs so a three-class LiteUAVDETR head is valid:
car=0, truck=1, bus=2. Ignore-region files are intentionally excluded, matching
the supervised results, while out-of-view and occluded boxes remain included.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image


SOURCE_TO_RELEASE_ID = {1: 0, 2: 1, 3: 2}
CATEGORIES = [
    {"id": 0, "name": "car", "supercategory": "vehicle"},
    {"id": 1, "name": "truck", "supercategory": "vehicle"},
    {"id": 2, "name": "bus", "supercategory": "vehicle"},
]
EXPECTED_IMAGES = {"train": 24_143, "test": 16_592}


def parse_args() -> argparse.Namespace:
    """Parse raw UAVDT locations and the destination directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing M*/img*.jpg sequences.",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        required=True,
        help="Directory containing M*_gt_whole.txt files.",
    )
    parser.add_argument(
        "--split",
        type=Path,
        required=True,
        help="Extracted M_attr directory containing train/ and test/ subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination for the two COCO JSON files.",
    )
    return parser.parse_args()


def split_sequences(split_root: Path, split: str) -> list[str]:
    """Read one official sequence list from M_attr."""
    split_dir = split_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing official UAVDT split directory: {split_dir}")
    sequences = sorted(path.stem[:5] for path in split_dir.glob("M*_attr.txt"))
    if not sequences:
        raise FileNotFoundError(f"No M*_attr.txt files found in {split_dir}")
    if len(sequences) != len(set(sequences)):
        raise ValueError(f"Duplicate UAVDT sequences found in {split_dir}")
    return sequences


def frame_index(path: Path) -> int:
    """Return the integer frame index from img000001.jpg."""
    if not path.stem.startswith("img"):
        raise ValueError(f"Unexpected UAVDT frame name: {path.name}")
    return int(path.stem.removeprefix("img"))


def convert_split(
    sequences: list[str], images_root: Path, gt_root: Path
) -> tuple[dict, Counter]:
    """Convert one official split without changing boxes or frame membership."""
    images: list[dict] = []
    annotations: list[dict] = []
    category_counts: Counter = Counter()

    for sequence in sequences:
        sequence_dir = images_root / sequence
        frames = sorted(sequence_dir.glob("*.jpg"))
        if not frames:
            raise FileNotFoundError(f"No UAVDT frames found in {sequence_dir}")

        with Image.open(frames[0]) as image:
            width, height = image.size

        frame_to_image_id: dict[int, int] = {}
        for frame in frames:
            image_id = len(images)
            index = frame_index(frame)
            if index in frame_to_image_id:
                raise ValueError(f"Duplicate frame index {index} in {sequence_dir}")
            frame_to_image_id[index] = image_id
            images.append(
                {
                    "file_name": f"{sequence}/{frame.name}",
                    "height": height,
                    "width": width,
                    "id": image_id,
                }
            )

        gt_path = gt_root / f"{sequence}_gt_whole.txt"
        if not gt_path.is_file():
            raise FileNotFoundError(f"Missing UAVDT detection ground truth: {gt_path}")

        for line_number, line in enumerate(
            gt_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) < 9:
                raise ValueError(
                    f"Malformed {gt_path}:{line_number}: expected at least 9 columns"
                )
            index = int(fields[0])
            if index not in frame_to_image_id:
                raise ValueError(
                    f"Ground truth references a missing frame at {gt_path}:{line_number}"
                )

            x, y, box_width, box_height = map(int, fields[2:6])
            if box_width <= 0 or box_height <= 0:
                continue
            source_category = int(fields[8])
            if source_category not in SOURCE_TO_RELEASE_ID:
                continue
            category_id = SOURCE_TO_RELEASE_ID[source_category]
            annotations.append(
                {
                    "image_id": frame_to_image_id[index],
                    "bbox": [x, y, box_width, box_height],
                    "category_id": category_id,
                    "id": len(annotations),
                    "iscrowd": 0,
                    "segmentation": [],
                    "area": box_width * box_height,
                }
            )
            category_counts[category_id] += 1

    return {
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }, category_counts


def main() -> None:
    """Convert both official splits and enforce the paper's image counts."""
    args = parse_args()
    images_root = args.images.expanduser().resolve()
    gt_root = args.gt.expanduser().resolve()
    split_root = args.split.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    train_sequences = split_sequences(split_root, "train")
    test_sequences = split_sequences(split_root, "test")
    overlap = sorted(set(train_sequences) & set(test_sequences))
    if overlap:
        raise ValueError(f"Official UAVDT train/test split overlaps: {overlap}")

    output_root.mkdir(parents=True, exist_ok=True)
    for split, sequences in (("train", train_sequences), ("test", test_sequences)):
        coco, counts = convert_split(sequences, images_root, gt_root)
        image_count = len(coco["images"])
        if image_count != EXPECTED_IMAGES[split]:
            raise ValueError(
                f"UAVDT {split} contains {image_count:,} images; expected {EXPECTED_IMAGES[split]:,}. "
                "Check that M_attr and all official sequences are complete."
            )
        destination = output_root / f"UAVDT-DET_{split}_coco.json"
        destination.write_text(json.dumps(coco), encoding="utf-8")
        print(
            f"{split}: images={image_count:,} annotations={len(coco['annotations']):,} "
            f"car={counts[0]:,} truck={counts[1]:,} bus={counts[2]:,} output={destination}"
        )


if __name__ == "__main__":
    main()
