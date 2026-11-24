# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""COCO JSON data contract for the D-FINE-derived LiteUAVDETR training path.

The Ultralytics integration is implemented by LiteUAVDETR without converting
the original COCO annotations to YOLO labels.
LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.v2.functional as tvf
from faster_coco_eval import COCO
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode

from .augment import LiteUAVDETRTrainTransform, as_xyxy_boxes

Image.MAX_IMAGE_PIXELS = None


def resolve_category_ids(category_ids, nc: int) -> list[int]:
    """Map model class indices to the raw category IDs stored in a COCO JSON file."""
    nc = int(nc)
    if category_ids is None:
        return list(range(nc))
    if not isinstance(category_ids, (list, tuple)):
        raise TypeError("LiteUAVDETR dataset 'category_ids' must be a list ordered by model class index.")
    if len(category_ids) != nc:
        raise ValueError(f"LiteUAVDETR dataset has {len(category_ids)} category_ids but {nc} names.")
    if any(isinstance(category_id, bool) or not isinstance(category_id, int) for category_id in category_ids):
        raise TypeError("LiteUAVDETR dataset 'category_ids' entries must be integers.")
    if len(set(category_ids)) != len(category_ids):
        raise ValueError("LiteUAVDETR dataset 'category_ids' entries must be unique.")
    return list(category_ids)


def resolve_annotation_file(data: dict, mode: str) -> Path:
    """Resolve train/val/test COCO annotation keys against the dataset root."""
    split = "val" if mode == "val" else mode
    key = f"{split}_annotations"
    annotation = data.get(key)
    if annotation is None:
        raise KeyError(f"LiteUAVDETR dataset YAML requires '{key}'.")
    annotation = Path(annotation)
    if not annotation.is_absolute():
        annotation = Path(data["path"]) / annotation
    if not annotation.is_file():
        raise FileNotFoundError(f"COCO annotation file does not exist: {annotation}")
    return annotation


class LiteUAVDETRCocoDataset(Dataset):
    """Read COCO boxes while emitting the standard Ultralytics detection batch contract."""

    def __init__(self, img_path, ann_file, imgsz=640, nc=12, category_ids=None):
        self.img_path = Path(img_path)
        self.ann_file = Path(ann_file)
        self.imgsz = int(imgsz)
        self.nc = int(nc)
        self.category_ids = resolve_category_ids(category_ids, self.nc)
        self.category_id_to_index = {
            category_id: class_index for class_index, category_id in enumerate(self.category_ids)
        }
        self.coco = COCO(self.ann_file)
        self.ids = sorted(self.coco.getImgIds())
        self.images = {image["id"]: image for image in self.coco.loadImgs(self.ids)}
        self.im_files = [str(self.img_path / self.images[image_id]["file_name"]) for image_id in self.ids]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        info = self.images[image_id]
        image_file = self.img_path / info["file_name"]
        with Image.open(image_file) as image:
            image = image.convert("RGB")
            width, height = image.size
            resized = tvf.resize(
                image,
                [self.imgsz, self.imgsz],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            image_tensor = tvf.pil_to_tensor(resized).float() / 255.0

        classes, boxes = [], []
        for annotation in self.coco.imgToAnns.get(image_id, []):
            if annotation.get("iscrowd", 0):
                continue
            category = self.category_id_to_index.get(int(annotation["category_id"]))
            if category is None:
                continue
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x1, y1 = max(x, 0.0), max(y, 0.0)
            x2, y2 = min(x + box_width, float(width)), min(y + box_height, float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            classes.append([float(category)])
            boxes.append(
                [
                    ((x1 + x2) * 0.5) / width,
                    ((y1 + y2) * 0.5) / height,
                    (x2 - x1) / width,
                    (y2 - y1) / height,
                ]
            )

        cls = torch.tensor(classes, dtype=torch.float32).reshape(-1, 1)
        bboxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        return {
            "img": image_tensor,
            "cls": cls,
            "bboxes": bboxes,
            "batch_idx": torch.zeros(len(cls), dtype=torch.float32),
            "ori_shape": (height, width),
            "resized_shape": (self.imgsz, self.imgsz),
            "ratio_pad": ((self.imgsz / height, self.imgsz / width), (0.0, 0.0)),
            "im_file": str(image_file),
            "image_id": image_id,
        }

    @staticmethod
    def collate_fn(batch):
        """Stack images and concatenate variable-length detection targets."""
        new_batch = {}
        batch = [dict(sorted(sample.items())) for sample in batch]
        keys = batch[0].keys()
        values = list(zip(*[sample.values() for sample in batch]))
        for key, value in zip(keys, values):
            if key == "img":
                value = torch.stack(value, 0)
            elif key in {"bboxes", "cls"}:
                value = torch.cat(value, 0)
            new_batch[key] = value
        batch_indices = list(new_batch["batch_idx"])
        for image_index, indices in enumerate(batch_indices):
            indices += image_index
        new_batch["batch_idx"] = torch.cat(batch_indices, 0)
        return new_batch


class LiteUAVDETRTrainDataset(Dataset):
    """Read COCO JSON and apply only the released LiteUAVDETR training recipe."""

    def __init__(
        self,
        img_path,
        ann_file,
        imgsz=640,
        nc=12,
        stop_epoch=196,
        fraction=1.0,
        category_ids=None,
    ):
        self.img_path = Path(img_path)
        self.ann_file = Path(ann_file)
        self.imgsz = int(imgsz)
        self.nc = int(nc)
        self.category_ids = resolve_category_ids(category_ids, self.nc)
        self.category_id_to_index = {
            category_id: class_index for class_index, category_id in enumerate(self.category_ids)
        }
        self.coco = COCO(self.ann_file)
        self.ids = sorted(self.coco.getImgIds())
        if fraction < 1.0:
            self.ids = self.ids[: max(1, round(len(self.ids) * float(fraction)))]
        self.images = {image["id"]: image for image in self.coco.loadImgs(self.ids)}
        self.im_files = [str(self.img_path / self.images[image_id]["file_name"]) for image_id in self.ids]
        self.transform = LiteUAVDETRTrainTransform(self.imgsz, stop_epoch=stop_epoch)
        self.epoch = -1

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch: int):
        """Expose the current epoch to the stop-epoch augmentation policy."""
        self.epoch = int(epoch)

    def load_item(self, index):
        """Load an unaugmented RGB image and its valid XYXY boxes."""
        image_id = self.ids[index]
        info = self.images[image_id]
        image_file = self.img_path / info["file_name"]
        with Image.open(image_file) as source:
            image = source.convert("RGB")
        width, height = image.size

        labels, boxes = [], []
        for annotation in self.coco.imgToAnns.get(image_id, []):
            if annotation.get("iscrowd", 0):
                continue
            category = self.category_id_to_index.get(int(annotation["category_id"]))
            if category is None:
                continue
            x, y, box_width, box_height = map(float, annotation["bbox"])
            x1, y1 = max(x, 0.0), max(y, 0.0)
            x2, y2 = min(x + box_width, float(width)), min(y + box_height, float(height))
            if x2 <= x1 or y2 <= y1:
                continue
            labels.append(category)
            boxes.append((x1, y1, x2, y2))

        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": as_xyxy_boxes(boxes, height, width),
            "labels": torch.tensor(labels, dtype=torch.int64),
        }
        return image, target

    def __getitem__(self, index):
        image_id = self.ids[index]
        info = self.images[image_id]
        image_file = self.img_path / info["file_name"]
        width, height = int(info["width"]), int(info["height"])
        image, target = self.load_item(index)
        image, target = self.transform(image, target, self)
        boxes = (
            target["boxes"].as_subclass(torch.Tensor) if hasattr(target["boxes"], "as_subclass") else target["boxes"]
        )
        labels = target["labels"].reshape(-1, 1).float()
        resized_shape = tuple(image.shape[-2:])
        return {
            "img": image.as_subclass(torch.Tensor) if hasattr(image, "as_subclass") else image,
            "cls": labels,
            "bboxes": boxes.reshape(-1, 4).float(),
            "batch_idx": torch.zeros(len(labels), dtype=torch.float32),
            "ori_shape": (height, width),
            "resized_shape": resized_shape,
            "ratio_pad": ((resized_shape[0] / height, resized_shape[1] / width), (0.0, 0.0)),
            "im_file": str(image_file),
            "image_id": image_id,
        }
