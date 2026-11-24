# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Beginner-friendly YOLO-label data contract for LiteUAVDETR OpenBox.

Ultralytics supplies the image discovery, label validation, and cache format.
Training still uses the released D-FINE-derived LiteUAVDETR augmentation recipe.
LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

import torch
import torchvision.transforms.v2.functional as tvf
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode

from ultralytics.data.augment import Compose
from ultralytics.data.dataset import YOLODataset

from .augment import LiteUAVDETRTrainTransform, as_xyxy_boxes

Image.MAX_IMAGE_PIXELS = None


class LiteUAVDETROpenBoxDataset(YOLODataset):
    """Read standard Ultralytics detection labels and emit the LiteUAVDETR batch contract."""

    def __init__(self, *args, stop_epoch=196, **kwargs):
        self.epoch = -1
        self.openbox_train = bool(kwargs.get("augment", False))
        self.liteuav_transform = LiteUAVDETRTrainTransform(kwargs.get("imgsz", 640), stop_epoch=stop_epoch)
        super().__init__(*args, **kwargs)

    def build_transforms(self, hyp=None):
        """Keep Ultralytics label scanning while delegating augmentation to LiteUAVDETR."""
        del hyp
        return Compose([])

    def set_epoch(self, epoch: int):
        """Expose the current epoch to the released stop-epoch augmentation policy."""
        self.epoch = int(epoch)

    def load_item(self, index):
        """Load one unaugmented RGB image and convert normalized YOLO boxes to pixel XYXY."""
        image_file = self.im_files[index]
        with Image.open(image_file) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size

        label = self.labels[index]
        classes = torch.as_tensor(label["cls"], dtype=torch.int64).reshape(-1)
        xywh = torch.as_tensor(label["bboxes"], dtype=torch.float32).reshape(-1, 4)
        if len(xywh):
            center_x, center_y, box_width, box_height = xywh.unbind(1)
            boxes = torch.stack(
                (
                    (center_x - box_width * 0.5) * width,
                    (center_y - box_height * 0.5) * height,
                    (center_x + box_width * 0.5) * width,
                    (center_y + box_height * 0.5) * height,
                ),
                dim=1,
            )
            boxes[:, 0::2].clamp_(0, width)
            boxes[:, 1::2].clamp_(0, height)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes, classes = boxes[valid], classes[valid]
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)

        target = {
            "boxes": as_xyxy_boxes(boxes, height, width),
            "labels": classes,
        }
        return image, target

    def __getitem__(self, index):
        """Return a train or validation sample compatible with the LiteUAVDETR trainer."""
        image, target = self.load_item(index)
        width, height = image.size
        if self.openbox_train:
            image, target = self.liteuav_transform(image, target, self)
            image_tensor = image.as_subclass(torch.Tensor) if hasattr(image, "as_subclass") else image
            boxes = (
                target["boxes"].as_subclass(torch.Tensor)
                if hasattr(target["boxes"], "as_subclass")
                else target["boxes"]
            )
        else:
            image = tvf.resize(
                image,
                [self.imgsz, self.imgsz],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            image_tensor = tvf.pil_to_tensor(image).float() / 255.0
            boxes = torch.as_tensor(self.labels[index]["bboxes"], dtype=torch.float32).reshape(-1, 4)
            target["labels"] = torch.as_tensor(self.labels[index]["cls"], dtype=torch.int64).reshape(-1)

        classes = target["labels"].reshape(-1, 1).float()
        resized_shape = tuple(image_tensor.shape[-2:])
        return {
            "img": image_tensor,
            "cls": classes,
            "bboxes": boxes.reshape(-1, 4).float(),
            "batch_idx": torch.zeros(len(classes), dtype=torch.float32),
            "ori_shape": (height, width),
            "resized_shape": resized_shape,
            "ratio_pad": ((resized_shape[0] / height, resized_shape[1] / width), (0.0, 0.0)),
            "im_file": self.im_files[index],
            "image_id": index,
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
