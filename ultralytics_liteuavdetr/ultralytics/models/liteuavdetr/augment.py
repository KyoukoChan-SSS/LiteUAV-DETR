# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Final LiteUAVDETR training augmentations adapted from D-FINE and RT-DETR.

D-FINE source: https://github.com/Peterande/D-FINE
RT-DETR source: https://github.com/lyuwenyu/RT-DETR
LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as nnf
import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as tvf
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes


def as_xyxy_boxes(boxes: torch.Tensor, height: int, width: int) -> BoundingBoxes:
    """Wrap an XYXY tensor so torchvision v2 transforms update boxes with the image."""
    return BoundingBoxes(boxes, format="XYXY", canvas_size=(height, width))


class Mosaic2(T.Transform):
    """Compose four resized images, then apply the released affine-and-crop step."""

    def __init__(self, size=(640, 640), probability=1.0):
        super().__init__()
        self.resize = T.Resize(size=list(size))
        self.crop = T.RandomCrop(size=list(size))
        self.probability = float(probability)
        self.random_affine = T.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.5, 1.5),
            fill=114,
        )

    def forward(self, image, target, dataset):
        if random.random() > self.probability:
            return image, target, dataset

        image, target = self.resize(image, target)
        images, targets = [image], [target]
        for index in random.choices(range(len(dataset)), k=3):
            extra_image, extra_target = dataset.load_item(index)
            extra_image, extra_target = self.resize(extra_image, extra_target)
            images.append(extra_image)
            targets.append(extra_target)

        height, width = tvf.get_size(images[0])
        offsets = ((0, 0), (width, 0), (0, height), (width, height))
        mosaic = Image.new(images[0].mode, (width * 2, height * 2), color=0)
        for tile, offset in zip(images, offsets):
            mosaic.paste(tile, offset)

        box_offsets = torch.tensor(offsets).repeat(1, 2)
        boxes = torch.cat(
            [sample["boxes"].as_subclass(torch.Tensor) + box_offsets[i] for i, sample in enumerate(targets)]
        )
        labels = torch.cat([sample["labels"] for sample in targets])
        target = {
            "boxes": as_xyxy_boxes(boxes, height * 2, width * 2),
            "labels": labels,
        }
        mosaic, target = self.random_affine(mosaic, target)
        mosaic, target = self.crop(mosaic, target)
        return mosaic, target, dataset


class MixUp(T.Transform):
    """Blend a transformed sample with one raw dataset sample."""

    def __init__(self, alpha=1.5, probability=0.2):
        super().__init__()
        self.alpha = float(alpha)
        self.probability = float(probability)

    def forward(self, image, target, dataset):
        if random.random() > self.probability:
            return image, target, dataset

        mix_image, mix_target = dataset.load_item(random.randint(0, len(dataset) - 1))
        lam = random.betavariate(self.alpha, self.alpha)
        width, height = image.size
        mixed_image = Image.blend(image, mix_image.resize((width, height)), 1 - lam)

        mix_width, mix_height = mix_image.size
        mix_boxes = mix_target["boxes"].as_subclass(torch.Tensor).clone()
        mix_boxes[:, 0::2] *= width / mix_width
        mix_boxes[:, 1::2] *= height / mix_height
        boxes = torch.cat((target["boxes"].as_subclass(torch.Tensor), mix_boxes))
        labels = torch.cat((target["labels"], mix_target["labels"]))
        mixed_target = {
            "boxes": as_xyxy_boxes(boxes, height, width),
            "labels": labels,
        }
        return mixed_image, mixed_target, dataset


class LiteUAVDETRTrainTransform:
    """The final StripFact-Res VisDrone augmentation sequence."""

    def __init__(self, size=640, stop_epoch=196):
        self.size = int(size)
        self.stop_epoch = int(stop_epoch)
        self.mosaic = Mosaic2((self.size, self.size), probability=1.0)
        self.mixup = MixUp(alpha=1.5, probability=0.2)
        self.photometric = T.RandomPhotometricDistort(p=0.5)
        self.zoom_out = T.RandomZoomOut(fill=0)
        self.iou_crop = T.RandomIoUCrop()
        self.sanitize = T.SanitizeBoundingBoxes(min_size=1)
        self.horizontal_flip = T.RandomHorizontalFlip()
        self.resize = T.Resize([self.size, self.size])

    def __call__(self, image, target, dataset):
        if dataset.epoch < self.stop_epoch:
            image, target, _ = self.mosaic(image, target, dataset)
            image, target, _ = self.mixup(image, target, dataset)
            image, target = self.photometric(image, target)
            image, target = self.zoom_out(image, target)
            if torch.rand(1) < 0.8:
                image, target = self.iou_crop(image, target)

        image, target = self.sanitize(image, target)
        image, target = self.horizontal_flip(image, target)
        image, target = self.resize(image, target)
        image, target = self.sanitize(image, target)

        canvas_height, canvas_width = target["boxes"].canvas_size
        boxes = torchvision.ops.box_convert(
            target["boxes"].as_subclass(torch.Tensor),
            in_fmt="xyxy",
            out_fmt="cxcywh",
        )
        scale = torch.tensor(
            (canvas_width, canvas_height, canvas_width, canvas_height),
            dtype=boxes.dtype,
        )
        target["boxes"] = boxes / scale
        image = tvf.pil_to_tensor(image).float() / 255.0
        return image, target


def generate_scales(base_size: int, base_size_repeat: int) -> list[int]:
    """Generate D-FINE's repeated 0.75x-to-1.25x stride-aligned scale list."""
    scale_repeat = (base_size - int(base_size * 0.75 / 32) * 32) // 32
    scales = [int(base_size * 0.75 / 32) * 32 + i * 32 for i in range(scale_repeat)]
    scales += [base_size] * base_size_repeat
    scales += [int(base_size * 1.25 / 32) * 32 - i * 32 for i in range(scale_repeat)]
    return scales


class LiteUAVDETRCollate:
    """Collate Ultralytics batches with D-FINE's optional batch-level multiscale."""

    def __init__(self, base_size=640, base_size_repeat=None, stop_epoch=None):
        self.base_size = int(base_size)
        self.scales = (
            generate_scales(self.base_size, int(base_size_repeat))
            if base_size_repeat is not None
            else None
        )
        self.stop_epoch = int(stop_epoch) if stop_epoch is not None else 1_000_000_000
        self.epoch = -1

    def set_epoch(self, epoch: int):
        """Set the epoch visible to worker-side collate calls."""
        self.epoch = int(epoch)

    def __call__(self, batch):
        images = torch.stack([sample["img"] for sample in batch])
        if self.scales is not None and self.epoch < self.stop_epoch:
            images = nnf.interpolate(images, size=random.choice(self.scales))

        classes = torch.cat([sample["cls"] for sample in batch])
        boxes = torch.cat([sample["bboxes"] for sample in batch])
        batch_indices = torch.cat(
            [
                torch.full((len(sample["cls"]),), index, dtype=torch.float32)
                for index, sample in enumerate(batch)
            ]
        )
        return {
            "img": images,
            "cls": classes,
            "bboxes": boxes,
            "batch_idx": batch_indices,
            "ori_shape": tuple(sample["ori_shape"] for sample in batch),
            "resized_shape": tuple(sample["resized_shape"] for sample in batch),
            "ratio_pad": tuple(sample["ratio_pad"] for sample in batch),
            "im_file": tuple(sample["im_file"] for sample in batch),
            "image_id": tuple(sample["image_id"] for sample in batch),
        }
