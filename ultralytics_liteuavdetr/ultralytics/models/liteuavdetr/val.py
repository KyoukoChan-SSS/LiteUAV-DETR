# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Validation support for Ultralytics-native LiteUAVDETR.

Evaluation follows D-FINE (https://github.com/Peterande/D-FINE) and reports
official COCO metrics from faster-coco-eval. LiteUAVDETR modifications
Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision
from faster_coco_eval import COCO, COCOeval_faster
from torch.utils.data import DataLoader

from ultralytics.engine.validator import BaseValidator
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.torch_utils import smart_inference_mode

from .dataset import LiteUAVDETRCocoDataset, resolve_annotation_file, resolve_category_ids


class LiteUAVDETRValidator(RTDETRValidator):
    """Validate LiteUAVDETR directly against COCO JSON annotations without NMS."""

    # D-FINE validation selects checkpoints by COCO AP and does not recompute
    # auxiliary/denoising training losses from the inference-only eval output.
    compute_validation_loss = False

    @smart_inference_mode(False)
    def __call__(self, trainer=None, model=None):
        """Run with D-FINE's no-grad semantics so device moves keep ordinary parameters.

        The upstream validator enters ``torch.inference_mode`` before AutoBackend moves a
        CPU model to CUDA. With PyTorch 2.10 this creates inference-tensor parameters and
        changes the attention dispatch used by the released LiteUAVDETR checkpoints.
        Calling the undecorated base implementation under ``no_grad`` preserves the
        original D-FINE evaluation path while retaining the Ultralytics validation loop.
        """
        return BaseValidator.__call__.__wrapped__(self, trainer=trainer, model=model)

    def init_metrics(self, model):
        """Initialize metrics while preserving the dataset's raw COCO category IDs."""
        super().init_metrics(model)
        configured_category_ids = self.data.get("category_ids")
        self.class_map = resolve_category_ids(configured_category_ids, len(model.names))
        self.explicit_category_ids = configured_category_ids is not None
        self.is_coco = True
        self.is_lvis = False
        self.args.save_json = True
        native_model = getattr(model, "model", None)
        self.num_top_queries = getattr(getattr(native_model, "postprocessor", None), "num_top_queries", 350)

    def preprocess(self, batch):
        """Move CPU-normalized D-FINE inputs to the selected inference device."""
        if not batch["img"].is_floating_point():
            raise TypeError("LiteUAVDETR validation images must be normalized to float32 on the CPU.")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = batch["img"].half() if self.args.quantize == 16 else batch["img"].float()
        return batch

    def build_dataset(self, img_path, mode="val", batch=None):
        """Build the COCO JSON dataset declared by the LiteUAVDETR data YAML."""
        self.annotation_file = resolve_annotation_file(self.data, mode)
        return LiteUAVDETRCocoDataset(
            img_path=img_path,
            ann_file=self.annotation_file,
            imgsz=self.args.imgsz,
            nc=len(self.data["names"]),
            category_ids=self.data.get("category_ids"),
        )

    def get_dataloader(self, dataset_path, batch_size):
        """Build a deterministic validation dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=dataset.collate_fn,
        )

    def postprocess(self, preds):
        """Apply the exact focal sigmoid and global query-by-class top-k selection."""
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(preds, torch.Tensor):
            boxes, scores, labels = preds.split((4, 1, 1), dim=-1)
            scores, labels = scores.squeeze(-1), labels.squeeze(-1).long()
        elif isinstance(preds, dict):
            logits, boxes = preds["pred_logits"], preds["pred_boxes"]
            nc = logits.shape[-1]
            topk = min(self.num_top_queries, logits.shape[1] * nc)
            scores, flat_index = torch.topk(logits.sigmoid().flatten(1), topk, dim=-1)
            labels = flat_index % nc
            query_index = flat_index // nc
            boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
        else:
            raise TypeError(f"Unsupported LiteUAVDETR validation output: {type(preds).__name__}")
        boxes = torchvision.ops.box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy")

        outputs = []
        for bbox, score, label in zip(boxes, scores, labels):
            keep = score > self.args.conf
            outputs.append({"bboxes": bbox[keep], "conf": score[keep], "cls": label[keep]})
        return outputs

    def _prepare_batch(self, si, batch):
        """Prepare normalized targets so prediction boxes never take a 640 round trip."""
        index = batch["batch_idx"] == si
        cls = batch["cls"][index].squeeze(-1)
        bbox = ops.xywh2xyxy(batch["bboxes"][index])
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": batch["ori_shape"][si],
            "imgsz": batch["img"].shape[2:],
            "ratio_pad": batch["ratio_pad"][si],
            "im_file": batch["im_file"][si],
            "image_id": int(batch["image_id"][si]),
        }

    def pred_to_json(self, predn, pbatch):
        """Serialize full-precision predictions using original COCO image/category ids."""
        box = predn["bboxes"].clone()
        height, width = pbatch["ori_shape"]
        box *= torch.tensor((width, height, width, height), device=box.device)
        xmin, ymin, xmax, ymax = box.unbind(1)
        box = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)
        for bbox, score, category in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": pbatch["image_id"],
                    "category_id": self.class_map[int(category)],
                    "bbox": bbox,
                    "score": float(score),
                }
            )

    def _sync_coco_ap(self, evaluator):
        """Expose official per-class COCO AP through the normal Ultralytics metrics object."""
        precision = np.asarray(evaluator.eval["precision"])
        category_ids = list(evaluator.params.catIds)
        current_ids = [int(index) for index in self.metrics.box.ap_class_index]
        selected_ids = current_ids or list(range(len(self.class_map)))
        all_ap, valid_ids = [], []
        for class_index in selected_ids:
            if not 0 <= class_index < len(self.class_map):
                continue
            category_id = self.class_map[class_index]
            if category_id not in category_ids:
                continue
            category_index = category_ids.index(category_id)
            category_precision = precision[:, :, category_index, 0, -1]
            if not np.any(category_precision > -1):
                continue
            per_iou = []
            for values in category_precision:
                valid = values[values > -1]
                per_iou.append(float(valid.mean()) if len(valid) else 0.0)
            all_ap.append(per_iou)
            valid_ids.append(class_index)
        self.metrics.box.all_ap = np.asarray(all_ap, dtype=np.float64).reshape(-1, precision.shape[0])
        self.metrics.box.ap_class_index = np.asarray(valid_ids, dtype=np.int64)

    def _evaluate_coco(self, predictions):
        """Evaluate a JSON path or an in-memory prediction list against the declared annotations."""
        LOGGER.info(f"\nEvaluating LiteUAVDETR COCO metrics using {self.annotation_file}...")
        annotation = COCO(Path(self.annotation_file))
        results = annotation.loadRes(predictions)
        evaluator = COCOeval_faster(
            annotation,
            results,
            iouType="bbox",
            print_function=LOGGER.info,
        )
        evaluator.params.imgIds = list(self.dataloader.dataset.ids)
        annotation_category_ids = set(annotation.getCatIds())
        if self.explicit_category_ids:
            missing = [category_id for category_id in self.class_map if category_id not in annotation_category_ids]
            if missing:
                raise ValueError(f"Dataset category_ids are missing from the COCO categories table: {missing}")
            evaluator.params.catIds = list(self.class_map)
        else:
            evaluator.params.catIds = [
                category_id for category_id in self.class_map if category_id in annotation_category_ids
            ]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        self._sync_coco_ap(evaluator)
        torch.save(evaluator.eval, self.save_dir / "eval.pth")
        self.coco_eval = evaluator
        return evaluator.stats_as_dict

    def _apply_coco_stats(self, stats, values):
        """Replace approximate matching mAP and fitness with official COCO values."""
        stats.update(self.metrics.results_dict)
        stats.update(
            {
                "metrics/mAP50-95(B)": values["AP_all"],
                "metrics/mAP50(B)": values["AP_50"],
                "metrics/mAP75(B)": values["AP_75"],
                "metrics/mAP_small(B)": values["AP_small"],
                "metrics/mAP_medium(B)": values["AP_medium"],
                "metrics/mAP_large(B)": values["AP_large"],
                "metrics/AR100(B)": values["AR_all"],
                "fitness": values["AP_all"],
            }
        )
        return stats

    def get_stats(self):
        """Use exact COCO AP during training so stage checkpoints use the paper metric."""
        stats = super().get_stats()
        if self.training and self.jdict:
            stats = self._apply_coco_stats(stats, self._evaluate_coco(self.jdict))
        return stats

    def eval_json(self, stats):
        """Run COCO evaluation against the annotation path declared in the data YAML."""
        pred_json = self.save_dir / "predictions.json"
        return self._apply_coco_stats(stats, self._evaluate_coco(pred_json))
