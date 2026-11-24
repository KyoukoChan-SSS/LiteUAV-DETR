# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Prediction support for Ultralytics-native LiteUAVDETR.

Adapted from D-FINE (https://github.com/Peterande/D-FINE). LiteUAVDETR
modifications Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class LiteUAVDETRPredictor(BasePredictor):
    """Convert LiteUAVDETR focal logits and normalized boxes into Ultralytics Results."""

    def postprocess(self, preds, img, orig_imgs):
        """Apply the released global query-by-class top-k selection without NMS."""
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(preds, torch.Tensor):
            boxes, scores, labels = preds.split((4, 1, 1), dim=-1)
            scores, labels = scores.squeeze(-1), labels.squeeze(-1).long()
        elif isinstance(preds, dict):
            logits, boxes = preds["pred_logits"], preds["pred_boxes"]
            nc = logits.shape[-1]
            native_model = getattr(self.model, "model", None)
            num_top_queries = getattr(
                getattr(native_model, "postprocessor", None), "num_top_queries", 350
            )
            topk = min(num_top_queries, logits.shape[1] * nc)
            scores, flat_index = torch.topk(logits.sigmoid().flatten(1), topk, dim=-1)
            labels = flat_index % nc
            query_index = flat_index // nc
            boxes = boxes.gather(1, query_index.unsqueeze(-1).expand(-1, -1, 4))
        else:
            raise TypeError(f"Unsupported LiteUAVDETR prediction output: {type(preds).__name__}")
        boxes = ops.xywh2xyxy(boxes)

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for bbox, score, label, orig_img, img_path in zip(boxes, scores, labels, orig_imgs, self.batch[0]):
            keep = score > self.args.conf
            if self.args.classes is not None:
                selected = torch.tensor(self.args.classes, device=label.device)
                keep &= (label[:, None] == selected[None]).any(1)
            bbox, score, label = bbox[keep], score[keep], label[keep]
            if self.args.max_det > 0:
                bbox, score, label = (
                    bbox[: self.args.max_det],
                    score[: self.args.max_det],
                    label[: self.args.max_det],
                )
            oh, ow = orig_img.shape[:2]
            bbox[..., [0, 2]] *= ow
            bbox[..., [1, 3]] *= oh
            pred = torch.cat((bbox, score[:, None], label[:, None].to(bbox.dtype)), dim=-1)
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """Stretch every image to the square input used by the released model."""
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=image) for image in im]
