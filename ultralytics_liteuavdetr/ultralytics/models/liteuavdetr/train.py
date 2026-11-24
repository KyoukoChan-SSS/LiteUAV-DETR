# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Ultralytics-native training for the released LiteUAVDETR recipe.

The optimization, augmentation-stage, warmup and EMA behavior is adapted from
D-FINE (https://github.com/Peterande/D-FINE), which is itself derived from
RT-DETR. LiteUAVDETR modifications Copyright (c) 2026 The LiteUAVDETR Authors.
"""

from __future__ import annotations

import shutil
from copy import copy, deepcopy
from pathlib import Path

import torch
from torch import distributed as dist, nn
from torch.utils.data import DataLoader, DistributedSampler

from ultralytics.data.build import seed_worker
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.liteuavdetr import infer_dfine_variant
from ultralytics.nn.liteuavdetr.task import LiteUAVDETRDetectionModel
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils import LOGGER, RANK, ROOT
from ultralytics.utils.patches import torch_load
from ultralytics.utils.torch_utils import EarlyStopping, ModelEMA, torch_distributed_zero_first, unwrap_model

from .augment import LiteUAVDETRCollate
from .dataset import LiteUAVDETRCocoDataset, LiteUAVDETRTrainDataset, resolve_annotation_file
from .val import LiteUAVDETRValidator


class LiteUAVDETRTrainer(DetectionTrainer):
    """Run the final D-FINE-derived LiteUAVDETR training recipe through Ultralytics."""

    # D-FINE saves model, EMA, and optimizer tensors in FP32. This also avoids
    # quantizing the best stage-1 state before the stage-2 restart.
    preserve_checkpoint_precision = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_stage1 = self.wdir / "best_stg1.pt"
        self.best_stage2 = self.wdir / "best_stg2.pt"
        self._optimizer_updates = 0
        self._stage_transition_done = False
        self.add_callback("on_train_epoch_start", self._on_train_epoch_start)

    def _recipe(self) -> dict:
        """Return the immutable training recipe embedded in the model YAML."""
        recipe = unwrap_model(self.model).yaml.get("liteuav_train")
        if not isinstance(recipe, dict):
            raise KeyError("LiteUAVDETR model YAML requires a 'liteuav_train' mapping.")
        return recipe

    def setup_model(self):
        """Build from YAML while accepting released D-FINE ``.pth`` weights natively."""
        if isinstance(self.model, nn.Module):
            self._release_frozen_names = {
                name for name, parameter in unwrap_model(self.model).named_parameters() if not parameter.requires_grad
            }
            return None

        pretrained = self.args.pretrained
        cfg = self.model
        if isinstance(self.model, (str, Path)) and Path(self.model).suffix == ".pth":
            pretrained = self.model
            variant = infer_dfine_variant(self.model)
            cfg = ROOT / "cfg" / "models" / "liteuavdetr" / f"liteuavdetr-{variant}.yaml"

        if isinstance(pretrained, (str, Path)) and Path(pretrained).suffix == ".pth":
            self.model = self.get_model(cfg=cfg, weights=pretrained, verbose=RANK in {-1, 0})
            self._release_frozen_names = {
                name for name, parameter in self.model.named_parameters() if not parameter.requires_grad
            }
            return None
        checkpoint = super().setup_model()
        self._release_frozen_names = {
            name for name, parameter in unwrap_model(self.model).named_parameters() if not parameter.requires_grad
        }
        return checkpoint

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build LiteUAVDETR and optionally load released D-FINE or native weights."""
        model_cfg = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        if weights is None and self.args.pretrained is True:
            # LiteUAVDETR: reproduce D-FINE's released training initialization
            # without forcing an unnecessary backbone download for inference or
            # when a complete LiteUAVDETR checkpoint is supplied.
            model_cfg["liteuav_backbone"]["pretrained"] = True
        model = LiteUAVDETRDetectionModel(
            model_cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            if isinstance(weights, (str, Path)) and Path(weights).suffix == ".pth":
                model.load_dfine(weights)
            else:
                model.load(weights)
        return model

    def build_dataset(self, img_path, mode="train", batch=None):
        """Read the release COCO JSON directly, without a YOLO-label conversion."""
        annotation_file = resolve_annotation_file(self.data, mode)
        common = {
            "img_path": img_path,
            "ann_file": annotation_file,
            "imgsz": self.args.imgsz,
            "nc": self.data["nc"],
            "category_ids": self.data.get("category_ids"),
        }
        if mode == "train":
            return LiteUAVDETRTrainDataset(
                **common,
                stop_epoch=self._recipe()["stop_epoch"],
                fraction=self.args.fraction,
            )
        return LiteUAVDETRCocoDataset(**common)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Build finite loaders so each epoch receives the current augmentation policy."""
        if mode not in {"train", "val"}:
            raise ValueError(f"Unsupported LiteUAVDETR dataloader mode: {mode}")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode=mode, batch=batch_size)

        is_train = mode == "train"
        sampler = DistributedSampler(dataset, shuffle=True) if is_train and rank != -1 else None
        collate = (
            LiteUAVDETRCollate(
                base_size=self.args.imgsz,
                base_size_repeat=2,
                stop_epoch=self._recipe()["stop_epoch"],
            )
            if is_train
            else dataset.collate_fn
        )
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + RANK)
        workers = max(int(self.args.workers), 0)
        return DataLoader(
            dataset,
            batch_size=min(int(batch_size), len(dataset)),
            shuffle=is_train and sampler is None,
            sampler=sampler,
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=collate,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=False,
            persistent_workers=False,
            prefetch_factor=4 if workers else None,
        )

    def preprocess_batch(self, batch):
        """Move already CPU-normalized D-FINE samples to the training device."""
        if not batch["img"].is_floating_point():
            raise TypeError("LiteUAVDETR training images must be CPU-normalized floating-point tensors.")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = batch["img"].float()
        return batch

    def set_class_weights(self):
        """Keep the released criterion unchanged; it does not use YOLO class weights."""

    def plot_training_labels(self):
        """Skip YOLO-label plots because this task consumes COCO JSON directly."""

    def _build_train_pipeline(self):
        """Disable nominal-batch accumulation and weight-decay rescaling."""
        self.args.nbs = self.batch_size
        super()._build_train_pipeline()

    def build_optimizer(self, model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Build the four exact D-FINE AdamW parameter groups."""
        del name, lr, momentum, decay, iterations
        recipe = self._recipe()
        base_lr = float(recipe["lr"])
        backbone_lr = float(recipe["backbone_lr"])
        weight_decay = float(recipe["weight_decay"])

        groups = {"backbone": [], "backbone_norm": [], "no_decay": [], "default": []}
        names = {key: [] for key in groups}
        for parameter_name, parameter in unwrap_model(model).named_parameters():
            if parameter_name in getattr(self, "_release_frozen_names", set()):
                parameter.requires_grad_(False)
                continue
            if not parameter.requires_grad:
                continue
            lowered = parameter_name.lower()
            is_backbone = "backbone" in lowered
            is_norm = "norm" in lowered or "bn" in lowered
            if is_backbone and not is_norm:
                group = "backbone"
            elif is_backbone:
                group = "backbone_norm"
            elif ("encoder" in lowered or "decoder" in lowered) and (is_norm or "bias" in lowered):
                group = "no_decay"
            else:
                group = "default"
            groups[group].append(parameter)
            names[group].append(parameter_name)

        parameter_groups = [
            {
                "params": groups["backbone"],
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "liteuav_base_lr": backbone_lr,
                "param_group": "backbone",
            },
            {
                "params": groups["backbone_norm"],
                "lr": backbone_lr,
                "weight_decay": 0.0,
                "liteuav_base_lr": backbone_lr,
                "param_group": "backbone_norm",
            },
            {
                "params": groups["no_decay"],
                "lr": base_lr,
                "weight_decay": 0.0,
                "liteuav_base_lr": base_lr,
                "param_group": "encoder_decoder_no_decay",
            },
            {
                "params": groups["default"],
                "lr": base_lr,
                "weight_decay": weight_decay,
                "liteuav_base_lr": base_lr,
                "param_group": "default",
            },
        ]
        if any(not group["params"] for group in parameter_groups):
            empty = [group["param_group"] for group in parameter_groups if not group["params"]]
            raise RuntimeError(f"LiteUAVDETR optimizer produced empty parameter groups: {empty}")

        optimizer = torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999))
        LOGGER.info("LiteUAVDETR AdamW groups: " + ", ".join(f"{key}={len(value)}" for key, value in names.items()))
        return optimizer

    def _setup_scheduler(self):
        """Use D-FINE's epoch scheduler; milestone 500 is beyond both release runs."""
        self.lf = lambda _: 1.0
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[500], gamma=0.1)

    def _get_warmup_iterations(self, num_batches):
        """Disable Ultralytics epoch warmup; optimizer_step implements the 500-step recipe."""
        del num_batches
        return 0

    def _configure_ema(self):
        """Replace Ultralytics defaults with D-FINE's 0.9999/1000 EMA schedule."""
        previous = self.ema
        updates = int(previous.updates) if previous is not None else 0
        configured = ModelEMA(
            self.model,
            decay=float(self._recipe()["ema_decay"]),
            tau=float(self._recipe()["ema_warmups"]),
            updates=updates,
        )
        if previous is not None:
            configured.ema.load_state_dict(previous.ema.float().state_dict(), strict=True)
        self.ema = configured

    def _apply_warmup_lr(self):
        """Set the LR used by the next optimizer update."""
        duration = int(self._recipe()["warmup_steps"])
        factor = min(1.0, (self._optimizer_updates + 1) / duration) if duration else 1.0
        for group in self.optimizer.param_groups:
            group["lr"] = group["liteuav_base_lr"] * factor

    def _setup_train(self):
        """Configure the common lifecycle, then restore release-specific EMA/warmup."""
        if self.args.compile:
            raise ValueError("LiteUAVDETR training requires compile=False to preserve the released loss path.")
        super()._setup_train()
        if self.args.imgsz != 640:
            raise ValueError("The released LiteUAVDETR training recipe requires imgsz=640.")
        if self._recipe()["stop_epoch"] < self.epochs and not self.args.val:
            raise ValueError("Two-stage LiteUAVDETR training requires val=True to select best_stg1.pt.")
        self._configure_ema()
        self._optimizer_updates = int(self.ema.updates)
        self._apply_warmup_lr()

    def _model_train(self):
        """Enter train mode and publish the epoch to dataset/collate workers."""
        super()._model_train()
        dataset = self.train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.epoch)
        if hasattr(self.train_loader.collate_fn, "set_epoch"):
            self.train_loader.collate_fn.set_epoch(self.epoch)
        self._apply_warmup_lr()

    def optimizer_step(self):
        """Apply D-FINE gradient clipping, AdamW, EMA and linear step warmup."""
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=float(self._recipe()["clip_grad_norm"]),
            foreach=False if self.device.type == "npu" else None,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        self._optimizer_updates += 1
        self._apply_warmup_lr()

    def get_validator(self):
        """Return a validator that uses the declared COCO annotation file for fitness."""
        validator = LiteUAVDETRValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )
        validator.annotation_file = resolve_annotation_file(self.data, "val")
        return validator

    def _on_train_epoch_start(self, trainer):
        """Start stage 2 from the best stage-1 EMA when heavy augmentation stops."""
        stop_epoch = int(self._recipe()["stop_epoch"])
        if trainer.epoch != stop_epoch or self._stage_transition_done:
            return
        if RANK != -1:
            dist.barrier()
        if not self.best_stage1.is_file():
            raise FileNotFoundError(f"Stage 2 requires the evaluated stage-1 checkpoint: {self.best_stage1}")

        checkpoint = torch_load(self.best_stage1, map_location="cpu")
        checkpoint_model = checkpoint.get("ema") or checkpoint.get("model")
        if not isinstance(checkpoint_model, nn.Module):
            raise TypeError(f"Checkpoint has no loadable EMA/model module: {self.best_stage1}")
        unwrap_model(self.model).load_state_dict(checkpoint_model.float().state_dict(), strict=True)
        self._load_checkpoint_state(checkpoint)
        self._configure_ema()
        self._optimizer_updates = int(self.ema.updates)
        self.best_fitness = None
        self.fitness = None
        self.stopper = EarlyStopping(patience=self.args.patience)
        self.stop = False
        self._stage_transition_done = True
        self._apply_warmup_lr()
        LOGGER.info(
            f"LiteUAVDETR stage 2 starts at epoch {trainer.epoch}: restored {self.best_stage1.name} "
            "and disabled Mosaic/MixUp/heavy augmentation."
        )
        if RANK != -1:
            dist.barrier()

    def save_model(self):
        """Keep standard Ultralytics checkpoints plus explicit stage-1/stage-2 best files."""
        is_best = self.best_fitness == self.fitness
        saved = super().save_model()
        if saved and is_best and self.best.is_file():
            stage_path = self.best_stage2 if self.epoch >= int(self._recipe()["stop_epoch"]) else self.best_stage1
            shutil.copy2(self.best, stage_path)
        return saved
