"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from ._solver import BaseSolver
from .det_engine import evaluate, train_one_epoch


class DetSolver(BaseSolver):
    def fit(self):
        self.train()
        args = self.cfg
        metric_names = ["AP50:95", "AP50", "AP75", "APsmall", "APmedium", "APlarge"]

        if self.use_wandb:
            import wandb

            wandb.init(
                project=args.yaml_cfg["project_name"],
                name=args.yaml_cfg["exp_name"],
                config=args.yaml_cfg,
            )
            wandb.watch(self.model)
#
        n_parameters, model_stats = stats(self.cfg)
        # use for debug
        # try:
        #     n_parameters, model_stats = stats(self.cfg)
        # except Exception as e:
        #     print(f"[Warning] calflops skipped (custom decoder): {e}")
        #     n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #     model_stats = None
        #     print(f"Trainable parameters: {n_parameters:,}")
        print(model_stats)
        print("-" * 42 + "Start training" + "-" * 43)
        top1 = 0
        best_stat = {
            "epoch": -1,
        }
        if self.last_epoch > 0:
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                self.last_epoch,
                self.use_wandb
            )
            for k in test_stats:
                best_stat["epoch"] = self.last_epoch
                best_stat[k] = test_stats[k][0]
                top1 = test_stats[k][0]
                print(f"best_stat: {best_stat}")

        best_stat_print = best_stat.copy()
        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epochs):
            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            if epoch == self.train_dataloader.collate_fn.stop_epoch:
                dist_utils.synchronize()
                self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                dist_utils.synchronize()
                if self.ema:
                    self.ema.decay = self.train_dataloader.collate_fn.ema_restart_decay
                    print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                epochs=args.epochs,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                use_wandb=self.use_wandb,
                output_dir=self.output_dir,
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir and epoch < self.train_dataloader.collate_fn.stop_epoch:
                checkpoint_paths = [self.output_dir / "last.pth"]
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device,
                epoch,
                self.use_wandb,
                output_dir=self.output_dir,
            )

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f"Test/{k}_{i}".format(k), v, epoch)

                if k in best_stat:
                    best_stat["epoch"] = (
                        epoch if test_stats[k][0] > best_stat[k] else best_stat["epoch"]
                    )
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat["epoch"] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat[k] > top1:
                    best_stat_print["epoch"] = epoch
                    top1 = best_stat[k]
                    if self.output_dir:
                        if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                        else:
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg1.pth"
                            )

                best_stat_print[k] = max(best_stat[k], top1)
                print(f"best_stat: {best_stat_print}")  # global best

                if best_stat["epoch"] == epoch and self.output_dir:
                    if epoch >= self.train_dataloader.collate_fn.stop_epoch:
                        if test_stats[k][0] > top1:
                            top1 = test_stats[k][0]
                            dist_utils.save_on_master(
                                self.state_dict(), self.output_dir / "best_stg2.pth"
                            )
                    else:
                        top1 = max(test_stats[k][0], top1)
                        dist_utils.save_on_master(
                            self.state_dict(), self.output_dir / "best_stg1.pth"
                        )

                elif epoch >= self.train_dataloader.collate_fn.stop_epoch:
                    best_stat = {
                        "epoch": -1,
                    }
                    if self.ema:
                        self.ema.decay -= 0.0001
                        dist_utils.synchronize()
                        self.load_resume_state(str(self.output_dir / "best_stg1.pth"))
                        dist_utils.synchronize()
                        print(f"Refresh EMA at epoch {epoch} with decay {self.ema.decay}")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            map_save_threshold = args.yaml_cfg.get("map_save_threshold", None)
            if (
                self.output_dir
                and dist_utils.is_main_process()
                and map_save_threshold is not None
                and "coco_eval_bbox" in test_stats
            ):
                ap = float(test_stats["coco_eval_bbox"][0])
                threshold = float(map_save_threshold)
                if ap >= threshold:
                    default_name = f"map_ge_{threshold * 100:.1f}".replace(".", "_")
                    map_save_dir = args.yaml_cfg.get("map_save_dir", None)
                    map_save_dir = (
                        self.output_dir / default_name
                        if map_save_dir is None
                        else self.output_dir / map_save_dir
                    )
                    map_save_dir.mkdir(parents=True, exist_ok=True)

                    ckpt_name = f"epoch{epoch:04d}_map{ap * 100:.3f}.pth"
                    ckpt_path = map_save_dir / ckpt_name
                    if not ckpt_path.exists():
                        dist_utils.save_on_master(self.state_dict(), ckpt_path)

                    record = {
                        "epoch": epoch,
                        "map": ap,
                        "map_percent": ap * 100,
                        "checkpoint": ckpt_name,
                    }
                    with (map_save_dir / "records.jsonl").open("a") as f:
                        f.write(json.dumps(record) + "\n")
                    csv_path = map_save_dir / "records.csv"
                    if not csv_path.exists():
                        csv_path.write_text("epoch,map,map_percent,checkpoint\n")
                    with csv_path.open("a") as f:
                        f.write(
                            f"{epoch},{ap:.12f},{ap * 100:.6f},{ckpt_name}\n"
                        )

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / "eval").mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ["latest.pth"]
                        if epoch % 50 == 0:
                            filenames.append(f"{epoch:03}.pth")
                        for name in filenames:
                            torch.save(
                                coco_evaluator.coco_eval["bbox"].eval,
                                self.output_dir / "eval" / name,
                            )

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

    def val(self):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module,
            self.criterion,
            self.postprocessor,
            self.val_dataloader,
            self.evaluator,
            self.device,
            epoch=-1,
            use_wandb=False,
        )

        if self.output_dir:
            dist_utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth"
            )

        return
