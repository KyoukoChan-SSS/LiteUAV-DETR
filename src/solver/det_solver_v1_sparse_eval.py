"""DetSolver v1 with sparse evaluation schedule for slow-eval datasets (e.g., AITOD).

Subclass of DetSolver. Replaces the unconditional per-epoch evaluation with
a phase-based schedule:

    Phase 1 (epoch < phase1_end):      eval every `eval_freq_phase1` epochs
    Phase 2 (epoch < phase2_end):      eval every `eval_freq_phase2` epochs
    Phase 3 (epoch >= phase2_end):     eval every 1 epoch

`phase1_end` and `phase2_end` are computed from `epochs` via fractions
`phase1_frac` (default 0.5) and `phase2_frac` (default 0.9).

Forced eval at the following epochs regardless of phase frequency, to keep
`best_stg1.pth` fresh and capture LR-drop bumps:

    - epoch 0  (sanity check)
    - last `force_eval_pre_stop` epochs before `aug_stop_epoch`
    - `aug_stop_epoch` itself (stage2 entry)
    - any LR scheduler milestone epoch and the next 2
    - last 2 epochs of training

Inference and checkpoint logic are inherited unchanged. When eval is skipped
on a given epoch, no `best_stg1.pth` update is attempted for that epoch
(that's correct behavior — best_stg1 is only updated on a new evaluated best).

Use this solver via `task: detection_sparse_eval` in the config.
"""

import datetime
import json
import time

import torch

from ..misc import dist_utils, stats
from .det_engine import evaluate, train_one_epoch
from .det_solver import DetSolver


class DetSolverV1SparseEval(DetSolver):
    def _resolve_eval_schedule(self):
        cfg = self.cfg
        yaml_cfg = getattr(cfg, "yaml_cfg", {}) or {}
        sparse = yaml_cfg.get("sparse_eval", {}) or {}

        epochs = int(getattr(cfg, "epochs", 0)) or int(yaml_cfg.get("epochs", 0))
        phase1_frac = float(sparse.get("phase1_frac", 0.5))
        phase2_frac = float(sparse.get("phase2_frac", 0.9))

        phase1_end = max(1, int(epochs * phase1_frac))
        phase2_end = max(phase1_end + 1, int(epochs * phase2_frac))

        eval_freq_phase1 = int(sparse.get("eval_freq_phase1", 20))
        eval_freq_phase2 = int(sparse.get("eval_freq_phase2", 10))
        eval_freq_phase3 = int(sparse.get("eval_freq_phase3", 1))

        force_eval_pre_stop = int(sparse.get("force_eval_pre_stop", 5))
        force_eval_post_milestone = int(sparse.get("force_eval_post_milestone", 2))

        try:
            aug_stop = int(self.train_dataloader.collate_fn.stop_epoch)
        except Exception:
            aug_stop = epochs

        forced = set()
        if epochs > 0:
            forced.add(0)
            forced.add(epochs - 1)
            forced.add(epochs - 2)
        if 0 <= aug_stop < epochs:
            forced.add(aug_stop)
            for k in range(1, force_eval_pre_stop + 1):
                if aug_stop - k >= 0:
                    forced.add(aug_stop - k)

        try:
            milestones = list(getattr(self.lr_scheduler, "milestones", []) or [])
        except Exception:
            milestones = []
        if isinstance(milestones, dict):
            milestones = list(milestones.keys())
        for m in milestones:
            try:
                m = int(m)
            except Exception:
                continue
            if 0 <= m < epochs:
                for k in range(0, force_eval_post_milestone + 1):
                    if m + k < epochs:
                        forced.add(m + k)

        return {
            "epochs": epochs,
            "phase1_end": phase1_end,
            "phase2_end": phase2_end,
            "eval_freq_phase1": max(1, eval_freq_phase1),
            "eval_freq_phase2": max(1, eval_freq_phase2),
            "eval_freq_phase3": max(1, eval_freq_phase3),
            "forced": forced,
            "aug_stop": aug_stop,
            "milestones": milestones,
        }

    def _should_eval(self, epoch, schedule):
        if epoch in schedule["forced"]:
            return True
        if epoch < schedule["phase1_end"]:
            freq = schedule["eval_freq_phase1"]
        elif epoch < schedule["phase2_end"]:
            freq = schedule["eval_freq_phase2"]
        else:
            freq = schedule["eval_freq_phase3"]
        return ((epoch + 1) % freq == 0)

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

        n_parameters, model_stats = stats(self.cfg)
        print(model_stats)

        schedule = self._resolve_eval_schedule()
        print("-" * 42 + "Start training" + "-" * 43)
        print(
            f"[SparseEval] epochs={schedule['epochs']}  "
            f"phase1_end={schedule['phase1_end']} (every {schedule['eval_freq_phase1']})  "
            f"phase2_end={schedule['phase2_end']} (every {schedule['eval_freq_phase2']})  "
            f"phase3 (every {schedule['eval_freq_phase3']})  "
            f"aug_stop={schedule['aug_stop']}  "
            f"milestones={schedule['milestones']}  "
            f"forced_eval_epochs={sorted(schedule['forced'])}"
        )

        top1 = 0
        best_stat = {"epoch": -1}
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
                self.use_wandb,
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
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f"checkpoint{epoch:04}.pth")
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            do_eval = self._should_eval(epoch, schedule)
            if not do_eval:
                print(f"[SparseEval] epoch {epoch}: eval SKIPPED (next eval at the next scheduled epoch)")
                continue

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
                print(f"best_stat: {best_stat_print}")

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
                    best_stat = {"epoch": -1}
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

            if self.use_wandb:
                wandb_logs = {}
                for idx, metric_name in enumerate(metric_names):
                    wandb_logs[f"metrics/{metric_name}"] = test_stats["coco_eval_bbox"][idx]
                wandb_logs["epoch"] = epoch
                wandb.log(wandb_logs)

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

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
