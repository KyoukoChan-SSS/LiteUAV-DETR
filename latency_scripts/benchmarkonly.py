#!/usr/bin/env python3
"""
benchmarkonly: latency-only detector benchmark, torch fp32, batch=1.

LiteUAVDETR model loading is adapted from D-FINE:
https://github.com/Peterande/D-FINE

Derived from benchmarkv1.py but deliberately slim — it keeps ONLY the two
latency tests we report and drops accuracy, FLOPs tracing, kernel profiling,
component/op hooks, and non-fp32 precisions.

Two clearly separated tests
---------------------------
  B (inference): the model's whole per-frame inference — forward AND
                 postprocess (NMS / DETR decode) — on ONE preloaded,
                 preprocessed input (a real image when a dataset is configured,
                 matching the CoreML B convention; uniform random fallback).
                 NO image read, NO preprocessing in the loop, NO hooks.
                 Timed as a single fused CUDA-event window (e0 fwd e1 post e2)
                 with the forward/postprocess breakdown from the sub-windows.
                 --> benchmark_inference()

  C (end2end)  : per-image real-world latency = dataloading (read + preprocess
                 + H2D) + forward + postprocess. The complete deployed pipeline.
                 Each window is timed separately and also summed.
                 --> benchmark_end2end()

Both B and C include the family-correct postprocess; the ONLY difference is
that C adds the image read + preprocessing that B omits.

Both tests time the GPU with CUDA events bracketed by torch.cuda.synchronize,
so the numbers are pure device-side latency (the trailing sync only makes the
event readable; it is not inside the measured window). B additionally asserts
no forward/backward hooks are attached, so nothing contaminates the timing.

Postprocess is dispatched per model family so C is fair across detectors:
  * LiteUAVDETR family         : LiteUAVDETRPostProcessor (topk decode)
  * YOLO   (yolo_master/v13/hic): ops.non_max_suppression  (fork defaults)
  * YOLO   end2end (yolo26/v10): confidence filter only (NMS-free head)
  * RT-DETR (csfpr / uavdetr)  : DETR decode (xywh2xyxy + conf + topk, no NMS)
  * DOME   (Dome-DETR)         : the repo's own cfg.postprocessor

No graph surgery: fuse/deploy is NOT applied, so every model is timed as its
native eval graph (fused numbers come from the ONNX/TRT stage). See --help.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

Tensor = torch.Tensor
PRECISION = "fp32"  # this benchmark is fp32-only by design


@dataclass
class LoadedModel:
    name: str
    family: str
    model: nn.Module
    forward: Callable[[Tensor], Any]
    # postprocess(outputs, orig_sizes) -> list[dict]; orig_sizes is [[w, h]].
    postprocess: Optional[Callable[[Any, Tensor], List[Dict[str, Tensor]]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# stats helpers (identical math to benchmarkv1)
# ─────────────────────────────────────────────────────────────────────────────
def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize_ms(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    if not vals:
        return {}
    mean = statistics.fmean(vals)
    return {
        "count": len(vals),
        "mean_ms": mean,
        "std_ms": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "min_ms": min(vals),
        "max_ms": max(vals),
        "p50_ms": percentile(vals, 50),
        "p90_ms": percentile(vals, 90),
        "p95_ms": percentile(vals, 95),
        "p99_ms": percentile(vals, 99),
        "fps": 1000.0 / mean if mean > 0 else 0.0,
    }


def current_cpu_rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 1024**2
    except Exception:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


_TEGRA_GR3D = re.compile(r"GR3D_FREQ\s+(\d+)%")
_TEGRA_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")
# Orin/Xavier: "VDD_IN 5678mW/5600mW". Older Nano/TX2: "POM_5V_GPU 400/380" (mW,
# no unit). Cover both so power/energy work across Jetson generations.
_TEGRA_RAILS = re.compile(r"([A-Z0-9_]+)\s+(\d+)mW/(\d+)mW")
_TEGRA_POM = re.compile(r"(POM_[A-Z0-9_]+)\s+(\d+)/(\d+)")


def _tegra_rails(line: str) -> dict[str, float]:
    rails = {n: float(c) for n, c, _ in _TEGRA_RAILS.findall(line)}
    rails.update({n: float(c) for n, c, _ in _TEGRA_POM.findall(line)})
    return rails


def _tegra_total_mw(rails: dict[str, float]) -> float:
    # Prefer a whole-module input rail (*_IN); else sum the compute rails.
    for k, v in rails.items():
        if k.endswith("_IN"):
            return v
    return sum(
        v for k, v in rails.items() if any(t in k for t in ("GPU", "SOC", "CV", "CPU"))
    )


class ResourceSampler:
    """Background GPU util / power sampler, portable across our targets.

    Auto-detects the platform:
      * nvidia-smi present (RTX 5090, T4)  -> poll nvidia-smi.
      * tegrastats present (Jetson)        -> stream tegrastats (Jetson has no
        working nvidia-smi; GPU load is GR3D_FREQ, power comes from the VDD
        rails in mW). Energy = integral of power over wall time.
      * neither                            -> no-op (empty summary).
    """

    def __init__(self, device: torch.device, interval_s: float = 0.05):
        self.device = device
        self.interval_s = interval_s
        self.samples: List[Dict[str, float]] = []
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.proc: Optional[subprocess.Popen] = None
        self.backend = self._detect()
        self.t0 = 0.0
        self.t1 = 0.0

    def _detect(self) -> Optional[str]:
        if self.device.type != "cuda":
            return None
        if shutil.which("nvidia-smi"):
            return "nvidia-smi"
        if shutil.which("tegrastats"):
            return "tegrastats"
        return None

    def start(self) -> None:
        if self.backend is None:
            return
        self.t0 = time.perf_counter()
        target = self._loop_nvsmi if self.backend == "nvidia-smi" else self._loop_tegra
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def _loop_nvsmi(self) -> None:
        index = self.device.index if self.device.index is not None else 0
        cmd = [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self.stop_event.is_set():
            try:
                out = subprocess.check_output(
                    cmd, text=True, stderr=subprocess.DEVNULL, timeout=1
                )
                p = [float(x.strip()) for x in out.strip().split(",")[:4]]
                if len(p) == 4:
                    self.samples.append(
                        {
                            "gpu_util_pct": p[0],
                            "mem_util_pct": p[1],
                            "mem_used_mb": p[2],
                            "power_w": p[3],
                        }
                    )
            except Exception:
                pass
            time.sleep(self.interval_s)

    def _loop_tegra(self) -> None:
        interval_ms = max(1, int(self.interval_s * 1000))
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return
        for line in self.proc.stdout:  # type: ignore[union-attr]
            if self.stop_event.is_set():
                break
            sample: Dict[str, float] = {}
            m = _TEGRA_GR3D.search(line)
            if m:
                sample["gpu_util_pct"] = float(m.group(1))
            m = _TEGRA_RAM.search(line)
            if m:
                sample["mem_used_mb"] = float(m.group(1))
            rails = _tegra_rails(line)
            if rails:
                sample["power_w"] = _tegra_total_mw(rails) / 1000.0
                sample["rails_mw"] = rails  # type: ignore[assignment]
            if sample:
                self.samples.append(sample)

    def stop(self) -> Dict[str, float]:
        self.stop_event.set()
        self.t1 = time.perf_counter()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2)
        result: Dict[str, float] = {"backend": self.backend or "none"}  # type: ignore[dict-item]
        for key in ("gpu_util_pct", "mem_util_pct", "mem_used_mb", "power_w"):
            vals = [s[key] for s in self.samples if key in s]
            if vals:
                result[f"{key}_mean"] = statistics.fmean(vals)
                result[f"{key}_max"] = max(vals)
        power_vals = [s["power_w"] for s in self.samples if "power_w" in s]
        if power_vals:
            elapsed = max(1e-9, self.t1 - self.t0)
            result["energy_j"] = statistics.fmean(power_vals) * elapsed
            result["samples"] = len(power_vals)
        return result


def _cuda_time(fn: Callable[[], Any], device: torch.device) -> Tuple[Any, float]:
    """Run fn() once, return (result, elapsed_ms). CUDA-event timed with sync."""
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        out = fn()
        end.record()
        torch.cuda.synchronize(device)
        return out, float(start.elapsed_time(end))
    t0 = time.perf_counter()
    out = fn()
    return out, (time.perf_counter() - t0) * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# import / checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────
def prepare_repo_on_path(repo_root: Optional[str]) -> Optional[Path]:
    """Put a LiteUAVDETR/Dome-style repo (must contain src/) first on sys.path."""
    if not repo_root:
        return None
    root = Path(repo_root).expanduser().resolve()
    if not (root / "src").exists():
        raise FileNotFoundError(f"repo must contain src/: {root}")
    root_str = str(root)
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)
    importlib.invalidate_caches()
    return root


def import_ultralytics_model(ultralytics_root: Optional[str], api: str):
    if ultralytics_root:
        root = Path(ultralytics_root).expanduser().resolve()
        if not (root / "ultralytics").exists():
            raise FileNotFoundError(
                f"--ultralytics-root must contain ultralytics/: {root}"
            )
        sys.path.insert(0, str(root))
    try:
        if api == "liteuavdetr":
            from ultralytics import LiteUAVDETR  # type: ignore

            return LiteUAVDETR
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"Could not import ultralytics api={api} (root={ultralytics_root})."
        ) from exc
    return YOLO


def load_checkpoint_state(path: str) -> Dict[str, Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        return checkpoint
    if (
        "ema" in checkpoint
        and isinstance(checkpoint["ema"], dict)
        and "module" in checkpoint["ema"]
    ):
        return checkpoint["ema"]["module"]
    for key in ("model", "state_dict", "module"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# per-family postprocess builders  (the C-test fairness lives here)
# ─────────────────────────────────────────────────────────────────────────────
def _scale_xyxy_stretch(
    det: Tensor, orig_wh: Tuple[float, float], in_wh: Tuple[int, int]
) -> Tensor:
    """Rescale xyxy boxes from model-input pixels to original image pixels.

    Preprocess here is a stretch resize (no letterbox), so x and y scale
    independently. Box CORRECTNESS is not the point (no accuracy check) — this
    keeps C-test outputs sane and, more importantly, exercises the same tensor
    ops a real deploy postprocess would.
    """
    if det.numel() == 0:
        return det
    w0, h0 = orig_wh
    in_w, in_h = in_wh
    det[:, [0, 2]] *= w0 / in_w
    det[:, [1, 3]] *= h0 / in_h
    return det


def resolve_ultralytics_predict_params(args: argparse.Namespace) -> Dict[str, Any]:
    """Resolve postprocess args the same way YOLO26's predict path does.

    YOLO26's Model.predict() adds method defaults on top of default.yaml:
    conf=0.25 for predict, while iou/max_det come from DEFAULT_CFG
    (default.yaml: iou=0.7, max_det=300). CLI overrides here are only for
    deliberate sweeps; the default path mirrors the fork.
    """
    try:
        from ultralytics.utils import DEFAULT_CFG  # type: ignore

        default_iou = DEFAULT_CFG.iou if DEFAULT_CFG.iou is not None else 0.7
        default_max_det = (
            DEFAULT_CFG.max_det if DEFAULT_CFG.max_det is not None else 300
        )
        default_agnostic = bool(getattr(DEFAULT_CFG, "agnostic_nms", False))
    except Exception:
        default_iou, default_max_det, default_agnostic = 0.7, 300, False

    conf = args.ultralytics_conf if args.ultralytics_conf is not None else 0.25
    iou = args.ultralytics_iou if args.ultralytics_iou is not None else default_iou
    max_det = (
        args.ultralytics_max_det
        if args.ultralytics_max_det is not None
        else default_max_det
    )
    return {
        "conf": float(conf),
        "iou": float(iou),
        "max_det": int(max_det),
        "classes": None,
        "agnostic_nms": default_agnostic,
        "source": "YOLO26 Model.predict defaults + DEFAULT_CFG unless CLI-overridden",
    }


def build_ultralytics_postprocess(
    inner: nn.Module, input_size: Tuple[int, int], args: argparse.Namespace
):
    """Return (family, postprocess_fn, parameters) for the detection head.

    YOLO detect head  -> NMS  (fork's own default conf/iou).
    end2end head      -> confidence filter only (NMS-free, v10/26-style).
    RT-DETR decoder   -> DETR decode (xywh2xyxy + conf + topk), no NMS.
    """
    try:
        ul_ops = importlib.import_module("ultralytics.utils.ops")
    except ImportError as exc:
        print(
            f"[warn] ultralytics.utils.ops unavailable ({exc}); postprocess is a no-op."
        )
        return (
            "ultralytics_unknown",
            (lambda outputs, orig: []),
            {"source": "unavailable"},
        )

    if bool(getattr(inner, "liteuavdetr", False)):
        postprocessor = inner.postprocessor.to(next(inner.parameters()).device).eval()
        postprocessor.deploy_mode = True
        predict_params = {
            "num_top_queries": int(postprocessor.num_top_queries),
            "source": "LiteUAVDETR release postprocessor",
        }

        def postprocess(outputs: Any, orig_sizes: Tensor) -> Any:
            return postprocessor(outputs, orig_sizes)

        print(
            f"[postprocess] ultralytics_liteuavdetr: topk={postprocessor.num_top_queries}"
        )
        return "ultralytics_liteuavdetr", postprocess, predict_params

    head = inner.model[-1] if hasattr(inner, "model") else None
    head_name = type(head).__name__ if head is not None else ""
    nms_free = bool(getattr(head, "end2end", False))
    is_rtdetr = (
        "RTDETR" in head_name or "Transformer" in head_name or "DETR" in head_name
    )
    predict_params = resolve_ultralytics_predict_params(args)
    conf = predict_params["conf"]
    iou = predict_params["iou"]
    max_det = predict_params["max_det"]
    classes = predict_params["classes"]
    agnostic_nms = predict_params["agnostic_nms"]
    in_h, in_w = int(input_size[0]), int(input_size[1])
    in_wh = (in_w, in_h)

    if is_rtdetr:
        family = "rtdetr"

        def postprocess(outputs: Any, orig_sizes: Tensor) -> List[Dict[str, Tensor]]:
            preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            nd = preds.shape[-1]
            bboxes, scores = preds.split((4, nd - 4), dim=-1)
            results: List[Dict[str, Tensor]] = []
            for i in range(preds.shape[0]):
                bbox = ul_ops.xywh2xyxy(bboxes[i])  # normalized -> xyxy(0..1)
                max_score, cls = scores[i].max(-1, keepdim=True)
                idx = max_score.squeeze(-1) > conf  # YOLO26 predict default: 0.25
                pred = torch.cat([bbox, max_score, cls.float()], dim=-1)[idx]
                pred = pred[pred[:, 4].argsort(descending=True)][:max_det]
                if pred.numel():
                    w0, h0 = float(orig_sizes[i, 0]), float(orig_sizes[i, 1])
                    pred[:, [0, 2]] *= w0  # coords are 0..1
                    pred[:, [1, 3]] *= h0
                results.append(
                    {"boxes": pred[:, :4], "scores": pred[:, 4], "labels": pred[:, 5]}
                )
            return results

        print(f"[postprocess] {family}: conf={conf} max_det={max_det}")
        return family, postprocess, predict_params

    family = "yolo_end2end" if nms_free else "yolo_nms"

    # Mirror the fork's own DetectionPredictor design exactly: ONE call to its
    # non_max_suppression for every detect model — the function itself handles
    # end2end heads internally (shape[-1]==6 / end2end flag -> conf filter,
    # no NMS). Thresholds come from the fork's DEFAULT_CFG the way the
    # predictor resolves them (conf None -> 0.25 predict default, iou from
    # default.yaml, max_det 300) — NOT this benchmark's own numbers.
    try:
        nms_mod = importlib.import_module("ultralytics.utils.nms")  # new forks
    except ImportError:
        nms_mod = ul_ops  # older forks
    nms_fn = nms_mod.non_max_suppression
    import inspect

    nms_sig = inspect.signature(nms_fn).parameters
    nms_kwargs: Dict[str, Any] = {}
    if "classes" in nms_sig:
        nms_kwargs["classes"] = classes
    if "agnostic" in nms_sig:
        nms_kwargs["agnostic"] = agnostic_nms
    if "max_det" in nms_sig:
        nms_kwargs["max_det"] = max_det
    if "nc" in nms_sig:
        nms_kwargs["nc"] = 0
    if "rotated" in nms_sig:
        nms_kwargs["rotated"] = False
    if "end2end" in nms_sig:
        nms_kwargs["end2end"] = nms_free
    print(
        f"[postprocess] {family}: nms={nms_mod.__name__} conf={conf} "
        f"iou={iou} max_det={max_det} agnostic={agnostic_nms}"
    )

    def postprocess(outputs: Any, orig_sizes: Tensor) -> List[Dict[str, Tensor]]:
        preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        dets = nms_fn(preds, conf, iou, **nms_kwargs)
        results: List[Dict[str, Tensor]] = []
        for i, det in enumerate(dets):
            det = det.clone()
            det = _scale_xyxy_stretch(
                det, (float(orig_sizes[i, 0]), float(orig_sizes[i, 1])), in_wh
            )
            results.append(
                {"boxes": det[:, :4], "scores": det[:, 4], "labels": det[:, 5]}
            )
        return results

    return family, postprocess, predict_params


# ─────────────────────────────────────────────────────────────────────────────
# model loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_liteuavdetr(
    args: argparse.Namespace, device: torch.device, *, family: str = "liteuavdetr"
) -> LoadedModel:
    """Load LiteUAVDETR or a compatible Dome model via D-FINE's YAMLConfig.

    DOME-DETR shares the D-FINE repo layout (src.core.YAMLConfig + a
    postprocessor on cfg), so it loads through this exact path — point
    --repo at the Dome-DETR checkout and pass its config. cfg.postprocessor
    then provides DOME's own correct decode.
    """
    external_root = prepare_repo_on_path(args.repo)
    from src.core import YAMLConfig, yaml_utils

    importlib.import_module("src.zoo")

    if not args.config:
        raise ValueError("--config is required for LiteUAVDETR / DOME models")
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    cfg = YAMLConfig(args.config, resume=args.weights or None, **update_dict)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
        if (
            external_root is not None
            and (external_root / "weight" / "hgnetv2").exists()
        ):
            cfg.yaml_cfg["HGNetv2"]["local_model_dir"] = str(
                external_root / "weight" / "hgnetv2"
            )

    model = cfg.model
    weight_status = "random_init"
    if args.weights:
        state = load_checkpoint_state(args.weights)
        missing, unexpected = model.load_state_dict(state, strict=False)
        weight_status = "loaded"
        if family == "liteuavdetr" and (missing or unexpected):
            raise RuntimeError(
                f"LiteUAVDETR checkpoint/config mismatch: missing={list(missing)[:20]} "
                f"unexpected={list(unexpected)[:20]}"
            )
        if missing:
            print(f"[warn] missing checkpoint keys: {len(missing)}")
        if unexpected:
            print(f"[warn] unexpected checkpoint keys: {len(unexpected)}")
    model.to(device).eval()

    # Optional fold-all protocol (--fold): fold Conv+BN via the model's own
    # deployment path. Default stays unfused.
    fold = bool(getattr(args, "fold", False))
    if fold:
        if hasattr(model, "deploy"):
            model = model.deploy()
        else:
            for m in model.modules():
                if hasattr(m, "convert_to_deploy"):
                    m.convert_to_deploy()
        model.eval()

    postprocessor = getattr(cfg, "postprocessor", None)
    if postprocessor is not None:
        postprocessor.to(device).eval()
        # Deploy postprocess path (labels/boxes/scores tensors), NOT the eval
        # path: eval builds per-image dicts and, with remap_mscoco_category,
        # runs a per-element .item() python loop — eval-only bookkeeping that
        # would contaminate latency. Mirrors DetQuant _set_deploy_postprocess.
        # (This selects an output path only; it is not Conv+BN graph surgery,
        # so it applies in BOTH fold and no-fold protocols.)
        if hasattr(postprocessor, "deploy_mode"):
            postprocessor.deploy_mode = True

    def postprocess(outputs: Any, orig_sizes: Tensor) -> List[Dict[str, Tensor]]:
        if postprocessor is None:
            return []
        return postprocessor(outputs, orig_sizes)

    return LoadedModel(
        name=family,
        family=family,
        model=model,
        forward=lambda x: model(x),
        postprocess=postprocess,
        metadata={"weight_status": weight_status, "fold": fold},
    )


class WrappedUltralytics(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, x: Tensor) -> Any:
        return self.inner(x)


def load_ultralytics(args: argparse.Namespace, device: torch.device) -> LoadedModel:
    UltralyticsModel = import_ultralytics_model(
        args.ultralytics_root, args.ultralytics_api
    )
    # Some legacy local Ultralytics/RT-DETR forks saved full model objects.
    # PyTorch 2.6 defaults torch.load(weights_only=True), which rejects those
    # trusted experiment checkpoints before the fork can build the model.
    original_torch_load = torch.load

    def legacy_torch_load(*load_args: Any, **load_kwargs: Any) -> Any:
        load_kwargs.setdefault("weights_only", False)
        return original_torch_load(*load_args, **load_kwargs)

    torch.load = legacy_torch_load  # type: ignore[assignment]
    try:
        facade = UltralyticsModel(args.weights)
    finally:
        torch.load = original_torch_load  # type: ignore[assignment]
    inner = facade.model.to(device).eval()
    # Optional fold-all protocol (--fold): ultralytics' own Conv+BN fold —
    # the family-equivalent of LiteUAVDETR deploy(). Default stays unfused.
    fold = bool(getattr(args, "fold", False))
    if fold and bool(getattr(inner, "liteuavdetr", False)) and hasattr(inner, "deploy"):
        inner = inner.deploy()
    elif fold and hasattr(inner, "fuse"):
        try:
            inner = inner.fuse()
        except Exception as exc:
            print(
                f"[warn] .fuse() failed ({type(exc).__name__}: {exc}); running unfused"
            )
            fold = False
    # Put the head in inference/export mode so forward strips training-only
    # auxiliary outputs — closer to what predict() runs at deploy time.
    try:
        last = inner.model[-1] if hasattr(inner, "model") else None
        if last is not None and hasattr(last, "export"):
            last.export = True
    except Exception:
        pass
    model = WrappedUltralytics(inner).to(device).eval()
    family, postprocess, predict_params = build_ultralytics_postprocess(
        inner, tuple(args.input_size), args
    )
    return LoadedModel(
        name=family,
        family=family,
        model=model,
        forward=lambda x: model(x),
        postprocess=postprocess,
        metadata={
            "head": type(inner.model[-1]).__name__ if hasattr(inner, "model") else "?",
            "ultralytics_api": args.ultralytics_api,
            "predict_params": predict_params,
            "fold": fold,
        },
    )


def load_model(args: argparse.Namespace, device: torch.device) -> LoadedModel:
    if args.model_type == "liteuavdetr":
        return load_liteuavdetr(args, device, family="liteuavdetr")
    if args.model_type == "dome":
        return load_liteuavdetr(args, device, family="dome")
    if args.model_type == "ultralytics":
        return load_ultralytics(args, device)
    raise ValueError(f"Unsupported model type: {args.model_type}")


# ─────────────────────────────────────────────────────────────────────────────
# B test — pure model inference latency (forward + postprocess), synthetic input
# ─────────────────────────────────────────────────────────────────────────────
def _assert_no_forward_hooks(model: nn.Module) -> int:
    """Count any forward/pre hooks. B must be pure model latency, so anything
    hooked (profilers, feature extractors) would contaminate the timing. We
    only observe — return the count so main() can record/flag it."""
    n = 0
    for m in model.modules():
        n += len(getattr(m, "_forward_hooks", {}))
        n += len(getattr(m, "_forward_pre_hooks", {}))
        n += len(getattr(m, "_backward_hooks", {}))
    return n


def _b_input(args: argparse.Namespace, device: torch.device) -> Tuple[Tensor, str]:
    """B input: prefer ONE real preprocessed image (same convention as the
    CoreML B test, and it gives realistic NMS candidate counts — random input
    produces near-zero boxes above conf, under-measuring YOLO postprocess).
    Falls back to uniform random in [0,1) — the normalized-image range —
    when no dataset is configured."""
    h, w = args.input_size
    try:
        paths = iter_image_paths(args)
    except Exception:
        paths = []
    if paths:
        from PIL import Image
        import torchvision.transforms as T

        transform = T.Compose([T.Resize((h, w)), T.ToTensor()])
        x = transform(Image.open(paths[0]).convert("RGB")).unsqueeze(0).to(device)
        return x, f"first preprocessed image ({paths[0].name})"
    return torch.rand(1, 3, h, w, device=device), "uniform random [0,1)"


def benchmark_inference(
    loaded: LoadedModel,
    args: argparse.Namespace,
    device: torch.device,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    """B: the model's whole per-frame inference — forward AND postprocess
    (NMS / DETR decode) — on one preloaded, preprocessed input. No image read,
    no preprocessing in the loop, no hooks: pure model inference latency, batch=1.

    Timing is ONE stream pass with three CUDA events (e0 forward e1 post e2):
    inference = e0->e2 is a single fused window (includes the host gap between
    stages, exactly what deployment pays); forward = e0->e1 and postprocess =
    e1->e2 are its breakdown. No host sync between stages inside the window.
    """
    h, w = args.input_size
    x, input_source = _b_input(args, device)
    orig = torch.tensor([[float(w), float(h)]], device=device)
    has_post = loaded.postprocess is not None

    fwd_ms: List[float] = []
    post_ms: List[float] = []
    infer_ms: List[float] = []
    postprocess_errors = 0

    hooks = _assert_no_forward_hooks(loaded.model)
    use_events = device.type == "cuda"

    with torch.inference_mode():
        for _ in range(warmup):
            out = loaded.forward(x)
            if has_post:
                try:
                    loaded.postprocess(out, orig)
                except Exception:
                    pass
        if use_events:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        for _ in range(iters):
            if use_events:
                e0 = torch.cuda.Event(enable_timing=True)
                e1 = torch.cuda.Event(enable_timing=True)
                e2 = torch.cuda.Event(enable_timing=True)
                e0.record()
                out = loaded.forward(x)
                e1.record()
                if has_post:
                    try:
                        loaded.postprocess(out, orig)
                    except Exception as exc:
                        postprocess_errors += 1
                        if postprocess_errors == 1:
                            print(
                                f"[warn] B postprocess raised {type(exc).__name__}: {exc}"
                            )
                e2.record()
                torch.cuda.synchronize(device)
                f = float(e0.elapsed_time(e1))
                p = float(e1.elapsed_time(e2))
                total = float(e0.elapsed_time(e2))
            else:
                t0 = time.perf_counter()
                out = loaded.forward(x)
                t1 = time.perf_counter()
                if has_post:
                    try:
                        loaded.postprocess(out, orig)
                    except Exception as exc:
                        postprocess_errors += 1
                        if postprocess_errors == 1:
                            print(
                                f"[warn] B postprocess raised {type(exc).__name__}: {exc}"
                            )
                t2 = time.perf_counter()
                f = (t1 - t0) * 1000.0
                p = (t2 - t1) * 1000.0
                total = (t2 - t0) * 1000.0
            fwd_ms.append(f)
            post_ms.append(p)
            infer_ms.append(total)

    result: Dict[str, Any] = {
        "scope": "model inference = forward + postprocess, single fused window (batch=1, no hooks)",
        "input_source": input_source,
        "forward_hooks_detected": hooks,
        "postprocess_errors": postprocess_errors,
        "inference_latency": summarize_ms(infer_ms),
        "forward_latency": summarize_ms(fwd_ms),
        "postprocess_latency": summarize_ms(post_ms),
        "per_iter_inference_ms": infer_ms,
        "per_iter_forward_ms": fwd_ms,
        "per_iter_post_ms": post_ms,
        "memory": {"cpu_rss_mb": current_cpu_rss_mb()},
    }
    if device.type == "cuda":
        result["memory"]["cuda_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**2
        )
        result["memory"]["cuda_peak_reserved_mb"] = (
            torch.cuda.max_memory_reserved(device) / 1024**2
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# C test — end-to-end per-image latency (real images)
# ─────────────────────────────────────────────────────────────────────────────
def iter_image_paths(args: argparse.Namespace) -> List[Path]:
    if args.dataset_type == "coco":
        if not args.coco_ann or not args.coco_images:
            raise ValueError(
                "--coco-ann and --coco-images required for dataset-type=coco."
            )
        from pycocotools.coco import COCO

        coco = COCO(args.coco_ann)
        ids = coco.getImgIds()
        if args.num_images:
            ids = ids[: args.num_images]
        base = Path(args.coco_images)
        return [base / coco.loadImgs([i])[0]["file_name"] for i in ids]
    if args.dataset_type == "folder":
        if not args.image_folder:
            raise ValueError("--image-folder required for dataset-type=folder.")
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
        paths = sorted(
            p
            for p in Path(args.image_folder).rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
        return paths[: args.num_images] if args.num_images else paths
    raise ValueError(f"Unsupported dataset-type: {args.dataset_type}")


def benchmark_end2end(
    loaded: LoadedModel, args: argparse.Namespace, device: torch.device
) -> Dict[str, Any]:
    from PIL import Image
    import torchvision.transforms as T

    paths = iter_image_paths(args)
    if not paths:
        raise ValueError("Dataset produced zero image paths.")

    h, w = args.input_size
    transform = T.Compose([T.Resize((h, w)), T.ToTensor()])

    # Warmup forward AND postprocess so the first real image pays neither CUDA
    # lazy init / cudnn autotune nor first-call postprocess (NMS kernel) cost.
    x_syn = torch.rand(1, 3, h, w, device=device)
    orig_syn = torch.tensor([[float(w), float(h)]], device=device)
    with torch.inference_mode():
        for _ in range(max(1, args.warmup)):
            out_syn = loaded.forward(x_syn)
            if loaded.postprocess is not None:
                try:
                    loaded.postprocess(out_syn, orig_syn)
                except Exception:
                    pass
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    load_ms: List[float] = []
    fwd_ms: List[float] = []
    post_ms: List[float] = []
    total_ms: List[float] = []
    postprocess_errors = 0

    sampler = ResourceSampler(device)
    sampler.start()

    with torch.inference_mode():
        for p in paths:
            # ── dataloading window: read + preprocess + H2D ──
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            image = Image.open(p).convert("RGB")
            w0, h0 = image.size
            x = transform(image).unsqueeze(0).to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            load = (time.perf_counter() - t0) * 1000.0

            # ── forward window ──
            outputs, fwd = _cuda_time(lambda: loaded.forward(x), device)

            # ── postprocess window (family-correct) ──
            post = 0.0
            if loaded.postprocess is not None:
                orig = torch.tensor([[float(w0), float(h0)]], device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                tp = time.perf_counter()
                try:
                    loaded.postprocess(outputs, orig)
                except Exception as exc:
                    postprocess_errors += 1
                    if postprocess_errors == 1:
                        print(
                            f"[warn] postprocess raised {type(exc).__name__}: {exc} "
                            "(further errors counted silently; see postprocess_errors)"
                        )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                post = (time.perf_counter() - tp) * 1000.0

            load_ms.append(load)
            fwd_ms.append(fwd)
            post_ms.append(post)
            total_ms.append(load + fwd + post)

    utilization = sampler.stop()
    result: Dict[str, Any] = {
        "scope": "end2end per-image (dataloading + forward + postprocess)",
        "family": loaded.family,
        "num_images": len(paths),
        "postprocess_errors": postprocess_errors,
        "end2end_total_latency": summarize_ms(total_ms),
        "forward_latency": summarize_ms(fwd_ms),
        "dataloading_latency": summarize_ms(load_ms),
        "postprocess_latency": summarize_ms(post_ms),
        "per_image_total_ms": total_ms,
        "per_image_forward_ms": fwd_ms,
        "per_image_load_ms": load_ms,
        "per_image_post_ms": post_ms,
        "utilization": utilization,
    }
    if device.type == "cuda":
        result["memory"] = {
            "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
            "cuda_peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# report
# ─────────────────────────────────────────────────────────────────────────────
def _row(label: str, s: Dict[str, float]) -> str:
    if not s:
        return f"  {label:<22s}  (no data)"
    return (
        f"  {label:<22s}  mean={s['mean_ms']:8.3f}  std={s.get('std_ms', 0.0):7.3f}  "
        f"p50={s['p50_ms']:8.3f}  p95={s['p95_ms']:8.3f}  fps={s['fps']:8.2f}"
    )


def build_text_report(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    meta = data.get("meta", {})
    lines.append("=" * 78)
    lines.append(
        f"benchmarkonly  tag={meta.get('tag')}  family={meta.get('family')}  "
        f"precision=fp32  fold={'ON' if meta.get('fold') else 'off'}  device={meta.get('device')}"
    )
    lines.append(
        f"input_size={meta.get('input_size')}  warmup={meta.get('warmup')}  iters={meta.get('iters')}"
    )
    lines.append("=" * 78)
    if "B_inference" in data:
        b = data["B_inference"]
        lines.append(
            "[B] model inference = forward + postprocess (synthetic, batch=1, no hooks):"
        )
        lines.append(_row("inference total", b.get("inference_latency", {})))
        lines.append(_row("  forward", b.get("forward_latency", {})))
        lines.append(_row("  postprocess", b.get("postprocess_latency", {})))
        if b.get("forward_hooks_detected"):
            lines.append(
                f"  !! forward_hooks_detected={b['forward_hooks_detected']} (B is meant to be hook-free)"
            )
        if b.get("postprocess_errors"):
            lines.append(f"  !! postprocess_errors={b['postprocess_errors']}")
    if "C_end2end" in data:
        c = data["C_end2end"]
        lines.append(
            f"[C] end2end per-image (n={c.get('num_images')}, family={c.get('family')}):"
        )
        lines.append(_row("end2end total", c.get("end2end_total_latency", {})))
        lines.append(_row("  forward", c.get("forward_latency", {})))
        lines.append(_row("  dataloading", c.get("dataloading_latency", {})))
        lines.append(_row("  postprocess", c.get("postprocess_latency", {})))
        if c.get("postprocess_errors"):
            lines.append(f"  !! postprocess_errors={c['postprocess_errors']}")
    lines.append("=" * 78)
    return "\n".join(lines)


def save_json(path: str, data: Dict[str, Any]) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        return str(obj)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(convert(data), indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# cli
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="benchmarkonly — fp32 latency-only detector benchmark (B: model, C: end2end).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--model-type", choices=["liteuavdetr", "dome", "ultralytics"], required=True
    )
    p.add_argument(
        "--repo",
        default=os.environ.get("BENCH_REPO_ROOT"),
        help="Repo root containing src/ for liteuavdetr/dome (or set BENCH_REPO_ROOT).",
    )
    p.add_argument("--config", default=None, help="YAML config for liteuavdetr/dome.")
    p.add_argument("--weights", default="", help="Checkpoint / .pt path.")
    p.add_argument(
        "-u",
        "--update",
        nargs="+",
        help="LiteUAVDETR YAML overrides (matches train.py -u).",
    )
    p.add_argument(
        "--ultralytics-root",
        default=None,
        help="Folder containing the ultralytics/ fork under test.",
    )
    p.add_argument(
        "--ultralytics-api",
        choices=["yolo", "liteuavdetr"],
        default="yolo",
        help="Ultralytics facade used to load --weights. Use liteuavdetr for this repository's bundled fork.",
    )
    p.add_argument(
        "--ultralytics-conf",
        type=float,
        default=None,
        help="Override YOLO predict conf; default mirrors YOLO26 predict conf=0.25.",
    )
    p.add_argument(
        "--ultralytics-iou",
        type=float,
        default=None,
        help="Override YOLO NMS IoU; default comes from the fork's DEFAULT_CFG.",
    )
    p.add_argument(
        "--ultralytics-max-det",
        type=int,
        default=None,
        help="Override YOLO max detections; default comes from the fork's DEFAULT_CFG.",
    )
    p.add_argument(
        "--fold",
        action="store_true",
        help="Fold Conv+BN for EVERY model before timing (LiteUAVDETR via "
        "deploy()/convert_to_deploy, ultralytics via .fuse()). Default off: "
        "all models run their native unfused eval graph.",
    )
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--input-size", nargs=2, type=int, default=[640, 640], metavar=("H", "W")
    )
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument(
        "--tests",
        nargs="+",
        choices=["B", "C"],
        default=["B", "C"],
        help="Which tests to run: B (model forward), C (end2end). Default both.",
    )
    # C-test dataset
    p.add_argument("--dataset-type", choices=["coco", "folder"], default="coco")
    p.add_argument("--coco-images", default=None)
    p.add_argument("--coco-ann", default=None)
    p.add_argument("--image-folder", default=None)
    p.add_argument("--num-images", type=int, default=500)
    p.add_argument("--output", default="latency_scripts/results/benchmarkonly.json")
    p.add_argument(
        "--log", default=None, help="Text report path (defaults to --output with .txt)."
    )
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    # fp32, no TF32: deterministic and comparable across models.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    loaded = load_model(args, device)
    print(f"[load] {loaded.name} family={loaded.family} meta={loaded.metadata}")

    data: Dict[str, Any] = {
        "meta": {
            "tag": args.tag,
            "family": loaded.family,
            "model_type": args.model_type,
            "precision": "fp32",
            "device": str(device),
            "input_size": list(args.input_size),
            "warmup": args.warmup,
            "iters": args.iters,
            "fold": bool(args.fold),
            "weights": args.weights,
            "metadata": loaded.metadata,
        }
    }

    if "B" in args.tests:
        print("[B] model inference latency (forward + postprocess) ...")
        data["B_inference"] = benchmark_inference(
            loaded, args, device, args.warmup, args.iters
        )

    if "C" in args.tests:
        print("[C] end2end per-image latency ...")
        data["C_end2end"] = benchmark_end2end(loaded, args, device)

    report = build_text_report(data)
    print(report)
    save_json(args.output, data)
    log_path = args.log or str(Path(args.output).with_suffix(".txt"))
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).write_text(report + "\n", encoding="utf-8")
    print(f"[saved] json={args.output}\n[saved] log={log_path}")


if __name__ == "__main__":
    main()
