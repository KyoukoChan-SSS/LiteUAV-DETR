#!/usr/bin/env python3
"""TensorRT B/C latency for LiteUAVDETR post-mode engines — Jetson-ready.

LiteUAVDETR model implementation is adapted from https://github.com/Peterande/D-FINE.

Same B/C protocol as the torch benchmarkonly.py, so numbers line up:

  B (inference): engine.execute on ONE preloaded, preprocessed input. The engine
                 is a POST-mode graph (forward + postprocess fused, outputs
                 labels/boxes/scores), so B is the whole per-frame inference.
  C (end2end)  : real image read + preprocess + H2D + engine.execute, per image.
                 C - B is the dataloading cost.

batch=1, fp16 (engine precision), warmup=50, iters/images=500, stats = mean+std.
Timed with CUDA events (device timeline). On Jetson, GPU util / power / energy
come from tegrastats; on discrete GPUs from nvidia-smi.

Runtime deps: tensorrt (Python), torch (Jetson JetPack build), numpy, pillow,
torchvision. No dependency on the rest of the repo — drop this folder on Jetson.
"""
from __future__ import annotations

import argparse
import collections
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
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# stats (identical math to benchmarkonly.py)
# ─────────────────────────────────────────────────────────────────────────────
def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def summarize_ms(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        "mean_ms": mean,
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min_ms": min(values),
        "max_ms": max(values),
        "p50_ms": percentile(values, 50),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "p99_ms": percentile(values, 99),
        "fps": 1000.0 / mean if mean > 0 else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# resource sampler (nvidia-smi | Jetson tegrastats), mirrors benchmarkonly
# ─────────────────────────────────────────────────────────────────────────────
_GR3D = re.compile(r"GR3D_FREQ\s+(\d+)%")
_RAM = re.compile(r"RAM\s+(\d+)/(\d+)MB")
_RAILS = re.compile(r"([A-Z0-9_]+)\s+(\d+)mW/(\d+)mW")
_POM = re.compile(r"(POM_[A-Z0-9_]+)\s+(\d+)/(\d+)")


class ResourceSampler:
    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self.util: List[float] = []
        self.power: List[float] = []
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None
        self.backend = "tegrastats" if shutil.which("tegrastats") else (
            "nvidia-smi" if shutil.which("nvidia-smi") else None)
        self.t0 = self.t1 = 0.0

    def start(self) -> None:
        if not self.backend:
            return
        self.t0 = time.perf_counter()
        self.thread = threading.Thread(
            target=self._tegra if self.backend == "tegrastats" else self._nvsmi, daemon=True)
        self.thread.start()

    def _nvsmi(self) -> None:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
               "--format=csv,noheader,nounits"]
        while not self.stop.is_set():
            try:
                out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=1)
                u, p = [float(x) for x in out.strip().splitlines()[0].split(",")[:2]]
                self.util.append(u)
                self.power.append(p)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def _tegra(self) -> None:
        try:
            self.proc = subprocess.Popen(
                ["tegrastats", "--interval", str(max(1, int(self.interval_s * 1000)))],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            return
        for line in self.proc.stdout:  # type: ignore[union-attr]
            if self.stop.is_set():
                break
            m = _GR3D.search(line)
            if m:
                self.util.append(float(m.group(1)))
            rails = {n: float(c) for n, c, _ in _RAILS.findall(line)}
            rails.update({n: float(c) for n, c, _ in _POM.findall(line)})
            if rails:
                total = next((v for k, v in rails.items() if k.endswith("_IN")), None)
                if total is None:
                    total = sum(v for k, v in rails.items()
                                if any(t in k for t in ("GPU", "SOC", "CV", "CPU")))
                self.power.append(total / 1000.0)

    def summary(self) -> Dict[str, Any]:
        self.stop.set()
        self.t1 = time.perf_counter()
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=2)
        out: Dict[str, Any] = {"backend": self.backend or "none"}
        if self.util:
            out["gpu_util_pct_mean"] = statistics.fmean(self.util)
            out["gpu_util_pct_max"] = max(self.util)
        if self.power:
            out["power_w_mean"] = statistics.fmean(self.power)
            out["power_w_max"] = max(self.power)
            out["energy_j"] = statistics.fmean(self.power) * max(1e-9, self.t1 - self.t0)
            out["samples"] = len(self.power)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# minimal TensorRT runner (torch buffers, execute_v2) — inlined for portability
# ─────────────────────────────────────────────────────────────────────────────
class TRTRunner:
    def __init__(self, engine_path: str, device: str = "cuda:0") -> None:
        import tensorrt as trt
        import torch

        self.torch = torch
        self.device = device
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        self.bindings: "collections.OrderedDict[str, Any]" = collections.OrderedDict()
        self.input_names: List[str] = []
        self.output_names: List[str] = []
        for name in self.engine:
            shape = list(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            if shape and shape[0] == -1:
                shape[0] = 1
                if is_input:
                    self.context.set_input_shape(name, shape)
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            self.bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())
            (self.input_names if is_input else self.output_names).append(name)
        self.addr = collections.OrderedDict((n, v.ptr) for n, v in self.bindings.items())

    def infer(self, blob: Dict[str, Any]) -> Dict[str, Any]:
        for n in self.input_names:
            t = blob[n]
            if t.dtype is not self.bindings[n].data.dtype:
                t = t.to(dtype=self.bindings[n].data.dtype)
            if list(t.shape) != list(self.bindings[n].shape):
                self.context.set_input_shape(n, tuple(t.shape))
                self.bindings[n] = self.bindings[n]._replace(shape=list(t.shape))
            self.addr[n] = t.data_ptr()
            blob[n] = t
        self.context.execute_v2(list(self.addr.values()))
        return {n: self.bindings[n].data for n in self.output_names}


# ─────────────────────────────────────────────────────────────────────────────
# input helpers
# ─────────────────────────────────────────────────────────────────────────────
def list_images(coco_ann: str | None, coco_images: str | None,
                image_folder: str | None, num: int) -> List[Path]:
    if coco_ann and coco_images:
        from pycocotools.coco import COCO
        coco = COCO(coco_ann)
        ids = coco.getImgIds()[:num] if num else coco.getImgIds()
        base = Path(coco_images)
        return [base / coco.loadImgs([i])[0]["file_name"] for i in ids]
    if image_folder:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        paths = sorted(p for p in Path(image_folder).rglob("*")
                       if p.is_file() and p.suffix.lower() in exts)
        return paths[:num] if num else paths
    return []


def preprocess(path: Path, size: int, device: str):
    import torch
    import torchvision.transforms as T
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    x = T.Compose([T.Resize((size, size)), T.ToTensor()])(im).unsqueeze(0).to(device)
    orig = torch.tensor([[w, h]], device=device)
    return x, orig


# ─────────────────────────────────────────────────────────────────────────────
# B and C
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_b(runner: TRTRunner, args, device: str) -> Dict[str, Any]:
    torch = runner.torch
    paths = list_images(args.coco_ann, args.coco_images, args.image_folder, args.num_images)
    if paths:
        x, orig = preprocess(paths[0], args.image_size, device)
        source = f"first preprocessed image ({paths[0].name})"
    else:
        x = torch.rand(1, 3, args.image_size, args.image_size, device=device)
        orig = torch.tensor([[args.image_size, args.image_size]], device=device)
        source = "uniform random [0,1)"
    blob = {"images": x, "orig_target_sizes": orig}

    for _ in range(args.warmup):
        runner.infer(dict(blob))
    torch.cuda.synchronize()
    sampler = ResourceSampler()
    sampler.start()
    times: List[float] = []
    for _ in range(args.iters):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        runner.infer(dict(blob))
        e1.record()
        torch.cuda.synchronize()
        times.append(float(e0.elapsed_time(e1)))
    util = sampler.summary()
    return {
        "scope": "engine inference (post-mode: forward+postprocess fused), batch=1",
        "input_source": source,
        "inference_latency": summarize_ms(times),
        "per_iter_inference_ms": times,
        "utilization": util,
    }


def benchmark_c(runner: TRTRunner, args, device: str) -> Dict[str, Any]:
    torch = runner.torch
    paths = list_images(args.coco_ann, args.coco_images, args.image_folder, args.num_images)
    if not paths:
        raise SystemExit("C test needs --coco-images/--coco-ann or --image-folder")

    # warmup on synthetic input (engine + first-call cost)
    xs = torch.rand(1, 3, args.image_size, args.image_size, device=device)
    orig_s = torch.tensor([[args.image_size, args.image_size]], device=device)
    for _ in range(args.warmup):
        runner.infer({"images": xs, "orig_target_sizes": orig_s})
    torch.cuda.synchronize()

    sampler = ResourceSampler()
    sampler.start()
    load_ms: List[float] = []
    infer_ms: List[float] = []
    total_ms: List[float] = []
    for p in paths:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x, orig = preprocess(p, args.image_size, device)  # read+resize+ToTensor+H2D
        torch.cuda.synchronize()
        load = (time.perf_counter() - t0) * 1000.0

        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        runner.infer({"images": x, "orig_target_sizes": orig})
        e1.record()
        torch.cuda.synchronize()
        inf = float(e0.elapsed_time(e1))

        load_ms.append(load)
        infer_ms.append(inf)
        total_ms.append(load + inf)
    util = sampler.summary()
    return {
        "scope": "end2end per-image (dataload+preprocess+H2D + engine infer)",
        "num_images": len(paths),
        "end2end_total_latency": summarize_ms(total_ms),
        "inference_latency": summarize_ms(infer_ms),
        "dataloading_latency": summarize_ms(load_ms),
        "per_image_total_ms": total_ms,
        "per_image_infer_ms": infer_ms,
        "per_image_load_ms": load_ms,
        "utilization": util,
    }


def manifest_model_entry(manifest_path: str | None, tag: str) -> Dict[str, Any] | None:
    if not manifest_path:
        return None
    path = Path(manifest_path)
    if not path.exists():
        return {"manifest": str(path), "warning": "manifest_not_found"}
    data = json.loads(path.read_text(encoding="utf-8"))
    for model in data.get("models", []):
        if model.get("tag") == tag:
            return model
    return {"manifest": str(path), "warning": f"tag_not_found:{tag}"}


def main() -> int:
    p = argparse.ArgumentParser(description="TensorRT B/C latency for LiteUAVDETR post-mode engines.")
    p.add_argument("--engine", required=True, help="Path to the .engine file.")
    p.add_argument("--tag", default=None)
    p.add_argument("--manifest", default=None, help="Optional trt_manifest.json; matching model entry is copied into output meta.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image-size", type=int, default=640)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--num-images", type=int, default=500)
    p.add_argument("--coco-images", default=os.environ.get("COCO_IMAGES"))
    p.add_argument("--coco-ann", default=os.environ.get("COCO_ANN"))
    p.add_argument("--image-folder", default=None)
    p.add_argument("--tests", nargs="+", choices=["B", "C"], default=["B", "C"])
    p.add_argument("--output", required=True)
    args = p.parse_args()

    runner = TRTRunner(args.engine, args.device)
    tag = args.tag or Path(args.engine).stem
    print(f"[trt] {tag} inputs={runner.input_names} outputs={runner.output_names}")

    data: Dict[str, Any] = {
        "meta": {
            "tag": tag, "backend": "tensorrt", "precision": "fp16",
            "engine": str(args.engine), "device": args.device,
            "image_size": args.image_size, "warmup": args.warmup, "iters": args.iters,
        }
    }
    entry = manifest_model_entry(args.manifest, tag)
    if entry is not None:
        data["meta"]["manifest_model"] = entry
    if "B" in args.tests:
        print("[B] engine inference latency ...")
        data["B_inference"] = benchmark_b(runner, args, args.device)
    if "C" in args.tests:
        print("[C] end2end per-image latency ...")
        data["C_end2end"] = benchmark_c(runner, args, args.device)

    b = data.get("B_inference", {}).get("inference_latency", {})
    c = data.get("C_end2end", {}).get("end2end_total_latency", {})
    if b:
        print(f"  B inference  mean={b['mean_ms']:.3f}  std={b['std_ms']:.3f}  fps={b['fps']:.1f}")
    if c:
        print(f"  C end2end    mean={c['mean_ms']:.3f}  std={c['std_ms']:.3f}  fps={c['fps']:.1f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
