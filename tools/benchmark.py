"""
tools/benchmark.py — Model profiling CLI for LiteUAVDETR.

Measures:
  • End-to-end latency (batch_size=1) over a full evaluation pass.
  • Per-component parameter count, GFLOPs, and percentages.
  • Per-component latency breakdown via CUDA-event hooks.

LiteUAVDETR's model path is adapted from D-FINE: https://github.com/Peterande/D-FINE
The profiler uses generic hasattr-based module discovery.

Three-layer override system
---------------------------
Settings are merged in this order (later layers win):

  1. Base config  (-c)   full YAML with __include__ chains — never modified
  2. Experiment   (-e)   plain YAML with ONLY the keys that differ per run
  3. CLI flags    (-u)   quick one-off overrides, dot-notation KEY=VALUE

This means you maintain ONE base config and ONE small experiment YAML per
experiment variant. No hardcoding anywhere, no duplicated full configs.

Experiment YAML (-e) format
----------------------------
Plain YAML dict — no __include__ needed, just the keys you want to change.

  # configs/experiments/4level_800px_B3.yml
  HGNetv2:
    return_idx: [0, 1, 2, 3]

  HybridEncoder:
    in_channels: [96, 384, 768, 1536]
    feat_strides: [4, 8, 16, 32]
    use_encoder_idx: [3]
    depth_mult: 0.67

  LiteUAVDETRTransformer:
    feat_channels: [256, 256, 256, 256]
    feat_strides:  [4, 8, 16, 32]
    num_levels: 4
    num_points: [4, 4, 4, 4]

  eval_spatial_size: [800, 800]
  epochs: 300

  train_dataloader:
    total_batch_size: 8
    num_workers: 12
    collate_fn:
      base_size: 800
      stop_epoch: 276
      base_size_repeat: 2
      ema_restart_decay: 0.9999
    dataset:
      transforms:
        policy:
          epoch: 276
        ops:
          5:
            size: [800, 800]

  val_dataloader:
    total_batch_size: 16
    num_workers: 12
    dataset:
      transforms:
        ops:
          0:
            size: [800, 800]

  lr_scheduler:
    type: MultiStepLR
    milestones: [120, 180]
    gamma: 0.8

  optimizer:
    lr: 0.0004
    weight_decay: 0.000065
    params:
      0:
        lr: 0.00003
      1:
        lr: 0.00003

  LiteUAVDETRPostProcessor:
    num_top_queries: 350

Usage example
-------------
  python tools/benchmark.py \\
      -c configs/liteuavdetr/custom/liteuavdetr_fast_visdrone_640.yml \\
      -r checkpoints/stripfact_res_m_visdrone_best_stg2.pth \\
      --device cuda --warmup 20 \\
      --output profile.json \\
      -u LiteUAVDETRPostProcessor.num_top_queries=350
"""

import argparse
import os
import sys

# ── Path setup — mirrors train.py ─────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import yaml
import torch

from src.core import YAMLConfig, yaml_utils
from src.misc.model_profiler import ModelProfiler


# ─────────────────────────────────────────────────────────────────────────────
# Override merging
# ─────────────────────────────────────────────────────────────────────────────

# Benchmark-only defaults — lowest priority, always overridable by -e or -u
_BENCHMARK_DEFAULTS = {
    "val_dataloader": {"total_batch_size": 1},   # latency always uses bs=1
}


def _build_update_dict(exp_cfg_path, user_update_args):
    """
    Build the unified kwargs dict that will be merged on top of the base config.

    Merge order (later wins):
      _BENCHMARK_DEFAULTS  ←  experiment delta YAML (-e)  ←  CLI flags (-u)
    """
    combined = dict(_BENCHMARK_DEFAULTS)  # start from benchmark defaults (copy)

    # Layer 2: experiment delta YAML
    if exp_cfg_path:
        with open(exp_cfg_path, "r") as fh:
            exp_overrides = yaml.safe_load(fh) or {}
        if not isinstance(exp_overrides, dict):
            raise ValueError(
                f"-e file must be a YAML mapping (dict), got {type(exp_overrides)}"
            )
        combined = yaml_utils.merge_dict(combined, exp_overrides)

    # Layer 3: CLI -u flags (highest priority)
    if user_update_args:
        cli_overrides = yaml_utils.parse_cli(user_update_args)
        combined = yaml_utils.merge_dict(combined, cli_overrides)

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auto_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_checkpoint(model: torch.nn.Module, path: str, device: torch.device) -> None:
    """Load a LiteUAVDETR checkpoint using the D-FINE-compatible state layout."""
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        state = (
            ckpt.get("model")
            or ckpt.get("state_dict")
            or ckpt.get("ema")
            or ckpt
        )
        if isinstance(state, dict) and "module" in state:
            state = state["module"]   # EMA wraps under .module
    else:
        state = ckpt

    if isinstance(state, dict):
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  [warn] Missing keys ({len(missing)}): {missing[:5]} …")
        if unexpected:
            print(f"  [warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")
    else:
        print("  [warn] Could not parse checkpoint — running with random weights.")
    print(f"  Loaded checkpoint : {path}")


def _get_base_size(cfg: YAMLConfig, update_dict: dict) -> int:
    """Read base_size: config train_dataloader → eval_spatial_size → 640."""
    # Check override dict first (in case it was set via -e or -u)
    try:
        return update_dict["train_dataloader"]["collate_fn"]["base_size"]
    except (KeyError, TypeError):
        pass
    try:
        return cfg.train_dataloader.collate_fn.base_size
    except Exception:
        pass
    try:
        es = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
        return es[0] if isinstance(es, (list, tuple)) else int(es)
    except Exception:
        return 640


def _flat_print(d: dict, prefix: str = "    ") -> None:
    """Recursively print a nested dict as flat KEY = VALUE lines."""
    def _recurse(obj, path):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _recurse(v, f"{path}.{k}" if path else k)
        else:
            print(f"{prefix}{path} = {obj!r}")
    _recurse(d, "")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device) if args.device else _auto_device()

    # ── Build layered override dict ───────────────────────────────────────
    update_dict = _build_update_dict(args.exp_cfg, args.update)

    # ── Print run summary ─────────────────────────────────────────────────
    sep = "─" * 70
    print(f"\n{sep}")
    print("  LiteUAVDETR Model Profiler")
    print(sep)
    print(f"  Base config : {args.config}")
    print(f"  Exp  delta  : {args.exp_cfg or '(none)'}")
    print(f"  Checkpoint  : {args.resume or '(none — random weights)'}")
    print(f"  Device      : {device}")
    print(f"  Warmup      : {args.warmup} batches")
    print("  Active overrides (defaults ← -e ← -u):")
    _flat_print(update_dict)
    print(sep + "\n")

    # ── Load config: base + all overrides merged at once ─────────────────
    cfg = YAMLConfig(args.config, **update_dict)

    # Suppress backbone pretrained download when benchmarking
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # ── Build model ───────────────────────────────────────────────────────
    print("  Building model …")
    model: torch.nn.Module = cfg.model.deploy().to(device)
    if args.resume:
        _load_checkpoint(model, args.resume, device)
    model.eval()
    print(f"  Model class : {type(model).__name__}\n")

    # ── Input shape ───────────────────────────────────────────────────────
    base_size   = _get_base_size(cfg, update_dict)
    input_shape = (1, 3, base_size, base_size)
    print(f"  Input shape : {input_shape}")

    # ── Profiler — params + FLOPs ─────────────────────────────────────────
    profiler   = ModelProfiler(model, device, input_shape=input_shape)
    components = profiler.discover_components()

    print(f"\n  Discovered {len(components)} components:")
    for label in components:
        print(f"    {'  ' * label.count('.')}{label}")
    print()

    print("  Computing parameter counts …")
    params  = profiler.compute_params(components)
    print(f"  Total params : {params.get('__total__', 0) / 1e6:.3f} M\n")

    print("  Computing GFLOPs (single dummy forward) …")
    flops   = profiler.compute_flops(components)
    total_g = flops.get("__total__", 0.0)
    if total_g > 0:
        print(f"  Total GFLOPs : {total_g:.3f}\n")
    else:
        print("  GFLOPs       : n/a  (pip install fvcore for per-component breakdown)\n")

    # ── Latency — full eval loop ──────────────────────────────────────────
    print("  Building val dataloader …")
    try:
        val_loader = cfg.val_dataloader
        n  = len(val_loader) if hasattr(val_loader, "__len__") else "?"
        bs = (update_dict.get("val_dataloader") or {}).get("total_batch_size", 1)
        print(f"  Val dataloader ready — {n} batches (bs={bs})")
    except Exception as exc:
        raise RuntimeError(
            f"Could not build val_dataloader: {exc}\n"
            "Check dataset paths in your config or pass them via -e / -u."
        ) from exc

    profiler.register_timing_hooks(components)
    lat_stats, component_stats = profiler.run_latency_benchmark(
        val_loader,
        warmup=args.warmup,
    )
    profiler.remove_hooks()

    # ── Report ────────────────────────────────────────────────────────────
    profiler.print_report(params, flops, lat_stats, component_stats, components)

    if args.output:
        profiler.save_json(args.output, params, flops, lat_stats, component_stats)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LiteUAVDETR model profiler — latency + params/FLOPs breakdown",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "-c", "--config", type=str, required=True,
        help="Base YAML config file (with __include__ chains). Never modified.",
    )
    p.add_argument(
        "-e", "--exp-cfg", type=str, default=None, dest="exp_cfg",
        help=(
            "Experiment delta YAML — plain YAML dict with ONLY the keys that "
            "differ for this run (no __include__ needed). Merged on top of the "
            "base config, before -u CLI flags. Keeps your base config clean and "
            "makes each experiment self-contained in a small file."
        ),
    )
    p.add_argument(
        "-r", "--resume", type=str, default=None,
        help="Checkpoint path (.pth). Omit to profile random-init weights.",
    )
    p.add_argument(
        "-d", "--device", type=str, default=None,
        help="cpu | cuda | cuda:0 | cuda:1  (default: auto-detect)",
    )
    p.add_argument(
        "--warmup", type=int, default=20,
        help="Warm-up batches discarded before timing (default: 20)",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Write profile results to this JSON file (optional)",
    )
    p.add_argument(
        "-u", "--update", nargs="+", default=None,
        metavar="KEY=VALUE",
        help=(
            "Quick one-off overrides using dot-notation KEY=VALUE. "
            "Applied AFTER -e, so they always win. Examples:\n"
            "  -u LiteUAVDETRPostProcessor.num_top_queries=350\n"
            "  -u HGNetv2.name=B4\n"
            "  -u eval_spatial_size=[800,800]\n"
            "  -u val_dataloader.total_batch_size=1"
        ),
    )

    return p.parse_args()


if __name__ == "__main__":
    main(_parse_args())
