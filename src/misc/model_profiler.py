"""
Model Profiler — Latency + Params/FLOPs breakdown.

Measures:
  1. End-to-end latency (batch_size=1) averaged over a full evaluation run.
  2. Per-component params, GFLOPs, and percentages.
  3. Per-component latency breakdown via CUDA-event forward hooks.

LiteUAVDETR integration is adapted from D-FINE: https://github.com/Peterande/D-FINE
Generic traversal supports LiteUAVDETR and compatible module layouts.
"""

import warnings
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _total_params(module: nn.Module) -> int:
    """Recursive param count for a module."""
    return sum(p.numel() for p in module.parameters())


def _group_total_params(modules: List[nn.Module]) -> int:
    """Total params across a list of modules, deduplicating shared tensors."""
    seen: set = set()
    count = 0
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                count += p.numel()
    return count


def _flatten_modules(comp: Any) -> List[nn.Module]:
    """Turn a single module or a list-of-modules into a flat list."""
    if isinstance(comp, nn.Module):
        return [comp]
    if isinstance(comp, (list, tuple)):
        flat = []
        for item in comp:
            flat.extend(_flatten_modules(item))
        return flat
    return []


# ─────────────────────────────────────────────────────────────────────────────
# ModelProfiler
# ─────────────────────────────────────────────────────────────────────────────

class ModelProfiler:
    """
    Profile any PyTorch detection model for params, FLOPs, and latency.

    Usage
    -----
    profiler = ModelProfiler(model, device, input_shape=(1, 3, 640, 640))
    components = profiler.discover_components()
    params     = profiler.compute_params(components)
    flops      = profiler.compute_flops(components)
    profiler.register_timing_hooks(components)
    lat_stats, comp_times = profiler.run_latency_benchmark(val_loader, warmup=10)
    profiler.remove_hooks()
    profiler.print_report(params, flops, lat_stats, comp_times, components)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        input_shape: Tuple[int, ...] = (1, 3, 640, 640),
    ):
        self.model = model
        self.device = device
        self.input_shape = input_shape
        self._hook_handles: List = []
        self._pending_events: Dict[str, List[Tuple]] = defaultdict(list)
        self._component_timings: Dict[str, List[float]] = defaultdict(list)

    # ─────────────────────────────────────────────────────────────────────
    # Component Discovery — generic, no hard-coded class names
    # ─────────────────────────────────────────────────────────────────────

    def discover_components(self) -> "OrderedDict[str, Any]":
        """
        Walk model attributes to build a component tree via hasattr checks.

        Returns
        -------
        OrderedDict mapping label → module-or-list-of-modules.
        Compound groups (fpn, pan) are stored as [ModuleList, ModuleList].
        """
        model = self.model
        components: OrderedDict = OrderedDict()

        # ── Backbone ─────────────────────────────────────────────────────
        if hasattr(model, "backbone"):
            components["backbone"] = model.backbone

        # ── Encoder ──────────────────────────────────────────────────────
        if hasattr(model, "encoder"):
            enc = model.encoder
            components["encoder"] = enc

            if hasattr(enc, "input_proj"):
                components["encoder.input_proj"] = enc.input_proj

            if hasattr(enc, "encoder") and len(enc.encoder) > 0:
                components["encoder.transformer"] = enc.encoder

            fpn_parts = [
                getattr(enc, a)
                for a in ("lateral_convs", "fpn_blocks")
                if hasattr(enc, a)
            ]
            if fpn_parts:
                components["encoder.fpn"] = fpn_parts

            pan_parts = [
                getattr(enc, a)
                for a in ("downsample_convs", "pan_blocks")
                if hasattr(enc, a)
            ]
            if pan_parts:
                components["encoder.pan"] = pan_parts

        # ── Decoder ──────────────────────────────────────────────────────
        if hasattr(model, "decoder"):
            dec = model.decoder
            components["decoder"] = dec

            # ── Query-init submodules ─────────────────────────────────
            for attr, label in [
                ("input_proj",     "decoder.input_proj"),
                ("query_pos_head", "decoder.query_pos_head"),
                ("enc_output",     "decoder.enc_output"),
                ("enc_score_head", "decoder.enc_score_head"),
                ("enc_bbox_head",  "decoder.enc_bbox_head"),
                ("pre_bbox_head",  "decoder.pre_bbox_head"),
            ]:
                if hasattr(dec, attr):
                    components[label] = getattr(dec, attr)

            # ── Integral / project ────────────────────────────────────
            if hasattr(dec, "integral"):
                components["decoder.integral"] = dec.integral

            # ── Per-layer breakdown ───────────────────────────────────
            # LiteUAVDETRTransformer normally exposes dec.decoder.layers;
            # retain the direct dec.layers fallback for compatible models.
            inner_dec = None
            if hasattr(dec, "decoder") and hasattr(dec.decoder, "layers"):
                inner_dec = dec.decoder
            elif hasattr(dec, "layers"):
                inner_dec = dec

            if inner_dec is not None:
                for i, layer in enumerate(inner_dec.layers):
                    prefix = f"decoder.layer{i}"

                    if hasattr(layer, "self_attn"):
                        components[f"{prefix}.self_attn"] = layer.self_attn
                    if hasattr(layer, "cross_attn"):
                        components[f"{prefix}.cross_attn"] = layer.cross_attn
                    if hasattr(layer, "gateway"):
                        components[f"{prefix}.gate"] = layer.gateway
                    elif hasattr(layer, "gate"):
                        components[f"{prefix}.gate"] = layer.gate

                    # FFN: MoE or standard linear1/linear2
                    if hasattr(layer, "moe_ffn"):
                        components[f"{prefix}.ffn"] = layer.moe_ffn
                    elif hasattr(layer, "linear1") and hasattr(layer, "linear2"):
                        # Group linear1 + linear2 as compound FFN
                        components[f"{prefix}.ffn"] = [layer.linear1, layer.linear2]

                # LQE layers (skip Identity placeholders in deploy mode)
                if hasattr(inner_dec, "lqe_layers"):
                    for i, lqe in enumerate(inner_dec.lqe_layers):
                        if not isinstance(lqe, nn.Identity):
                            components[f"decoder.layer{i}.lqe"] = lqe

            # ── Per-layer output heads ────────────────────────────────
            if hasattr(dec, "dec_score_head"):
                for i, head in enumerate(dec.dec_score_head):
                    components[f"decoder.layer{i}.score_head"] = head
            if hasattr(dec, "dec_bbox_head"):
                for i, head in enumerate(dec.dec_bbox_head):
                    components[f"decoder.layer{i}.bbox_head"] = head

            # ── Additional / Dome-style heads ─────────────────────────
            for attr in [
                "corner_head", "refer_head",
                "pre_score_head", "pre_corner_head",
                "query_embed", "tgt_embed",
                "query_pos_embed", "query_content_embed",
            ]:
                if hasattr(dec, attr):
                    components[f"decoder.{attr}"] = getattr(dec, attr)

        return components

    # ─────────────────────────────────────────────────────────────────────
    # Params
    # ─────────────────────────────────────────────────────────────────────

    def compute_params(self, components: OrderedDict) -> Dict[str, int]:
        """
        Compute total (recursive) parameter count for each component.

        Returns dict label → param_count, plus '__total__' for the full model.
        Compound groups use deduplicated counting to avoid double-counting
        shared parameters.
        """
        result: Dict[str, int] = {}
        for label, comp in components.items():
            mods = _flatten_modules(comp)
            if mods:
                result[label] = _group_total_params(mods)
            else:
                result[label] = 0

        result["__total__"] = _total_params(self.model)
        return result

    # ─────────────────────────────────────────────────────────────────────
    # FLOPs
    # ─────────────────────────────────────────────────────────────────────

    def compute_flops(self, components: OrderedDict) -> Dict[str, float]:
        """
        Compute GFLOPs per component.

        Tries fvcore (single forward pass, per-module) first.
        Falls back to calflops total-only if fvcore is not available.

        Returns dict label → GFLOPs (float), plus '__total__'.
        """
        dummy = torch.zeros(self.input_shape, device=self.device)

        try:
            return self._compute_flops_fvcore(components, dummy)
        except ImportError:
            warnings.warn(
                "fvcore not installed — computing total FLOPs only "
                "(pip install fvcore for per-component breakdown)"
            )
        except Exception as exc:
            warnings.warn(
                f"fvcore tracing failed ({type(exc).__name__}: {exc}). "
                "This often happens with models that use dynamic control flow "
                "or non-standard forward signatures. Falling back to calflops."
            )

        # ── calflops fallback — total only ────────────────────────────
        total_gflops = 0.0
        try:
            from calflops import calculate_flops  # noqa: F401
            import copy

            model_copy = copy.deepcopy(self.model)
            flops_val, _, _ = calculate_flops(
                model=model_copy,
                input_shape=self.input_shape,
                output_as_string=False,
                print_detailed=False,
            )
            total_gflops = flops_val / 1e9
            del model_copy
        except Exception as exc:
            warnings.warn(f"calflops also failed: {exc}")

        result = {label: 0.0 for label in components}
        result["__total__"] = total_gflops
        return result

    def _compute_flops_fvcore(
        self, components: OrderedDict, dummy: torch.Tensor
    ) -> Dict[str, float]:
        from fvcore.nn import FlopCountAnalysis  # raises ImportError if missing

        with torch.no_grad():
            fa = FlopCountAnalysis(self.model, dummy)
            fa.unsupported_ops_warnings(False)
            fa.uncalled_modules_warnings(False)
            per_module: Dict[str, int] = dict(fa.by_module())  # path → flops (recursive)

        # Build id(module) → fvcore_path for fast lookup
        id_to_path: Dict[int, str] = {
            id(m): name for name, m in self.model.named_modules()
        }

        def _flops_for_module(m: nn.Module) -> int:
            path = id_to_path.get(id(m))
            if path is None:
                return 0
            if path in per_module:
                return per_module[path]
            # Root module has empty string path in fvcore
            if path == "" and "" in per_module:
                return per_module[""]
            return 0

        result: Dict[str, float] = {}
        for label, comp in components.items():
            mods = _flatten_modules(comp)
            total = sum(_flops_for_module(m) for m in mods)
            result[label] = total / 1e9

        total_all = fa.total()
        result["__total__"] = total_all / 1e9
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Timing Hooks
    # ─────────────────────────────────────────────────────────────────────

    def register_timing_hooks(self, components: OrderedDict) -> None:
        """
        Attach CUDA-event forward hooks to each component's module(s).

        Strategy depends on component type:
        • Single nn.Module (backbone, encoder, cross_attn, …):
            Hook pre+post on that module — gives exact elapsed time.
        • Compound list of leaf modules [linear1, linear2]:
            Hook pre on first + post on last — measures total sequential time.
        • Compound list containing ModuleLists (fpn, pan):
            ModuleLists have no callable forward(), so hooks cannot fire.
            Timing is skipped; latency column shows "—" for these rows.
        """
        self._hook_handles.clear()
        self._pending_events = defaultdict(list)
        self._component_timings = defaultdict(list)

        use_cuda = torch.cuda.is_available()

        for label, comp in components.items():
            if isinstance(comp, nn.Module):
                # Single module — straightforward hook
                self._attach_hook_pair(label, comp, comp, use_cuda)

            elif isinstance(comp, (list, tuple)):
                # Compound group — check whether members are hookable leaf modules
                # (i.e., not ModuleList/Sequential containers that wrap child calls)
                leaf_mods = [
                    m for m in _flatten_modules(comp)
                    if not isinstance(m, (nn.ModuleList, nn.Sequential, nn.ModuleDict))
                ]
                if len(leaf_mods) == 0:
                    # Only containers — skip timing (e.g., fpn/pan ModuleLists)
                    continue
                elif len(leaf_mods) == 1:
                    self._attach_hook_pair(label, leaf_mods[0], leaf_mods[0], use_cuda)
                else:
                    # Multiple leaf modules (e.g., linear1 + linear2 for FFN):
                    # Start timer on first module's pre_hook,
                    # stop timer on last module's post_hook.
                    self._attach_hook_pair(label, leaf_mods[0], leaf_mods[-1], use_cuda)

    def _attach_hook_pair(
        self,
        label: str,
        start_module: nn.Module,
        end_module: nn.Module,
        use_cuda: bool,
    ) -> None:
        """
        Attach timing hooks:
          • pre_hook on start_module → records start event
          • post_hook on end_module  → records end event + appends to pending

        start_module and end_module may be the same (single-module case) or
        different (first/last of a compound sequential group).
        """
        profiler_ref = self
        # Shared mutable box so pre_hook can pass the start event to post_hook
        # even when they're on different modules.
        _event_box: List = []  # will hold [ev_s, ev_e] after pre_hook fires

        def pre_hook(m: nn.Module, inp):
            if use_cuda:
                ev_s = torch.cuda.Event(enable_timing=True)
                ev_e = torch.cuda.Event(enable_timing=True)
                ev_s.record()
                _event_box.clear()
                _event_box.append(ev_s)
                _event_box.append(ev_e)
            else:
                import time as _time
                _event_box.clear()
                _event_box.append(_time.perf_counter())

        def post_hook(m: nn.Module, inp, out):
            if not _event_box:
                return
            if use_cuda:
                ev_e = _event_box[1]
                ev_e.record()
                profiler_ref._pending_events[label].append((_event_box[0], ev_e))
            else:
                import time as _time
                elapsed_ms = (_time.perf_counter() - _event_box[0]) * 1000.0
                profiler_ref._component_timings[label].append(elapsed_ms)

        h1 = start_module.register_forward_pre_hook(pre_hook)
        h2 = end_module.register_forward_hook(post_hook)
        self._hook_handles.extend([h1, h2])

    def _flush_timing_events(self) -> None:
        """
        Synchronize the CUDA stream and collect elapsed times from all
        pending event pairs. Call after each forward pass.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        for label, pairs in list(self._pending_events.items()):
            for ev_s, ev_e in pairs:
                try:
                    elapsed = ev_s.elapsed_time(ev_e)  # ms
                    self._component_timings[label].append(elapsed)
                except Exception:
                    pass
        self._pending_events = defaultdict(list)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    # ─────────────────────────────────────────────────────────────────────
    # Latency Benchmark
    # ─────────────────────────────────────────────────────────────────────

    def run_latency_benchmark(
        self,
        dataloader,
        warmup: int = 10,
        device_move_fn=None,
        milestone_pcts: Tuple[int, ...] = (10, 25, 40, 50, 75, 100),
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Run a full eval-loop latency benchmark with batch_size=1.

        Parameters
        ----------
        dataloader     : iterable yielding (samples, targets)
        warmup         : number of initial batches to discard before timing
        device_move_fn : optional callable(batch, device) → (samples, targets)
        milestone_pcts : dataset-completion percentages at which to print
                         a running-statistics snapshot (default every 10/25/40/50/75/100 %)

        Returns
        -------
        lat_stats      : dict with mean/median/p95/p99/std/fps/gpu_mem_peak
        component_stats: dict label → {mean_ms, pct}
        """
        self.model.eval()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)

        e2e_timings: List[float] = []
        n_total = len(dataloader) if hasattr(dataloader, "__len__") else None
        n_timed = max(n_total - warmup, 0) if n_total else None  # batches that will be timed

        # Pre-compute which timed-batch indices trigger each milestone print
        _milestone_triggered = set()

        if use_cuda:
            ev_s_e2e = torch.cuda.Event(enable_timing=True)
            ev_e_e2e = torch.cuda.Event(enable_timing=True)

        # ── Header ────────────────────────────────────────────────────
        bar_width = 40
        total_str = str(n_total) if n_total else "?"
        print(f"\n  {'─'*60}")
        print(f"  Latency benchmark")
        print(f"  {'─'*60}")
        print(f"  Total batches : {total_str}  |  Warmup : {warmup}  |  "
              f"Timed : {n_timed if n_timed else '?'}")
        print(f"  {'─'*60}")

        def _progress_bar(pct: float, width: int = bar_width) -> str:
            filled = int(width * pct / 100)
            return f"[{'█' * filled}{'░' * (width - filled)}] {pct:5.1f}%"

        def _print_milestone(pct: float, n_done: int, timings: List[float]) -> None:
            arr = np.array(timings, dtype=np.float64)
            mean_ms  = float(arr.mean())
            med_ms   = float(np.median(arr))
            fps      = 1000.0 / mean_ms if mean_ms > 0 else 0.0
            bar      = _progress_bar(pct)
            mem_str  = ""
            if use_cuda:
                mem_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
                mem_str = f"  GPU mem: {mem_mb:.0f} MB"
            print(f"\n  {bar}  [{n_done}/{n_timed if n_timed else '?'} batches]")
            print(f"    mean: {mean_ms:7.2f} ms   median: {med_ms:7.2f} ms   "
                  f"fps: {fps:6.1f}{mem_str}")

        # ── Benchmark loop ────────────────────────────────────────────
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):

                # ── Move to device ─────────────────────────────────────
                if device_move_fn is not None:
                    samples, targets = device_move_fn(batch, self.device)
                else:
                    samples, targets = self._default_move(batch)

                # ── Warmup pass (no timing) ────────────────────────────
                is_warmup = batch_idx < warmup
                if is_warmup:
                    pct_warmup = (batch_idx + 1) / warmup * 100 if warmup > 0 else 100
                    bar = _progress_bar(pct_warmup)
                    print(f"\r  Warming up  {bar}  [{batch_idx + 1}/{warmup}]",
                          end="", flush=True)
                    try:
                        self.model(samples)
                    except Exception:
                        pass
                    if use_cuda:
                        torch.cuda.synchronize(self.device)
                    self._pending_events = defaultdict(list)
                    if batch_idx + 1 == warmup:
                        print()   # newline after warmup bar completes
                    continue

                # ── Timed forward pass ─────────────────────────────────
                if use_cuda:
                    torch.cuda.synchronize(self.device)
                    ev_s_e2e.record()
                else:
                    import time as _time
                    t0 = _time.perf_counter()

                try:
                    self.model(samples)
                except Exception:
                    pass

                if use_cuda:
                    ev_e_e2e.record()
                    torch.cuda.synchronize(self.device)
                    elapsed = ev_s_e2e.elapsed_time(ev_e_e2e)
                else:
                    elapsed = (_time.perf_counter() - t0) * 1000.0  # type: ignore

                e2e_timings.append(elapsed)
                self._flush_timing_events()

                # ── Inline progress bar (updates every batch) ──────────
                n_done = len(e2e_timings)
                if n_timed and n_timed > 0:
                    pct_done = n_done / n_timed * 100
                    bar = _progress_bar(pct_done)
                    avg_ms = sum(e2e_timings[-20:]) / min(20, len(e2e_timings))
                    print(f"\r  Benchmarking {bar}  [{n_done}/{n_timed}]  "
                          f"avg {avg_ms:.1f} ms", end="", flush=True)

                    # ── Milestone snapshot ────────────────────────────
                    for m in sorted(milestone_pcts):
                        if m not in _milestone_triggered and pct_done >= m:
                            _milestone_triggered.add(m)
                            print()  # end inline bar line first
                            _print_milestone(pct_done, n_done, e2e_timings)
                else:
                    # Unknown total — print every 50 batches
                    if n_done % 50 == 0:
                        avg_ms = sum(e2e_timings[-50:]) / min(50, len(e2e_timings))
                        print(f"\r  batch {n_done}  avg {avg_ms:.1f} ms", flush=True)

        # ── Aggregate ─────────────────────────────────────────────────
        if not e2e_timings:
            warnings.warn("No batches timed — check warmup vs dataloader length.")
            return {}, {}

        arr = np.array(e2e_timings, dtype=np.float64)
        lat_stats: Dict[str, float] = {
            "n_batches":   float(len(arr)),
            "mean_ms":     float(arr.mean()),
            "median_ms":   float(np.median(arr)),
            "p95_ms":      float(np.percentile(arr, 95)),
            "p99_ms":      float(np.percentile(arr, 99)),
            "std_ms":      float(arr.std()),
            "fps":         float(1000.0 / arr.mean()),
        }
        if use_cuda:
            lat_stats["gpu_mem_peak_mb"] = (
                torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            )

        mean_e2e = lat_stats["mean_ms"]
        component_stats: Dict[str, Dict[str, float]] = {}
        for label, times in self._component_timings.items():
            if times:
                ta = np.array(times, dtype=np.float64)
                component_stats[label] = {
                    "mean_ms": float(ta.mean()),
                    "pct":     float(ta.mean() / mean_e2e * 100) if mean_e2e > 0 else 0.0,
                }

        return lat_stats, component_stats

    def _default_move(self, batch):
        """Best-effort move of dataloader batch to self.device."""
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            samples = batch[0]
            targets = batch[1] if len(batch) > 1 else None
        else:
            samples = batch
            targets = None

        if isinstance(samples, torch.Tensor):
            samples = samples.to(self.device)
        elif isinstance(samples, (list, tuple)):
            samples = [
                s.to(self.device) if isinstance(s, torch.Tensor) else s
                for s in samples
            ]

        if isinstance(targets, (list, tuple)):
            targets = [
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in t.items()}
                if isinstance(t, dict) else t
                for t in targets
            ]
        return samples, targets

    # ─────────────────────────────────────────────────────────────────────
    # Report
    # ─────────────────────────────────────────────────────────────────────

    def print_report(
        self,
        params: Dict[str, int],
        flops: Dict[str, float],
        lat_stats: Dict[str, float],
        component_stats: Dict[str, Dict[str, float]],
        components: OrderedDict,
    ) -> None:
        """Pretty-print all three profiling tables."""
        try:
            from tabulate import tabulate as _tabulate  # type: ignore
            _has_tabulate = True
        except ImportError:
            _has_tabulate = False

        total_params = max(params.get("__total__", 1), 1)
        total_flops  = max(flops.get("__total__", 0.0), 1e-9)
        has_flops    = flops.get("__total__", 0.0) > 0

        # ── Table 1: Params & FLOPs ──────────────────────────────────
        W = 90
        print("\n" + "═" * W)
        print("  Model Component Breakdown — Params & GFLOPs")
        print("═" * W)

        rows1 = []
        for label in components:
            p = params.get(label, 0)
            g = flops.get(label, 0.0)
            p_pct = p / total_params * 100
            g_pct = g / total_flops * 100 if has_flops else 0.0
            indent = "  " * label.count(".")
            rows1.append([
                indent + label,
                f"{p / 1e6:>8.3f} M",
                f"{p_pct:>6.1f}%",
                f"{g:>8.3f}" if has_flops and g > 0 else "       —",
                f"{g_pct:>6.1f}%" if has_flops and g > 0 else "      —",
            ])
        # Totals row
        rows1.append([
            "── TOTAL ──",
            f"{total_params / 1e6:>8.3f} M",
            "100.0%",
            f"{flops.get('__total__', 0):>8.3f}" if has_flops else "—",
            "100.0%" if has_flops else "—",
        ])

        hdrs1 = ["Component", "Params", "Params%", "GFLOPs", "FLOPs%"]
        if _has_tabulate:
            print(_tabulate(rows1, headers=hdrs1, tablefmt="simple",
                            colalign=("left", "right", "right", "right", "right")))
        else:
            _print_table(rows1, hdrs1, col_widths=[48, 11, 8, 10, 8])

        # ── Table 2: Latency Breakdown ───────────────────────────────
        if component_stats:
            print("\n" + "═" * 65)
            print("  Per-component Latency (averaged over timed batches)")
            print("═" * 65)

            rows2 = []
            for label in components:
                cs = component_stats.get(label)
                if cs:
                    indent = "  " * label.count(".")
                    rows2.append([
                        indent + label,
                        f"{cs['mean_ms']:>8.3f} ms",
                        f"{cs['pct']:>6.1f}%",
                    ])
            if lat_stats:
                rows2.append([
                    "── END-TO-END ──",
                    f"{lat_stats['mean_ms']:>8.3f} ms",
                    "100.0%",
                ])
            hdrs2 = ["Component", "Avg Latency", "%"]
            if _has_tabulate:
                print(_tabulate(rows2, headers=hdrs2, tablefmt="simple",
                                colalign=("left", "right", "right")))
            else:
                _print_table(rows2, hdrs2, col_widths=[48, 13, 8])

        # ── Table 3: Overall latency stats ──────────────────────────
        if lat_stats:
            print("\n" + "═" * 65)
            print("  Overall Latency Statistics")
            print("═" * 65)
            n = int(lat_stats.get("n_batches", 0))
            print(f"  Batches timed (bs=1) : {n}")
            print(f"  Mean                 : {lat_stats['mean_ms']:.2f} ms"
                  f"   ({lat_stats['fps']:.1f} FPS)")
            print(f"  Median               : {lat_stats['median_ms']:.2f} ms")
            print(f"  P95                  : {lat_stats['p95_ms']:.2f} ms")
            print(f"  P99                  : {lat_stats['p99_ms']:.2f} ms")
            print(f"  Std                  : {lat_stats['std_ms']:.2f} ms")
            if "gpu_mem_peak_mb" in lat_stats:
                print(f"  GPU Mem Peak         : {lat_stats['gpu_mem_peak_mb']:.0f} MB")
        print("═" * 65 + "\n")

    def save_json(
        self,
        path: str,
        params: Dict,
        flops: Dict,
        lat_stats: Dict,
        component_stats: Dict,
    ) -> None:
        """Save profiling results to a JSON file."""
        import json

        # Convert numpy scalars to plain Python types for JSON serialisation
        def _to_python(obj):
            if isinstance(obj, dict):
                return {k: _to_python(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_python(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return obj

        data = {
            "params":             _to_python(params),
            "gflops":             _to_python(flops),
            "latency":            _to_python(lat_stats),
            "component_latency":  _to_python(component_stats),
        }
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        print(f"  Saved profile JSON → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Table-printing fallback (no tabulate)
# ─────────────────────────────────────────────────────────────────────────────

def _print_table(
    rows: List[List[str]],
    headers: List[str],
    col_widths: Optional[List[int]] = None,
) -> None:
    if col_widths is None:
        col_widths = [max(len(str(r[c])) for r in [headers] + rows) + 2
                      for c in range(len(headers))]
    fmt = " ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * (sum(col_widths) + len(col_widths) - 1))
    for row in rows:
        padded = list(row) + [""] * max(0, len(col_widths) - len(row))
        print(fmt.format(*padded[:len(col_widths)]))
