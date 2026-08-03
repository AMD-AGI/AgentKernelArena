#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Correctness and performance harness for sqrt-softplus MoE top-k routing.

Ground truth is AMD's fused ``aiter.topk_softplus`` op (DeepSeek-V4-Pro
routing). The pure-torch reference in ``model.py`` is validated against it per
shape:

  * top-k expert ids: EXACT set match. Disagreements are tolerated ONLY when
    they sit on a genuine selection-score tie (straddling experts within
    ``_TIE_TOL``); any non-tie id mismatch fails.
  * top-k weights: normalized max error ``max|ref - aiter| / max|ref|`` over the
    matched expert ids must stay <= ``REL_TOL``.

If a FlyDSL ``kernel.py`` is present (GEAK's target) it is additionally compared
against the reference with the same gate. The check asserts and exits non-zero on
any failure.

Modes:
  --correctness     compare the reference (and kernel.py if present) vs aiter
  --full-benchmark  time the fused op vs the torch reference and write a report
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from _aka_benchmark import benchmark_cuda_graph_or_events

KERNEL_FILE = "kernel.py"
MODEL_FILE = "model.py"


def _resolve_kernel_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, KERNEL_FILE)):
        return here
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, KERNEL_FILE)):
        return cwd
    return here


def _load_module(kernel_dir, filename, alias):
    entry = os.path.join(kernel_dir, filename)
    if not os.path.isfile(entry):
        return None
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    spec = importlib.util.spec_from_file_location(alias, entry)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_KERNEL_DIR = _resolve_kernel_dir()

# Real sqrt-softplus routed MoE shapes (router gating [tokens, experts]):
#   DeepSeek-V4-Pro routing: E=256, topk=8, renorm, route_scale=2.5; plus a
#   384-expert (Kimi-style) config and a smaller dense-MoE config.
SHAPES = [
    {"name": "dsv4_t64_e256_k8", "tokens": 64, "experts": 256, "topk": 8, "renormalize": True, "route_scale": 2.5},
    {"name": "dsv4_t1024_e256_k8", "tokens": 1024, "experts": 256, "topk": 8, "renormalize": True, "route_scale": 2.5},
    {"name": "t256_e384_k8", "tokens": 256, "experts": 384, "topk": 8, "renormalize": True, "route_scale": 2.5},
    {"name": "t64_e128_k4_norenorm", "tokens": 64, "experts": 128, "topk": 4, "renormalize": False, "route_scale": 1.0},
]

REL_TOL = 1e-2
# Selection-score tie tolerance (~100x the kernel's exp2/log2 approximation
# noise on O(1) scores). Disagreements within this band are genuine ties whose
# routing choice is semantically irrelevant; larger gaps are real bugs.
_TIE_TOL = 1e-4
SEED = 20260602


def _make_gating(experts, tokens, dtype, device, gen):
    """Shuffled-uniform gating so every row has unique, well-separated values."""
    import torch

    base = torch.arange(-1, 1, 2.0 / experts, device=device)[:experts]
    g = base.repeat(tokens, 1).to(dtype=dtype)
    perm = torch.argsort(torch.rand(g.shape, generator=gen, device=device), dim=-1)
    return torch.gather(g, -1, perm).contiguous()


def _build_model(mmod, shape, device="cuda"):
    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = mmod.Model(
        num_experts=shape["experts"], topk=shape["topk"],
        renormalize=shape["renormalize"], route_scale=shape["route_scale"],
    ).to(device).eval()
    gen = torch.Generator(device=device).manual_seed(SEED + shape["tokens"] + shape["experts"])
    gating = _make_gating(shape["experts"], shape["tokens"], torch.bfloat16, device, gen)
    return model, gating


def _retry(fn, tries=8, what="aiter op"):
    """Call fn with backoff: aiter may JIT-compile its kernel on first use."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            msg = str(exc).lower()
            transient = any(k in msg for k in ("out of memory", "hip", "jit", "compil", "busy"))
            if not transient and i >= 1:
                break
            time.sleep(3.0 * (i + 1))
    raise RuntimeError(f"{what} failed after {tries} tries: {last}")


def _aiter_softplus(aiter, gating, bias, topk, renormalize, route_scale):
    import torch

    tokens = gating.shape[0]
    w = torch.empty((tokens, topk), dtype=torch.float32, device=gating.device)
    idx = torch.empty((tokens, topk), dtype=torch.int32, device=gating.device)

    def call():
        aiter.topk_softplus(w, idx, gating, bias, renormalize, route_scale)
        torch.cuda.synchronize()
        return w, idx

    return _retry(call, what="aiter.topk_softplus")


def _compare_routing(ref_w, ref_id, out_w, out_id, sel, topk):
    """Return (genuine_mismatch_tokens, weight_norm_err). Ids match as sets except
    at selection-score ties; weights compared by matched id."""

    ref_id_c = ref_id.cpu()
    out_id_c = out_id.cpu()
    ref_w_c = ref_w.float().cpu()
    out_w_c = out_w.float().cpu()
    sel_c = sel.float().cpu()

    sorted_sel, _ = sel_c.sort(dim=-1, descending=True)
    cutoff = sorted_sel[:, topk - 1]

    genuine = 0
    ref_scale = ref_w_c.abs().max().item() + 1e-9
    max_w_err = 0.0
    for t in range(out_id_c.shape[0]):
        kset = set(out_id_c[t].tolist())
        rset = set(ref_id_c[t].tolist())
        if not (kset == rset and len(kset) == topk):
            thr = cutoff[t].item()
            extra = kset - rset
            missing = rset - kset
            excused = (
                len(kset) == topk and len(rset) == topk
                and all(sel_c[t, e].item() >= thr - _TIE_TOL for e in extra)
                and all(sel_c[t, e].item() <= thr + _TIE_TOL for e in missing)
            )
            if not excused:
                genuine += 1
        ref_pos = {int(ref_id_c[t, k]): k for k in range(topk)}
        for k in range(topk):
            kid = int(out_id_c[t, k])
            if kid in ref_pos:
                max_w_err = max(
                    max_w_err, abs(out_w_c[t, k].item() - ref_w_c[t, ref_pos[kid]].item())
                )
    return genuine, max_w_err / ref_scale


def run_correctness(verbose=True):
    import torch

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None
    import aiter

    failures = []
    for shape in SHAPES:
        model, gating = _build_model(mmod, shape)
        bias = model.correction_bias.detach().float()
        with torch.no_grad():
            ref_w, ref_id = model(gating)
            a_w, a_id = _aiter_softplus(
                aiter, gating, bias, shape["topk"], shape["renormalize"], shape["route_scale"]
            )
            sel = mmod.selection_scores(gating, bias)
        torch.cuda.synchronize()

        genuine, w_err = _compare_routing(ref_w, ref_id, a_w, a_id, sel, shape["topk"])
        ok = genuine == 0 and w_err <= REL_TOL
        if verbose:
            print(
                f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                f"(T{shape['tokens']}/E{shape['experts']}/k{shape['topk']}) "
                f"[ref-vs-aiter] id_mismatch={genuine} weight_norm_err={w_err:.2e} (tol={REL_TOL})"
            )
        if not ok:
            failures.append(shape["name"])

        if has_kernel:
            try:
                k_w, k_id = kmod.flydsl_topk_softplus(
                    gating, bias, shape["topk"], shape["renormalize"], shape["route_scale"]
                )
            except NotImplementedError:
                has_kernel = False
                if verbose:
                    print(
                        "        SKIP: kernel.py FlyDSL target not implemented yet "
                        "(reference validated against the aiter op above)"
                    )
            else:
                torch.cuda.synchronize()
                kg, kw = _compare_routing(ref_w, ref_id, k_w, k_id, sel, shape["topk"])
                kok = kg == 0 and kw <= REL_TOL
                if verbose:
                    print(
                        f"        [kernel-vs-ref] id_mismatch={kg} weight_norm_err={kw:.2e} "
                        f"{'PASS' if kok else 'FAIL'}"
                    )
                if not kok:
                    failures.append(shape["name"] + "[kernel]")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch
    import aiter

    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert mmod is not None, "cannot load model.py"
    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    has_kernel = kmod is not None

    if has_kernel:
        s0 = SHAPES[0]
        model0, gating0 = _build_model(mmod, s0)
        bias0 = model0.correction_bias.detach().float()
        try:
            with torch.no_grad():
                kmod.flydsl_topk_softplus(
                    gating0, bias0, s0["topk"], s0["renormalize"], s0["route_scale"]
                )
        except NotImplementedError:
            has_kernel = False
            print(
                "SKIP: kernel.py FlyDSL target not implemented yet "
                "(benchmarking aiter op instead)"
            )
        del model0, gating0
        torch.cuda.empty_cache()

    latencies, speedups, report = [], [], []
    print(f"{'Config':<24} {'Ref':>10} {'Fused':>10} {'Speedup':>10}")
    print("-" * 60)
    for idx, shape in enumerate(SHAPES):
        model, gating = _build_model(mmod, shape)
        bias = model.correction_bias.detach().float()
        topk, renorm, rs = shape["topk"], shape["renormalize"], shape["route_scale"]

        with torch.no_grad():
            if has_kernel:
                def run_fused():
                    return kmod.flydsl_topk_softplus(gating, bias, topk, renorm, rs)
            else:
                fused_w = torch.empty(
                    (shape["tokens"], topk),
                    dtype=torch.float32,
                    device=gating.device,
                )
                fused_idx = torch.empty(
                    (shape["tokens"], topk),
                    dtype=torch.int32,
                    device=gating.device,
                )

                def run_fused():
                    aiter.topk_softplus(
                        fused_w,
                        fused_idx,
                        gating,
                        bias,
                        renorm,
                        rs,
                    )
                    return fused_w, fused_idx

                def prepare_fused():
                    run_fused()
                    torch.cuda.synchronize()

                _retry(prepare_fused, what="aiter.topk_softplus")

            run_fused()
            torch.cuda.synchronize()
            for _ in range(warmup):
                run_fused()
            torch.cuda.synchronize()
            fused_ms, fused_bench_meta = benchmark_cuda_graph_or_events(
                run_fused, warmup=0, repetition=iters
            )

            ref_ms, ref_bench_meta = benchmark_cuda_graph_or_events(
                lambda: model(gating), warmup=0, repetition=iters
            )

        methods_match = fused_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"]
        speedup = (
            ref_ms / fused_ms if methods_match and fused_ms > 0 else None
        )
        speedup_display = (
            format(speedup, ">8.2f") + "x"
            if speedup is not None
            else f"{'N/A':>9}"
        )
        latencies.append(fused_ms)
        if speedup is not None:
            speedups.append(speedup)
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": fused_ms,
            **fused_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": fused_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [shape["tokens"], shape["experts"], shape["topk"]],
            "params": {k: shape[k] for k in ("tokens", "experts", "topk", "renormalize", "route_scale")},
        })
        if verbose:
            print(f"{shape['name']:<24} {ref_ms:>8.4f}ms {fused_ms:>8.4f}ms {speedup_display}")
        del model, gating
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(x) for x in speedups) / len(speedups)) if speedups else None
    geomean_speedup_display = (
        format(geomean_speedup, ".2f") + "x"
        if geomean_speedup is not None
        else "N/A"
    )

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 60)
    print(f"Geometric mean latency: {geomean_latency:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup_display}")
    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl moe_topk_softplus harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("torch2flydsl MoE top-k routing (sqrt-softplus, vs aiter ground truth)")
    print("=" * 60)

    if args.correctness:
        try:
            run_correctness()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_benchmark(warmup=args.warmup, iters=args.iterations)
