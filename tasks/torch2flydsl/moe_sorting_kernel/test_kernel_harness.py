#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl MoE token-sorting task (EXACT match).

This op is an INTEGER counting sort, so correctness is an EXACT equality of the
integer outputs -- there is NO float tolerance band:

  * ``num_valid_ids``    : exact equality ([total_padded_token_count, M]).
  * ``sorted_token_ids`` : exact, position-by-position over the valid range
                           [0:num_valid] (packed ids (slot<<24)|token), AND the
                           padding region must be the sentinel (topk<<24)|M.
                           Both the reference and the kernel emit tokens in
                           ascending (token, slot) order within each expert and
                           pad each expert run to a multiple of block_size, so
                           the layouts are bit-for-bit identical.
  * ``sorted_expert_ids``: exact over the valid blocks (num_valid//block_size).
  * ``sorted_weights``   : the kernel copies the raw 32-bit weight through, so
                           it is bit-identical; gated at max|diff| == 0.

The check ASSERTS and exits non-zero on ANY mismatch (it prints an example of
expected vs actual on failure). It also OPTIONALLY cross-checks the pure-torch
reference against ``aiter.fused_moe.moe_sorting`` (the CK host op) to confirm
the layout/ordering convention matches (skipped if aiter is unavailable).

Modes:
  --correctness     assert EXACT integer match vs the pure-torch reference
  --full-benchmark  time the FlyDSL kernel vs the torch reference, write report
"""
import argparse
import importlib.util
import json
import math
import os
import sys
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

# Representative MoE routing shapes, all within the ONESHOT path
# (M <= min(sub_tokens, ONESHOT_MAX_T) on gfx950). block_size (unit_size) = 32.
SHAPES = [
    {"name": "deepseek_M16_E256_k8", "M": 16, "E": 256, "topk": 8},
    {"name": "deepseek_M8_E256_k8", "M": 8, "E": 256, "topk": 8},
    {"name": "M16_E32_k2", "M": 16, "E": 32, "topk": 2},
    {"name": "M64_E32_k2", "M": 64, "E": 32, "topk": 2},
    {"name": "M16_E8_k2", "M": 16, "E": 8, "topk": 2},
    {"name": "M64_E8_k2", "M": 64, "E": 8, "topk": 2},
]

BLOCK_SIZE = 32
SEED = 20260616


def _make_inputs(mmod, shape, device="cuda"):
    topk_ids, topk_weights = mmod._gen_topk(
        shape["M"], shape["topk"], shape["E"], seed=SEED + shape["M"] * 1000 + shape["E"]
    )
    return topk_ids.to(device), topk_weights.to(device)


def _ref_outputs(mmod, shape, topk_ids, topk_weights):
    model = mmod.Model(num_experts=shape["E"], topk=shape["topk"], block_size=BLOCK_SIZE)
    return model(topk_ids, topk_weights)


def _make_prepared_reference(shape, topk_ids, topk_weights):
    """Resolve data-dependent expert run sizes once for graph-safe replay."""
    import torch

    M, topk = topk_ids.shape
    num_experts = shape["E"]
    max_tokens = M * topk + num_experts * BLOCK_SIZE - topk
    max_blocks = (max_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    sentinel = (topk << 24) | M

    sorted_token_ids = torch.empty(max_tokens, dtype=torch.int32, device=topk_ids.device)
    sorted_weights = torch.empty(max_tokens, dtype=torch.float32, device=topk_ids.device)
    sorted_expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=topk_ids.device)

    plans = []
    ids_cursor = 0
    block_cursor = 0
    for expert in range(num_experts):
        token_id, topk_pos = torch.where(topk_ids == expert)
        count = token_id.numel()
        if not count:
            continue
        blocks = (count + BLOCK_SIZE - 1) // BLOCK_SIZE
        plans.append(
            (expert, token_id, topk_pos, ids_cursor, count, block_cursor, blocks)
        )
        ids_cursor += blocks * BLOCK_SIZE
        block_cursor += blocks

    num_valid_ids = torch.empty(2, dtype=torch.int32, device=topk_ids.device)
    num_valid_template = torch.tensor(
        [ids_cursor, M], dtype=torch.int32, device=topk_ids.device
    )

    def run_ref():
        sorted_token_ids.fill_(sentinel)
        sorted_weights.zero_()
        sorted_expert_ids.fill_(-1)
        num_valid_ids.copy_(num_valid_template)
        for expert, token_id, topk_pos, start, count, block_start, blocks in plans:
            packed = (topk_pos.to(torch.int32) << 24) | token_id.to(torch.int32)
            sorted_token_ids[start : start + count].copy_(packed)
            sorted_weights[start : start + count].copy_(
                topk_weights[token_id, topk_pos].to(torch.float32)
            )
            sorted_expert_ids[block_start : block_start + blocks].fill_(expert)
        return sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids

    return run_ref


def _exact_check(shape, ref, out, verbose=True):
    """Return (ok, detail). Asserts exact integer match; bitwise weights."""
    import torch

    ref_ids, ref_w, ref_eids, ref_nv = ref
    out_ids, out_w, out_eids, out_nv = out

    name = shape["name"]
    topk, M = shape["topk"], shape["M"]

    # 1. num_valid_ids — exact
    nv_ok = torch.equal(ref_nv.cpu(), out_nv.cpu())
    num_valid = int(ref_nv[0].item())

    # 2. sorted_token_ids — exact over [0:num_valid]. This range includes the
    # intra-run padding sentinels (each expert run is padded to a block_size
    # multiple with the sentinel (topk<<24)|M), so the exact equality below
    # verifies both the packed token ids AND the padding placement bit-for-bit.
    # Positions >= num_valid are uninitialized in both the kernel output and the
    # CK/AITER op (allocated via torch.empty), so they are intentionally ignored.
    ids_ok = torch.equal(ref_ids[:num_valid].cpu(), out_ids[:num_valid].cpu())

    # 3. sorted_expert_ids — exact over valid blocks
    n_blocks = num_valid // BLOCK_SIZE
    eids_ok = torch.equal(ref_eids[:n_blocks].cpu(), out_eids[:n_blocks].cpu())

    # 4. sorted_weights — bitwise (kernel copies raw 32-bit weight through)
    w_max_err = (
        (ref_w[:num_valid] - out_w[:num_valid]).abs().max().item() if num_valid > 0 else 0.0
    )
    w_ok = w_max_err == 0.0

    ok = nv_ok and ids_ok and eids_ok and w_ok

    if verbose:
        print(
            f"  {'PASS' if ok else 'FAIL'}: {name} (M{M}/E{shape['E']}/k{topk}) "
            f"num_valid={num_valid} blocks={n_blocks} "
            f"[nv={'ok' if nv_ok else 'X'} ids={'ok' if ids_ok else 'X'} "
            f"eids={'ok' if eids_ok else 'X'} "
            f"w_maxerr={w_max_err:.1e}]"
        )

    detail = None
    if not ok:
        lines = [f"MISMATCH in {name}:"]
        lines.append(f"  ref num_valid_ids={ref_nv.cpu().tolist()} out={out_nv.cpu().tolist()}")
        if not ids_ok:
            r = ref_ids[:num_valid].cpu()
            o = out_ids[:num_valid].cpu()
            diff = (r != o).nonzero(as_tuple=True)[0][:8]
            for idx in diff.tolist():
                rv, ov = int(r[idx]), int(o[idx])
                lines.append(
                    f"  sorted_token_ids[{idx}]: ref=(slot={rv >> 24},tok={rv & 0xFFFFFF}) "
                    f"out=(slot={ov >> 24},tok={ov & 0xFFFFFF})"
                )
        if not eids_ok:
            r = ref_eids[:n_blocks].cpu()
            o = out_eids[:n_blocks].cpu()
            diff = (r != o).nonzero(as_tuple=True)[0][:8]
            for idx in diff.tolist():
                lines.append(f"  sorted_expert_ids[{idx}]: ref={int(r[idx])} out={int(o[idx])}")
        if not w_ok:
            lines.append(f"  sorted_weights max|diff|={w_max_err:.3e} (expected 0)")
        detail = "\n".join(lines)
    return ok, detail


def _cross_check_aiter(mmod, shape, topk_ids, topk_weights, ref, verbose=True):
    """Confirm the pure-torch reference matches aiter.fused_moe.moe_sorting."""
    import torch

    try:
        from aiter.fused_moe import moe_sorting as aiter_moe_sorting
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  [aiter cross-check] SKIP (aiter unavailable: {type(e).__name__})")
        return None

    ref_ids, ref_w, ref_eids, ref_nv = ref
    try:
        a_ids, a_w, a_eids, a_nv, _ = aiter_moe_sorting(
            topk_ids, topk_weights, shape["E"], model_dim=512,
            moebuf_dtype=torch.bfloat16, block_size=BLOCK_SIZE,
        )
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"  [aiter cross-check] SKIP (aiter call failed: {type(e).__name__}: {e})")
        return None

    num_valid = int(ref_nv[0].item())
    nv_ok = torch.equal(ref_nv.cpu(), a_nv.cpu())
    ids_ok = torch.equal(ref_ids[:num_valid].cpu(), a_ids[:num_valid].cpu())
    n_blocks = num_valid // BLOCK_SIZE
    eids_ok = torch.equal(ref_eids[:n_blocks].cpu(), a_eids[:n_blocks].cpu())
    ok = bool(nv_ok and ids_ok and eids_ok)
    if verbose:
        print(
            f"  [aiter cross-check] {'MATCH' if ok else 'MISMATCH'} "
            f"(nv={'ok' if nv_ok else 'X'} ids={'ok' if ids_ok else 'X'} "
            f"eids={'ok' if eids_ok else 'X'})"
        )
    return ok


def run_correctness(verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert kmod is not None and mmod is not None, "cannot load kernel.py / model.py"

    failures = []
    aiter_results = []
    for shape in SHAPES:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        topk_ids, topk_weights = _make_inputs(mmod, shape)
        with torch.no_grad():
            ref = _ref_outputs(mmod, shape, topk_ids, topk_weights)
            out = kmod.flydsl_moe_sorting(topk_ids, topk_weights, shape["E"], unit_size=BLOCK_SIZE)
        torch.cuda.synchronize()

        ok, detail = _exact_check(shape, ref, out, verbose=verbose)
        if not ok:
            failures.append(shape["name"])
            if detail:
                print(detail)

        ac = _cross_check_aiter(mmod, shape, topk_ids, topk_weights, ref, verbose=verbose)
        if ac is not None:
            aiter_results.append(ac)
            if ac is False:
                failures.append(shape["name"] + "[aiter]")

    if aiter_results:
        print(f"aiter cross-check: {sum(aiter_results)}/{len(aiter_results)} matched")
    else:
        print("aiter cross-check: skipped (aiter unavailable)")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    assert kmod is not None and mmod is not None, "cannot load kernel.py / model.py"

    latencies, speedups, report = [], [], []
    print(f"{'Config':<24} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 60)
    for idx, shape in enumerate(SHAPES):
        torch.manual_seed(SEED)
        topk_ids, topk_weights = _make_inputs(mmod, shape)
        model = mmod.Model(num_experts=shape["E"], topk=shape["topk"], block_size=BLOCK_SIZE)
        run_ref = _make_prepared_reference(shape, topk_ids, topk_weights)

        with torch.no_grad():
            def run_kernel():
                return kmod.flydsl_moe_sorting(
                    topk_ids, topk_weights, shape["E"], unit_size=BLOCK_SIZE
                )

            run_kernel()
            torch.cuda.synchronize()
            for _ in range(warmup):
                run_kernel()
            torch.cuda.synchronize()
            kernel_ms, kernel_bench_meta = benchmark_cuda_graph_or_events(
                run_kernel, warmup=0, repetition=iters
            )

            ref_ms, ref_bench_meta = benchmark_cuda_graph_or_events(
                run_ref, warmup=0, repetition=iters
            )

        methods_match = kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"]
        speedup = (
            ref_ms / kernel_ms if methods_match and kernel_ms > 0 else None
        )
        speedup_display = (
            format(speedup, ">8.2f") + "x"
            if speedup is not None
            else f"{'N/A':>9}"
        )
        latencies.append(kernel_ms)
        if speedup is not None:
            speedups.append(speedup)
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            **kernel_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [shape["M"], shape["E"], shape["topk"]],
            "params": {k: shape[k] for k in ("M", "E", "topk")},
        })
        if verbose:
            print(f"{shape['name']:<24} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}")
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
    parser = argparse.ArgumentParser(description="torch2flydsl moe_sorting harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 60)
    print("torch2flydsl MoE token-sorting (integer counting sort, EXACT match)")
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
