#!/usr/bin/env python3
"""Task runner for triton2flydsl/sglang/triton_mrope_fused.

Standalone harness for sglang's fused multimodal (sectioned) rotary embedding
Triton kernel (`triton_mrope_fused` -> `_triton_mrope_forward_fused`): the
Qwen2-VL / Qwen2.5-VL M-RoPE applied in place to q and k. Each token has three
positions (t, h, w); the rotary half-dim is split into t/h/w sections (contiguous
for the non-interleaved layout, modulo-3 for the interleaved layout) and each
rotary index draws cos/sin from the cache row of its governing position.

Modes:
  --compile        : ast-parse + import source, assert symbols.
  --correctness    : Triton M-RoPE vs torch fp32 sectioned-rope reference, close.
  --full-benchmark : graph-first GPU timing, write build/performance_report.json
"""
import sys
import os
import json
import time
import argparse
import importlib.util
from _aka_benchmark import benchmark_cuda_graph_or_events

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(TASK_DIR)

TASK_NAME = "triton2flydsl/sglang/triton_mrope_fused"
SOURCE_FILE = os.path.join(TASK_DIR, "triton_mrope_fused.py")
MAX_POS = 4096

# Real Qwen2-VL / Qwen2.5-VL M-RoPE shapes. mrope_section sums to rotary_dim//2.
# n_qh / n_kh from GQA (Qwen2-VL-7B: 28 q heads, 4 kv heads). Both NEOX and GPT-J
# rotate styles, contiguous (non-interleaved) and modulo-3 interleaved sections.
TEST_SHAPES = [
    {"nt": 64, "n_qh": 28, "n_kh": 4, "hd": 128, "rd": 128,
     "section": [16, 24, 24], "interleaved": False, "neox": True},
    {"nt": 128, "n_qh": 28, "n_kh": 4, "hd": 128, "rd": 128,
     "section": [16, 24, 24], "interleaved": False, "neox": True},
    {"nt": 32, "n_qh": 16, "n_kh": 2, "hd": 128, "rd": 128,
     "section": [24, 20, 20], "interleaved": False, "neox": True},
    {"nt": 64, "n_qh": 28, "n_kh": 4, "hd": 128, "rd": 128,
     "section": [16, 24, 24], "interleaved": False, "neox": False},  # gpt-j
    {"nt": 48, "n_qh": 12, "n_kh": 2, "hd": 128, "rd": 64,
     "section": [8, 12, 12], "interleaved": False, "neox": True},  # rd < hd
    {"nt": 96, "n_qh": 28, "n_kh": 4, "hd": 128, "rd": 128,
     "section": [16, 24, 24], "interleaved": True, "neox": True},   # qwen2.5-vl
    {"nt": 16, "n_qh": 8, "n_kh": 1, "hd": 128, "rd": 128,
     "section": [16, 24, 24], "interleaved": True, "neox": False},  # interleaved gpt-j
    {"nt": 1, "n_qh": 28, "n_kh": 4, "hd": 128, "rd": 128,
     "section": [16, 24, 24], "interleaved": False, "neox": True},  # single token
]
WARMUP_ITERATIONS = 10
BENCHMARK_ITERATIONS = 100
MAX_OOM_RETRIES = 5


def load_module():
    spec = importlib.util.spec_from_file_location("triton_mrope_fused_src", SOURCE_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_oom(err):
    return "out of memory" in str(err).lower()


def _retry_oom(fn):
    import torch
    delay = 1.0
    for attempt in range(MAX_OOM_RETRIES):
        try:
            return fn()
        except RuntimeError as e:
            if _is_oom(e) and attempt < MAX_OOM_RETRIES - 1:
                torch.cuda.empty_cache()
                time.sleep(delay)
                delay *= 2.0
                continue
            raise


def make_inputs(cfg, device="cuda"):
    import torch
    nt, hd, rd = cfg["nt"], cfg["hd"], cfg["rd"]
    q = torch.randn(nt, cfg["n_qh"] * hd, device=device, dtype=torch.bfloat16)
    k = torch.randn(nt, cfg["n_kh"] * hd, device=device, dtype=torch.bfloat16)
    # cos_sin_cache: [max_pos, rd] = [cos(half_rd) | sin(half_rd)], fp32.
    half = rd // 2
    inv = 1.0 / (10000.0 ** (torch.arange(0, half, device=device).float() / half))
    pos = torch.arange(MAX_POS, device=device).float()[:, None] * inv[None, :]
    cache = torch.cat([torch.cos(pos), torch.sin(pos)], dim=1).contiguous()  # [P, rd]
    positions = torch.randint(0, MAX_POS, (3, nt), device=device, dtype=torch.int64)
    axis_map = torch.zeros(half, device=device, dtype=torch.int32)
    return q, k, cache, positions, axis_map


def _section_masks(cfg, device):
    import torch
    rd = cfg["rd"]
    half = rd // 2
    st, sh, sw = cfg["section"]
    offs = torch.arange(half, device=device)
    if cfg["interleaved"]:
        h_mask = ((offs % 3) == 1) & (offs <= 3 * sh)
        w_mask = ((offs % 3) == 2) & (offs <= 3 * sw)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = st
        h_end = st + sh
        t_mask = offs < st
        h_mask = (t_end <= offs) & (offs < h_end)
        w_mask = (h_end <= offs) & (offs < half)
    return t_mask, h_mask, w_mask


def _apply_rope(x_flat, n_h, cfg, cos, sin):
    # x_flat: [nt, n_h*hd]; cos/sin: [nt, half]. Returns rotated [nt, n_h*hd] fp32.
    nt, hd, rd = cfg["nt"], cfg["hd"], cfg["rd"]
    half = rd // 2
    x = x_flat.view(nt, n_h, hd).float().clone()
    xr = x[..., :rd]
    cb = cos[:, None, :]
    sb = sin[:, None, :]
    if cfg["neox"]:
        x1 = xr[..., :half]
        x2 = xr[..., half:rd]
        o1 = x1 * cb - x2 * sb
        o2 = x2 * cb + x1 * sb
        x[..., :half] = o1
        x[..., half:rd] = o2
    else:
        xe = xr[..., 0::2]
        xo = xr[..., 1::2]
        oe = xe * cb - xo * sb
        oo = xo * cb + xe * sb
        x[..., 0:rd:2] = oe
        x[..., 1:rd:2] = oo
    return x.view(nt, n_h * hd)


def reference(q, k, cache, positions, cfg):
    device = q.device
    rd = cfg["rd"]
    half = rd // 2
    cos_cache = cache[:, :half]  # [P, half]
    sin_cache = cache[:, half:rd]
    t_mask, h_mask, w_mask = _section_masks(cfg, device)
    pt, ph, pw = positions[0], positions[1], positions[2]
    cos = (cos_cache[pt] * t_mask + cos_cache[ph] * h_mask + cos_cache[pw] * w_mask)
    sin = (sin_cache[pt] * t_mask + sin_cache[ph] * h_mask + sin_cache[pw] * w_mask)
    q_out = _apply_rope(q, cfg["n_qh"], cfg, cos, sin).to(q.dtype)
    k_out = _apply_rope(k, cfg["n_kh"], cfg, cos, sin).to(k.dtype)
    return q_out, k_out


def _shape_of(cfg):
    return [cfg["nt"], cfg["n_qh"], cfg["n_kh"], cfg["hd"], cfg["rd"]]


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE, "r") as f:
            ast.parse(f.read())
        mod = load_module()
        assert hasattr(mod, "triton_mrope_fused"), "Missing entry triton_mrope_fused"
        assert hasattr(mod, "_triton_mrope_forward_fused"), \
            "Missing @triton.jit _triton_mrope_forward_fused"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    import torch
    try:
        mod = load_module()
    except Exception as e:
        return False, f"Failed to load module: {e}", []

    atol, rtol = 1e-2, 1e-2
    details = []
    for i, cfg in enumerate(TEST_SHAPES):
        shape = _shape_of(cfg)
        try:
            torch.manual_seed(42 + i)
            q, k, cache, positions, axis_map = make_inputs(cfg, "cuda")
            q_in, k_in = q.clone(), k.clone()
            _retry_oom(lambda: mod.triton_mrope_fused(
                q_in, k_in, cache, positions, cfg["section"], cfg["hd"], cfg["rd"],
                cfg["interleaved"], False, cfg["neox"], axis_map))
            torch.cuda.synchronize()
            q_ref, k_ref = reference(q, k, cache, positions, cfg)
            finite = bool(torch.isfinite(q_in).all().item() and
                          torch.isfinite(k_in).all().item())
            qd = (q_in.float() - q_ref.float()).abs().max().item()
            kd = (k_in.float() - k_ref.float()).abs().max().item()
            q_close = bool(torch.allclose(q_in.float(), q_ref.float(),
                                          atol=atol, rtol=rtol))
            k_close = bool(torch.allclose(k_in.float(), k_ref.float(),
                                          atol=atol, rtol=rtol))
            passed = finite and q_close and k_close
            details.append({"shape_id": i + 1, "shape": shape,
                            "interleaved": cfg["interleaved"], "neox": cfg["neox"],
                            "q_diff": qd, "k_diff": kd, "passed": passed})
            if not passed:
                return False, (f"Shape {i+1} {shape} il={cfg['interleaved']} "
                               f"neox={cfg['neox']}: q_diff={qd:.4e} k_diff={kd:.4e} "
                               f"finite={finite}"), details
        except Exception as e:
            details.append({"shape_id": i + 1, "shape": shape, "error": str(e)})
            return False, f"Shape {i+1} {shape}: exception: {e}", details
    return True, None, details


def run_performance():
    import torch
    try:
        mod = load_module()
    except Exception:
        return []

    test_cases = []
    for ti, cfg in enumerate(TEST_SHAPES):
        params = {"shape": _shape_of(cfg), "interleaved": cfg["interleaved"],
                  "neox": cfg["neox"]}
        try:
            torch.manual_seed(42 + ti)
            q, k, cache, positions, axis_map = make_inputs(cfg, "cuda")
            qc, kc = q.clone(), k.clone()

            def prepare_fn():
                qc.copy_(q)
                kc.copy_(k)

            def fn():
                mod.triton_mrope_fused(
                    qc, kc, cache, positions, cfg["section"], cfg["hd"], cfg["rd"],
                    cfg["interleaved"], False, cfg["neox"], axis_map)

            prepare_fn()
            _retry_oom(fn)
            for _ in range(WARMUP_ITERATIONS):
                prepare_fn()
                fn()
            torch.cuda.synchronize()
            elapsed_ms, bench_meta = benchmark_cuda_graph_or_events(
                fn, warmup=0, repetition=BENCHMARK_ITERATIONS,
                prepare_fn=prepare_fn,
            )
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": elapsed_ms,
                               **bench_meta,
                               "params": params})
        except Exception:
            test_cases.append({"test_case_id": f"perf{ti+1}",
                               "execution_time_ms": -1.0,
                               "benchmark_method": "benchmark_failed",
                               "benchmark_fallback_reason": "performance case failed before timing completed",
                               "params": params})
    return test_cases


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("--compile", dest="mode", action="store_const", const="compile")
    parser.add_argument("--correctness", dest="mode", action="store_const", const="correctness")
    parser.add_argument("--full-benchmark", dest="mode", action="store_const", const="performance")
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(build_dir, "compile_report.json"), "w"), indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err, details = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err,
                   "num_shapes": len(TEST_SHAPES), "details": details},
                  open(os.path.join(build_dir, "correctness_report.json"), "w"), indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        for d in details:
            if "passed" in d:
                print(f"  shape {d['shape_id']} {d['shape']} il={d['interleaved']} "
                      f"neox={d['neox']}: q_diff={d['q_diff']:.4e} "
                      f"k_diff={d['k_diff']:.4e} -> {'PASS' if d['passed'] else 'FAIL'}")
            elif "error" in d:
                print(f"  shape {d['shape_id']} {d['shape']}: ERROR {d['error']}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        test_cases = run_performance()
        json.dump(test_cases, open(os.path.join(build_dir, "performance_report.json"), "w"), indent=2)
        if test_cases:
            total = sum(c["execution_time_ms"] for c in test_cases if c["execution_time_ms"] > 0)
            print(f"Performance: measured {len(test_cases)} case(s), total {total:.4f} ms")
            for c in test_cases:
                print(f"  {c['test_case_id']} {c['params']}: {c['execution_time_ms']:.4f} ms")
        else:
            print("Performance: FAILED - no test cases measured")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
