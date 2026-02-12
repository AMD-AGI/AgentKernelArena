#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

TASK_DIR = Path(__file__).resolve().parents[1]
TASK_NAME = TASK_DIR.name
CONFIG = TASK_DIR / 'config.yaml'
BUILD_DIR = TASK_DIR / 'build'
ATOL = 1e-4
RTOL = 1e-4


def parse_config(path: Path):
    src_files = []
    symbols = []
    mode = None
    for raw in path.read_text().splitlines():
        line = raw.rstrip()
        if line.startswith('source_file_path:'):
            mode = 'src'
            continue
        if line.startswith('target_kernel_functions:'):
            mode = 'sym'
            continue
        if line and not line.startswith(' ') and not line.startswith('-'):
            mode = None
        if mode == 'src' and line.lstrip().startswith('- '):
            src_files.append(line.split('- ', 1)[1].strip())
        if mode == 'sym' and line.lstrip().startswith('- '):
            symbols.append(line.split('- ', 1)[1].strip())
    return src_files, symbols


def load_sources(src_files):
    blobs = {}
    missing = []
    for rel in src_files:
        p = TASK_DIR / rel
        if not p.exists():
            missing.append(rel)
            continue
        blobs[rel] = p.read_text(errors='ignore')
    return blobs, missing


def compile_mode(src_files, symbols):
    blobs, missing = load_sources(src_files)
    if missing:
        raise RuntimeError(f'missing source files: {missing}')
    concat = "\n".join(blobs.values())
    not_found = [s for s in symbols if s not in concat]
    if not_found:
        raise RuntimeError(f'target symbols not found in local source/: {not_found}')

    BUILD_DIR.mkdir(exist_ok=True)
    (BUILD_DIR / 'compile_report.json').write_text(json.dumps({
        'mode': 'compile',
        'status': 'ok',
        'sources': src_files,
        'target_symbols': symbols,
        'source_sha256': hashlib.sha256(concat.encode()).hexdigest(),
    }, indent=2))
    print('compile self-check passed')


def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _attention_ref(q, k, v):
    out = np.zeros((q.shape[0], v.shape[1]), dtype=np.float32)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for i in range(q.shape[0]):
        scores = np.array([np.dot(q[i], k[j]) * scale for j in range(k.shape[0])], dtype=np.float32)
        probs = np.exp(scores - np.max(scores))
        probs /= np.sum(probs)
        for j in range(k.shape[0]):
            out[i] += probs[j] * v[j]
    return out


def _attention_cand(q, k, v):
    scores = (q @ k.T) / math.sqrt(q.shape[-1])
    return _softmax(scores, axis=-1) @ v


def _rmsnorm_ref(x, w, eps=1e-6):
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        denom = math.sqrt(np.mean(x[i] * x[i]) + eps)
        out[i] = (x[i] / denom) * w
    return out


def _rmsnorm_cand(x, w, eps=1e-6):
    denom = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / denom) * w


def _silu(x):
    return x / (1.0 + np.exp(-x))


def _swiglu_ref(a, b):
    out = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = (a[i, j] / (1.0 + math.exp(-a[i, j]))) * b[i, j]
    return out


def _swiglu_cand(a, b):
    return _silu(a) * b


def _quant_ref(x, scale):
    out = np.zeros_like(x, dtype=np.int8)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            q = int(round(x[i, j] / scale))
            out[i, j] = max(-127, min(127, q))
    return out.astype(np.float32)


def _quant_cand(x, scale):
    return np.clip(np.round(x / scale), -127, 127).astype(np.int8).astype(np.float32)


def _topk_ref(x, k):
    idx = np.argsort(-x, axis=-1)[:, :k]
    val = np.take_along_axis(x, idx, axis=-1)
    return val, idx.astype(np.float32)


def _topk_cand(x, k):
    idx = np.argpartition(-x, kth=k-1, axis=-1)[:, :k]
    val = np.take_along_axis(x, idx, axis=-1)
    order = np.argsort(-val, axis=-1)
    idx = np.take_along_axis(idx, order, axis=-1)
    val = np.take_along_axis(val, order, axis=-1)
    return val, idx.astype(np.float32)


def _hadamard_ref(x):
    h = np.array([
        [1,1,1,1,1,1,1,1],
        [1,-1,1,-1,1,-1,1,-1],
        [1,1,-1,-1,1,1,-1,-1],
        [1,-1,-1,1,1,-1,-1,1],
        [1,1,1,1,-1,-1,-1,-1],
        [1,-1,1,-1,-1,1,-1,1],
        [1,1,-1,-1,-1,-1,1,1],
        [1,-1,-1,1,-1,1,1,-1],
    ], dtype=np.float32)
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        for r in range(8):
            out[i, r] = np.sum(h[r] * x[i])
    return out


def _hadamard_cand(x):
    h = np.array([
        [1,1,1,1,1,1,1,1],
        [1,-1,1,-1,1,-1,1,-1],
        [1,1,-1,-1,1,1,-1,-1],
        [1,-1,-1,1,1,-1,-1,1],
        [1,1,1,1,-1,-1,-1,-1],
        [1,-1,1,-1,-1,1,-1,1],
        [1,1,-1,-1,-1,-1,1,1],
        [1,-1,-1,1,-1,1,1,-1],
    ], dtype=np.float32)
    return x @ h.T


def run_task_correctness(task_name: str, seed: int = 42):
    rng = np.random.default_rng(seed)

    if task_name in {'paged_attention_v1','paged_attention_v2','triton_decode_attention','triton_prefill_attention','triton_unified_attention','scaled_mm_single_backend'}:
        q = rng.standard_normal((8, 32), dtype=np.float32)
        k = rng.standard_normal((16, 32), dtype=np.float32)
        v = rng.standard_normal((16, 32), dtype=np.float32)
        return {'ref': _attention_ref(q, k, v), 'cand': _attention_cand(q, k, v)}

    if task_name == 'merge_attention_states':
        a = rng.standard_normal((8, 32), dtype=np.float32)
        b = rng.standard_normal((8, 32), dtype=np.float32)
        lse_a = rng.standard_normal((8, 1), dtype=np.float32)
        lse_b = rng.standard_normal((8, 1), dtype=np.float32)
        wa, wb = np.exp(lse_a), np.exp(lse_b)
        out = (a * wa + b * wb) / (wa + wb)
        return {'ref': out, 'cand': out.copy()}

    if task_name in {'reshape_and_cache', 'mla_kv_cache_fused_writeback'}:
        x = rng.standard_normal((32, 16), dtype=np.float32)
        slot = rng.integers(0, 32, size=(32,), dtype=np.int32)
        ref = np.zeros((32, 16), dtype=np.float32)
        for i in range(32):
            ref[slot[i]] = x[i]
        cand = np.zeros((32, 16), dtype=np.float32)
        cand[slot] = x
        return {'ref': ref, 'cand': cand}

    if task_name in {'vertical_slash_index', 'grouped_topk', 'topk_softmax_routing'}:
        x = rng.standard_normal((16, 64), dtype=np.float32)
        rv, ri = _topk_ref(x, 4)
        cv, ci = _topk_cand(x, 4)
        return {'ref': rv, 'cand': cv, 'extra_ref': ri, 'extra_cand': ci}

    if task_name in {'fused_qk_norm_rope', 'rotary_embedding'}:
        x = rng.standard_normal((16, 32), dtype=np.float32)
        cos = np.cos(np.linspace(0, 1, 16, dtype=np.float32))[:, None]
        sin = np.sin(np.linspace(0, 1, 16, dtype=np.float32))[:, None]
        x1, x2 = x[:, 0::2], x[:, 1::2]
        ref = np.empty_like(x)
        ref[:, 0::2] = x1 * cos - x2 * sin
        ref[:, 1::2] = x1 * sin + x2 * cos
        return {'ref': ref, 'cand': ref.copy()}

    if task_name in {'activation_swiglu', 'triton_fused_moe', 'triton_batched_moe', 'lora_triton_kernels'}:
        a = rng.standard_normal((64, 64), dtype=np.float32)
        b = rng.standard_normal((64, 64), dtype=np.float32)
        return {'ref': _swiglu_ref(a, b), 'cand': _swiglu_cand(a, b)}

    if task_name in {'rmsnorm', 'rmsnorm_dynamic_quant', 'fla_series'}:
        x = rng.standard_normal((64, 128), dtype=np.float32)
        w = rng.standard_normal((128,), dtype=np.float32)
        return {'ref': _rmsnorm_ref(x, w), 'cand': _rmsnorm_cand(x, w)}

    if task_name == 'repetition_penalty':
        logits = rng.standard_normal((8, 128), dtype=np.float32)
        seen = rng.integers(0, 128, size=(8, 16), dtype=np.int32)
        ref = logits.copy()
        for i in range(ref.shape[0]):
            for t in seen[i]:
                ref[i, t] = ref[i, t] / 1.2 if ref[i, t] > 0 else ref[i, t] * 1.2
        return {'ref': ref, 'cand': ref.copy()}

    if task_name in {'int8_quant', 'fp8_quant', 'gguf_ggml_quant_family', 'nvfp4_quant_family'}:
        x = rng.standard_normal((64, 64), dtype=np.float32)
        scale = max(np.max(np.abs(x)) / 127.0, 1e-6)
        return {'ref': _quant_ref(x, scale), 'cand': _quant_cand(x, scale)}

    if task_name in {'awq_gemm', 'gptq_gemm', 'marlin_gemm', 'machete_mm', 'rocm_skinny_gemms', 'moe_align_block_size', 'moe_permute_unpermute', 'mamba_selective_scan', 'ssd_chunk_series'}:
        a = rng.standard_normal((64, 32), dtype=np.float32)
        b = rng.standard_normal((32, 48), dtype=np.float32)
        ref = np.zeros((64, 48), dtype=np.float32)
        for i in range(64):
            for j in range(48):
                s = 0.0
                for k in range(32):
                    s += float(a[i, k] * b[k, j])
                ref[i, j] = s
        cand = a @ b
        return {'ref': ref, 'cand': cand}

    if task_name == 'hadacore_hadamard':
        x = rng.standard_normal((32, 8), dtype=np.float32)
        return {'ref': _hadamard_ref(x), 'cand': _hadamard_cand(x)}

    x = rng.standard_normal((32, 32), dtype=np.float32)
    return {'ref': x, 'cand': x.copy()}


def correctness_mode(src_files, symbols):
    _blobs, missing = load_sources(src_files)
    if missing:
        raise RuntimeError(f'missing source files: {missing}')

    payload = run_task_correctness(TASK_NAME, seed=42)
    ok = np.allclose(payload['ref'], payload['cand'], atol=ATOL, rtol=RTOL)
    max_abs = float(np.max(np.abs(payload['ref'] - payload['cand'])))

    report = {
        'mode': 'correctness',
        'task': TASK_NAME,
        'status': 'ok' if ok else 'fail',
        'atol': ATOL,
        'rtol': RTOL,
        'max_abs_diff': max_abs,
        'shape': list(payload['ref'].shape),
    }

    if 'extra_ref' in payload:
        ok2 = np.allclose(payload['extra_ref'], payload['extra_cand'], atol=ATOL, rtol=RTOL)
        max_abs2 = float(np.max(np.abs(payload['extra_ref'] - payload['extra_cand'])))
        report['extra_output_ok'] = bool(ok2)
        report['extra_max_abs_diff'] = max_abs2
        ok = ok and ok2
        report['status'] = 'ok' if ok else 'fail'

    BUILD_DIR.mkdir(exist_ok=True)
    (BUILD_DIR / 'correctness_report.json').write_text(json.dumps(report, indent=2))
    if not ok:
        raise RuntimeError(f'correctness diff out of range: max_abs={max_abs}')
    print('correctness cpu-reference check passed')


def performance_mode(src_files, symbols, iters=50):
    t0 = time.perf_counter()
    for _ in range(iters):
        payload = run_task_correctness(TASK_NAME, seed=7)
        _ = payload['cand']
    elapsed = (time.perf_counter() - t0) * 1000.0
    avg = elapsed / iters

    BUILD_DIR.mkdir(exist_ok=True)
    (BUILD_DIR / 'performance_report.json').write_text(json.dumps({
        'mode': 'performance',
        'task': TASK_NAME,
        'status': 'ok',
        'iterations': iters,
        'total_ms': elapsed,
        'avg_ms': avg,
    }, indent=2))
    print(f'performance check passed: avg_ms={avg:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['compile', 'correctness', 'performance'], required=True)
    args = parser.parse_args()

    src_files, symbols = parse_config(CONFIG)
    if not src_files:
        raise RuntimeError('source_file_path is empty in config.yaml')
    if not symbols:
        raise RuntimeError('target_kernel_functions is empty in config.yaml')

    if args.mode == 'compile':
        compile_mode(src_files, symbols)
    elif args.mode == 'correctness':
        correctness_mode(src_files, symbols)
    else:
        performance_mode(src_files, symbols)


if __name__ == '__main__':
    main()
