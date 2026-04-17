#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import venv
from pathlib import Path

TASK_NAME = "repository/aiter/mla_decode_rope"
REPO_SUBDIR = "aiter"
VENV_PACKAGES = [
    "psutil",
    "pytest",
    "pybind11",
    "ninja",
    "pyyaml",
    "pandas",
    "einops",
    "packaging",
]


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _repo_root() -> Path:
    return _workspace_root() / REPO_SUBDIR


def _report_root() -> Path:
    return _workspace_root() / "build"


def _venv_dir() -> Path:
    return _workspace_root() / ".task-venv"


def _venv_python() -> Path:
    return _venv_dir() / "bin" / "python"


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _ensure_venv() -> None:
    venv_dir = _venv_dir()
    venv_python = _venv_python()
    ready_marker = venv_dir / ".ready"
    script_path = Path(__file__).resolve()

    if not venv_python.exists():
        builder = venv.EnvBuilder(with_pip=True, system_site_packages=True)
        builder.create(str(venv_dir))

    if not ready_marker.exists():
        _run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "-q",
                *VENV_PACKAGES,
            ],
            cwd=_workspace_root(),
        )
        ready_marker.write_text("\n".join(VENV_PACKAGES) + "\n")

    current_python = Path(sys.executable).resolve()
    if current_python != venv_python.resolve():
        env = os.environ.copy()
        env.setdefault("ENABLE_CK", "0")
        env.setdefault("AITER_LOG_LEVEL", "WARNING")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{_repo_root()}{os.pathsep}{existing}" if existing else str(_repo_root())
        )
        os.execve(
            str(venv_python),
            [str(venv_python), str(script_path), *sys.argv[1:]],
            env,
        )


def _configure_runtime() -> None:
    os.environ.setdefault("ENABLE_CK", "0")
    os.environ.setdefault("AITER_LOG_LEVEL", "WARNING")
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)


def _benchmark_ms(fn, warmup: int = 10, rep: int = 30) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(rep):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _write_performance_report(results: list[dict]) -> None:
    report_root = _report_root()
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "performance_report.json").write_text(json.dumps(results, indent=2))


def _make_case(
    *,
    B: int,
    H: int,
    S: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    rotary_dim: int,
    equal_seqlens: bool,
    is_neox_style: bool,
):
    import torch
    from op_tests.triton_tests.attention.test_mla_decode_rope import (
        input_helper,
        ref_preprocess,
    )

    num_kv_splits = 2
    dtype = torch.bfloat16
    device = "cuda"
    torch.manual_seed(0)
    (
        kv_indptr,
        kv_indices,
        q,
        kv_cache,
        attn_logits,
        rotary_emb,
        positions,
        _,
    ) = input_helper(
        B,
        H,
        S,
        kv_lora_rank,
        rotary_dim,
        qk_rope_head_dim,
        num_kv_splits,
        dtype,
        device,
        equal_seqlens=equal_seqlens,
        is_neox_style=is_neox_style,
    )
    k_input, v_input = ref_preprocess(kv_cache, kv_lora_rank)
    k_pe_tokens = torch.empty(B, qk_rope_head_dim, dtype=kv_cache.dtype, device=device)
    o = torch.empty(B, H, kv_lora_rank, dtype=kv_cache.dtype, device=device)
    return {
        "params": {
            "B": B,
            "H": H,
            "S": S,
            "kv_lora_rank": kv_lora_rank,
            "qk_rope_head_dim": qk_rope_head_dim,
            "rotary_dim": rotary_dim,
            "equal_seqlens": equal_seqlens,
            "is_neox_style": is_neox_style,
        },
        "num_kv_splits": num_kv_splits,
        "q": q,
        "k_input": k_input,
        "v_input": v_input,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "attn_logits": attn_logits,
        "rotary_emb": rotary_emb,
        "positions": positions,
        "k_pe_tokens": k_pe_tokens,
        "o": o,
    }


def _run_kernel(case: dict) -> None:
    from aiter.ops.triton.attention.mla_decode_rope import decode_attention_fwd_grouped_rope

    params = case["params"]
    decode_attention_fwd_grouped_rope(
        case["q"],
        case["k_input"],
        case["v_input"],
        case["o"],
        case["kv_indptr"],
        case["kv_indices"],
        case["k_pe_tokens"],
        params["kv_lora_rank"],
        params["rotary_dim"],
        case["rotary_emb"].cos_sin_cache,
        case["positions"],
        case["attn_logits"],
        case["num_kv_splits"],
        1.0,
        0.0,
        True,
        params["is_neox_style"],
    )


def run_compile() -> None:
    case = _make_case(
        B=1,
        H=8,
        S=32,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        rotary_dim=16,
        equal_seqlens=True,
        is_neox_style=True,
    )
    _run_kernel(case)
    print(f"{TASK_NAME} compile smoke: PASS")


def run_correctness() -> None:
    import torch
    from op_tests.triton_tests.attention.test_mla_decode_rope import ref_compute_full_fwd

    cases = [
        _make_case(
            B=1,
            H=8,
            S=32,
            kv_lora_rank=32,
            qk_rope_head_dim=16,
            rotary_dim=16,
            equal_seqlens=True,
            is_neox_style=True,
        ),
        _make_case(
            B=2,
            H=8,
            S=48,
            kv_lora_rank=32,
            qk_rope_head_dim=16,
            rotary_dim=16,
            equal_seqlens=False,
            is_neox_style=False,
        ),
    ]

    for idx, case in enumerate(cases):
        _run_kernel(case)
        params = case["params"]
        ref_logits, ref_o, ref_k_pe_tokens = ref_compute_full_fwd(
            case["q"],
            case["k_input"],
            case["v_input"],
            params["kv_lora_rank"],
            case["kv_indptr"],
            case["kv_indices"],
            case["num_kv_splits"],
            1.0,
            0.0,
            case["rotary_emb"],
            case["positions"],
            True,
            device="cuda",
        )
        torch.testing.assert_close(ref_logits, case["attn_logits"], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(ref_o, case["o"], atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(
            ref_k_pe_tokens,
            case["k_pe_tokens"].squeeze(),
            atol=1e-2,
            rtol=1e-2,
        )
        print(f"Correctness case {idx}: PASS")


def run_performance() -> None:
    benchmark_cases = [
        (
            "mla_decode_small",
            _make_case(
                B=1,
                H=8,
                S=32,
                kv_lora_rank=32,
                qk_rope_head_dim=16,
                rotary_dim=16,
                equal_seqlens=True,
                is_neox_style=True,
            ),
        ),
        (
            "mla_decode_medium",
            _make_case(
                B=2,
                H=16,
                S=96,
                kv_lora_rank=64,
                qk_rope_head_dim=32,
                rotary_dim=32,
                equal_seqlens=True,
                is_neox_style=True,
            ),
        ),
    ]

    results: list[dict] = []
    for test_case_id, case in benchmark_cases:
        _run_kernel(case)
        time_ms = _benchmark_ms(lambda: _run_kernel(case))
        params = case["params"]
        results.append(
            {
                "test_case_id": test_case_id,
                "shape": [
                    params["B"],
                    params["H"],
                    params["S"],
                    params["kv_lora_rank"],
                    params["qk_rope_head_dim"],
                ],
                "execution_time_ms": time_ms,
                "metadata": params,
            }
        )
        print(f"{test_case_id}: {time_ms:.4f} ms")

    _write_performance_report(results)


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    _ensure_venv()
    _configure_runtime()

    if args.mode == "compile":
        run_compile()
    elif args.mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
