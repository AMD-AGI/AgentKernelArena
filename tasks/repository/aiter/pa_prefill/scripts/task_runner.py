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

TASK_NAME = "repository/aiter/pa_prefill"
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
    BS: int,
    MAX_SEQ_LEN: int,
    MAX_CTX_LEN: int,
    cache_size: int,
    block_size: int,
    max_block_per_request: int,
    num_heads: int,
    head_size: int,
    num_queries_per_kv: int,
    sliding_window: int,
):
    import torch
    from op_tests.triton_tests.attention.test_pa_prefill import input_helper

    (
        query,
        k,
        v,
        output,
        k_cache,
        v_cache,
        block_table,
        b_start_loc,
        b_seq_len,
        max_input_len,
        k_scale,
        v_scale,
        _,
    ) = input_helper(
        BS=BS,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        MAX_CTX_LEN=MAX_CTX_LEN,
        cache_size=cache_size,
        block_size=block_size,
        max_block_per_request=max_block_per_request,
        num_heads=num_heads,
        head_size=head_size,
        num_queries_per_kv=num_queries_per_kv,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        device="cuda:0",
        use_alibi_slope=False,
    )
    return {
        "params": {
            "BS": BS,
            "MAX_SEQ_LEN": MAX_SEQ_LEN,
            "MAX_CTX_LEN": MAX_CTX_LEN,
            "cache_size": cache_size,
            "block_size": block_size,
            "max_block_per_request": max_block_per_request,
            "num_heads": num_heads,
            "head_size": head_size,
            "num_queries_per_kv": num_queries_per_kv,
            "sliding_window": sliding_window,
        },
        "query": query,
        "k": k,
        "v": v,
        "output": output,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "block_table": block_table,
        "b_start_loc": b_start_loc,
        "b_seq_len": b_seq_len,
        "max_input_len": max_input_len,
        "k_scale": k_scale,
        "v_scale": v_scale,
    }


def _run_kernel(case: dict) -> None:
    from aiter.ops.triton.attention.pa_prefill import context_attention_fwd

    context_attention_fwd(
        case["query"],
        case["k"],
        case["v"],
        case["output"],
        "auto",
        case["k_cache"],
        case["v_cache"],
        case["block_table"],
        case["b_start_loc"],
        case["b_seq_len"],
        case["max_input_len"],
        case["k_scale"],
        case["v_scale"],
        sliding_window=case["params"]["sliding_window"],
    )


def run_compile() -> None:
    case = _make_case(
        BS=2,
        MAX_SEQ_LEN=32,
        MAX_CTX_LEN=32,
        cache_size=64,
        block_size=16,
        max_block_per_request=4,
        num_heads=64,
        head_size=24,
        num_queries_per_kv=64,
        sliding_window=0,
    )
    _run_kernel(case)
    print(f"{TASK_NAME} compile smoke: PASS")


def run_correctness() -> None:
    import torch
    from op_tests.triton_tests.attention.test_pa_prefill import context_attention_fwd_torch

    cases = [
        _make_case(
            BS=2,
            MAX_SEQ_LEN=32,
            MAX_CTX_LEN=32,
            cache_size=64,
            block_size=16,
            max_block_per_request=4,
            num_heads=64,
            head_size=24,
            num_queries_per_kv=64,
            sliding_window=0,
        ),
        _make_case(
            BS=2,
            MAX_SEQ_LEN=64,
            MAX_CTX_LEN=64,
            cache_size=128,
            block_size=16,
            max_block_per_request=8,
            num_heads=16,
            head_size=64,
            num_queries_per_kv=1,
            sliding_window=16,
        ),
    ]

    for idx, case in enumerate(cases):
        ref = torch.empty_like(case["output"])
        _run_kernel(case)
        context_attention_fwd_torch(
            case["query"],
            case["k"],
            case["v"],
            ref,
            case["k_cache"],
            case["v_cache"],
            case["b_start_loc"],
            case["b_seq_len"],
            case["k_scale"],
            case["v_scale"],
            None,
            case["params"]["sliding_window"],
        )
        torch.testing.assert_close(case["output"], ref, atol=2e-2, rtol=2e-2)
        print(f"Correctness case {idx}: PASS")


def run_performance() -> None:
    benchmark_cases = [
        (
            "pa_prefill_small",
            _make_case(
                BS=2,
                MAX_SEQ_LEN=64,
                MAX_CTX_LEN=64,
                cache_size=128,
                block_size=16,
                max_block_per_request=8,
                num_heads=16,
                head_size=64,
                num_queries_per_kv=1,
                sliding_window=0,
            ),
        ),
        (
            "pa_prefill_medium",
            _make_case(
                BS=4,
                MAX_SEQ_LEN=128,
                MAX_CTX_LEN=128,
                cache_size=256,
                block_size=16,
                max_block_per_request=16,
                num_heads=64,
                head_size=24,
                num_queries_per_kv=64,
                sliding_window=32,
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
                    params["BS"],
                    params["num_heads"],
                    params["head_size"],
                    params["MAX_SEQ_LEN"],
                    params["MAX_CTX_LEN"],
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
