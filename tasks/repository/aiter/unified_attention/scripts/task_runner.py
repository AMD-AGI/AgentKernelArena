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

TASK_NAME = "repository/aiter/unified_attention"
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
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    sliding_window: int | None,
    num_blocks: int,
):
    import torch

    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads, num_kv_heads = num_heads
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    query = torch.randn(
        sum(query_lens),
        num_query_heads,
        head_size,
        dtype=torch.float32,
        device="cuda",
    ).bfloat16()
    key_cache = torch.randn(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        dtype=torch.float32,
        device="cuda",
    ).bfloat16()
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device="cuda"
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0,
        num_blocks,
        (len(seq_lens), max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16, device="cuda")
    output = torch.empty_like(query)
    return {
        "params": {
            "seq_lens": seq_lens,
            "num_heads": num_heads,
            "head_size": head_size,
            "block_size": block_size,
            "sliding_window": sliding_window,
            "num_blocks": num_blocks,
        },
        "query_lens": query_lens,
        "kv_lens": kv_lens,
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "cu_query_lens": cu_query_lens,
        "kv_lens_tensor": kv_lens_tensor,
        "block_tables": block_tables,
        "sinks": sinks,
        "output": output,
        "max_query_len": max_query_len,
        "max_kv_len": max_kv_len,
    }


def _run_kernel(case: dict) -> None:
    from aiter.ops.triton.attention.unified_attention import unified_attention

    params = case["params"]
    window_size = (
        (params["sliding_window"] - 1, 0)
        if params["sliding_window"] is not None
        else (-1, -1)
    )
    scale = params["head_size"] ** -0.5
    unified_attention(
        q=case["query"],
        k=case["key_cache"],
        v=case["value_cache"],
        out=case["output"],
        cu_seqlens_q=case["cu_query_lens"],
        seqused_k=case["kv_lens_tensor"],
        max_seqlen_q=case["max_query_len"],
        max_seqlen_k=case["max_kv_len"],
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=case["block_tables"],
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=case["sinks"],
        output_scale=None,
    )


def run_compile() -> None:
    case = _make_case(
        seq_lens=[(1, 32), (3, 48)],
        num_heads=(4, 4),
        head_size=128,
        block_size=16,
        sliding_window=None,
        num_blocks=64,
    )
    _run_kernel(case)
    print(f"{TASK_NAME} compile smoke: PASS")


def run_correctness() -> None:
    import torch
    from op_tests.triton_tests.attention.test_unified_attention import ref_paged_attn

    cases = [
        _make_case(
            seq_lens=[(1, 32), (3, 48)],
            num_heads=(4, 4),
            head_size=128,
            block_size=16,
            sliding_window=None,
            num_blocks=64,
        ),
        _make_case(
            seq_lens=[(1, 64), (1, 96)],
            num_heads=(8, 2),
            head_size=128,
            block_size=16,
            sliding_window=64,
            num_blocks=96,
        ),
    ]

    for idx, case in enumerate(cases):
        _run_kernel(case)
        ref = ref_paged_attn(
            query=case["query"],
            key_cache=case["key_cache"],
            value_cache=case["value_cache"],
            query_lens=case["query_lens"],
            kv_lens=case["kv_lens"],
            block_tables=case["block_tables"],
            scale=case["params"]["head_size"] ** -0.5,
            out_dtype=torch.bfloat16,
            sliding_window=case["params"]["sliding_window"],
            sinks=case["sinks"],
        )
        torch.testing.assert_close(case["output"].float(), ref.float(), atol=1.5e-2, rtol=1e-2)
        print(f"Correctness case {idx}: PASS")


def run_performance() -> None:
    benchmark_cases = [
        (
            "unified_attention_small",
            _make_case(
                seq_lens=[(1, 64), (2, 96)],
                num_heads=(4, 4),
                head_size=128,
                block_size=16,
                sliding_window=None,
                num_blocks=96,
            ),
        ),
        (
            "unified_attention_medium",
            _make_case(
                seq_lens=[(1, 128), (1, 256)],
                num_heads=(8, 2),
                head_size=128,
                block_size=16,
                sliding_window=128,
                num_blocks=256,
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
                    params["num_heads"][0],
                    params["num_heads"][1],
                    params["head_size"],
                    max(q for q, _ in params["seq_lens"]),
                    max(k for _, k in params["seq_lens"]),
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
