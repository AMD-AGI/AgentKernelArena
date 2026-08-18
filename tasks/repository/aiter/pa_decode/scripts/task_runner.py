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
from _aka_benchmark import benchmark_cuda_graph_or_events_samples

TASK_NAME = "repository/aiter/pa_decode"
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


def _inherit_parent_site_packages(venv_python: Path) -> None:
    """Expose packages from the image's parent venv inside the task venv."""

    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    child_site = subprocess.check_output(
        [str(venv_python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        text=True,
    ).strip()
    parent_sites = [
        entry
        for entry in sys.path
        if "site-packages" in entry and Path(entry).is_dir()
    ]
    (Path(child_site) / "aka_parent_site_packages.pth").write_text(
        "\n".join(
            f"import site; site.addsitedir({entry!r})" for entry in parent_sites
        ) + "\n"
    )


def _ensure_venv() -> None:
    os.environ.setdefault("USER", "agentkernelarena")
    os.environ.setdefault("LOGNAME", "agentkernelarena")
    venv_dir = _venv_dir()
    venv_python = _venv_python()
    ready_marker = venv_dir / ".ready"
    script_path = Path(__file__).resolve()

    if not venv_python.exists():
        builder = venv.EnvBuilder(with_pip=True, system_site_packages=True)
        builder.create(str(venv_dir))

    _inherit_parent_site_packages(venv_python)

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


def _benchmark_ms(fn, warmup: int = 10, rep: int = 30) -> tuple[float, dict]:
    samples, metadata = benchmark_cuda_graph_or_events_samples(
        fn, warmup=warmup, repetition=rep,
    )
    return statistics.median(samples), metadata


def _write_performance_report(results: list[dict]) -> None:
    report_root = _report_root()
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "performance_report.json").write_text(json.dumps(results, indent=2))


def _make_case(
    *,
    B: int,
    H_Q: int,
    H_KV: int,
    D: int,
    KV_BLK_SZ: int,
    SEQ_LEN: int,
    num_blocks: int,
):
    import torch
    import triton.language as tl
    from op_tests.triton_tests.attention.test_pa_decode import input_helper

    query, output, key_cache, value_cache, key_cache_tri, value_cache_tri, context_lens, block_tables, max_context_len = input_helper(
        B=B,
        H_Q=H_Q,
        H_KV=H_KV,
        D=D,
        KV_BLK_SZ=KV_BLK_SZ,
        SEQ_LEN=SEQ_LEN,
        dtype=torch.float16,
        kv_cache_dtype=torch.float16,
        output_type=torch.float16,
        num_blocks=num_blocks,
        random_seed=0,
    )
    return {
        "params": {
            "B": B,
            "H_Q": H_Q,
            "H_KV": H_KV,
            "D": D,
            "KV_BLK_SZ": KV_BLK_SZ,
            "SEQ_LEN": SEQ_LEN,
            "num_blocks": num_blocks,
        },
        "query": query,
        "output": output,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "key_cache_tri": key_cache_tri,
        "value_cache_tri": value_cache_tri,
        "context_lens": context_lens,
        "block_tables": block_tables,
        "max_context_len": max_context_len,
        "compute_type": tl.float16,
        "k_scale": torch.tensor([1.0], device=query.device),
        "v_scale": torch.tensor([1.0], device=query.device),
    }


def _run_kernel(case: dict) -> None:
    from aiter.ops.triton.attention.pa_decode import paged_attention_decode

    D = case["params"]["D"]
    paged_attention_decode(
        case["output"],
        case["query"],
        case["key_cache_tri"],
        case["value_cache_tri"],
        case["context_lens"],
        case["block_tables"],
        1.0 / (D**0.5),
        case["max_context_len"],
        case["compute_type"],
        case["k_scale"],
        case["v_scale"],
    )


def run_compile() -> None:
    case = _make_case(B=1, H_Q=1, H_KV=1, D=64, KV_BLK_SZ=4, SEQ_LEN=32, num_blocks=8)
    _run_kernel(case)
    print(f"{TASK_NAME} compile smoke: PASS")


def run_correctness() -> None:
    import torch
    from op_tests.triton_tests.attention.test_pa_decode import paged_attention_decode_ref

    cases = [
        _make_case(B=1, H_Q=1, H_KV=1, D=64, KV_BLK_SZ=4, SEQ_LEN=32, num_blocks=8),
        _make_case(B=2, H_Q=16, H_KV=16, D=128, KV_BLK_SZ=16, SEQ_LEN=96, num_blocks=16),
    ]

    for idx, case in enumerate(cases):
        ref = torch.zeros_like(case["output"])
        _run_kernel(case)
        paged_attention_decode_ref(
            ref,
            case["query"],
            case["key_cache"],
            case["value_cache"],
            case["block_tables"],
            case["context_lens"],
        )
        torch.testing.assert_close(case["output"], ref, rtol=1e-2, atol=1e-2)
        print(f"Correctness case {idx}: PASS")


def run_performance() -> None:
    benchmark_cases = [
        ("pa_decode_small", _make_case(B=1, H_Q=1, H_KV=1, D=64, KV_BLK_SZ=4, SEQ_LEN=64, num_blocks=8)),
        ("pa_decode_medium", _make_case(B=4, H_Q=16, H_KV=16, D=128, KV_BLK_SZ=16, SEQ_LEN=256, num_blocks=32)),
    ]

    results: list[dict] = []
    for test_case_id, case in benchmark_cases:
        _run_kernel(case)
        time_ms, benchmark_meta = _benchmark_ms(lambda: _run_kernel(case))
        params = case["params"]
        results.append(
            {
                "test_case_id": test_case_id,
                "shape": [
                    params["B"],
                    params["H_Q"],
                    params["H_KV"],
                    params["D"],
                    params["SEQ_LEN"],
                ],
                "execution_time_ms": time_ms,
                **benchmark_meta,
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
