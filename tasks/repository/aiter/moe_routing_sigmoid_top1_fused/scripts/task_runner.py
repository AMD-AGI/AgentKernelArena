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

TASK_NAME = "repository/aiter/moe_routing_sigmoid_top1_fused"
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


def _make_case(*, M: int, N: int, K: int):
    import torch

    torch.manual_seed(7)
    dtype = torch.bfloat16
    x = torch.randint(-2, 3, (M, K), device="cuda").to(dtype)
    w = torch.randint(-2, 3, (K, N), device="cuda").to(dtype)
    dummy_ids = torch.ones((M, 1), dtype=torch.int32, device="cuda") * N
    dummy_weights = torch.ones((M, 1), dtype=torch.float32, device="cuda")
    return {
        "params": {"M": M, "N": N, "K": K},
        "x": x,
        "w": w,
        "dummy_ids": dummy_ids,
        "dummy_weights": dummy_weights,
    }


def _run_kernel(case: dict):
    from aiter.ops.triton.moe.moe_routing_sigmoid_top1_fused import routing_sigmoid_top1

    return routing_sigmoid_top1(
        case["x"],
        case["w"],
        1,
        fused_shared_experts=True,
    )


def run_compile() -> None:
    case = _make_case(M=256, N=32, K=64)
    _run_kernel(case)
    print(f"{TASK_NAME} compile smoke: PASS")


def run_correctness() -> None:
    import torch
    from op_tests.triton_tests.moe.test_moe_routing_sigmoid_top1_fused import (
        torch_routing_sigmoid_top1,
    )

    cases = [
        _make_case(M=256, N=32, K=64),
        _make_case(M=1024, N=128, K=128),
    ]

    for idx, case in enumerate(cases):
        ids, weights = _run_kernel(case)
        ref_ids, ref_weights = torch_routing_sigmoid_top1(
            case["x"],
            case["w"],
            1,
            fused_shared_experts=True,
            dummy_ids=case["dummy_ids"],
            dummy_weights=case["dummy_weights"],
        )
        torch.testing.assert_close(ids, ref_ids, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(weights, ref_weights, atol=1e-5, rtol=1e-5)
        print(f"Correctness case {idx}: PASS")


def run_performance() -> None:
    benchmark_cases = [
        ("moe_routing_small", _make_case(M=1024, N=16, K=128)),
        ("moe_routing_medium", _make_case(M=4096, N=128, K=128)),
    ]

    results: list[dict] = []
    for test_case_id, case in benchmark_cases:
        _run_kernel(case)
        time_ms, benchmark_meta = _benchmark_ms(lambda: _run_kernel(case))
        params = case["params"]
        results.append(
            {
                "test_case_id": test_case_id,
                "shape": [params["M"], params["N"], params["K"]],
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
