# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
GEAK-v3 Triton agent: full geak-preprocess + geak-orchestrate pipeline.

Runs inside the GEAK Docker container (invoked by run_geak_triton.sh via docker exec).
Output directories are placed as a sibling of workspace (_logs/) to prevent
recursive worktree bloat when GEAK copies repo_root for parallel agents.
"""
import json
import logging
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent


def _read_stream(stream, lines: list, prefix: str, log_func):
    """Read from a subprocess stream in a background thread."""
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            raw = line.rstrip()
            if raw.strip():
                lines.append(raw)
                log_func(f"{prefix} {raw}")
    finally:
        stream.close()


def _run_pipeline_step(
    cmd: str,
    *,
    env: dict[str, str],
    cwd: str,
    label: str,
    logger: logging.Logger,
    timeout: int = 36000,
) -> tuple[int, list[str], list[str]]:
    """Run a pipeline step with real-time streaming."""
    logger.info(f"[{label}] Running: {cmd}")
    logger.info(f"[{label}] cwd: {cwd}")

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        bufsize=1,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    t_out = threading.Thread(
        target=_read_stream, args=(proc.stdout, stdout_lines, f"[{label}]", logger.info), daemon=True
    )
    t_err = threading.Thread(
        target=_read_stream, args=(proc.stderr, stderr_lines, f"[{label} ERR]", logger.warning), daemon=True
    )
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"[{label}] Timed out after {timeout}s; killing")
        proc.kill()

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    logger.info(f"[{label}] exit code: {proc.returncode}")
    return proc.returncode, stdout_lines, stderr_lines


def _apply_best_patch(workspace: str, logs_dir: Path, logger: logging.Logger) -> bool:
    """Find and apply the best optimized kernel from orchestrator output back to workspace.

    Strategy (in order):
    1. Read round_N_evaluation.json files (reverse order) to find the highest
       verified speedup, then locate the kernel file in the evaluation worktree.
    2. Try applying the patch from final_report.json.
    3. Fallback: try per-round best_results.json patches.
    4. Fallback: best_patch_r*.diff files.
    """
    import shutil

    kernel_name = "kernel.py"
    ws_kernel = Path(workspace) / kernel_name

    # --- Primary: find best verified round and copy kernel from worktree ---
    best_speedup = 0.0
    best_round = None
    best_task = None
    for eval_file in sorted(logs_dir.glob("round_*_evaluation.json"), reverse=True):
        try:
            data = json.loads(eval_file.read_text())
            if data.get("status") == "patch_failed":
                continue
            speedup = float(data.get("benchmark_speedup", 0))
            if speedup > best_speedup:
                best_speedup = speedup
                best_round = data.get("round")
                best_task = data.get("best_task")
        except Exception as e:
            logger.warning(f"Error reading {eval_file}: {e}")

    if best_round and best_task:
        # Find the best task's patch file and reconstruct the kernel
        task_dir = logs_dir / "results" / f"round_{best_round}" / best_task
        best_results_file = task_dir / "best_results.json"
        if best_results_file.exists():
            try:
                br_data = json.loads(best_results_file.read_text())
                patch_id = br_data.get("best_patch_id", "")
                # The patch test output contains the verified result
                test_file = task_dir / f"{patch_id}_test.txt"
                if test_file.exists():
                    test_output = test_file.read_text()
                    if "ALL PASS" in test_output or "PASS" in test_output:
                        logger.info(
                            f"Best round {best_round}, task {best_task}, "
                            f"patch {patch_id} (speedup: {best_speedup:.2f}x)"
                        )
            except Exception:
                pass

        # Search all worktree slots from best round for modified kernels
        # Try to apply each and validate via correctness check
        round_dir = logs_dir / "results" / f"round_{best_round}"
        candidates = []
        original_text = ws_kernel.read_text() if ws_kernel.exists() else ""
        for slot_dir in sorted(round_dir.glob("worktrees/slot_*")):
            if not slot_dir.is_dir() or "_logs" in slot_dir.name:
                continue
            candidate = slot_dir / kernel_name
            if candidate.exists() and candidate.read_text() != original_text:
                candidates.append(candidate)

        # Try each candidate; pick one that passes correctness
        for candidate in candidates:
            logger.info(f"Trying optimized kernel from {candidate.parent.name}")
            shutil.copy2(str(candidate), str(ws_kernel))
            # Quick correctness check
            check = subprocess.run(
                ["python3", "test_kernel_harness.py", "--correctness"],
                cwd=workspace, capture_output=True, text=True, timeout=120,
            )
            if check.returncode == 0 and "FAIL" not in check.stdout:
                logger.info(
                    f"Optimized kernel from round {best_round} "
                    f"{candidate.parent.name} passes correctness (speedup: {best_speedup:.2f}x)"
                )
                return True
            logger.warning(f"{candidate.parent.name} failed correctness, trying next")

        # If none passed, restore original
        if candidates:
            logger.warning("No worktree kernel passed correctness; restoring original")
            ws_kernel.write_text(original_text)

    # --- Fallback: use patch from final_report.json ---
    final_report = logs_dir / "final_report.json"
    if final_report.exists():
        try:
            data = json.loads(final_report.read_text())
            patch_file = data.get("best_patch")
            if patch_file and Path(patch_file).exists():
                logger.info(f"Applying best patch from final_report.json: {patch_file}")
                result = subprocess.run(
                    ["patch", "-p1", "--dry-run", "-i", str(patch_file)],
                    cwd=workspace, capture_output=True, text=True,
                )
                if result.returncode == 0:
                    subprocess.run(
                        ["patch", "-p1", "-i", str(patch_file)],
                        cwd=workspace, capture_output=True, text=True, check=True,
                    )
                    logger.info(f"Patch applied successfully (speedup: {data.get('best_speedup_verified', 'N/A')}x)")
                    return True
                logger.warning(f"final_report patch dry-run failed: {result.stderr}")
        except Exception as e:
            logger.warning(f"Error reading final_report.json: {e}")

    # --- Fallback: per-round best_results.json ---
    for rdir in sorted(logs_dir.glob("results/round_*"), reverse=True):
        for task_dir in sorted(rdir.iterdir()):
            if not task_dir.is_dir() or task_dir.name == "worktrees":
                continue
            best = task_dir / "best_results.json"
            if not best.exists():
                continue
            try:
                data = json.loads(best.read_text())
                patch_file = data.get("best_patch_file")
                if not patch_file or not Path(patch_file).exists():
                    continue
                logger.info(f"Applying best patch (fallback): {patch_file}")
                result = subprocess.run(
                    ["patch", "-p1", "--dry-run", "-i", str(patch_file)],
                    cwd=workspace, capture_output=True, text=True,
                )
                if result.returncode == 0:
                    subprocess.run(
                        ["patch", "-p1", "-i", str(patch_file)],
                        cwd=workspace, capture_output=True, text=True, check=True,
                    )
                    logger.info("Patch applied successfully (fallback)")
                    return True
                logger.warning(f"Patch dry-run failed: {result.stderr}")
            except Exception as e:
                logger.warning(f"Error applying patch from {best}: {e}")

    # --- Fallback: best_patch_r*.diff files ---
    for diff in sorted(logs_dir.glob("best_patch_r*.diff"), reverse=True):
        try:
            result = subprocess.run(
                ["patch", "-p1", "--dry-run", "-i", str(diff)],
                cwd=workspace, capture_output=True, text=True,
            )
            if result.returncode == 0:
                subprocess.run(
                    ["patch", "-p1", "-i", str(diff)],
                    cwd=workspace, capture_output=True, text=True, check=True,
                )
                logger.info(f"Applied fallback patch: {diff}")
                return True
        except Exception as e:
            logger.warning(f"Fallback patch {diff} failed: {e}")

    logger.warning("No applicable patch found")
    return False


@register_agent("geak_v3_triton")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch GEAK-v3 Triton agent via the unified ``geak`` CLI.

    Calls: geak --kernel-url <kernel> --eval <harness> --gpu-ids <gpus>
           --max-rounds <N> [--heterogeneous] --yolo -o <logs_dir>

    This is the same entrypoint used by HIP kernels (geak_v3), ensuring
    a single pipeline for both Triton and HIP.
    """
    logger = logging.getLogger(__name__)

    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open() as f:
        agent_config = yaml.safe_load(f) or {}

    with open(task_config_dir) as f:
        task_config = yaml.safe_load(f) or {}

    workspace_path = Path(workspace).resolve()
    kernel_path = workspace_path / (task_config.get("source_file_path", ["kernel.py"])[0])
    harness_path = workspace_path / task_config.get("harness_path", "test_kernel_harness.py")

    if not kernel_path.is_file():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")
    if not harness_path.is_file():
        raise FileNotFoundError(f"Harness not found: {harness_path}")

    # Logs dir as sibling of workspace (prevents recursive copy bloat)
    logs_dir = workspace_path.parent / f"{workspace_path.name}_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    preprocess_dir = logs_dir / "preprocess"

    # Build environment with all GEAK_* vars
    run_env = os.environ.copy()
    for k, v in (agent_config.get("geak_env") or {}).items():
        run_env[k] = str(v)

    gpu_ids = os.environ.get("GEAK_GPU_IDS", eval_config.get("gpu_ids", "0,1,2,3"))

    orch_config = agent_config.get("orchestrate", {})
    max_rounds = orch_config.get("max_rounds", 3)
    model = orch_config.get("model", "claude-opus-4.6")

    active_config_name = os.environ.get("GEAK_CONFIG_NAME", "heterogeneous_memory_off")
    heterogeneous = True
    for cfg in agent_config.get("configs", []):
        if cfg.get("name") == active_config_name:
            heterogeneous = cfg.get("heterogeneous", True)
            for ek, ev in (cfg.get("extra_env") or {}).items():
                run_env[ek] = str(ev)
            break

    # PYTHONPATH: prefer GEAK repo from home dir (mounted into container)
    geak_src = os.environ.get("GEAK_SRC", "/home/sapmajum/GEAK-agent-filtering-and-cli-unification/src")
    if not Path(geak_src).is_dir():
        geak_src = "/workspace/src"
    run_env["PYTHONPATH"] = f"{geak_src}:{run_env.get('PYTHONPATH', '')}"

    timeout = int(agent_config.get("timeout_seconds", 36000))

    logger.info("=" * 60)
    logger.info("  GEAK-v3 Triton Pipeline (via geak CLI)")
    logger.info("=" * 60)
    logger.info(f"  kernel:        {kernel_path}")
    logger.info(f"  harness:       {harness_path}")
    logger.info(f"  workspace:     {workspace_path}")
    logger.info(f"  logs_dir:      {logs_dir}")
    logger.info(f"  gpu_ids:       {gpu_ids}")
    logger.info(f"  max_rounds:    {max_rounds}")
    logger.info(f"  model:         {model}")
    logger.info(f"  heterogeneous: {heterogeneous}")
    logger.info(f"  config:        {active_config_name}")
    for k, v in sorted(run_env.items()):
        if k.startswith("GEAK_"):
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # Build unified geak command — same entrypoint for Triton and HIP
    hetero_flag = "--heterogeneous" if heterogeneous else ""
    geak_cmd = (
        f"geak"
        f" --kernel-url {kernel_path}"
        f" --eval {harness_path}"
        f" --gpu-ids {gpu_ids}"
        f" --max-rounds {max_rounds}"
        f" --model {model}"
        f" {hetero_flag}"
        f" --yolo"
        f" -o {logs_dir}"
    )

    rc, out, err = _run_pipeline_step(
        geak_cmd,
        env=run_env,
        cwd=str(workspace_path),
        label="geak",
        logger=logger,
        timeout=timeout,
    )

    all_output: list[str] = []
    all_output.extend(out)
    if rc != 0:
        logger.warning(f"geak exited with code {rc}")
        all_output.extend(err)

    # Apply best patch to workspace for AKA evaluator
    # The geak CLI writes results directly to logs_dir (not logs_dir/preprocess)
    # Try logs_dir first, fall back to preprocess subdir for backward compat
    patch_search_dir = logs_dir if (logs_dir / "final_report.json").exists() else preprocess_dir
    if patch_search_dir.exists():
        _apply_best_patch(workspace, patch_search_dir, logger)
    else:
        logger.warning(f"No results found in {logs_dir} or {preprocess_dir}")

    return "\n".join(all_output)
