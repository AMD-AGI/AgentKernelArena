# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
GEAK-v3 Triton agent: geak-preprocess + geak-orchestrate pipeline.

Runs inside the GEAK Docker container (invoked via docker exec).
Output directories are placed as a sibling of workspace (_logs/) to prevent
recursive worktree bloat when GEAK copies repo_root for parallel agents.

Uses the two-step pipeline from af/heterogeneous-postprocess branch:
  1. geak-preprocess <kernel_path> --harness <harness> --gpu <gpu> -o <logs_dir>
  2. geak-orchestrate --preprocess-dir <logs_dir> --gpu-ids <gpu_ids>
     --max-rounds <N> [--heterogeneous] --model <model>
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


def _try_patch_with_strip(
    patch_file: str, workspace: str, logger: logging.Logger
) -> bool:
    """Try applying a patch with increasing -p strip levels (p1 through p8).

    GEAK generates patches with nested paths like
    ``a/tasks/triton2triton/geak_eval/L2/topk/kernel.py`` but the AKA
    workspace is flat (kernel.py at root).  ``-p1`` strips only ``a/``,
    leaving the rest unresolvable.  We try higher levels until the file
    is found.
    """
    for p in range(1, 9):
        result = subprocess.run(
            ["patch", f"-p{p}", "--dry-run", "-i", str(patch_file)],
            cwd=workspace, capture_output=True, text=True,
        )
        if result.returncode == 0:
            logger.info(f"patch -p{p} dry-run succeeded, applying")
            subprocess.run(
                ["patch", f"-p{p}", "-i", str(patch_file)],
                cwd=workspace, capture_output=True, text=True, check=True,
            )
            return True
    return False


def _apply_best_patch(workspace: str, logs_dir: Path, logger: logging.Logger) -> tuple[bool, float]:
    """Find and apply the best optimized kernel from orchestrator output back to workspace.

    Returns (applied, best_verified_speedup) — best verified speedup across all
    rounds regardless of whether the patch was successfully applied.

    Strategy (in order):
    1. Read round_N_evaluation.json files to find the highest verified speedup,
       then locate the kernel file in the evaluation worktree.
    2. Try applying the patch from the best verified round's evaluation.json.
    3. Try applying the patch from final_report.json.
    4. Fallback: try per-round best_results.json patches.
    5. Fallback: best_patch_r*.diff files.
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
            # Use FULL_BENCHMARK verified_speedup, not select_agent benchmark_speedup
            fb = data.get("full_benchmark", {})
            speedup = float(fb.get("verified_speedup", 0)) if isinstance(fb, dict) else 0.0
            if speedup <= 0:
                speedup = float(data.get("benchmark_speedup", 0))
            if speedup > best_speedup:
                best_speedup = speedup
                best_round = data.get("round")
                best_task = data.get("best_task")
        except Exception as e:
            logger.warning(f"Error reading {eval_file}: {e}")

    if best_round and best_task:
        task_dir = logs_dir / "results" / f"round_{best_round}" / best_task
        best_results_file = task_dir / "best_results.json"
        if best_results_file.exists():
            try:
                br_data = json.loads(best_results_file.read_text())
                patch_id = br_data.get("best_patch_id", "")
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

        # Search all worktree slots from best round for modified kernels.
        # Kernel files are nested under tasks/triton2triton/geak_eval/.../kernel.py
        # inside each slot, so we search recursively.
        round_dir = logs_dir / "results" / f"round_{best_round}"
        candidates = []
        original_text = ws_kernel.read_text() if ws_kernel.exists() else ""
        for slot_dir in sorted(round_dir.glob("worktrees/slot_*")):
            if not slot_dir.is_dir() or "_logs" in slot_dir.name:
                continue
            # Search recursively for kernel.py (may be nested under tasks/...)
            for candidate in slot_dir.rglob(kernel_name):
                if candidate.read_text() != original_text:
                    candidates.append(candidate)
                    break  # one per slot

        for candidate in candidates:
            slot_name = None
            for p in candidate.parents:
                if p.parent.name == "worktrees":
                    slot_name = p.name
                    break
            logger.info(f"Trying optimized kernel from {slot_name or candidate.parent.name}")
            shutil.copy2(str(candidate), str(ws_kernel))
            check = subprocess.run(
                ["python3", "test_kernel_harness.py", "--correctness"],
                cwd=workspace, capture_output=True, text=True, timeout=120,
            )
            if check.returncode == 0 and "FAIL" not in check.stdout:
                logger.info(
                    f"Optimized kernel from round {best_round} "
                    f"{slot_name or candidate.parent.name} passes correctness (speedup: {best_speedup:.2f}x)"
                )
                return True, best_speedup
            logger.warning(f"{slot_name or candidate.parent.name} failed correctness, trying next")

        if candidates:
            logger.warning("No worktree kernel passed correctness; restoring original")
            ws_kernel.write_text(original_text)

    # --- Fallback: apply patch from best verified round's evaluation ---
    if best_round:
        best_eval = logs_dir / f"round_{best_round}_evaluation.json"
        if best_eval.exists():
            try:
                eval_data = json.loads(best_eval.read_text())
                patch_file = eval_data.get("best_patch")
                if patch_file and Path(patch_file).exists():
                    logger.info(f"Applying patch from best verified round {best_round}: {patch_file}")
                    if _try_patch_with_strip(patch_file, workspace, logger):
                        logger.info(f"Patch from round {best_round} applied (verified: {best_speedup:.2f}x)")
                        return True, best_speedup
                    logger.warning(f"Patch from best verified round {best_round} failed")
            except Exception as e:
                logger.warning(f"Error applying patch from round {best_round} evaluation: {e}")

    # --- Fallback: use patch from final_report.json ---
    # Patches have nested paths (a/tasks/triton2triton/.../kernel.py), so we
    # try multiple -p levels to find one that works.
    final_report = logs_dir / "final_report.json"
    if final_report.exists():
        try:
            data = json.loads(final_report.read_text())
            patch_file = data.get("best_patch")
            if patch_file and Path(patch_file).exists():
                logger.info(f"Applying best patch from final_report.json: {patch_file}")
                applied = _try_patch_with_strip(patch_file, workspace, logger)
                if applied:
                    logger.info(f"Patch applied successfully (speedup: {data.get('best_speedup_verified', 'N/A')}x)")
                    return True, best_speedup
                logger.warning("final_report patch failed at all strip levels")
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
                if _try_patch_with_strip(patch_file, workspace, logger):
                    logger.info("Patch applied successfully (fallback)")
                    return True, best_speedup
                logger.warning(f"Patch failed at all strip levels: {patch_file}")
            except Exception as e:
                logger.warning(f"Error applying patch from {best}: {e}")

    # --- Fallback: best_patch_r*.diff files ---
    for diff in sorted(logs_dir.glob("best_patch_r*.diff"), reverse=True):
        try:
            if _try_patch_with_strip(str(diff), workspace, logger):
                logger.info(f"Applied fallback patch: {diff}")
                return True, best_speedup
        except Exception as e:
            logger.warning(f"Fallback patch {diff} failed: {e}")

    logger.warning("No applicable patch found")
    return False, best_speedup


@register_agent("geak_v3_triton")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch GEAK-v3 Triton agent via two-step pipeline:
      1. geak-preprocess <kernel> --harness <harness> -o <logs_dir>
      2. geak-orchestrate --preprocess-dir <logs_dir> --gpu-ids <gpus>
         --max-rounds <N> [--heterogeneous] --model <model>

    GEAK_SRC env var selects which GEAK branch to use (sets PYTHONPATH).
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

    # Build environment with all GEAK_* vars
    run_env = os.environ.copy()
    for k, v in (agent_config.get("geak_env") or {}).items():
        run_env[k] = str(v)

    gpu_ids = os.environ.get("GEAK_GPU_IDS", eval_config.get("gpu_ids", "0,1,2,3"))
    first_gpu = gpu_ids.split(",")[0]

    orch_config = agent_config.get("orchestrate", {})
    max_rounds = orch_config.get("max_rounds", 3)
    model = orch_config.get("model", "claude-opus-4-6")

    active_config_name = os.environ.get("GEAK_CONFIG_NAME", "heterogeneous_memory_off")
    heterogeneous = True
    for cfg in agent_config.get("configs", []):
        if cfg.get("name") == active_config_name:
            heterogeneous = cfg.get("heterogeneous", True)
            for ek, ev in (cfg.get("extra_env") or {}).items():
                run_env[ek] = str(ev)
            break

    # PYTHONPATH: use GEAK_SRC (stream-specific branch) for pipeline modules
    geak_src = os.environ.get("GEAK_SRC")
    if geak_src and Path(geak_src).is_dir():
        run_env["PYTHONPATH"] = f"{geak_src}:{run_env.get('PYTHONPATH', '')}"
    else:
        run_env["PYTHONPATH"] = f"/workspace/src:{run_env.get('PYTHONPATH', '')}"

    timeout = int(agent_config.get("timeout_seconds", 36000))

    logger.info("=" * 60)
    logger.info("  GEAK-v3 Triton Pipeline (two-step: preprocess + orchestrate)")
    logger.info("=" * 60)
    logger.info(f"  kernel:        {kernel_path}")
    logger.info(f"  harness:       {harness_path}")
    logger.info(f"  workspace:     {workspace_path}")
    logger.info(f"  logs_dir:      {logs_dir}")
    logger.info(f"  gpu_ids:       {gpu_ids}")
    logger.info(f"  first_gpu:     {first_gpu}")
    logger.info(f"  max_rounds:    {max_rounds}")
    logger.info(f"  model:         {model}")
    logger.info(f"  heterogeneous: {heterogeneous}")
    logger.info(f"  config:        {active_config_name}")
    logger.info(f"  PYTHONPATH:    {run_env.get('PYTHONPATH', '')[:120]}")
    for k, v in sorted(run_env.items()):
        if k.startswith("GEAK_"):
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    all_output: list[str] = []

    # Bootstrap a git repo in the workspace so GEAK's .git walk-up stops here
    # (prevents it from finding AgentKernelArena/.git and creating broken worktrees)
    if not (workspace_path / ".git").exists():
        import subprocess as _sp
        # Create .gitignore BEFORE git init to keep patches clean (kernel.py only)
        _gi = workspace_path / ".gitignore"
        if not _gi.exists():
            _gi.write_text(
                "baseline_metrics.json\nprofile.json\n.optimization_strategies.md\n"
                "baseline_perf.yaml\nconfig.yaml\n__pycache__/\n*.pyc\naiter/\n"
            )
        _sp.run(["git", "init"], cwd=str(workspace_path), capture_output=True)
        _sp.run(["git", "add", "."], cwd=str(workspace_path), capture_output=True)
        _sp.run(["git", "commit", "-m", "baseline"], cwd=str(workspace_path), capture_output=True)

    # -- Step 1: geak-preprocess ----------------------------------------
    preprocess_cmd = (
        f"python3 -m minisweagent.run.preprocess.preprocessor"
        f" {kernel_path}"
        f" --harness {harness_path}"
        f" --gpu {first_gpu}"
        f" --model {model}"
        f" -o {logs_dir}"
    )

    rc_pre, out_pre, err_pre = _run_pipeline_step(
        preprocess_cmd,
        env=run_env,
        cwd=str(workspace_path),
        label="preprocess",
        logger=logger,
        timeout=3600,  # 1hr for preprocessing
    )
    all_output.extend(out_pre)

    # Patch resolved.json so GEAK uses workspace dir as repo_root
    # (prevents GEAK from walking up to AKA's .git and creating broken worktrees)
    resolved_json = logs_dir / "resolved.json"
    if resolved_json.exists():
        import json as _json
        resolved_data = _json.loads(resolved_json.read_text())
        if not resolved_data.get("local_repo_path"):
            resolved_data["local_repo_path"] = str(workspace_path)
            resolved_json.write_text(_json.dumps(resolved_data, indent=2, default=str))

    if rc_pre != 0:
        logger.warning(f"geak-preprocess exited with code {rc_pre}")
        all_output.extend(err_pre)
        # Don't abort — orchestrator may still work with partial artifacts

    # -- Step 2: geak-orchestrate ----------------------------------------
    hetero_flag = "--heterogeneous" if heterogeneous else ""
    orchestrate_cmd = (
        f"python3 -m minisweagent.run.orchestrator"
        f" --preprocess-dir {logs_dir}"
        f" --gpu-ids {gpu_ids}"
        f" --max-rounds {max_rounds}"
        f" --model {model}"
        f" {hetero_flag}"
    )

    rc_orch, out_orch, err_orch = _run_pipeline_step(
        orchestrate_cmd,
        env=run_env,
        cwd=str(workspace_path),
        label="orchestrate",
        logger=logger,
        timeout=timeout,
    )
    all_output.extend(out_orch)

    if rc_orch != 0:
        logger.warning(f"geak-orchestrate exited with code {rc_orch}")
        all_output.extend(err_orch)

    # Apply best patch to workspace for AKA evaluator
    if logs_dir.exists():
        applied, best_verified = _apply_best_patch(workspace, logs_dir, logger)
        logger.info(f"Best verified speedup across all rounds: {best_verified:.4f}x (applied={applied})")
        summary = {"best_verified_speedup": best_verified, "patch_applied": applied}
        (logs_dir / "geak_summary.json").write_text(json.dumps(summary, indent=2))
    else:
        logger.warning(f"No results found in {logs_dir}")

    return "\n".join(all_output)
