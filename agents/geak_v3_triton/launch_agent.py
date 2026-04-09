# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
GEAK-v3 Triton agent: unified ``geak`` CLI invocation.

Uses the same ``geak`` CLI entry point as the HIP agent (geak_v3),
with ``--test-command`` pointing to the Triton harness.  GEAK
auto-promotes the test command to harness mode when it detects
argparse ``--correctness``/``--benchmark`` modes.

Output goes to a sibling ``_logs/`` directory.  After the run,
the best patch or kernel is promoted back into the AKA workspace.
"""
import json
import logging
import os
import shlex
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

import yaml

from agents import register_agent


def _read_stream(stream, lines: list, prefix: str, log_func):
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


def _try_patch_with_strip(
    patch_file: str, workspace: str, logger: logging.Logger
) -> bool:
    """Try applying a patch with increasing -p strip levels (p1 through p8)."""
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
    """Find and apply the best optimized kernel from GEAK output back to workspace.

    Returns (applied, best_verified_speedup).

    Strategy (in order):
    1. Read round_N_evaluation.json for highest verified speedup,
       then locate the kernel in the evaluation worktree.
    2. Apply patch from the best verified round's evaluation.json.
    3. Apply patch from final_report.json.
    4. Per-round best_results.json patches.
    5. best_patch_r*.diff files.
    """
    kernel_name = "kernel.py"
    ws_kernel = Path(workspace) / kernel_name

    best_speedup = 0.0
    best_round = None
    best_task = None
    for eval_file in sorted(logs_dir.glob("round_*_evaluation.json"), reverse=True):
        try:
            data = json.loads(eval_file.read_text())
            if data.get("status") == "patch_failed":
                continue
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
                logger.info(
                    f"Best round {best_round}, task {best_task}, "
                    f"patch {patch_id} (speedup: {best_speedup:.2f}x)"
                )
            except Exception:
                pass

        round_dir = logs_dir / "results" / f"round_{best_round}"
        candidates = []
        original_text = ws_kernel.read_text() if ws_kernel.exists() else ""
        for slot_dir in sorted(round_dir.glob("worktrees/slot_*")):
            if not slot_dir.is_dir() or "_logs" in slot_dir.name:
                continue
            for candidate in slot_dir.rglob(kernel_name):
                if candidate.read_text() != original_text:
                    candidates.append(candidate)
                    break

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
                    f"{slot_name or candidate.parent.name} passes correctness "
                    f"(speedup: {best_speedup:.2f}x)"
                )
                return True, best_speedup
            logger.warning(f"{slot_name or candidate.parent.name} failed correctness, trying next")

        if candidates:
            logger.warning("No worktree kernel passed correctness; restoring original")
            ws_kernel.write_text(original_text)

    if best_round:
        best_eval = logs_dir / f"round_{best_round}_evaluation.json"
        if best_eval.exists():
            try:
                eval_data = json.loads(best_eval.read_text())
                patch_file = eval_data.get("best_patch")
                if patch_file and Path(patch_file).exists():
                    logger.info(f"Applying patch from best round {best_round}: {patch_file}")
                    if _try_patch_with_strip(patch_file, workspace, logger):
                        logger.info(f"Patch from round {best_round} applied ({best_speedup:.2f}x)")
                        return True, best_speedup
            except Exception as e:
                logger.warning(f"Error applying patch from round {best_round}: {e}")

    final_report = logs_dir / "final_report.json"
    if final_report.exists():
        try:
            data = json.loads(final_report.read_text())
            patch_file = data.get("best_patch")
            if patch_file and Path(patch_file).exists():
                logger.info(f"Applying patch from final_report.json: {patch_file}")
                if _try_patch_with_strip(patch_file, workspace, logger):
                    logger.info(f"Patch applied ({data.get('best_speedup_verified', 'N/A')}x)")
                    return True, best_speedup
                logger.warning("final_report patch failed at all strip levels")
        except Exception as e:
            logger.warning(f"Error reading final_report.json: {e}")

    for rdir in sorted(logs_dir.glob("results/round_*"), reverse=True):
        for td in sorted(rdir.iterdir()):
            if not td.is_dir() or td.name == "worktrees":
                continue
            best = td / "best_results.json"
            if not best.exists():
                continue
            try:
                data = json.loads(best.read_text())
                patch_file = data.get("best_patch_file")
                if not patch_file or not Path(patch_file).exists():
                    continue
                logger.info(f"Applying fallback patch: {patch_file}")
                if _try_patch_with_strip(patch_file, workspace, logger):
                    logger.info("Fallback patch applied")
                    return True, best_speedup
            except Exception as e:
                logger.warning(f"Error applying patch from {best}: {e}")

    for diff in sorted(logs_dir.glob("best_patch_r*.diff"), reverse=True):
        try:
            if _try_patch_with_strip(str(diff), workspace, logger):
                logger.info(f"Applied diff patch: {diff}")
                return True, best_speedup
        except Exception as e:
            logger.warning(f"Diff patch {diff} failed: {e}")

    # Last resort: scan ALL worktree kernels across all rounds for any
    # modification, even when round evaluation and patch apply all failed.
    original_text = ws_kernel.read_text() if ws_kernel.exists() else ""
    for wk in sorted(logs_dir.rglob("worktrees/slot_*/kernel.py"), reverse=True):
        try:
            if wk.read_text() != original_text:
                logger.info(f"Trying worktree kernel (last resort): {wk}")
                shutil.copy2(str(wk), str(ws_kernel))
                check = subprocess.run(
                    ["python3", "test_kernel_harness.py", "--correctness"],
                    cwd=workspace, capture_output=True, text=True, timeout=120,
                )
                if check.returncode == 0 and "FAIL" not in check.stdout:
                    logger.info(f"Worktree kernel passes correctness (last resort)")
                    return True, best_speedup
                logger.warning(f"Worktree kernel failed correctness, trying next")
        except Exception as e:
            logger.warning(f"Error trying worktree kernel {wk}: {e}")

    if original_text and ws_kernel.exists() and ws_kernel.read_text() != original_text:
        ws_kernel.write_text(original_text)

    logger.warning("No applicable patch found")
    return False, best_speedup


def _build_task_prompt(task_config: dict, workspace_path: Path) -> str:
    """Build a task prompt from the Triton task config."""
    source_files = task_config.get("source_file_path", ["kernel.py"])
    target_kernels = task_config.get("target_kernel_functions", [])
    instructions = (task_config.get("prompt") or {}).get("instructions", "")

    sections = []
    sections.append("## Task Info\n")
    sections.append("**Source files:**")
    for f in source_files:
        sections.append(f"  - {f}")
    if target_kernels:
        sections.append("\n**Target kernel functions:**")
        for k in target_kernels:
            sections.append(f"  - {k}")

    if instructions:
        sections.append(f"\n## Instructions\n\n{instructions}")
    else:
        sections.append("\nOptimize the kernel in the workspace directory.")

    sections.append("\nUse heterogeneous mode for diverse optimization strategies.")
    sections.append(f"\n### Workspace Directory\nYour working directory is: `{workspace_path}`\n")
    return "\n".join(sections)


@register_agent("geak_v3_triton")
def launch_agent(eval_config: dict[str, Any], task_config_dir: str, workspace: str) -> str:
    """
    Launch GEAK-v3 Triton agent via the unified ``geak`` CLI.

    Uses ``--test-command`` with the harness path.  GEAK auto-promotes
    the test command to harness mode when it detects the harness has
    ``--correctness``/``--benchmark`` argparse modes.
    """
    logger = logging.getLogger(__name__)

    AGENT = "geak"
    if not shutil.which(AGENT):
        raise RuntimeError(
            f"Command '{AGENT}' not found. Install GEAK (pip install -e /path/to/GEAK) "
            f"and ensure it is on your PATH."
        )

    config_path = Path(__file__).with_name("agent_config.yaml")
    with config_path.open() as f:
        agent_config = yaml.safe_load(f) or {}

    with open(task_config_dir) as f:
        task_config = yaml.safe_load(f) or {}

    workspace_path = Path(workspace).resolve()
    source_files = task_config.get("source_file_path", ["kernel.py"])
    if isinstance(source_files, list):
        kernel_file = source_files[0]
    else:
        kernel_file = source_files
    kernel_path = workspace_path / kernel_file

    if not kernel_path.is_file():
        raise FileNotFoundError(f"Kernel not found: {kernel_path}")

    # Build test command: prefer harness_path, fall back to command chain
    harness_file = task_config.get("harness_path")
    if harness_file and (workspace_path / harness_file).is_file():
        test_cmd = f"python3 {workspace_path / harness_file}"
    else:
        cmds = []
        seen = set()
        for cmd_list in [task_config.get("compile_command", []),
                         task_config.get("correctness_command", []),
                         task_config.get("performance_command", [])]:
            if isinstance(cmd_list, str):
                cmd_list = [cmd_list]
            for c in (cmd_list or []):
                c = c.strip()
                if c and c not in seen:
                    seen.add(c)
                    cmds.append(c)
        test_cmd = " && ".join(cmds) if cmds else None

    logs_dir = workspace_path.parent / f"{workspace_path.name}_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_env = os.environ.copy()
    for k, v in (agent_config.get("geak_env") or {}).items():
        run_env[k] = str(v)

    gpu_ids = os.environ.get("GEAK_GPU_IDS", eval_config.get("gpu_ids", "0,1,2,3"))
    num_parallel = len(gpu_ids.split(","))
    timeout = int(agent_config.get("timeout_seconds", 36000))

    prompt = _build_task_prompt(task_config, workspace_path)
    prompt_file = workspace_path / "task_prompt.md"
    prompt_file.write_text(prompt, encoding="utf-8")

    logger.info("=" * 60)
    logger.info("  GEAK-v3 Triton Agent (unified geak CLI)")
    logger.info("=" * 60)
    logger.info(f"  kernel:       {kernel_path}")
    logger.info(f"  test_cmd:     {test_cmd}")
    logger.info(f"  workspace:    {workspace_path}")
    logger.info(f"  logs_dir:     {logs_dir}")
    logger.info(f"  gpu_ids:      {gpu_ids}")
    logger.info(f"  num_parallel: {num_parallel}")
    logger.info(f"  timeout:      {timeout}s")
    for k, v in sorted(run_env.items()):
        if k.startswith("GEAK_"):
            logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    if not (workspace_path / ".git").exists():
        gi = workspace_path / ".gitignore"
        if not gi.exists():
            gi.write_text(
                "baseline_metrics.json\nprofile.json\n.optimization_strategies.md\n"
                "baseline_perf.yaml\noptimized_perf.yaml\nconfig.yaml\n__pycache__/\n"
                "*.pyc\naiter/\n.rocprofv3/\ntraj.json\ndo_task.sh\n"
            )
        subprocess.run(["git", "init"], cwd=str(workspace_path), capture_output=True)
        subprocess.run(["git", "add", "."], cwd=str(workspace_path), capture_output=True)
        subprocess.run(["git", "commit", "-m", "baseline"], cwd=str(workspace_path), capture_output=True)

    cmd = (
        f"{AGENT}"
        f" --kernel-url {shlex.quote(str(kernel_path))}"
        + (f" --test-command {shlex.quote(test_cmd)}" if test_cmd else "")
        + f" --gpu-ids {gpu_ids}"
        f" --num-parallel {num_parallel}"
        f" --yolo"
        f" --exit-immediately"
        f" -t {shlex.quote(str(prompt_file))}"
        f" -o {shlex.quote(str(logs_dir))}"
    )

    logger.info(f"Running: {cmd}")

    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(workspace_path),
        env=run_env,
        bufsize=1,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    t_out = threading.Thread(
        target=_read_stream, args=(proc.stdout, stdout_lines, "[GEAK]", logger.info), daemon=True
    )
    t_err = threading.Thread(
        target=_read_stream, args=(proc.stderr, stderr_lines, "[GEAK ERR]", logger.warning), daemon=True
    )
    t_out.start()
    t_err.start()

    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(f"GEAK timed out after {timeout}s; killing")
        proc.kill()

    t_out.join(timeout=5)
    t_err.join(timeout=5)

    logger.info(f"GEAK exited with code: {proc.returncode}")

    if logs_dir.exists():
        applied, best_verified = _apply_best_patch(workspace, logs_dir, logger)
        logger.info(f"Best verified speedup: {best_verified:.4f}x (applied={applied})")
        summary = {"best_verified_speedup": best_verified, "patch_applied": applied}
        (logs_dir / "geak_summary.json").write_text(json.dumps(summary, indent=2))
    else:
        logger.warning(f"No results found in {logs_dir}")

    output = "\n".join(stdout_lines)
    if stderr_lines:
        output += "\n=== STDERR ===\n" + "\n".join(stderr_lines[-50:])

    return output
