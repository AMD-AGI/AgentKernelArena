# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Held-out evaluation script.

Takes a completed run directory, copies each task workspace into two
sub-workspaces (orig/ and opt/), injects held-out shapes into both,
restores the original kernel in orig/, and measures baseline vs optimized
performance on the unseen shapes.

Usage:
    python held_out/run_heldout_eval.py \
        --run-dir workspace_MI300_cursor/run_20260417_142419 \
        --heldout-dir held_out_tests/ \
        --tasks-dir tasks/ \
        [--output-suffix _heldout]
"""
import argparse
import logging
import shutil
import sys
import yaml
import statistics
from pathlib import Path
from typing import Dict, Any, List, Optional

# Allow running from repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from held_out.injection import apply_all_injections
from src.evaluator import (
    evaluate_compilation,
    evaluate_correctness,
    measure_performance,
)
from src.performance import measure_baseline
from src.testcases import (
    TestCaseResult,
    save_performance_results,
    calculate_average_speedup,
)
from src.score import score as calc_score


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    log = logging.getLogger("heldout_eval")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    log.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        log.addHandler(fh)

    return log


def _valid_perf_cases(cases: List[TestCaseResult]) -> List[TestCaseResult]:
    return [c for c in cases if c.execution_time_ms is not None and c.execution_time_ms > 0]


def resolve_task_id(workspace_dir: Path) -> Optional[str]:
    """
    Derive the canonical task id (e.g. 'hip2hip/gpumode/SiLU') from the
    task_result.yaml written by the framework, falling back to config.yaml
    or directory-name heuristics.
    """
    task_result = workspace_dir / "task_result.yaml"
    if task_result.exists():
        try:
            data = yaml.safe_load(task_result.read_text()) or {}
            name = data.get("task_name", "")
            if "/" in name:
                return name
        except Exception:
            pass

    config = workspace_dir / "config.yaml"
    if config.exists():
        try:
            data = yaml.safe_load(config.read_text()) or {}
            task_type = data.get("task_type", "")
            # config.yaml is a copy of the original, so its parent path
            # doesn't help; use task_result.yaml's task_name or dir name
        except Exception:
            pass

    # Heuristic: dir name is like hip2hip_gpumode_SiLU_20260417_142639
    # or triton2triton_rocmbench_easy_test_add_kernel_20260417_142639
    # Strip trailing _YYYYMMDD_HHMMSS
    dir_name = workspace_dir.name
    parts = dir_name.rsplit("_", 2)
    if len(parts) >= 3:
        task_slug = "_".join(parts[:-2])
        # triton2triton/rocmbench has an extra level (easy/medium/hard)
        # so 3 slashes instead of 2
        if task_slug.startswith("triton2triton_rocmbench_"):
            return task_slug.replace("_", "/", 3)
        return task_slug.replace("_", "/", 2)

    return None


def _restore_original_kernel(
    orig_workspace: Path,
    task_dir: Path,
    task_config: Dict[str, Any],
    logger: logging.Logger,
) -> bool:
    """
    Overwrite the agent-optimized kernel in *orig_workspace* with the
    unoptimized original from the canonical tasks directory.

    For hip2hip: copies ``target_file_path``.
    For torch2hip: removes the agent-generated HIP file (the original is
    pure PyTorch -- there is no HIP kernel to restore).
    For triton2triton: copies every file in ``source_file_path``.

    Returns True on success, False on failure.
    """
    task_type = task_config.get("task_type", "")

    if task_type == "torch2hip":
        target = task_config.get("target_file_path")
        if target:
            hip_file = orig_workspace / target
            if hip_file.exists():
                hip_file.unlink()
                logger.info("Removed agent HIP kernel from orig/: %s", hip_file)
        logger.info("torch2hip: orig/ will use PyTorch baseline (no HIP kernel)")
        return True

    if task_type == "hip2hip":
        target = task_config.get("target_file_path")
        if not target:
            logger.error("config.yaml has no target_file_path for %s task", task_type)
            return False
        files_to_copy = [target]
    else:
        files_to_copy = task_config.get("source_file_path", [])
        if not files_to_copy:
            logger.error("config.yaml has no source_file_path for %s task", task_type)
            return False

    for rel_path in files_to_copy:
        src = task_dir / rel_path
        dst = orig_workspace / rel_path
        if not src.exists():
            logger.error("Original kernel not found: %s", src)
            return False
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info("Restored original kernel: %s -> %s", src, dst)

    return True


def _clear_build_artifacts(workspace: Path) -> None:
    """Remove stale build artifacts so performance reports are fresh."""
    build_dir = workspace / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)


def _classify_generalization(orig_correct: bool, opt_correct: bool) -> str:
    """
    Classify the held-out outcome into one of four quadrants:

    - both_pass:      orig ✓  opt ✓  — normal, compare speedups
    - opt_regression:  orig ✓  opt ✗  — optimization broke generalization
    - both_fail:       orig ✗  opt ✗  — shape exceeds kernel spec
    - opt_improvement: orig ✗  opt ✓  — agent improved robustness
    """
    if orig_correct and opt_correct:
        return "both_pass"
    if orig_correct and not opt_correct:
        return "opt_regression"
    if not orig_correct and not opt_correct:
        return "both_fail"
    return "opt_improvement"


def evaluate_single_task(
    original_workspace: Path,
    output_workspace: Path,
    heldout_config: dict,
    task_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Create orig/ and opt/ workspace copies, inject held-out shapes into
    both, restore the original kernel in orig/, and evaluate both.

    - ``orig/``: original unoptimized kernel — compile, correctness, performance
    - ``opt/``: agent's optimized kernel — compile, correctness, performance

    The ``generalization_status`` field classifies the outcome:

    - ``both_pass``:      orig correct, opt correct — compare speedups
    - ``opt_regression``:  orig correct, opt failed — genuine generalization failure
    - ``both_fail``:       both incorrect — shape exceeds kernel design spec
    - ``opt_improvement``: orig failed, opt correct — agent improved robustness

    Returns a result dict for heldout_task_result.yaml.
    """
    task_id = resolve_task_id(original_workspace) or original_workspace.name

    run_result: Dict[str, Any] = {}
    run_result_file = original_workspace / "task_result.yaml"
    if run_result_file.exists():
        run_result = yaml.safe_load(run_result_file.read_text()) or {}

    result: Dict[str, Any] = {
        "task_name": task_id,
        "heldout": True,
        # opt kernel results
        "opt_pass_compilation": False,
        "opt_pass_correctness": False,
        "opt_execution_time": 0.0,
        # orig kernel results on held-out shapes
        "orig_heldout_pass_compilation": False,
        "orig_heldout_pass_correctness": False,
        "orig_heldout_execution_time": 0.0,
        # comparison
        "generalization_status": "both_fail",
        "speedup_ratio": 0.0,
        "score": 0.0,
        # original run results (on original shapes, for reference)
        "original_run_pass_correctness": run_result.get("pass_correctness", False),
        "original_run_speedup_ratio": run_result.get("speedup_ratio", 0.0),
        "original_run_score": run_result.get("score", 0.0),
        "error": None,
    }

    # ── 1. Prepare opt/ and orig/ workspaces ────────────────────────────
    if output_workspace.exists():
        shutil.rmtree(output_workspace)
    output_workspace.mkdir(parents=True)

    opt_ws = output_workspace / "opt"
    orig_ws = output_workspace / "orig"

    shutil.copytree(original_workspace, opt_ws)
    shutil.copytree(original_workspace, orig_ws)
    logger.info(f"Created opt/ and orig/ workspaces under {output_workspace}")

    _clear_build_artifacts(opt_ws)
    _clear_build_artifacts(orig_ws)

    # ── 2. Restore original (unoptimized) kernel in orig/ ───────────────
    config_file = opt_ws / "config.yaml"
    if not config_file.exists():
        result["error"] = "config.yaml not found in workspace"
        _write_result(output_workspace, result)
        return result
    task_config = yaml.safe_load(config_file.read_text()) or {}

    if not _restore_original_kernel(orig_ws, task_dir, task_config, logger):
        result["error"] = "Failed to restore original kernel in orig/"
        _write_result(output_workspace, result)
        return result

    # ── 3. Inject held-out shapes into both workspaces ──────────────────
    for label, ws in [("opt", opt_ws), ("orig", orig_ws)]:
        if not apply_all_injections(ws, heldout_config, logger):
            result["error"] = f"Injection failed in {label}/"
            logger.error(f"[{task_id}] Injection failed in {label}/, skipping")
            _write_result(output_workspace, result)
            return result

    # ── 4. Evaluate ORIGINAL kernel (orig/) ─────────────────────────────
    is_torch2hip = task_config.get("task_type") == "torch2hip"

    if is_torch2hip:
        # torch2hip: the "original" is pure PyTorch -- no HIP kernel to
        # compile or correctness-check.  We treat the PyTorch module as
        # always correct and measure its baseline latency in step 7.
        logger.info(f"[{task_id}] torch2hip: skipping compilation/correctness for orig/ "
                     "(PyTorch baseline)")
        orig_comp = True
        orig_correct = True
        result["orig_heldout_pass_compilation"] = True
        result["orig_heldout_pass_correctness"] = True
    else:
        logger.info(f"[{task_id}] Compiling original kernel (orig/)...")
        orig_comp, orig_comp_err = evaluate_compilation(orig_ws, task_config, logger)
        result["orig_heldout_pass_compilation"] = orig_comp

        orig_correct = False
        if orig_comp:
            logger.info(f"[{task_id}] Running correctness on original kernel (orig/)...")
            orig_correct, _ = evaluate_correctness(orig_ws, task_config, logger)
            result["orig_heldout_pass_correctness"] = orig_correct
        else:
            logger.warning(f"[{task_id}] Original kernel failed compilation on held-out shapes")

    # ── 5. Evaluate OPTIMIZED kernel (opt/) ─────────────────────────────
    logger.info(f"[{task_id}] Compiling optimized kernel (opt/)...")
    opt_comp, opt_comp_err = evaluate_compilation(opt_ws, task_config, logger)
    result["opt_pass_compilation"] = opt_comp

    opt_correct = False
    if opt_comp:
        logger.info(f"[{task_id}] Running correctness on optimized kernel (opt/)...")
        opt_correct, opt_corr_err = evaluate_correctness(opt_ws, task_config, logger)
        result["opt_pass_correctness"] = opt_correct
        if not opt_correct:
            result["error"] = opt_corr_err
    else:
        result["error"] = opt_comp_err

    # ── 6. Classify generalization outcome ──────────────────────────────
    status = _classify_generalization(orig_correct, opt_correct)
    result["generalization_status"] = status
    logger.info(f"[{task_id}] Generalization status: {status}")

    # ── 7. Performance (only when the respective kernel is correct) ─────
    valid_baseline: List[TestCaseResult] = []
    valid_optimized: List[TestCaseResult] = []

    if orig_correct:
        if is_torch2hip:
            # For torch2hip the baseline is the PyTorch module, measured
            # via the performance script with --baseline_only (no HIP
            # compilation needed).  Use opt/ workspace which still has
            # the PyTorch source files and eval_tools intact.
            logger.info(f"[{task_id}] Measuring PyTorch baseline performance (opt/ --baseline_only)...")
            baseline_cases = measure_baseline(opt_ws, task_config, logger)
        else:
            logger.info(f"[{task_id}] Measuring baseline performance (orig/)...")
            baseline_cases = measure_baseline(orig_ws, task_config, logger)
        valid_baseline = _valid_perf_cases(baseline_cases)
        if valid_baseline:
            result["orig_heldout_execution_time"] = (
                sum(c.execution_time_ms for c in valid_baseline) / len(valid_baseline)
            )

    if opt_correct:
        logger.info(f"[{task_id}] Measuring optimized performance (opt/)...")
        optimized_cases = measure_performance(opt_ws, task_config, logger)
        valid_optimized = _valid_perf_cases(optimized_cases)
        if valid_optimized:
            result["opt_execution_time"] = (
                sum(c.execution_time_ms for c in valid_optimized) / len(valid_optimized)
            )
            save_performance_results(valid_optimized, opt_ws, "optimized_perf.yaml", logger)

    if status == "both_pass" and valid_baseline and valid_optimized:
        result["speedup_ratio"] = calculate_average_speedup(
            valid_baseline, valid_optimized, logger,
        )

    # ── 8. Score ────────────────────────────────────────────────────────
    result["score"] = calc_score(
        opt_comp,
        opt_correct,
        result["orig_heldout_execution_time"],
        result["opt_execution_time"],
        result["speedup_ratio"],
    )

    logger.info(
        f"[{task_id}] Held-out result: status={status}, "
        f"orig_correct={orig_correct}, opt_correct={opt_correct}, "
        f"speedup={result['speedup_ratio']:.2f}x "
        f"(original_run={result['original_run_speedup_ratio']:.2f}x)"
    )

    _write_result(output_workspace, result)
    return result


def _write_result(workspace: Path, result: dict) -> None:
    out_file = workspace / "heldout_task_result.yaml"
    with open(out_file, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)


def write_summary(
    output_dir: Path,
    results: List[Dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """Write aggregate heldout_summary.yaml and log a report."""
    total = len(results)
    if total == 0:
        logger.warning("No held-out results to summarize")
        return

    def _stats(vals):
        if not vals:
            return {"mean": 0.0, "median": 0.0, "std": 0.0}
        return {
            "mean": round(sum(vals) / len(vals), 4),
            "median": round(statistics.median(vals), 4),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
        }

    # ── Quadrant counts ────────────────────────────────────────────────
    quadrants = {"both_pass": 0, "opt_regression": 0, "both_fail": 0, "opt_improvement": 0}
    for r in results:
        status = r.get("generalization_status", "both_fail")
        quadrants[status] = quadrants.get(status, 0) + 1

    # ── Conditional correctness: P(opt correct | orig correct) ─────────
    orig_correct_tasks = [r for r in results if r["orig_heldout_pass_correctness"]]
    n_orig_correct = len(orig_correct_tasks)
    n_opt_correct_given_orig = sum(1 for r in orig_correct_tasks if r["opt_pass_correctness"])
    conditional_correctness = (
        round(n_opt_correct_given_orig / n_orig_correct * 100, 1)
        if n_orig_correct > 0 else 0.0
    )

    # ── Speedups (only for both_pass) ──────────────────────────────────
    both_pass = [r for r in results if r["generalization_status"] == "both_pass"]
    heldout_speedups = [r["speedup_ratio"] for r in both_pass if r["speedup_ratio"] > 0]
    orig_run_speedups = [r["original_run_speedup_ratio"] for r in both_pass if r["original_run_speedup_ratio"] > 0]

    # ── Per-task detail ────────────────────────────────────────────────
    per_task = []
    for r in results:
        per_task.append({
            "task_name": r["task_name"],
            "generalization_status": r["generalization_status"],
            "orig_heldout_correct": r["orig_heldout_pass_correctness"],
            "opt_heldout_correct": r["opt_pass_correctness"],
            "heldout_speedup": round(r["speedup_ratio"], 4),
            "original_run_speedup": round(r["original_run_speedup_ratio"], 4),
            "speedup_delta": round(r["speedup_ratio"] - r["original_run_speedup_ratio"], 4),
            "heldout_score": round(r["score"], 2),
            "original_run_score": round(r["original_run_score"], 2),
            "error": r.get("error"),
        })

    summary = {
        "total_tasks": total,
        "quadrant_counts": quadrants,
        "conditional_correctness": {
            "description": "P(opt correct on held-out | orig correct on held-out)",
            "orig_correct_count": n_orig_correct,
            "opt_also_correct_count": n_opt_correct_given_orig,
            "rate_pct": conditional_correctness,
        },
        "heldout_speedup_stats": {
            "description": "Speedup on held-out shapes (both_pass tasks only)",
            "count": len(heldout_speedups),
            **_stats(heldout_speedups),
        },
        "original_run_speedup_stats": {
            "description": "Speedup from original run (both_pass tasks only, for reference)",
            "count": len(orig_run_speedups),
            **_stats(orig_run_speedups),
        },
        "generalization_gap": {
            "correctness_retention_pct": conditional_correctness,
            "speedup_mean_delta": round(
                _stats(heldout_speedups)["mean"] - _stats(orig_run_speedups)["mean"], 4
            ),
        },
        "per_task": per_task,
    }

    summary_file = output_dir / "heldout_summary.yaml"
    with open(summary_file, "w") as f:
        yaml.dump(summary, f, default_flow_style=False, sort_keys=False)
    logger.info(f"Summary written to {summary_file}")

    # ── Log summary ───────────────────────────────────────────────────
    logger.info("=" * 80)
    logger.info("HELD-OUT EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total tasks evaluated: {total}")
    logger.info("")
    logger.info("Generalization quadrants:")
    logger.info(f"  both_pass (orig ✓ opt ✓):       {quadrants['both_pass']}")
    logger.info(f"  opt_regression (orig ✓ opt ✗):   {quadrants['opt_regression']}")
    logger.info(f"  both_fail (orig ✗ opt ✗):        {quadrants['both_fail']}")
    logger.info(f"  opt_improvement (orig ✗ opt ✓):  {quadrants['opt_improvement']}")
    logger.info("")
    logger.info(f"Conditional correctness:  {n_opt_correct_given_orig}/{n_orig_correct} "
                f"= {conditional_correctness:.1f}%  "
                f"(P(opt correct | orig correct))")
    s = _stats(heldout_speedups)
    logger.info(f"Held-out speedup:         mean={s['mean']:.2f}x, median={s['median']:.2f}x "
                f"(n={len(heldout_speedups)} both_pass tasks)")
    s2 = _stats(orig_run_speedups)
    logger.info(f"Original run speedup:     mean={s2['mean']:.2f}x, median={s2['median']:.2f}x "
                f"(same tasks, for reference)")
    gap = summary["generalization_gap"]
    logger.info(f"Speedup retention:        {gap['speedup_mean_delta']:+.4f}x mean delta")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Held-out evaluation for AgentKernelArena")
    parser.add_argument(
        "--run-dir", required=True,
        help="Path to the completed run directory (e.g. workspace_MI300_cursor/run_20260417_142419)",
    )
    parser.add_argument(
        "--heldout-dir", required=True,
        help="Path to held_out_tests/ directory containing per-task held_out_shapes.yaml",
    )
    parser.add_argument(
        "--tasks-dir", required=True,
        help="Path to the canonical tasks/ directory (for restoring original kernels)",
    )
    parser.add_argument(
        "--output-suffix", default="_heldout",
        help="Suffix to append to the run directory name for output (default: _heldout)",
    )
    parser.add_argument(
        "--tasks", nargs="*", default=None,
        help="Optional list of task IDs to evaluate (default: all tasks with held-out configs)",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    heldout_dir = Path(args.heldout_dir).resolve()
    tasks_dir = Path(args.tasks_dir).resolve()
    output_dir = run_dir.parent / (run_dir.name + args.output_suffix)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir / "heldout_eval.log")
    logger.info(f"Run directory:     {run_dir}")
    logger.info(f"Held-out data:     {heldout_dir}")
    logger.info(f"Tasks directory:   {tasks_dir}")
    logger.info(f"Output directory:  {output_dir}")

    # Discover task workspaces
    task_workspaces = sorted(
        p for p in run_dir.iterdir()
        if p.is_dir() and (p / "task_result.yaml").exists()
    )
    logger.info(f"Found {len(task_workspaces)} completed task workspaces")

    # Build list of evaluable tasks so we can show Task N/M progress
    eval_queue: List[tuple] = []
    for ws in task_workspaces:
        task_id = resolve_task_id(ws)
        if task_id is None:
            logger.warning(f"Could not resolve task ID for {ws.name}, skipping")
            continue
        if args.tasks and task_id not in args.tasks:
            continue
        heldout_yaml = heldout_dir / task_id / "held_out_shapes.yaml"
        if not heldout_yaml.exists():
            logger.debug(f"No held-out config for {task_id}, skipping")
            continue
        task_dir = tasks_dir / task_id
        if not task_dir.exists():
            logger.warning(f"Canonical task directory not found: {task_dir}, skipping")
            continue
        heldout_config = yaml.safe_load(heldout_yaml.read_text()) or {}
        eval_queue.append((ws, task_id, heldout_config, task_dir))

    total_tasks = len(eval_queue)
    logger.info(f"Tasks to evaluate: {total_tasks}")

    results: List[Dict[str, Any]] = []

    for idx, (ws, task_id, heldout_config, task_dir) in enumerate(eval_queue, 1):
        logger.info("=" * 80)
        logger.info(f"Task {idx}/{total_tasks}: {task_id}")
        logger.info("=" * 80)

        out_ws = output_dir / ws.name
        result = evaluate_single_task(ws, out_ws, heldout_config, task_dir, logger)
        results.append(result)

        status = result["generalization_status"]
        spd = result["speedup_ratio"]
        logger.info(
            f"Task {idx}/{total_tasks} done: {task_id} — "
            f"{status}, speedup={spd:.2f}x"
        )

    write_summary(output_dir, results, logger)
    logger.info(f"Done. {len(results)} tasks evaluated.")


if __name__ == "__main__":
    main()
