"""Image-owned schema-v3 entry point for Waitcheck and rocJITsu race detection.

    python -I aka_v3_entrypoint.py {waitcheck|rocjitsu} --workspace W --role R --output-dir O ...

1. Runs the workspace runner's ``ROLE sanitize capture`` lane on the GPU,
   which records the role's dispatched kernels (code objects, and for
   Triton/Gluon a replay record per launch) under ``O/capture``.
2. Waitcheck: lists every kernel of each captured code object, then runs the
   image-owned single-kernel bridge on each one recorded as dispatched (all of
   them for a HIP extension or FlyDSL module, whose code object holds only the
   role's own kernels).
   rocJITsu: builds one replay capsule per recorded launch and replays it
   under the race detector through the image-owned trusted replay helper.
3. Writes ``O/index.json``: one entry per analysed kernel or capsule with the
   exact log the plugin parses, plus what the capture skipped.

Like the other image-owned entry points it imports framework code only from
the tree that contains this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

_ENTRYPOINT = Path(__file__).resolve(strict=True)
_FRAMEWORK_ROOT = _ENTRYPOINT.parents[3]
_ADAPTERS = _FRAMEWORK_ROOT / "src" / "eval_tools" / "adapters"
INDEX_SCHEMA = 1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture(args, output: Path) -> tuple[dict | None, str | None]:
    capture_dir = output / "capture"
    capture_dir.mkdir(parents=True, exist_ok=True)
    cache = output / "capture-triton-cache"
    cache.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1",
           "AKA_SANITIZE_OUTPUT_DIR": str(capture_dir), "TRITON_CACHE_DIR": str(cache)}
    completed = subprocess.run(
        [args.python, "-m", "_aka_eval.evaluate", args.role, "sanitize", "capture"],
        cwd=args.workspace, env=env, capture_output=True, text=True, timeout=args.capture_timeout, check=False)
    (output / "capture.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (output / "capture.stderr.log").write_text(completed.stderr, encoding="utf-8")
    report = None
    for line in completed.stdout.splitlines():
        if line.startswith("ARENA_EVAL_RESULT="):
            report = json.loads(line[len("ARENA_EVAL_RESULT="):])
    if completed.returncode != 0 or not report or report.get("status") != "PASS":
        reason = (report or {}).get("reason") or completed.stderr[-2000:] or f"rc={completed.returncode}"
        return None, f"capture failed: {reason}"
    from src.eval_tools.adapters.aka_v3_capsules import load_manifest

    return load_manifest(capture_dir), None


def _inventory(waitcheck: Path, code_object: Path, target: str) -> list[dict]:
    completed = subprocess.run([str(waitcheck), str(code_object), "--target", target, "--list-kernels"],
                               capture_output=True, text=True, check=False, timeout=300)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "inventory failed")
    records = []
    for line in completed.stdout.splitlines():
        value = json.loads(line)
        if isinstance(value, dict) and "kernel_name" in value:
            records.append(value)
    return records


def _waitcheck(args, output: Path, manifest: dict) -> list[dict]:
    capture_dir = output / "capture"
    entries = []
    for number, item in enumerate(manifest.get("code_objects", [])):
        code_object = (capture_dir / item["code_object"]).resolve(strict=True)
        if not code_object.is_relative_to(capture_dir.resolve()) or _sha256(code_object) != item["sha256"]:
            entries.append({"code_object": item["code_object"], "error": "captured code object changed"})
            continue
        try:
            inventory = _inventory(args.waitcheck, code_object, args.target)
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
            entries.append({"code_object": item["code_object"], "error": f"inventory: {error}"})
            continue
        wanted = item.get("kernels")
        selected = inventory if wanted is None else [r for r in inventory if r.get("kernel_name") in wanted]
        if wanted is not None and {r.get("kernel_name") for r in selected} != set(wanted):
            entries.append({"code_object": item["code_object"], "error": f"dispatched kernels {wanted} are not "
                            "all in the code object's inventory"})
        for record in selected:
            log = output / "kernels" / f"{number:02d}-{len(entries):03d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(
                [args.python, "-I", str(_ADAPTERS / "waitcheck_entrypoint.py"), "--waitcheck", str(args.waitcheck),
                 "--capi-wrapper", str(args.capi_wrapper), "--code-object", str(code_object),
                 "--expected-sha256", item["sha256"], "--target", args.target,
                 "--expected-kernel", str(record["kernel_name"]), "--kernel-entry", str(record["kernel_entry"])],
                capture_output=True, text=True, check=False, timeout=args.kernel_timeout)
            log.write_text(completed.stdout + ("\n" + completed.stderr if completed.stderr else ""), encoding="utf-8")
            entries.append({"code_object": item["code_object"], "code_object_sha256": item["sha256"],
                            "kernel_name": record["kernel_name"], "kernel_entry": record["kernel_entry"],
                            "kernels_in_code_object": len(inventory),
                            "returncode": completed.returncode, "log": log.relative_to(output).as_posix()})
    return entries


def _task_id(args) -> str:
    if args.task_id:
        return args.task_id
    try:
        return str(json.loads((args.workspace / "_aka_eval" / "contract.json").read_text())["task_id"])
    except (OSError, ValueError, KeyError, TypeError):
        return args.workspace.name or "unknown"


def _rocjitsu(args, output: Path, manifest: dict) -> list[dict]:
    from src.eval_tools.adapters.aka_v3_capsules import build_capsule

    task_id = _task_id(args)
    capture_dir = output / "capture"
    entries = []
    for launch in manifest.get("launches", []):
        launch_dir = (capture_dir / launch["directory"]).resolve(strict=True)
        entry = {"directory": launch["directory"], "kernel_name": launch["kernel_name"]}
        try:
            capsule = build_capsule(launch_dir, manifest=manifest, task_id=task_id,
                                    image_digest=os.environ.get("AKA_EVAL_TOOL_RUNTIME_REF") or "unrecorded",
                                    rocm_version=args.rocm_version, framework_version=args.framework_version)
        except Exception as error:  # an unbuildable capsule is that launch's unverified result
            entries.append({**entry, "error": f"capsule: {type(error).__name__}: {error}"[:2000]})
            continue
        replay_dir = output / "replay" / launch["directory"]
        report_dir = replay_dir / "rocjitsu-report"
        report_dir.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "RJ_RACE": "1", "RJ_LOG": "1", "RJ_SINKS": "stderr,file", "RJ_SINK_DIR": str(report_dir)}
        completed = subprocess.run(
            [args.python, "-I", str(_ADAPTERS / "rocjitsu_replay_entrypoint.py"), "--capsule", str(capsule),
             "--output-dir", str(replay_dir / "launcher"), "--rocjitsu", str(args.rocjitsu), "--config",
             str(args.config), "--hipcc", args.hipcc, "--expected-adapter", "triton_aot", "--expected-arch",
             args.target, "--expected-kernel", launch["kernel_name"], "--expected-capsule-sha256",
             _sha256(capsule)],
            capture_output=True, text=True, env=env, check=False, timeout=args.kernel_timeout)
        log = replay_dir / "replay.log"
        log.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8")
        race_log = report_dir / "race.log"
        entries.append({**entry, "capsule": capsule.relative_to(output).as_posix(), "capsule_sha256": _sha256(capsule),
                        "returncode": completed.returncode, "log": log.relative_to(output).as_posix(),
                        "race_log": race_log.relative_to(output).as_posix() if race_log.is_file() else None})
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    sys.path.insert(0, str(_FRAMEWORK_ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=("waitcheck", "rocjitsu"))
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--role", required=True, choices=("candidate", "baseline"))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--python", default="/opt/venv/bin/python")
    parser.add_argument("--target", default="gfx950")
    parser.add_argument("--task-id", default="")
    parser.add_argument("--capture-timeout", type=float, default=900)
    parser.add_argument("--kernel-timeout", type=float, default=900)
    parser.add_argument("--waitcheck", type=Path)
    parser.add_argument("--capi-wrapper", type=Path)
    parser.add_argument("--rocjitsu", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--hipcc", default="/opt/rocm/bin/hipcc")
    parser.add_argument("--rocm-version", default="10.0.0")
    parser.add_argument("--framework-version", default="unknown")
    args = parser.parse_args(argv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    index = {"schema_version": INDEX_SCHEMA, "tool": args.tool, "role": args.role, "entries": [], "skipped": []}
    try:
        manifest, error = _capture(args, output)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        manifest, error = None, f"capture failed: {type(exc).__name__}: {exc}"
    if manifest is None:
        index["error"] = error
    else:
        index["language"] = manifest.get("language")
        index["skipped"] = list(manifest.get("skipped") or [])
        index["entries"] = _waitcheck(args, output, manifest) if args.tool == "waitcheck" else \
            _rocjitsu(args, output, manifest)
    (output / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    print("AKA_V3_INDEX " + json.dumps({"entries": len(index["entries"]), "error": index.get("error")}), flush=True)
    return 0 if manifest is not None else 3


if __name__ == "__main__":  # pragma: no cover - exercised by sidecar tests.
    raise SystemExit(main())
