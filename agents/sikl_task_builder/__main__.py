"""Inspect SIKL data or run the task generation campaign through Docker."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .bundle import ImportProblem, inspect_bundle
from .checkout import check_mountable, inspect_checkout
from .config import CheckoutConfig, load_config
from .orchestrator import REPO, resolve_paths, run, run_checkout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "run", "resume", "mount-info"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("agent_config.yaml"))
    parser.add_argument("--input-dir")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if args.action == "resume" and not args.run_id:
        parser.error("resume requires --run-id")
    if args.action == "run" and args.run_id:
        parser.error("use resume with --run-id")
    # The Docker runner captures mount-info's stdout, so its errors go where the user sees them.
    errors = sys.stderr if args.action == "mount-info" else sys.stdout
    try:
        config = load_config(args.config, args.input_dir)
        if isinstance(config, CheckoutConfig):
            return checkout_action(args, config)
    except ImportProblem as exc:
        print(json.dumps({"ok": False, "code": exc.code, "message": str(exc)}), file=errors)
        return 1
    if args.action == "inspect":
        tasks = inspect_bundle(Path(config.input_dir), config.selections)
        result = {"ok": True, "tasks": [t.summary() for t in tasks],
                  "task_count": len(tasks), "case_count": sum(len(t.rows) for t in tasks)}
    elif args.action == "mount-info":
        source, output, artifacts = resolve_paths(config)
        if not source.is_dir():
            raise ValueError(f"Missing input directory: {source}")
        print_mounts(args, config, source, output, artifacts,
                     " ".join(sorted({"codex", config.validator.get("backend", "codex")})))
        return 0
    else:
        logging.basicConfig(level=logging.INFO)
        result = run(config, args.run_id)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def checkout_action(args, config: CheckoutConfig) -> int:
    from src.preprocessing import _resolve_gfx_arch
    source, output, artifacts = resolve_paths(config)
    if args.action == "inspect":
        commit, tasks = inspect_checkout(source, config.tasks)
        arch = _resolve_gfx_arch(config.target_gpu_model)
        result = {"ok": True, "input": "sikl-v5", "commit": commit,
                  "tasks": [t.summary(arch, config.target_gpu_model) for t in tasks],
                  "task_count": len(tasks), "case_count": sum(len(t.cases) for t in tasks)}
    elif args.action == "mount-info":
        inspect_checkout(source, config.tasks)
        check_mountable(source)
        # A placeholder, not an empty line: `$(...)` would drop a trailing empty line.
        print_mounts(args, config, source, output, artifacts, "none")
        return 0
    else:
        logging.basicConfig(level=logging.INFO)
        result = run_checkout(config, args.run_id)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


def print_mounts(args, config, source: Path, output: Path, artifacts: Path, agents: str) -> None:
    values = [str(source), str(args.config.resolve()), str(artifacts.relative_to(REPO)),
              str(output.relative_to(REPO)), agents]
    if any("\n" in v or ":" in v for v in values):
        raise ValueError("Mount paths cannot contain newline or colon characters")
    artifacts.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    print("\n".join(values))


if __name__ == "__main__":
    raise SystemExit(main())
