"""Inspect SIKL data or run the task generation campaign through Docker."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .bundle import ImportProblem, inspect_bundle
from .config import RepoConfig, load_config
from .orchestrator import REPO, resolve_paths, run, run_repo
from .sikl_repo import SiklRepo


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
    try:
        config = load_config(args.config, args.input_dir)
    except ImportProblem as exc:
        print(json.dumps({"ok": False, "code": exc.code, "message": str(exc)}))
        return 1
    if isinstance(config, RepoConfig):
        try:
            return repo_action(args, config)
        except ImportProblem as exc:
            print(json.dumps({"ok": False, "code": exc.code, "message": str(exc)}))
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


def repo_action(args, config: RepoConfig) -> int:
    if args.action == "inspect":
        from src.preprocessing import _resolve_gfx_arch
        arch = _resolve_gfx_arch(config.target_gpu_model)
        sikl = SiklRepo.inspect(REPO / Path(config.input_dir).expanduser(), config.tasks)
        result = {"ok": True, "input": "sikl-v5", "commit": sikl.commit,
                  "tasks": [s.summary(arch, config.target_gpu_model) for s in sikl.schemas],
                  "task_count": len(sikl.schemas), "case_count": sum(len(s.cases) for s in sikl.schemas)}
    elif args.action == "mount-info":
        source, output, artifacts = resolve_paths(config)
        SiklRepo.inspect(source, config.tasks).check_mountable()
        # A placeholder, not an empty line: `$(...)` would drop a trailing empty line.
        print_mounts(args, config, source, output, artifacts, "none")
        return 0
    else:
        logging.basicConfig(level=logging.INFO)
        result = run_repo(config, args.run_id)
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
