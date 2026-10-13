"""Inspect SIKL data or run the task generation campaign through Docker."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .bundle import inspect_bundle
from .config import load_config
from .orchestrator import REPO, resolve_paths, run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "run", "resume", "mount-info"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("agent_config.yaml"))
    parser.add_argument("--input-dir")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.action == "inspect":
        tasks = inspect_bundle(Path(config.input_dir), config.selections)
        result = {"ok": True, "tasks": [t.summary() for t in tasks],
                  "task_count": len(tasks), "case_count": sum(len(t.rows) for t in tasks)}
    elif args.action == "mount-info":
        source, output, artifacts = resolve_paths(config)
        if not source.is_dir():
            raise ValueError(f"Missing input directory: {source}")
        values = [str(source), str(args.config.resolve()), str(artifacts.relative_to(REPO)),
                  str(output.relative_to(REPO)), " ".join(sorted({"codex", config.validator.get("backend", "codex")}))]
        if any("\n" in v or ":" in v for v in values):
            raise ValueError("Mount paths cannot contain newline or colon characters")
        artifacts.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        print("\n".join(values))
        return 0
    else:
        if args.action == "resume" and not args.run_id:
            parser.error("resume requires --run-id")
        if args.action == "run" and args.run_id:
            parser.error("use resume with --run-id")
        logging.basicConfig(level=logging.INFO)
        result = run(config, args.run_id)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
