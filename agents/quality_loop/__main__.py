# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

import yaml

from .config import load_config
from .orchestrator import QualityLoop


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("agent_config.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit, repair, harden, and publish AgentKernelArena tasks"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Override task selectors from the config (paths relative to tasks/)",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="List runnable/deferred tasks without GitHub, GPU, or agent execution",
    )
    parser.add_argument("--resume", metavar="RUN_ID", help="Resume a prior run")
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Run all hard preflights and audits but do not create issues, push, or open a PR",
    )
    parser.add_argument("--defer-github", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-preflight", action="store_true", help=argparse.SUPPRESS)
    return parser


def configure_logging() -> logging.Logger:
    logger = logging.getLogger("quality_loop")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(config_path)
    if args.tasks:
        config = dataclasses.replace(config, tasks=tuple(args.tasks))
    if args.no_publish:
        config = dataclasses.replace(
            config,
            github=dataclasses.replace(config.github, publish=False),
        )
    logger = configure_logging()
    workflow = QualityLoop(REPO_ROOT, config, logger=logger, defer_github=args.defer_github)
    if args.plan:
        print(yaml.safe_dump(workflow.plan(), sort_keys=False, allow_unicode=True))
        return 0
    report = workflow.run(
        resume_run_id=args.resume,
        skip_preflight=args.skip_preflight,
    )
    logger.info("quality_loop report: %s", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
