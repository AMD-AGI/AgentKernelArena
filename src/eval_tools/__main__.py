"""Command-line entry points for evaluation-tool sidecar diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runtime_client import UnixSocketRuntimeClient
from .worker import main as worker_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.eval_tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker", help="run a Unix-socket tool worker")
    worker.add_argument("worker_args", nargs=argparse.REMAINDER)

    health = subparsers.add_parser("health", help="query a worker")
    health.add_argument("--socket", required=True, type=Path)
    health.add_argument("--timeout-s", type=float, default=30.0)

    execute = subparsers.add_parser("execute", help="send an execute request")
    execute.add_argument("--socket", required=True, type=Path)
    request_source = execute.add_mutually_exclusive_group(required=True)
    request_source.add_argument("--request", help="inline JSON request params")
    request_source.add_argument("--request-file", type=Path)
    execute.add_argument("--timeout-s", type=float, default=300.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "worker":
        return worker_main(args.worker_args)

    client = UnixSocketRuntimeClient(args.socket, timeout_seconds=args.timeout_s)
    if args.command == "health":
        result = client.health()
    else:
        raw = args.request if args.request is not None else args.request_file.read_text(encoding="utf-8")
        request = json.loads(raw)
        if not isinstance(request, dict):
            raise SystemExit("execute request must be a JSON object")
        result = client.execute(request, timeout_seconds=args.timeout_s)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.command == "health" and result.get("status") != "ready":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
