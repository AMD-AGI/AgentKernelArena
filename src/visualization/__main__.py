"""Command-line interface for ``python -m src.visualization``."""

from __future__ import annotations

import argparse

from src.visualization.build_data import write_dashboard_data
from src.visualization.server import serve


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m src.visualization",
        description="Build and serve the AgentKernelArena experiment dashboard.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build dashboard data.")
    build_parser.add_argument(
        "--include-workspace-runs",
        action="store_true",
        help="Include workspace_*/run_*/reports directories.",
    )

    serve_parser = subparsers.add_parser("serve", help="Serve the dashboard.")
    _add_server_arguments(serve_parser)

    run_parser = subparsers.add_parser(
        "run", help="Build dashboard data, then serve the dashboard."
    )
    run_parser.add_argument(
        "--include-workspace-runs",
        action="store_true",
        help="Include workspace_*/run_*/reports directories.",
    )
    _add_server_arguments(run_parser)
    return parser


def _add_server_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host", default="127.0.0.1", help="Interface to bind. Default: 127.0.0.1"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="TCP port to bind. Default: 8080"
    )


def main() -> None:
    args = create_parser().parse_args()
    if args.command == "build":
        write_dashboard_data(include_workspace_runs=args.include_workspace_runs)
        return
    if args.command == "serve":
        serve(host=args.host, port=args.port)
        return

    write_dashboard_data(include_workspace_runs=args.include_workspace_runs)
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
