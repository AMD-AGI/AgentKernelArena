# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Derive weighted performance cases for kernel tasks from the AgentX serving corpus.

    python3 -m src.tools.agentx_workloads capture --model-path ... --corpus ... --image ... --out DIR --capture-id ID
    python3 -m src.tools.agentx_workloads census DIR/ID
    python3 -m src.tools.agentx_workloads propose tasks/<...>/<scenario> --census DIR/ID/kernel_census.json

``capture`` runs on a GPU node (SGLang in Docker); ``census`` and ``propose``
run on a CPU. ``replay`` is the corpus client that ``capture`` starts inside the
serving image. See docs/how-to/agentx-workloads.md.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m src.tools.agentx_workloads", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    from . import capture, census, propose, replay

    for name, module, help_text in (
            ("capture", capture, "serve a model, replay the corpus and record profiler windows (GPU node)"),
            ("census", census, "build kernel_census.json from a capture (CPU)"),
            ("propose", propose, "propose a weighted workloads.json for one task (CPU)"),
            ("replay", replay, "corpus replay client (runs inside the serving image)")):
        p = sub.add_parser(name, help=help_text, description=module.__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        module.add_arguments(p)
        p.set_defaults(handler=module.main)
    args = ap.parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    sys.exit(main())
