#!/usr/bin/env python3
"""Extract a compact structural contract from an explicitly supplied local archive.

This offline developer tool hashes the archive before restricted CPU loading.
It does not download files, contact a host, or alter an existing task contract.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.tools.head_kernel_archives.minimax import digest, positional_names, verified_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=("minimax", "kimi", "deepseek", "qwen", "glm"))
    parser.add_argument("--task", required=True, type=Path, help="existing local task directory")
    parser.add_argument("--archive", required=True, type=Path, help="local reference_io.pt archive")
    parser.add_argument("--output", required=True, type=Path, help="new JSON file; never overwrites")
    parser.add_argument("--positional-names", nargs="*", help="captured callable argument names in order")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    meta = json.loads((args.task / "ut/meta.json").read_text())
    meta = {**meta, **meta.get("archival_capture", {})}
    expected = meta.get("reference_io_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("task metadata must declare the original archive SHA-256")
    module = importlib.import_module("src.tools.head_kernel_archives." + args.family)
    import torch

    torch.serialization.clear_safe_globals()
    torch.set_num_threads(1)
    with verified_archive(args.archive, expected) as archive:
        blob = torch.load(archive, map_location="cpu", weights_only=True, mmap=True)
        if args.family in {"glm", "kimi"}:
            contract = module.extract_blob(blob, meta, args.task.name, torch)
        else:
            names = args.positional_names
            if names is None:
                has_positional = any(row.get("args") for row in blob.get("records", []))
                names = positional_names(args.task) if has_positional else []
            if args.family == "qwen":
                contract = module.extract_blob(blob, meta, torch, names)
            else:
                contract = module.extract_blob(blob, meta, names, torch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(contract, stream, separators=(",", ":"))
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output),
                      "bytes": args.output.stat().st_size, "source_sha256": expected}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
