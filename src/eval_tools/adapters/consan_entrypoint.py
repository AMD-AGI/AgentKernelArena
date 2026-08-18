#!/usr/bin/env python3
"""Run a native launcher under ConSan, then run its configured oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PREFIX = "AKA_CONSAN_RUN "


def _identities(path: Path) -> tuple[str, str]:
    sha = hashlib.sha256()
    fnv = 14695981039346656037
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
            for byte in chunk:
                fnv ^= byte
                fnv = (fnv * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return sha.hexdigest(), f"fnv1a64:{fnv:016x}"


def _forward(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hook", required=True, type=Path)
    parser.add_argument("--code-object", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-fingerprint", required=True)
    parser.add_argument("--mode", choices=("record-replay",), default="record-replay")
    parser.add_argument("--command-arg", action="append", default=[])
    parser.add_argument("--oracle-arg", action="append", default=[])
    args = parser.parse_args(argv)

    payload: dict[str, object] = {
        "schema_version": 1,
        "code_object_sha256": args.expected_sha256,
        "code_object_fingerprint": args.expected_fingerprint,
        "mode": args.mode,
        "instrumented_returncode": None,
        "oracle_returncode": None,
        "oracle_passed": False,
    }
    try:
        if not args.command_arg or not args.oracle_arg:
            raise RuntimeError("both instrumented and oracle argv must be non-empty")
        code_object = args.code_object.resolve(strict=True)
        observed_sha, observed_fingerprint = _identities(code_object)
        if observed_sha != args.expected_sha256:
            raise RuntimeError("code-object SHA-256 changed before ConSan execution")
        if observed_fingerprint != args.expected_fingerprint:
            raise RuntimeError("code-object FNV identity changed before ConSan execution")
        hook = args.hook.resolve(strict=True)

        instrumented_env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("RJ_CONSAN_")
        }
        instrumented_env.update(
            {
                "HSA_TOOLS_DISABLE_REGISTER": "1",
                "HSA_TOOLS_LIB": str(hook),
                "RJ_CONSAN_MODE": args.mode,
                "RJ_CONSAN_POLICY": "strict",
                "RJ_CONSAN_LOG": "1",
            }
        )
        instrumented = subprocess.run(
            args.command_arg,
            check=False,
            capture_output=True,
            text=True,
            env=instrumented_env,
        )
        _forward(instrumented)
        payload["instrumented_returncode"] = instrumented.returncode

        if instrumented.returncode == 0:
            oracle_env = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("RJ_CONSAN_")
                and key not in {"HSA_TOOLS_DISABLE_REGISTER", "HSA_TOOLS_LIB"}
            }
            oracle = subprocess.run(
                args.oracle_arg,
                check=False,
                capture_output=True,
                text=True,
                env=oracle_env,
            )
            _forward(oracle)
            payload["oracle_returncode"] = oracle.returncode
            payload["oracle_passed"] = oracle.returncode == 0
        print(PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))
        if instrumented.returncode != 0:
            return instrumented.returncode
        return int(payload["oracle_returncode"] or 0)
    except (OSError, RuntimeError, ValueError) as error:
        payload["error"] = str(error)
        print(PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))
        return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
