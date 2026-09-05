#!/usr/bin/env python3
"""Image-owned bridge from an attested HSACO to the waitcheck C API."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PREFIX = "AKA_WAITCHECK_RESULT "
CAPI_PREFIX = "AKA_WAITCHECK_CAPI_RESULT "


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _emit(payload: dict[str, object]) -> None:
    print(PREFIX + json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _kernel_inventory(binary: Path, code_object: Path, target: str) -> list[dict[str, object]]:
    completed = subprocess.run(
        [str(binary), str(code_object), "--target", target, "--list-kernels"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "waitcheck kernel inventory failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    records: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError("waitcheck emitted malformed kernel inventory JSON") from error
        if isinstance(value, dict) and "kernel_name" in value:
            records.append(value)
    return records


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waitcheck", required=True, type=Path)
    parser.add_argument("--capi-wrapper", required=True, type=Path)
    parser.add_argument("--code-object", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--expected-kernel", required=True)
    parser.add_argument("--kernel-entry", required=True, type=lambda value: int(value, 0))
    args = parser.parse_args(argv)

    base = {
        "schema_version": 1,
        "code_object_sha256": args.expected_sha256,
        "target": args.target,
        "expected_kernel": args.expected_kernel,
        "kernel_entry": args.kernel_entry,
    }
    try:
        code_object = args.code_object.resolve(strict=True)
        observed_sha256 = _sha256_file(code_object)
        if observed_sha256 != args.expected_sha256:
            raise RuntimeError(
                f"code-object SHA-256 mismatch: {observed_sha256} != {args.expected_sha256}"
            )
        inventory = _kernel_inventory(args.waitcheck, code_object, args.target)
        matching = [
            record
            for record in inventory
            if record.get("kernel_name") == args.expected_kernel
            and record.get("kernel_entry") == args.kernel_entry
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "expected exactly one matching kernel inventory record, "
                f"observed {len(matching)}"
            )
        completed = subprocess.run(
            [
                str(args.capi_wrapper),
                "--code-object",
                str(code_object),
                "--kernel-entry",
                str(args.kernel_entry),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        payloads = []
        for line in completed.stdout.splitlines():
            if line.startswith(CAPI_PREFIX):
                payloads.append(json.loads(line[len(CAPI_PREFIX) :]))
        if completed.stderr:
            sys.stderr.write(completed.stderr)
        if completed.returncode != 0 or len(payloads) != 1 or not isinstance(payloads[0], dict):
            raise RuntimeError(
                "waitcheck C API wrapper failed or emitted an invalid result record"
            )
        result = {**base, **payloads[0], "inventory_attested": True}
        _emit(result)
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        _emit({**base, "analysis_complete": False, "error": str(error)})
        return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
