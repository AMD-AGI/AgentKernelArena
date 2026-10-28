"""Exact historical-source views for pre-existing migration fingerprint tests.

The current bytes must match a reviewed repair delta. Current-behavior tests
read actual source directly; this helper only retains pre-existing contracts.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Keep historical migration fingerprints as exact pre-repair contracts. These
# inverses accept only pinned current bytes and restore every previous byte;
# they do not skip tasks or adjust old expected hashes. The new controls are
# exercised on their actual source by test_task_repair_sensitivity and GPU runs.
REPAIR_DELTAS = json.loads((Path(__file__).parent / "fixtures/task_semantic_repair_deltas.json").read_text())
REPAIR_BY_AFTER = {row["after_sha256"]: [r for r in REPAIR_DELTAS if r["after_sha256"] == row["after_sha256"]] for row in REPAIR_DELTAS}


def reviewed_repair_source(source, path=None):
    matches = REPAIR_BY_AFTER.get(hashlib.sha256(source.encode()).hexdigest())
    if matches is None:
        return source
    if path is not None:
        relative = Path(path).relative_to(ROOT).as_posix()
        matches = [row for row in matches if row["path"] == relative]
    assert matches and len({row["before_sha256"] for row in matches}) == 1, "Ambiguous reviewed task repair"
    record = matches[0]
    lines = source.splitlines(keepends=True)
    for hunk in reversed(record["reverse_hunks"]):
        assert "".join(lines[hunk["start"]:hunk["end"]]) == hunk["added"]
        lines[hunk["start"]:hunk["end"]] = [hunk["original"]]
    restored = "".join(lines)
    assert hashlib.sha256(restored.encode()).hexdigest() == record["before_sha256"]
    return restored


def reviewed_repair_bytes(raw, path=None):
    return reviewed_repair_source(raw.decode(), path=path).encode()
