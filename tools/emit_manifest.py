#!/usr/bin/env python3
"""Emit MANIFEST.tsv: every info-table row mapped to its task or its gap reason."""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = os.path.dirname(HERE)

COLS = ["row", "model", "info_kernel", "backend", "regime", "gpu_pct",
        "empirical_roofline", "optimized_roofline", "e2e_uplift",
        "tier", "status", "task", "source_package", "device_symbol",
        "target_callable", "reason"]


def main():
    man = json.load(open(os.path.join(HERE, "manifest.json")))
    tasks_dir = os.path.join(SUITE, "tasks", "headkernel")

    lines = ["\t".join(COLS)]
    tally = {"BUILT": 0, "BLOCKED": 0, "NOT_BUILT": 0, "GAP": 0}
    for r in man["rows"]:
        task = r.get("task")
        if not task:
            status = "GAP"
        elif not os.path.isfile(os.path.join(tasks_dir, task, "config.yaml")):
            status = "NOT_BUILT"
        elif r.get("blocked"):
            # Built and structurally sound, but known not to run: a declared file
            # the harness loads unconditionally is missing. Calling this BUILT
            # would promise something the task cannot deliver.
            status = "BLOCKED"
        else:
            status = "BUILT"
        tally[status] += 1
        row = {
            "row": r["row"], "model": r["model"], "info_kernel": r["info_kernel"],
            "backend": r.get("backend", ""), "regime": r.get("regime", ""),
            "gpu_pct": r.get("gpu_pct", ""),
            "empirical_roofline": r.get("empirical_roofline", ""),
            "optimized_roofline": r.get("optimized_roofline", ""),
            "e2e_uplift": r.get("e2e_uplift", ""),
            "tier": r.get("tier", ""), "status": status, "task": task or "",
            "source_package": r.get("pkg") or "",
            "device_symbol": r.get("device_symbol", ""),
            "target_callable": r.get("target_callable", ""),
            "reason": (r.get("gap_reason") or r.get("blocked") and
                       f"BLOCKED: {r['blocked']}. " + (r.get("note") or "")
                       or r.get("note") or "").replace("\t", " "),
        }
        lines.append("\t".join(str(row[c]).replace("\n", " ") for c in COLS))

    out = os.path.join(SUITE, "MANIFEST.tsv")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(man['rows'])} info rows -> {out}")
    print("  " + ", ".join(f"{v} {k}" for k, v in sorted(tally.items())))


if __name__ == "__main__":
    main()
