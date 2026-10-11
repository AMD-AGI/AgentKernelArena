#!/usr/bin/env python3
"""Exercise the ready postchecker against authentic saved Lean evidence."""
import argparse
import copy
import json
from pathlib import Path

import check_timing as checker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-dir", required=True)
    args = parser.parse_args(); directory = Path(args.measurement_dir)
    quality = checker.read_and_assess(directory)
    assert quality["status"] == "pass" and quality["comparison_status"] == "unchanged_source_control"
    assert quality["gain_eligible"] is False and quality["accepted_gain"] is False
    assert quality["cases"][0]["reference"]["singleton_work_classes"] > 0
    measurement = json.loads((directory / "trusted_measurement.json").read_text())
    reports = {leg: {phase: json.loads((directory / (leg + "_" + phase + ".json")).read_text())
                    for phase in ("compile", "correctness", "performance")} for leg in ("reference", "candidate")}
    checks = {"authentic_unchanged_control": True, "singleton_limit_reported": True}
    mutations = {
        "outer_reference_substituted_for_native": lambda m, r: m["cases"][0].update(reference_ms=m["cases"][0]["port_reference_ms"]),
        "outer_ratio_substituted_for_primary": lambda m, r: m.update(arithmetic_mean_speedup=m["port_to_port_speedup_ratio"]),
        "missing_performance_phase": lambda m, r: r["candidate"].pop("performance"),
        "wrong_physical_GPU": lambda m, r: r["candidate"]["compile"]["request"]["gpu"].update(rocr_uuid="GPU-wrong"),
        "missing_correctness_seed": lambda m, r: r["candidate"]["correctness"]["cases"][0].update(seeds=[0, 1]),
        "changed_realized_work_receipt": lambda m, r: r["candidate"]["performance"]["cases"][0]["paired_reference"]["input_schedule"]["measured_inputs"][0].update(work_value=1),
    }
    for name, mutate in mutations.items():
        m, r = copy.deepcopy(measurement), copy.deepcopy(reports); mutate(m, r)
        try:
            checker.validate_and_assess(m, r)
        except (ValueError, AssertionError, KeyError):
            checks[name] = True
        else:
            raise AssertionError("Invalid evidence was accepted: " + name)
    print(json.dumps({"status": "PASS_CPU_ONLY", "checks": checks, "GPU_actions": False}, indent=2))


if __name__ == "__main__":
    main()
