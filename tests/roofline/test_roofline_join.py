"""Joining floors with performance evidence, downgrades and the report (CPU only)."""
from __future__ import annotations

import json

from src.roofline import __main__ as cli
from src.roofline.join import COLD_CACHE, join, measurements, report_rows


def floor(task_id="t/a", case_id="c1", model="m@v1", t_floor_us=10.0, tier="estimated"):
    return {"task_id": task_id, "case_id": case_id, "operator": "op", "model": model, "t_floor_us": t_floor_us,
            "tier": tier, "bound": "memory", "notes": []}


def write_session(directory, task_id, cases, role="baseline", status="PASS", index=1):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "session.json").write_text(json.dumps({"version": 1, "task_id": task_id, "workspace": "w"}))
    record = {"invocation_id": "x", "phase": "task_validation",
              "result": {"protocol": 1, "role": role, "action": "performance", "status": status, "cases": cases}}
    (directory / f"action-{index:04d}-{role}-performance.json").write_text(json.dumps(record))


def case(case_id, ms, method=COLD_CACHE):
    return {"test_case_id": case_id, "execution_time_ms": ms, "benchmark_method": method}


def test_measurements_keep_cold_cache_cases_and_record_skips(tmp_path):
    write_session(tmp_path / "s1", "t/a", [case("c1", 0.02), case("c2", 0.02, method="cuda_graph")])
    write_session(tmp_path / "s2", "t/b", [case("c1", 0.02)], status="FAIL")
    rows, skipped = measurements(tmp_path)
    assert [(r["task_id"], r["case_id"], r["role"], r["us"]) for r in rows] == [("t/a", "c1", "baseline", 20.0)]
    reasons = sorted(s["reason"] for s in skipped)
    assert reasons[0].startswith("benchmark_method") and reasons[1] == "status FAIL"


def test_join_computes_fractions_from_the_fastest_measurement():
    measured = [{"task_id": "t/a", "case_id": "c1", "role": "baseline", "us": us, "method": COLD_CACHE, "file": "f"}
                for us in (40.0, 20.0)]
    joined, violations = join([floor()], measured)
    assert joined[0]["baseline_us"] == 20.0
    assert joined[0]["floor_fraction_baseline"] == 0.5
    assert violations == []


def test_a_measurement_faster_than_the_floor_downgrades_every_floor_of_its_model():
    floors = [floor(case_id="c1"), floor(case_id="c2"), floor(case_id="c3", model="other@v1")]
    measured = [{"task_id": "t/a", "case_id": "c1", "role": "candidate", "us": 9.4, "method": COLD_CACHE,
                 "file": "f"}]
    joined, violations = join(floors, measured)
    assert [v["case_id"] for v in violations] == ["c1"]
    assert [r["tier"] for r in joined] == ["none", "none", "estimated"]


def test_within_the_noise_margin_is_not_a_violation():
    measured = [{"task_id": "t/a", "case_id": "c1", "role": "baseline", "us": 9.6, "method": COLD_CACHE, "file": "f"}]
    assert join([floor()], measured)[1] == []


def test_report_summarizes_per_task():
    joined = [{**floor(case_id="c1"), "floor_fraction_baseline": 0.5},
              {**floor(case_id="c2"), "floor_fraction_baseline": 0.7}, floor(case_id="c3")]
    (row,) = report_rows(joined)
    assert (row["cases"], row["measured_cases"], row["median_fraction"], row["max_fraction"]) == (3, 2, 0.6, 0.7)


def test_join_and_report_commands_write_outside_the_repository(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "floors.jsonl").write_text(json.dumps(floor()) + "\n")
    write_session(tmp_path / "evidence" / "s1", "t/a", [case("c1", 0.0125)])
    assert cli.main(["join", "--floors", str(out / "floors.jsonl"), "--evidence", str(tmp_path / "evidence"),
                     "--out", str(out)]) == 0
    assert cli.main(["report", "--joined", str(out / "joined.jsonl"), "--out", str(out)]) == 0
    assert "| t/a | estimated | memory | 1/1 | 80% | 80% |" in (out / "report.md").read_text()
