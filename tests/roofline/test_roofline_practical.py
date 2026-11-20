"""Measured (practical) peaks in join and report: a second column, never the floor (CPU only)."""
from __future__ import annotations

import json

import pytest

from src.roofline import __main__ as cli
from src.roofline.arch import Peaks, load_peaks
from src.roofline.join import COLD_CACHE, join, practical_time_us, report_rows

MEASURED = Peaks("measured", "measured@sha256:test", mem_bw_gbps=6000.0,
                 tflops={"matrix_bf16": 1000.0, "matrix_fp8": 0.0, "vector_fp32": 100.0, "vector_bf16": 0.0})


def floor(t_floor_us=10.0, **counts):
    record = {"task_id": "t/a", "case_id": "c1", "operator": "op", "model": "m@v1", "t_floor_us": t_floor_us,
              "tier": "estimated", "bound": "memory", "notes": [], "peaks": {"source": "datasheet@sha256:x"},
              "flops_matrix_by_precision": {}, "flops_vector": 0, "bytes_min": 0}
    record.update(counts)
    return record


def baseline(us):
    return [{"task_id": "t/a", "case_id": "c1", "role": "baseline", "us": us, "method": COLD_CACHE, "file": "f"}]


def test_practical_time_is_the_floor_counts_at_the_measured_peaks():
    # 1e12 bf16 FLOPs at 1000 TFLOPS = 1000 us; 3e9 bytes at 6000 GB/s = 500 us; 5e9 vector FLOPs at the
    # fastest nonzero vector rate (100 TFLOPS) = 50 us. The maximum is 1000 us.
    record = floor(flops_matrix_by_precision={"matrix_bf16": 10 ** 12}, flops_vector=5 * 10 ** 9,
                   bytes_min=3 * 10 ** 9)
    assert practical_time_us(record, MEASURED) == pytest.approx(1000.0)
    assert practical_time_us(floor(bytes_min=3 * 10 ** 9), MEASURED) == pytest.approx(500.0)


def test_a_peak_the_microbench_could_not_measure_gives_no_practical_time():
    assert practical_time_us(floor(flops_matrix_by_precision={"matrix_fp8": 10 ** 9}), MEASURED) is None
    assert practical_time_us(floor(flops_matrix_by_precision={"matrix_fp4": 10 ** 9}), MEASURED) is None


def test_an_emulated_key_is_charged_at_its_measured_16_bit_peak_divided_by_its_products():
    bf16 = practical_time_us(floor(flops_matrix_by_precision={"matrix_bf16": 3 * 10 ** 9}), MEASURED)
    emulated = practical_time_us(floor(flops_matrix_by_precision={"matrix_bf16x3": 10 ** 9}), MEASURED)
    assert bf16 is not None and emulated == pytest.approx(bf16)


def test_practical_peaks_never_change_tiers_or_violations():
    # The baseline (400 us) beats the practical time (500 us) but not the floor (10 us): no violation.
    record = floor(bytes_min=3 * 10 ** 9)
    joined, violations = join([record], baseline(400.0), MEASURED)
    assert violations == []
    assert joined[0]["tier"] == "estimated"
    assert joined[0]["t_practical_us"] == pytest.approx(500.0)
    assert joined[0]["practical_fraction_baseline"] == pytest.approx(1.25)
    assert joined[0]["floor_fraction_baseline"] == pytest.approx(10.0 / 400.0)
    assert join([record], baseline(400.0))[0][0] == {k: v for k, v in joined[0].items()
                                                  if k not in ("t_practical_us", "practical_fraction_baseline",
                                                               "practical_peaks")}
    (row,) = report_rows(joined)
    assert (row["median_practical_fraction"], row["max_practical_fraction"]) == (pytest.approx(1.25),) * 2


def test_join_reads_a_microbench_json_and_report_shows_both_columns(tmp_path):
    microbench = {"name": "MI355X", "mem_bw_gbps": 6000.0, "memory_gb": 288.0, "source": "benchmark derived",
                  "max_achievable_tflops": {"matrix_bf16": 1000.0, "vector_fp32": 100.0}}
    peaks_file = tmp_path / "peaks" / "MI355X.json"
    peaks_file.parent.mkdir()
    peaks_file.write_text(json.dumps(microbench))
    assert load_peaks(str(peaks_file)).mem_bw_gbps == 6000.0

    out = tmp_path / "out"
    out.mkdir()
    (out / "floors.jsonl").write_text(json.dumps(floor(t_floor_us=8.0, bytes_min=60 * 10 ** 6)) + "\n")
    session = tmp_path / "evidence" / "s1"
    session.mkdir(parents=True)
    (session / "session.json").write_text(json.dumps({"task_id": "t/a"}))
    (session / "action-0001-baseline-performance.json").write_text(json.dumps({"result": {
        "status": "PASS", "cases": [{"test_case_id": "c1", "execution_time_ms": 0.0125,
                                     "benchmark_method": COLD_CACHE}]}}))
    assert cli.main(["join", "--floors", str(out / "floors.jsonl"), "--evidence", str(tmp_path / "evidence"),
                     "--out", str(out), "--practical-arch", str(peaks_file)]) == 0
    (record,) = [json.loads(line) for line in (out / "joined.jsonl").read_text().splitlines()]
    # 60 MB at 6000 GB/s = 10 us against a 12.5 us baseline; the floor stays 8 us.
    assert record["t_practical_us"] == pytest.approx(10.0)
    assert record["t_floor_us"] == 8.0
    assert cli.main(["report", "--joined", str(out / "joined.jsonl"), "--out", str(out)]) == 0
    report = (out / "report.md").read_text()
    assert "| datasheet median | datasheet max | practical median | practical max |" in report
    assert "| t/a | estimated | memory | 1/1 | 64% | 64% | 80% | 80% |" in report
