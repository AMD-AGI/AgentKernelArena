"""CPU regressions for headkernel replay failures and performance reports."""

import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


TASKS = Path(__file__).resolve().parents[1] / "tasks/headkernel"
SCRIPTS = TASKS / "qwen3.8-2.4t__paged_attention_decode/scripts"
MISSING = object()


def load_script(name):
    spec = importlib.util.spec_from_file_location("headkernel_test_" + name, SCRIPTS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCuda:
    def __init__(self, samples=(1.0,)):
        self.samples = iter(samples)

    def is_available(self):
        return True

    def synchronize(self):
        pass

    def empty_cache(self):
        pass

    def Event(self, *, enable_timing):
        assert enable_timing is True
        return SimpleNamespace(record=lambda: None, elapsed_time=lambda end: next(self.samples))


class BenchmarkExecutionTests(unittest.TestCase):
    def setUp(self):
        self.bench = load_script("_bench")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.ut = Path(self.temporary.name) / "ut"
        self.ut.mkdir()
        self.out = self.ut.parent / "result.json"
        self.calls = []

    def call(self, value):
        self.calls.append(value)
        if value == "fail":
            raise RuntimeError("candidate failed")

    def run_bench(self, blob=MISSING, *, samples=(1.0,), meta=None, entrypoint=False):
        if blob is not MISSING:
            (self.ut / "reference_io.pt").write_bytes(b"fake load; no tensor serialization")
        (self.ut / "meta.json").write_text(json.dumps(
            {"target_callable": "candidate:call"} if meta is None else meta
        ))
        fake_torch = SimpleNamespace(cuda=FakeCuda(samples), load=lambda *args, **kwargs: blob)
        argv = ["_bench.py", "--ut", str(self.ut), "--out", str(self.out),
                "--warmup", "0", "--iters", "1"]
        with (mock.patch.dict(sys.modules, {"torch": fake_torch,
                                           "candidate": SimpleNamespace(call=self.call)}),
              mock.patch.object(sys, "argv", argv), mock.patch.object(sys, "path", sys.path.copy()),
              mock.patch.object(self.bench, "resolve", return_value=self.call),
              contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO())):
            if entrypoint:
                return runpy.run_path(str(SCRIPTS / "_bench.py"), run_name="__main__")
            return self.bench.main()

    def native_builder(self, source):
        (self.ut / "harness_lib.py").write_text("# CPU test harness\n")
        (self.ut / "cases.py").write_text(source)

    def test_direct_replay_keeps_existing_timing_counts_and_statistics(self):
        calls = []
        result = self.bench.time_call(lambda: calls.append(True),
                                      SimpleNamespace(cuda=FakeCuda([3.0, 1.0, 2.0])), 2, 3)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result, {"mean_ms": 2.0, "median_ms": 2.0, "min_ms": 1.0})

    def test_missing_replay_support_has_explicit_capability_result(self):
        self.assertEqual(self.run_bench(), 4)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.calls, [])

    def test_entrypoint_keeps_explicit_unsupported_return_code(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_bench(entrypoint=True)
        self.assertEqual(raised.exception.code, 4)

    def test_entrypoint_rejects_candidate_exit_four_even_after_success(self):
        def candidate(value):
            self.calls.append(value)
            if value == "fail":
                raise SystemExit(4)

        self.call = candidate
        for values in (["fail"], ["good", "fail"]):
            with self.subTest(values=values):
                self.calls.clear()
                blob = {"records": [{"sig": value, "args": [value]} for value in values]}
                with self.assertRaises(SystemExit) as raised:
                    self.run_bench(blob, entrypoint=True)
                self.assertEqual(raised.exception.code, 1)
                self.assertEqual(self.calls, values)
                self.assertFalse(self.out.exists())

    def test_entrypoint_rejects_builder_exit_four(self):
        self.native_builder("raise SystemExit(4)\n")
        with self.assertRaises(SystemExit) as raised:
            self.run_bench(entrypoint=True)
        self.assertEqual(raised.exception.code, 1)
        self.assertFalse(self.out.exists())

    def test_entrypoint_preserves_argparse_help_exit(self):
        with (mock.patch.object(sys, "argv", ["_bench.py", "--help"]),
              contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as raised):
            runpy.run_path(str(SCRIPTS / "_bench.py"), run_name="__main__")
        self.assertEqual(raised.exception.code, 0)

    def test_optional_shared_pool_may_be_none(self):
        blob = {"records": [{"args": ["good"]}], "shared": None}
        self.assertEqual(self.run_bench(blob), 0)
        self.assertEqual(self.calls, ["good"])

    def test_unsupported_representation_can_use_native_builder(self):
        self.native_builder(
            "def timing_cases(h, meta):\n"
            "    return [{'sig': 'native', 'args': None}]\n"
            "def call(args):\n"
            "    return None\n"
        )
        blob = {"records": [{"args": [{"__repr__": "opaque_capture()"}]}]}
        self.assertEqual(self.run_bench(blob), 0)
        report = json.loads(self.out.read_text())
        self.assertEqual(report["cases"][0]["sig"], "native")
        self.assertEqual(self.calls, [])

    def test_unsupported_representation_without_builder_can_fall_back(self):
        blob = {"records": [{"args": [{"__repr__": "opaque_capture()"}]}]}
        self.assertEqual(self.run_bench(blob), 4)
        self.assertFalse(self.out.exists())
        self.assertEqual(self.calls, [])

    def test_direct_call_failure_invalidates_even_prior_successful_cases(self):
        for values in (["fail"], ["good", "fail"]):
            with self.subTest(values=values):
                self.calls.clear()
                blob = {"records": [{"sig": value, "args": [value]} for value in values]}
                with self.assertRaisesRegex(RuntimeError, "candidate failed"):
                    self.run_bench(blob)
                self.assertFalse(self.out.exists())
                self.assertEqual(self.calls, values)

    def test_native_call_failure_invalidates_even_prior_successful_cases(self):
        self.native_builder(
            "def timing_cases(h, meta):\n"
            "    return [{'sig': 'good', 'args': False}, {'sig': 'bad', 'args': True}]\n"
            "def call(fail):\n"
            "    if fail:\n"
            "        raise RuntimeError('native candidate failed')\n"
        )
        with self.assertRaisesRegex(RuntimeError, "native candidate failed"):
            self.run_bench()
        self.assertFalse(self.out.exists())

    def test_native_failure_after_unsupported_reconstruction_is_not_capability_failure(self):
        self.native_builder(
            "def timing_cases(h, meta):\n"
            "    return [{'sig': 'bad', 'args': None}]\n"
            "def call(args):\n"
            "    raise RuntimeError('native candidate failed')\n"
        )
        with self.assertRaisesRegex(RuntimeError, "native candidate failed"):
            self.run_bench({"records": [{"sig": "unreconstructable"}]})
        self.assertFalse(self.out.exists())

    def test_builder_import_and_selection_errors_do_not_fall_back(self):
        sources = [
            "raise ImportError('builder import failed')\n",
            "def selected_cases(meta, ids):\n"
            "    raise RuntimeError('selection failed')\n"
            "def case_map(meta):\n"
            "    return {'unselected': {'sig': 'unselected', 'args': None}}\n"
            "def timing_case(case):\n"
            "    return case\n"
            "def candidate_call(args):\n"
            "    return None\n",
        ]
        for source in sources:
            with self.subTest(source=source):
                self.native_builder(source)
                with self.assertRaises((ImportError, RuntimeError)):
                    self.run_bench()
                self.assertFalse(self.out.exists())

    def test_recognized_builder_must_produce_cases(self):
        self.native_builder("def timing_cases(h, meta): return []\ndef call(args): pass\n")
        with self.assertRaisesRegex(ValueError, "no timing cases"):
            self.run_bench()

    def test_corrupt_captures_are_not_unsupported_replay(self):
        blobs = [
            [], {"records": {}}, {"records": [None]},
            {"records": [{"args": ["good"]}], "shared": []},
            {"records": [{"args": [{"__shared__": "missing"}]}]},
            {"records": [{"args": {"value": "good"}}]},
            {"records": [{"args": [], "kwargs": []}]},
            {"records": [{"args": ["good"]}, {"args": [{"__shared__": "missing"}]}]},
        ]
        for blob in blobs:
            with self.subTest(blob=blob):
                with self.assertRaises((ValueError, KeyError, AttributeError)):
                    self.run_bench(blob)
                self.assertFalse(self.out.exists())

    def test_missing_target_is_a_capture_error(self):
        with self.assertRaisesRegex(ValueError, "no target"):
            self.run_bench({"records": [{"args": ["good"]}]}, meta={})

    def test_invalid_event_samples_cannot_produce_a_report(self):
        for duration in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "device-event duration"):
                    self.run_bench({"records": [{"args": ["good"]}]}, samples=[duration])
                self.assertFalse(self.out.exists())


class BenchmarkReportTests(unittest.TestCase):
    def setUp(self):
        self.runner = load_script("task_runner")
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runner.TASK_DIR = str(self.root)
        self.runner.BUILD_DIR = str(self.root / "build")
        self.runner.UT_DIR = str(self.root / "ut")
        (self.root / "ut").mkdir()
        (self.root / "build").mkdir()
        self.raw_path = self.root / "build/_bench_raw.json"
        self.report_path = self.root / "build/performance_report.json"
        self.valid = {
            "timer": "cuda_event", "warmup": 10, "iters": 100,
            "cases": [{"sig": "case", "mean_ms": 1.0, "median_ms": 1.0, "min_ms": 0.5}],
        }

    def run_performance(self, *, returncode=0, report=MISSING, error=None):
        def process(*args, **kwargs):
            self.assertFalse(self.raw_path.exists(), "previous benchmark output survived launch")
            if error is not None:
                raise error
            if report is not MISSING:
                self.raw_path.write_text(report if isinstance(report, str) else json.dumps(report))
            return subprocess.CompletedProcess(args[0], returncode, "", "")

        with (mock.patch.object(self.runner, "candidate_overlay", return_value=None),
              mock.patch.object(self.runner.subprocess, "run", side_effect=process),
              mock.patch.object(self.runner, "run_performance_via_ut", return_value=["fallback"]) as fallback,
              contextlib.redirect_stdout(io.StringIO())):
            result = self.runner.run_performance({}, 30)
        return result, fallback

    def assert_failure(self, result, fallback):
        self.assertEqual(result, [])
        fallback.assert_not_called()
        report = json.loads(self.report_path.read_text())
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["test_cases"], [])
        self.assertIs(report["fallback_used"], False)

    def test_valid_fresh_report_is_scoreable(self):
        result, fallback = self.run_performance(report=self.valid)
        fallback.assert_not_called()
        self.assertEqual(result[0]["test_case_id"], "case")
        self.assertEqual(result[0]["execution_time_ms"], 1.0)
        self.assertEqual(json.loads(self.report_path.read_text())["status"], "ok")

    def test_only_explicit_capability_result_without_report_uses_fallback(self):
        result, fallback = self.run_performance(returncode=4)
        self.assertEqual(result, ["fallback"])
        fallback.assert_called_once()
        self.assert_failure(*self.run_performance(returncode=4, report=self.valid))

    def test_process_failures_and_timeouts_never_use_fallback(self):
        for code in (1, 2, 3, 5, -9):
            with self.subTest(code=code):
                self.assert_failure(*self.run_performance(returncode=code))
        self.assert_failure(*self.run_performance(error=subprocess.TimeoutExpired("bench", 30)))

    def test_old_report_is_removed_before_launch_and_cannot_be_reused(self):
        self.raw_path.write_text(json.dumps(self.valid))
        self.assert_failure(*self.run_performance())

    def test_failed_process_cannot_score_even_with_output(self):
        self.assert_failure(*self.run_performance(returncode=1, report=self.valid))

    def test_malformed_empty_or_incomplete_reports_never_use_fallback(self):
        reports = ["{broken JSON", None, [], {}, {"cases": []},
                   dict(self.valid, cases=[None]), dict(self.valid, cases=[{}]),
                   dict(self.valid, warmup=9), dict(self.valid, iters=True)]
        for report in reports:
            with self.subTest(report=report):
                self.assert_failure(*self.run_performance(report=report))

    def test_missing_or_different_timer_never_uses_fallback(self):
        for timer in (None, "host_wall"):
            with self.subTest(timer=timer):
                report = copy.deepcopy(self.valid)
                if timer is None:
                    report.pop("timer")
                else:
                    report["timer"] = timer
                self.assert_failure(*self.run_performance(report=report))

    def test_any_invalid_timing_rejects_the_whole_report(self):
        for key in ("mean_ms", "median_ms", "min_ms"):
            for value in (float("nan"), float("inf"), 0, -1, True, "1.0", None):
                with self.subTest(key=key, value=value):
                    report = copy.deepcopy(self.valid)
                    bad = dict(report["cases"][0], sig="bad")
                    bad[key] = value
                    report["cases"].append(bad)
                    self.assert_failure(*self.run_performance(report=report))

    def test_valid_fallback_preserves_all_measurements(self):
        output = "GEAK_PER_CASE " + json.dumps([
            {"sig": "first", "optimized_ms": 1.0}, {"sig": "second", "optimized_ms": 2.0},
        ])
        process = subprocess.CompletedProcess([], 0, output, "")
        with (mock.patch.object(self.runner, "run_ut", return_value=(process, 0.1)),
              contextlib.redirect_stdout(io.StringIO())):
            result = self.runner.run_performance_via_ut({}, 30, "none", "unsupported")
        self.assertEqual([row["execution_time_ms"] for row in result], [1.0, 2.0])
        self.assertEqual(json.loads(self.report_path.read_text())["status"], "ok")

    def test_fallback_invalid_row_rejects_whole_result(self):
        for value in (float("inf"), float("nan"), True, -1, 0, None, "1.0"):
            with self.subTest(value=value):
                output = "GEAK_PER_CASE " + json.dumps([
                    {"sig": "good", "optimized_ms": 1.0}, {"sig": "bad", "optimized_ms": value},
                ])
                process = subprocess.CompletedProcess([], 0, output, "")
                with (mock.patch.object(self.runner, "run_ut", return_value=(process, 0.1)),
                      contextlib.redirect_stdout(io.StringIO())):
                    result = self.runner.run_performance_via_ut({}, 30, "none", "unsupported")
                self.assertEqual(result, [])
                report = json.loads(self.report_path.read_text())
                self.assertEqual(report["status"], "fail")
                self.assertEqual(report["test_cases"], [])
                self.assertEqual(report["rows_without_a_candidate_time"], 1)

    def test_fallback_timing_lines_preserve_valid_historical_forms(self):
        output = "\n".join([
            "unrelated diagnostic output",
            "timing:plain baseline_ms=2.0 candidate_ms=1.0",
            " timing:scientific baseline_ms=2e-3 candidate_ms=+1.0e-3 speedup=2.0 reps=3 ",
            "timing:optional-none baseline_ms=4.0 candidate_ms=2.0 speedup=None reps=None",
        ])
        process = subprocess.CompletedProcess([], 0, output, "")
        with (mock.patch.object(self.runner, "run_ut", return_value=(process, 0.1)),
              contextlib.redirect_stdout(io.StringIO())):
            result = self.runner.run_performance_via_ut({}, 30, "none", "unsupported")
        self.assertEqual([r["execution_time_ms"] for r in result], [1.0, 0.001, 2.0])
        self.assertEqual([r["reps"] for r in result], [None, 3, None])
        self.assertEqual(json.loads(self.report_path.read_text())["status"], "ok")

    def test_fallback_mixed_valid_and_invalid_timing_lines_reject_whole_result(self):
        good = "timing:good baseline_ms=1.0 candidate_ms=1.0"
        bad_rows = [
            "timing:bad baseline_ms=1.0 candidate_ms=" + value
            for value in ("nan", "NaN", "inf", "-inf", "None", "True", "0", "-1", "1.0oops")
        ] + [
            "timing:bad baseline_ms=nan candidate_ms=1.0",
            "timing:bad baseline_ms=1.0",  # Missing candidate measurement.
            "timing: baseline_ms=1.0 candidate_ms=1.0",
            "timing:bad baseline_ms=1.0 candidate_ms=1.0 unexpected",
            "timing:bad baseline_ms=1.0 candidate_ms=1.0 reps=broken",
        ]
        for bad in bad_rows:
            with self.subTest(row=bad):
                process = subprocess.CompletedProcess([], 0, good + "\n" + bad, "")
                with (mock.patch.object(self.runner, "run_ut", return_value=(process, 0.1)),
                      mock.patch.object(self.runner, "_per_case_from_result_json") as file_fallback,
                      contextlib.redirect_stdout(io.StringIO())):
                    result = self.runner.run_performance_via_ut({}, 30, "none", "unsupported")
                self.assertEqual(result, [])
                file_fallback.assert_not_called()
                report = json.loads(self.report_path.read_text())
                self.assertEqual(report["status"], "fail")
                self.assertEqual(report["test_cases"], [])
                self.assertIn("invalid fallback timing report", report["error"])

    def test_structured_fallback_cannot_hide_an_invalid_timing_line(self):
        rows = [{"sig": "good", "optimized_ms": 1.0}]
        output = "GEAK_PER_CASE " + json.dumps(rows) + "\ntiming:bad baseline_ms=1.0 candidate_ms=inf"
        process = subprocess.CompletedProcess([], 0, output, "")
        with (mock.patch.object(self.runner, "run_ut", return_value=(process, 0.1)),
              contextlib.redirect_stdout(io.StringIO())):
            result = self.runner.run_performance_via_ut({}, 30, "none", "unsupported")
        self.assertEqual(result, [])
        self.assertEqual(json.loads(self.report_path.read_text())["status"], "fail")

    def test_valid_structured_fallback_formats_remain_supported(self):
        rows = [{"sig": "first", "optimized_ms": 1.0}, {"sig": "second", "optimized_ms": 2.0}]
        for output in ("GEAK_PER_CASE=" + json.dumps(rows),
                       "GEAK_PER_CASE " + json.dumps(rows),
                       json.dumps({"per_case": rows}, indent=2)):
            with self.subTest(output=output):
                process = subprocess.CompletedProcess([], 0, output, "")
                with (mock.patch.object(self.runner, "run_ut", return_value=(process, 0.1)),
                      contextlib.redirect_stdout(io.StringIO())):
                    result = self.runner.run_performance_via_ut({}, 30, "none", "unsupported")
                self.assertEqual([r["execution_time_ms"] for r in result], [1.0, 2.0])


class SuiteCopiesTests(unittest.TestCase):
    def test_all_headkernel_benchmark_and_runner_copies_match(self):
        for filename in ("_bench.py", "task_runner.py"):
            paths = list(TASKS.glob("*/scripts/" + filename))
            self.assertEqual(len(paths), 16)
            template = TASKS.parents[1] / "tools/templates" / filename
            if filename == "task_runner.py":
                # The requested correctness fix is limited to the eight bound
                # non-Qwen tasks. Keep Qwen and the three Kimi runners unchanged.
                bound = [path for path in paths
                         if not path.parent.parent.name.startswith("qwen")
                         and (json.loads((path.parent.parent / "ut/meta.json").read_text())
                              .get("candidate_bind") or {}).get("file")]
                self.assertEqual(len(bound), 8)
                self.assertEqual(len({path.read_bytes() for path in bound}), 1)
                paths = [path for path in paths if path not in bound]
            self.assertEqual(len({path.read_bytes() for path in paths}), 1, filename)
            self.assertEqual(paths[0].read_bytes(), template.read_bytes(), filename)


if __name__ == "__main__":
    unittest.main()
