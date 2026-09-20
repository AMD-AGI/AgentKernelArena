"""CPU fixtures for optional device evidence; no GPU success is inferred."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

from src.tools import head_kernel_trace_worker as worker
from src.tools import trace_head_kernels as trace
from src.scripts import top5_head_kernels as launcher


def gpu_event(**args):
    return {"ph": "X", "cat": "kernel", "name": "real_device_symbol", "dur": 7.5, "args": args}


class DeviceTraceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_only_gpu_events_supply_kernel_names(self):
        events = [{"ph": "X", "cat": "cpu_op", "name": "pretend_kernel", "dur": 1},
                  {"ph": "X", "cat": "cuda_runtime", "name": "hipGraphLaunch", "dur": 2},
                  gpu_event(grid=[8, 1, 1], block=[256, 1, 1], device=0, stream=4, correlation=12)]
        result = worker.kernel_events({"traceEvents": events})
        self.assertEqual([event["symbol"] for event in result], ["real_device_symbol"])
        self.assertEqual(result[0]["grid"], [8, 1, 1])
        self.assertEqual(result[0]["block"], [256, 1, 1])
        self.assertEqual(result[0]["missing_metadata"], ["graph_node_id"])

    def test_missing_launch_metadata_is_explicit(self):
        event = worker.kernel_events({"traceEvents": [gpu_event()]})[0]
        self.assertIsNone(event["grid"])
        self.assertIn("grid", event["missing_metadata"])
        self.assertIn("block", event["missing_metadata"])

    def test_case_selection_covers_regimes_without_inventing_cases(self):
        rows = [{"sig": str(index), "regime": regime} for index, regime in
                enumerate(["decode", "decode", "prefill", "prefill", "decode", "decode"])]
        self.assertEqual(worker.select_indices(rows, 3), [0, 2, 5])
        self.assertEqual(worker.select_indices(rows, 2, [4, 1]), [4, 1])
        self.assertEqual(worker.select_indices(rows, 16), [0, 2, 1, 3, 4, 5])

    def test_invalid_case_selections_are_rejected(self):
        for selected in ([0, 0], [2], [-1], [0, 1]):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                worker.select_indices([{"sig": "a"}, {"sig": "b"}], 1, selected)

    def fake_torch(self, events, gpu=True):
        class Profiler:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def export_chrome_trace(self, path):
                Path(path).write_text(json.dumps({"traceEvents": events}))
        profiler = SimpleNamespace(ProfilerActivity=SimpleNamespace(CPU="CPU", CUDA="CUDA"),
                                   supported_activities=lambda: {"CPU", "CUDA"} if gpu else {"CPU"},
                                   profile=mock.Mock(return_value=Profiler()))
        return SimpleNamespace(profiler=profiler, cuda=SimpleNamespace(synchronize=mock.Mock()))

    def test_profile_replays_only_the_supplied_callback(self):
        torch = self.fake_torch([gpu_event(grid=[1, 1, 1])])
        callback = mock.Mock()
        result = worker.profile_call(callback, torch, self.root / "trace.json", 2)
        self.assertEqual(callback.call_count, 2)
        self.assertEqual(result[0]["symbol"], "real_device_symbol")
        self.assertEqual(torch.profiler.profile.call_args.kwargs["activities"], ["CPU", "CUDA"])

    def test_cpu_only_profiler_output_cannot_pass(self):
        torch = self.fake_torch([{"ph": "X", "cat": "cpu_op", "name": "kernel_name"}])
        with self.assertRaisesRegex(RuntimeError, "no GPU kernel events"):
            worker.profile_call(mock.Mock(), torch, self.root / "trace.json", 1)

    def test_unavailable_gpu_profiler_does_not_invoke_callback(self):
        callback = mock.Mock()
        with self.assertRaisesRegex(RuntimeError, "device activity"):
            worker.profile_call(callback, self.fake_torch([], gpu=False), self.root / "trace.json", 1)
        callback.assert_not_called()

    def test_timed_trace_observes_the_unchanged_scored_replay(self):
        state = SimpleNamespace(probe_enabled=False, restore=mock.Mock())
        replay = SimpleNamespace(replay=mock.Mock())
        timer = mock.Mock(return_value=([1.0] * 100, {"benchmark_method": "cuda_graph"}))
        row = {"args": {"x": "fixture"}}
        seen = {}
        def measure(actual_row, call, expected, module, h, meta, torch, warmup, iterations, benchmark):
            seen.update(row=actual_row, warmup=warmup, iterations=iterations, expected=expected)
            benchmark(lambda: call(actual_row["args"]), warmup=warmup, repetition=iterations,
                      prepare_fn=state.restore, timed_run=replay)
            return {"benchmark_method": "cuda_graph"}
        bench = SimpleNamespace(InputState=lambda *a: state, replay_probe=lambda *a: None,
                                collect_output=lambda *a: "output", cpu_copy=lambda value, torch: value,
                                output_transform=lambda *a: None, measure_case=measure)
        callback, metadata = worker.timed_baseline(row, mock.Mock(), None, None, {}, None,
                                                   bench, timer, 10, 100)
        self.assertIs(callback, replay.replay)
        self.assertIs(seen["row"], row)
        self.assertEqual((seen["warmup"], seen["iterations"]), (10, 100))
        self.assertEqual(set(seen["expected"]), {"base", "probe"})
        self.assertIs(timer.call_args.kwargs["timed_run"], replay)
        self.assertFalse(state.probe_enabled)

    def test_worker_launch_uses_only_the_reference_overlay(self):
        runner = SimpleNamespace(verify_fixtures=mock.Mock(), overlays=lambda: ("baseline", "candidate"),
                                 run_worker=mock.Mock(return_value=SimpleNamespace(returncode=0, stdout="", stderr="")))
        with mock.patch.object(worker, "__file__", str(self.root / "scripts/_device_trace.py")), \
                mock.patch.object(worker, "load", return_value=runner), \
                mock.patch.object(worker.sys, "argv", ["probe", "--mode", "eager"]):
            self.assertEqual(worker.main(), 0)
        arguments = runner.run_worker.call_args.args
        self.assertEqual(arguments[2], "baseline")
        self.assertIs(arguments[4], False)
        runner.verify_fixtures.assert_called_once()

    def test_artifact_mutation_and_empty_gpu_evidence_are_rejected(self):
        path = self.root / "trace.json"
        path.write_text(json.dumps({"traceEvents": [gpu_event()]}))
        artifact = {"path": path.name, **trace.fingerprint(path)}
        report = {"status": "trace_recorded", "selected_case_indexes": [0], "cases": [
            {"status": "trace_recorded", "case_index": 0, "kernel_events": [{"symbol": "real"}],
             "raw_trace": artifact}]}
        trace.validate_trace_artifacts(self.root, report)
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            trace.validate_trace_artifacts(self.root, report)
        report["cases"][0]["kernel_events"] = []
        with self.assertRaisesRegex(ValueError, "no GPU"):
            trace.validate_trace_artifacts(self.root, report)

    def test_host_execution_and_unbounded_options_are_refused(self):
        args = SimpleNamespace(max_cases=3, replays=1, timeout=600, mode="timed-graph", case_index=[])
        config = launcher.REPO_ROOT / "example_configs/top5_validator_glm_bf16_public_mi355x.yaml"
        with mock.patch.dict(os.environ, {"AGENT_KERNEL_ARENA_DOCKER": ""}), \
                self.assertRaisesRegex(ValueError, "Docker runner"):
            trace.trace(config, args)
        args.timeout = float("inf")
        with self.assertRaisesRegex(ValueError, "timeout"):
            trace.trace(config, args)

    def test_successful_coordinator_still_claims_no_score_or_equivalence(self):
        config_digest = "sha256:" + "a" * 64
        image = "example.invalid/runtime:v1"
        plan = {"image": image, "expected_image_id": config_digest, "tasks": ["head_kernels/test"]}
        environment = {"AGENT_KERNEL_ARENA_DOCKER": "1", "AGENT_KERNEL_ARENA_DOCKER_IMAGE": image,
                       "AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID": config_digest,
                       "AGENT_KERNEL_ARENA_DOCKER_CONFIG_DIGEST": config_digest,
                       "AGENT_KERNEL_ARENA_DOCKER_IDENTITY": "{}",
                       "AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS": "[]"}
        args = SimpleNamespace(max_cases=3, replays=1, timeout=600, mode="timed-graph", case_index=[])
        with mock.patch.object(trace, "plan_run", return_value=plan), \
                mock.patch.object(trace, "trace_task", return_value={"status": "trace_recorded"}) as execute, \
                mock.patch.dict(os.environ, environment):
            code, report_path = trace.trace(self.root / "run.yaml", args, self.root)
        report = json.loads(report_path.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(execute.call_count, 1)
        self.assertFalse(report["scoring_performed"])
        self.assertFalse(report["framework_PASS_claimed"])
        self.assertFalse(report["original_dispatch_equivalence_certified"])


if __name__ == "__main__":
    unittest.main()
