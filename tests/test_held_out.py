"""Held-out evaluation of finished schema-v3 candidates.

The trusted evaluator is substituted by a CPU fixture that reports what the
framework-written held-out contract asks for; these tests check the protocol
(evidence binding, inputs, isolation, scoring), not GPU correctness or timing.
"""
import io
import json
import logging
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

import yaml

from src.held_out import run_heldout_eval as heldout
from src.task_contract import case_seeds, load_task_contract, render_workspace
from src.task_execution import ExecutedAction
from src.task_protocol import ActionResult
from src.task_run import _completion_record
from src.task_runtime import bind_session_runtime
from src.task_session import TaskSession
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "tasks/normalization/rms_norm/bf16_h8192"
PILOT_ID = "normalization/rms_norm/bf16_h8192"
RUNTIME = {"gpu_arch": "gfx950", "gpu_name": "CPU protocol fixture, no GPU", "torch": "fixture"}
SALT = "0123456789abcdef"
LOGGER = logging.getLogger(__name__)


def _contract():
    return load_task_contract(PILOT / "task.yaml", task_id=PILOT_ID).resolved_contract("triton")


def completed_run(root: Path, *, accepted: bool = True) -> tuple[Path, Path]:
    """A framework-complete run of the pilot task whose candidate was edited."""
    run = root / "run_20261105_000000"
    workspace = run / "normalization_rms_norm_bf16_h8192_20261105_000000"
    shutil.copytree(PILOT, workspace, ignore=shutil.ignore_patterns("__pycache__"))
    render_workspace(PILOT, workspace, task_id=PILOT_ID, target_language="triton")
    spec = load_task_spec(PILOT / "task.yaml", task_id=PILOT_ID, target_language="triton")
    state = run / ".task-sessions" / workspace.name
    session = TaskSession.create(spec, workspace, state, LOGGER)
    with mock.patch("src.task_runtime._runtime_identity", return_value=RUNTIME):
        bind_session_runtime(session)
    kernel = workspace / "solution/kernel.py"
    kernel.write_text(kernel.read_text() + "\n# tuned by the agent\n")
    (workspace / "task_result.yaml").write_text(yaml.safe_dump({
        "task_name": PILOT_ID, "candidate_accepted": accepted, "speedup_ratio": 1.5, "score": 0.3,
        "evaluated_candidate_sources": session.candidate_source_evidence()["sources"]}))
    (state / "completion.json").write_text(json.dumps(_completion_record(session, "codex")))
    return run, workspace


class FakeEvaluator:
    """Answers each action from the contract in the workspace it runs in."""

    def __init__(self, *, fail=(), times=None, during=None):
        self.fail = set(fail)
        self.times = times or {"baseline": 2.0, "candidate": 1.0}
        self.during = during or {}
        self.calls = []
        self.contracts = []

    def __call__(self, spec, workspace, *, role, action, phase, manifest=None, logger=None):
        contract = json.loads((workspace / "_aka_eval/contract.json").read_text())
        self.calls.append((role, action, phase))
        self.contracts.append(contract)
        if (role, action) in self.during:
            self.during[(role, action)](workspace)
        rows = []
        for case in contract["cases"]:
            if action == "compile" or (action != "validate-task" and action not in case["checks"]):
                continue
            row = {"test_case_id": case["id"], "status": "PASS",
                   "params": {"axes": case["axes"], "scalars": case["scalars"]}}
            if action == "validate-task":
                row["checks"] = case["checks"]
            if action == "performance":
                row.update(execution_time_ms=self.times[role], benchmark_method="cuda_graph")
            rows.append(row)
        failed = (role, action) in self.fail
        result = ActionResult(role, action, "FAIL" if failed else "PASS", tuple(rows),
                              "fixture failure" if failed else None)
        return ExecutedAction(f"invocation-{len(self.calls)}", result, ())


class HeldOutInputTests(unittest.TestCase):
    def test_seeds_depend_on_the_salt_and_never_reuse_the_run_seeds(self):
        seeds = heldout.heldout_seeds(SALT, PILOT_ID, "t2048")
        self.assertEqual(seeds, heldout.heldout_seeds(SALT, PILOT_ID, "t2048"))
        self.assertNotEqual(seeds, heldout.heldout_seeds("fedcba9876543210", PILOT_ID, "t2048"))
        self.assertFalse(set(seeds) & set(case_seeds(PILOT_ID, "t2048")))
        self.assertTrue(all(0 <= seed < 2 ** 31 for seed in seeds))

    def test_contract_gets_new_seeds_for_every_case_and_validated_extra_cases(self):
        original = _contract()
        extra = [{"id": "h777", "axes": {"n_tokens": 777}, "checks": ["correctness"], "source": "held out"},
                 {"id": "h65536", "axes": {"n_tokens": 65536}, "scalars": {"epsilon": 1e-5},
                  "checks": ["correctness", "performance"]}]
        contract = heldout.heldout_contract(original, salt=SALT, extra_cases=extra)
        self.assertEqual([c["id"] for c in contract["cases"]],
                         [c["id"] for c in original["cases"]] + ["h777", "h65536"])
        old_seeds = {seed for case in original["cases"] for seed in case["seeds"]}
        for case in contract["cases"]:
            self.assertEqual(case["seeds"], heldout.heldout_seeds(SALT, PILOT_ID, case["id"]))
            self.assertFalse(set(case["seeds"]) & old_seeds)
        self.assertEqual(contract["cases"][-1]["scalars"], {"epsilon": 1e-5})
        self.assertNotIn("source", contract["cases"][-2])
        self.assertEqual({k: v for k, v in contract.items() if k != "cases"},
                         {k: v for k, v in original.items() if k != "cases"})
        self.assertEqual(len(original["cases"]), 9)  # The input is never modified.

    def test_extra_cases_outside_the_task_domain_are_rejected(self):
        bad = {
            "below minimum": {"id": "h", "axes": {"n_tokens": 0}, "checks": ["correctness"]},
            "above maximum": {"id": "h", "axes": {"n_tokens": 262145}, "checks": ["correctness"]},
            "not an int": {"id": "h", "axes": {"n_tokens": True}, "checks": ["correctness"]},
            "const axis": {"id": "h", "axes": {"n_tokens": 5, "hidden_size": 8192}, "checks": ["correctness"]},
            "missing axis": {"id": "h", "axes": {}, "checks": ["correctness"]},
            "unknown scalar": {"id": "h", "axes": {"n_tokens": 5}, "scalars": {"eps": 1.0},
                               "checks": ["correctness"]},
            "bad scalar": {"id": "h", "axes": {"n_tokens": 5}, "scalars": {"epsilon": "small"},
                           "checks": ["correctness"]},
            "bad checks": {"id": "h", "axes": {"n_tokens": 5}, "checks": ["performance"]},
            "existing id": {"id": "t2048", "axes": {"n_tokens": 5}, "checks": ["correctness"]},
            "existing inputs": {"id": "h", "axes": {"n_tokens": 2048}, "checks": ["correctness"]},
            "chosen seeds": {"id": "h", "axes": {"n_tokens": 5}, "checks": ["correctness"], "seeds": [1, 2]},
        }
        for name, case in bad.items():
            with self.subTest(name), self.assertRaises(ValueError):
                heldout.heldout_contract(_contract(), salt=SALT, extra_cases=[case])
        with self.assertRaises(ValueError):
            heldout.heldout_contract(_contract(), salt=SALT, extra_cases=[
                {"id": "a", "axes": {"n_tokens": 5}, "checks": ["correctness"]},
                {"id": "b", "axes": {"n_tokens": 5}, "checks": ["correctness"]}])

    def test_generalization_quadrants(self):
        self.assertEqual(heldout.classify(True, True), "both_pass")
        self.assertEqual(heldout.classify(True, False), "candidate_regression")
        self.assertEqual(heldout.classify(False, True), "candidate_improvement")
        self.assertEqual(heldout.classify(False, False), "both_fail")


class HeldOutEvaluationTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self.root = Path(self._directory.name).resolve()
        runtime = mock.patch("src.task_runtime._runtime_identity", return_value=RUNTIME)
        runtime.start()
        self.addCleanup(runtime.stop)
        self.addCleanup(self._directory.cleanup)

    def evaluate(self, evaluator, *, accepted=True, cases=None, change=None):
        run, workspace = completed_run(self.root, accepted=accepted)
        if change:
            change(workspace)
        cases_file = None
        if cases is not None:
            cases_file = self.root / "cases.json"
            cases_file.write_text(json.dumps({"cases": cases}))
        before = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        output = self.root / "out" / workspace.name
        with mock.patch.object(heldout, "run_action", evaluator):
            result = heldout.evaluate_task(workspace, output, salt=SALT, cases_file=cases_file, logger=LOGGER)
        after = {p.relative_to(run).as_posix(): p.read_bytes() for p in run.rglob("*") if p.is_file()}
        self.assertEqual(before, after, "held-out evaluation must not write into the original run")
        self.assertEqual(yaml.safe_load((output / "heldout_task_result.yaml").read_text()), result)
        return result, output, workspace

    def test_accepted_candidate_runs_on_heldout_inputs_with_interleaved_timing(self):
        evaluator = FakeEvaluator()
        result, output, _ = self.evaluate(
            evaluator, cases=[{"id": "h777", "axes": {"n_tokens": 777}, "checks": ["correctness"]}])
        self.assertIsNone(result["error"])
        self.assertEqual(result["generalization_status"], "both_pass")
        self.assertEqual(result["speedup_ratio"], 2.0)
        self.assertGreater(result["score"], 0)
        self.assertEqual(result["extra_case_ids"], ["h777"])
        self.assertEqual((result["salt"], result["runtime_identity"]), (SALT, RUNTIME))
        self.assertEqual(result["original_run_speedup_ratio"], 1.5)
        self.assertEqual([call[:2] for call in evaluator.calls], [
            ("task", "validate-task"), ("baseline", "compile"), ("baseline", "correctness"),
            ("candidate", "compile"), ("candidate", "correctness"), ("baseline", "performance"),
            ("candidate", "performance"), ("baseline", "performance"), ("candidate", "performance")])
        self.assertEqual(len(list((output / "actions").glob("*.json"))), 9)
        contract = evaluator.contracts[0]
        self.assertIn("h777", [case["id"] for case in contract["cases"]])
        original = {case["id"]: case["seeds"] for case in _contract()["cases"]}
        self.assertTrue(all(case["seeds"] != original.get(case["id"]) for case in contract["cases"]))
        held = output / "workspace"
        self.assertIn("# tuned by the agent", (held / "solution/kernel.py").read_text())
        self.assertEqual((held / "baseline/kernel.py").read_bytes(), (PILOT / "baseline/kernel.py").read_bytes())
        self.assertFalse((held / "task_result.yaml").exists())

    def test_candidate_failure_is_a_regression_without_a_speedup(self):
        evaluator = FakeEvaluator(fail={("candidate", "correctness")})
        result, _, _ = self.evaluate(evaluator)
        self.assertEqual(result["generalization_status"], "candidate_regression")
        self.assertEqual((result["speedup_ratio"], result["score"]), (0.0, 0.0))
        self.assertIn("candidate correctness", result["error"])
        self.assertNotIn("performance", [call[1] for call in evaluator.calls])

    def test_baseline_failure_on_heldout_inputs_is_reported_as_such(self):
        result, _, _ = self.evaluate(FakeEvaluator(fail={("baseline", "correctness")}))
        self.assertEqual(result["generalization_status"], "candidate_improvement")
        self.assertEqual(result["speedup_ratio"], 0.0)

    def test_failed_timing_round_gives_no_speedup(self):
        result, _, _ = self.evaluate(FakeEvaluator(fail={("candidate", "performance")}))
        self.assertEqual(result["generalization_status"], "both_pass")
        self.assertEqual((result["speedup_ratio"], result["score"]), (0.0, 0.0))
        self.assertIn("candidate performance", result["error"])

    def test_missing_evidence_and_protocol_violations_are_evaluation_errors(self):
        def edit_candidate(workspace):
            (workspace / "solution/kernel.py").write_text("def run(**_): pass\n")

        def rewrite_contract(workspace):
            path = workspace / "_aka_eval/contract.json"
            path.write_text(path.read_text().replace('"seeds"', '"seeds" '))

        def edit_candidate_during_compile(workspace):
            (workspace / "solution/kernel.py").write_text("def run(**_): pass\n")

        def omit_case(spec, workspace, **kwargs):
            executed = FakeEvaluator()(spec, workspace, **kwargs)
            result = executed.result
            return ExecutedAction(executed.invocation_id, ActionResult(
                result.role, result.action, result.status, result.cases[1:]), ())

        scenarios = {
            "not accepted": dict(evaluator=FakeEvaluator(), accepted=False, match="accepted"),
            "candidate changed after scoring": dict(evaluator=FakeEvaluator(), change=edit_candidate,
                                                    match="completion"),
            "invalid extra case": dict(evaluator=FakeEvaluator(), match="domain",
                                       cases=[{"id": "h", "axes": {"n_tokens": 0}, "checks": ["correctness"]}]),
            "contract rewritten by an action": dict(
                evaluator=FakeEvaluator(during={("candidate", "compile"): rewrite_contract}), match="contract.json"),
            "candidate rewritten by an action": dict(
                evaluator=FakeEvaluator(during={("baseline", "compile"): edit_candidate_during_compile}),
                match="changed the candidate"),
            "evaluator ignores the held-out cases": dict(evaluator=omit_case, match="held-out contract"),
        }
        for name, scenario in scenarios.items():
            with self.subTest(name):
                shutil.rmtree(self.root / "run_20261105_000000", ignore_errors=True)
                shutil.rmtree(self.root / "out", ignore_errors=True)
                match = scenario.pop("match")
                result, _, _ = self.evaluate(scenario.pop("evaluator"), **scenario)
                self.assertEqual(result["generalization_status"], "evaluation_error")
                self.assertEqual((result["speedup_ratio"], result["score"]), (0.0, 0.0))
                self.assertIn(match, result["error"])

    def test_a_changed_scoring_runtime_is_an_evaluation_error(self):
        run, workspace = completed_run(self.root)
        with mock.patch("src.task_runtime._runtime_identity", return_value={**RUNTIME, "torch": "another build"}), \
                mock.patch.object(heldout, "run_action", FakeEvaluator()):
            result = heldout.evaluate_task(workspace, self.root / "out", salt=SALT, logger=LOGGER)
        self.assertEqual(result["generalization_status"], "evaluation_error")
        self.assertIn("runtime", result["error"])

    def test_command_line_evaluates_every_v3_workspace_and_never_reuses_output(self):
        run, workspace = completed_run(self.root)
        cases = self.root / "cases" / PILOT_ID
        cases.mkdir(parents=True)
        (cases / heldout.CASES_FILE).write_text(json.dumps(
            {"cases": [{"id": "h5", "axes": {"n_tokens": 5}, "checks": ["correctness"]}]}))
        with mock.patch.object(heldout, "run_action", FakeEvaluator()):
            code = heldout.main(["--run-dir", str(run), "--salt", SALT, "--cases-dir", str(self.root / "cases")])
        self.assertEqual(code, 0)
        output = run.parent / (run.name + "_heldout")
        summary = yaml.safe_load((output / "heldout_summary.yaml").read_text())
        self.assertEqual(summary["salt"], SALT)
        self.assertEqual(summary["quadrant_counts"]["both_pass"], 1)
        self.assertEqual(summary["conditional_correctness"]["rate_pct"], 100.0)
        self.assertEqual(summary["heldout_speedup"]["geomean"], 2.0)
        task = yaml.safe_load((output / workspace.name / "heldout_task_result.yaml").read_text())
        self.assertEqual(task["extra_case_ids"], ["h5"])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            heldout.main(["--run-dir", str(run)])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            heldout.main(["--run-dir", str(run), "--output-suffix", "_other", "--salt", "not-hex"])


if __name__ == "__main__":
    unittest.main()
