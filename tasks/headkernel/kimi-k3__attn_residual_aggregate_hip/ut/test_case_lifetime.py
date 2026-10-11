"""CPU lifetime regression for full case traversal; no GPU allocations."""
import gc
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import pytest


TASK = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_aggregation_lifetime_runner", TASK / "scripts/task_runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("phase", ["compile", "correctness", "performance"])
def test_completed_case_storage_is_reclaimed_before_next_case(phase, tmp_path):
    manifest = json.loads((TASK / "cases.json").read_text())
    (tmp_path / "cases.json").write_text(json.dumps(manifest))
    policy = manifest["measurement"]
    probes, checks, captures, rows, collections = [], [], [], [], []
    active_work = False
    source_hashes = {"source/attn_res_hip.py": "synthetic-source"}

    class Storage:
        pass

    class Callbacks:
        def __init__(self, owner):
            self.owner = owner
            # Failed negative controls may leave a CPU truth token until the
            # case is released. Model that storage as well as device buffers.
            self.truth = Storage()
            probes.append(weakref.ref(self.truth))

        def check_once(self, seed):
            checks.append(self.owner.case["case_id"])

        def run_row(self, case, actual_policy, **kwargs):
            nonlocal active_work
            active_work = True
            try:
                assert actual_policy == policy
                rows.append(case["case_id"])
                return {"case": case, "correct": True}
            finally:
                active_work = False

        correctness_row = performance_row = run_row

    class Prepared:
        def __init__(self, case):
            self.case = case
            self.storage = Storage()
            self.callbacks = Callbacks(self)
            probes.extend((weakref.ref(self), weakref.ref(self.storage), weakref.ref(self.callbacks)))

        def invoke_candidate(self):
            pass

        def observe(self):
            return self.case

        def corrupt_outputs(self):
            pass

        def native_engagement(self):
            return {"source_sha256": source_hashes, "candidate_invoked": True,
                    "independent_reference_invoked": True}

    class Graph:
        def __init__(self, prepared):
            self.prepared = prepared
            probes.append(weakref.ref(self))

        def replay(self):
            pass

    def prepare(case):
        assert all(probe() is None for probe in probes), "previous case still owns storage"
        return Prepared(case)

    def capture(prepared, callbacks, seed, torch):
        captures.append(prepared.case["case_id"])
        graph = Graph(prepared)
        callbacks.replay = graph.replay
        return graph

    original_collect = gc.collect

    def collect():
        assert not active_work, "collection entered case measurement work"
        collections.append(len(checks))
        return original_collect()

    request = {"challenge_seed": 7, "source_sha256": source_hashes}
    torch = SimpleNamespace(version=SimpleNamespace(hip=True), cuda=SimpleNamespace(
        get_device_properties=lambda device: SimpleNamespace(gcnArchName="gfx950")))
    was_enabled = gc.isenabled()
    gc.disable()  # Reclamation must not depend on the process GC threshold.
    try:
        with patch.object(runner, "ROOT", tmp_path), patch.object(runner, "FreshCallbacks", Callbacks), \
                patch.object(runner, "validate_sources"), patch.object(runner, "request_for", return_value=request), \
                patch.object(runner, "load_runtime", return_value=SimpleNamespace(prepare_case=prepare)), \
                patch.object(runner, "capture", capture), patch.object(runner, "finalize_report", side_effect=lambda r, *args: r), \
                patch.object(gc, "collect", collect), patch.dict(sys.modules, torch=torch), \
                patch.object(sys, "argv", ["task_runner.py", phase]):
            runner.main()
        assert all(probe() is None for probe in probes), "final case still owns storage"
    finally:
        if was_enabled:
            gc.enable()
        original_collect()

    case_ids = [case["case_id"] for case in manifest["cases"]]
    assert checks == case_ids
    assert collections == list(range(1, len(case_ids) + 1))
    assert captures == ([] if phase == "compile" else case_ids)
    assert rows == ([] if phase == "compile" else case_ids)
    report = json.loads((tmp_path / "build" / (phase + "_report.json")).read_text())
    assert [case["case_id"] for case in report["compiled_specializations"]] == case_ids
