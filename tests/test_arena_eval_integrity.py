"""The evaluator detects candidate code that rebinds its timing and judging objects."""
import gc
import sys

import pytest

torch = pytest.importorskip("torch")

from src.tools.arena_eval import audit, evaluate, integrity  # noqa: E402

PACKAGE = "src.tools.arena_eval"


def _tampered(check, stage="test"):
    with pytest.raises(integrity.TamperError, match="Harness tampering detected after test"):
        check(stage)


def test_an_untouched_process_passes_and_lazy_attributes_are_allowed(monkeypatch):
    check = integrity.snapshot(PACKAGE)
    check("test")
    monkeypatch.setattr(torch, "_aka_lazily_added_submodule", object(), raising=False)
    check("test")


def test_the_reflective_route_of_the_reported_exploit_is_detected():
    check = integrity.snapshot(PACKAGE)
    original = evaluate._report
    # The exploit reached evaluator modules through torch's own imports, which
    # the static audit does not see: torch.sys.modules and torch.builtins.setattr.
    for name, module in list(torch.sys.modules.items()):
        if name == f"{PACKAGE}.evaluate":
            torch.builtins.setattr(module, "_report", lambda *a, **k: None)
    try:
        _tampered(check)
    finally:
        evaluate._report = original
    check("test")


def test_rebinding_a_torch_timing_method_is_detected(monkeypatch):
    check = integrity.snapshot(PACKAGE)
    monkeypatch.setattr(torch.cuda.Event, "elapsed_time", lambda self, end: 1e-4)
    _tampered(check)


def test_replacing_an_evaluator_module_in_sys_modules_is_detected(monkeypatch):
    check = integrity.snapshot(PACKAGE)
    monkeypatch.setitem(sys.modules, f"{PACKAGE}.compare", type(sys)("fake"))
    _tampered(check)


def test_interpreter_hooks_are_detected():
    check = integrity.snapshot(PACKAGE)
    previous = sys.getprofile()
    sys.setprofile(lambda *args: None)
    try:
        _tampered(check)
    finally:
        sys.setprofile(previous)
    callback = lambda *args: None  # noqa: E731
    gc.callbacks.append(callback)
    try:
        _tampered(check)
    finally:
        gc.callbacks.remove(callback)
    check("test")


def test_the_flush_cache_may_only_hold_none_or_the_flush_class(monkeypatch):
    check = integrity.snapshot(PACKAGE)
    monkeypatch.setattr(evaluate, "_FLUSH", object())
    # Without the workspace's _aka_benchmark there is no flush class, so only None passes.
    _tampered(check)
    monkeypatch.setattr(evaluate, "_FLUSH", None)
    check("test")


def test_rebinding_the_guard_after_the_snapshot_does_not_disable_it(monkeypatch):
    check = integrity.snapshot(PACKAGE)
    error = integrity.TamperError
    monkeypatch.setattr(integrity.HarnessIntegrity, "changes", lambda self, **kw: [])
    monkeypatch.setattr(integrity, "_hook_state", lambda: {})
    monkeypatch.setattr(integrity, "TamperError", ValueError)
    with pytest.raises(error, match="HarnessIntegrity.changes was rebound"):
        check("test")


def test_audited_call_checks_integrity_after_every_call():
    stages = []
    call = audit.audited_call(lambda **kw: None, language="triton", package="x", role="candidate",
                              enforce=False, integrity=stages.append)
    call()
    call()
    assert stages == ["a candidate call", "a candidate call"]


def test_a_tamper_error_fails_the_whole_action_as_harness_tampering(monkeypatch):
    def tamper(contract, role):
        raise integrity.TamperError("Harness tampering detected after a candidate call: [...]")

    monkeypatch.setattr(evaluate, "load_contract", lambda: object())
    monkeypatch.setattr(evaluate, "performance_action", tamper)
    report, code = evaluate.run(["candidate", "performance"])
    assert code == 1
    assert report["status"] == "FAIL"
    assert report["failure_kind"] == "harness_tampering"


def test_pytorchs_own_lazy_dynamo_import_is_not_tampering():
    """torch._dynamo rebinds torch.manual_seed when first imported, e.g. by a kernel language's first launch."""
    import subprocess
    from pathlib import Path

    code = (
        "import sys, torch\n"
        "assert 'torch._dynamo' not in sys.modules\n"
        f"from {PACKAGE} import integrity\n"
        f"check = integrity.snapshot({PACKAGE!r})\n"
        "import torch._dynamo, torch._inductor\n"
        "check('a candidate call')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-2000:]


def test_a_tamper_error_is_not_an_ordinary_exception():
    assert issubclass(integrity.TamperError, BaseException)
    assert not issubclass(integrity.TamperError, Exception)


def test_tampering_found_during_a_timing_sample_survives_the_timing_helper(monkeypatch):
    """The real timing helper wraps any Exception from its sample loop as a capture failure."""
    import types

    class TimedRun:
        after_sample = None

    def benchmark(fn, *, cache_flush, warmup, repetition, prepare_fn, timed_run):
        try:
            prepare_fn()
            fn()
            timed_run.after_sample(None)
            return [0.1], {"benchmark_method": "cuda_graph_cold_cache"}
        except Exception as exc:
            raise RuntimeError("CUDA-graph capture failed; timed_run cannot validate the fallback") from exc

    helper = types.ModuleType(f"{PACKAGE}._aka_benchmark")
    helper.TimedRun, helper.benchmark_cuda_graph_cold_cache_samples = TimedRun, benchmark
    monkeypatch.setitem(sys.modules, helper.__name__, helper)

    def check(stage):
        if stage == "a timing sample":
            raise integrity.TamperError(f"Harness tampering detected after {stage}: [...]")

    case = types.SimpleNamespace(checks=("correctness", "performance"), identity=lambda: {"test_case_id": "c0"})
    contract = types.SimpleNamespace(cases=[case], timing={})
    data = types.SimpleNamespace(inputs=[types.SimpleNamespace(kernel={})] * 2, references=[{}] * 2,
                                 baselines=None)
    monkeypatch.setattr(evaluate, "load_contract", lambda: contract)
    monkeypatch.setattr(evaluate, "_device", lambda: "cpu")
    monkeypatch.setattr(evaluate, "_role_calls", lambda contract, role: (lambda **kw: None, None, False, check))
    monkeypatch.setattr(evaluate, "_cache_flush", lambda nbytes: None)
    monkeypatch.setattr(evaluate, "_CaseData", lambda *args: data)
    monkeypatch.setattr(evaluate, "_check", lambda *args: (True, {}))
    from src.tools.arena_eval import values
    for name in ("clone_inputs", "allocate_outputs", "checked_values"):
        monkeypatch.setattr(values, name, lambda *args, **kwargs: {})
    for name in ("copy_into", "poison"):
        monkeypatch.setattr(values, name, lambda *args, **kwargs: None)

    report, code = evaluate.run(["candidate", "performance"])
    assert code == 1
    assert report["failure_kind"] == "harness_tampering"
    assert "a timing sample" in report["reason"]
