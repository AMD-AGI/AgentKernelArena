"""CPU regressions for candidate host-execution and evaluator-state boundaries.

These are structural checks on protected task code; they do not run kernels or
qualify any task on a GPU.
"""
import ast
import hashlib
import importlib.util
import inspect
from pathlib import Path
import shutil
import sys
import time
import types

import pytest
from tests.task_repair_history import reviewed_repair_source

from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec


ROOT = Path(__file__).resolve().parents[1]
# Every FlyDSL task runtime, selected by family rather than by content, so a
# runtime without the closure and evaluator-state checks fails these tests.
FLYDSL_CLOSURE_TASKS = sorted(
    path.parent for family in ("flydsl2flydsl", "torch2flydsl", "triton2flydsl")
    for path in (ROOT / "tasks" / family).rglob("task_runtime.py"))
FLYDSL = "import flydsl\nimport flydsl.compiler as flyc\n"


def _copy_task(tmp_path, relative):
    root = tmp_path / Path(relative).name
    shutil.copytree(ROOT / "tasks" / relative, root)
    return root, load_task_spec(root / "config.yaml", task_id=relative)


def _guard_result(root, spec, source, text):
    snapshot = snapshot_workspace_harness(root, task_root=root, task_spec=spec)
    original = source.read_text()
    source.write_text(text)
    try:
        verify_workspace_harness(snapshot)
        return None
    except RuntimeError as exc:
        return str(exc)
    finally:
        source.write_text(original)


BMM_ANCHOR = "@triton.jit\ndef bmm_kernel"
HOST_HOOK = 'def _hook(a):\n    a["c_ptr"].copy_(torch.bmm(a["a_ptr"], a["b_ptr"]))\n\n\n'


@pytest.mark.parametrize("helpers,decorator,message", [
    ("", '@triton.heuristics({"H": lambda a: (a["c_ptr"].copy_(torch.bmm(a["a_ptr"], a["b_ptr"])), 0)[1]})\n',
     "launch-time host computation"),
    ("", '@triton.autotune(configs=[triton.Config({}, pre_hook=lambda a: a["c_ptr"].zero_())], key=[])\n',
     "Host callback 'pre_hook'"),
    (HOST_HOOK, '@triton.autotune(configs=[triton.Config({}, pre_hook=_hook)], key=[])\n',
     "Host callback 'pre_hook'"),
    ("", '@triton.autotune(configs=[triton.Config({})], key=[], prune_configs_by={"early_config_prune": lambda c, a, **k: c})\n',
     "Host callback 'prune_configs_by'"),
    ('def _h(a):\n    return 0\n\n\n', '@triton.heuristics({"H": _h})\n', "literal lambdas"),
])
def test_editable_triton_decorators_cannot_run_host_code(tmp_path, helpers, decorator, message):
    root, spec = _copy_task(tmp_path, "triton2triton/vllm/triton_bmm")
    source = root / "source/triton_bmm.py"
    text = source.read_text()
    assert BMM_ANCHOR in text
    error = _guard_result(root, spec, source, text.replace(BMM_ANCHOR, helpers + decorator + BMM_ANCHOR, 1))
    assert error is not None and message in error


def test_editable_triton_launch_metadata_callback_is_rejected(tmp_path):
    root, spec = _copy_task(tmp_path, "triton2triton/vllm/triton_bmm")
    source = root / "source/triton_bmm.py"
    text = source.read_text().replace(BMM_ANCHOR, "@triton.jit(launch_metadata=lambda g, k, a: {})\ndef bmm_kernel", 1)
    assert "Host callback 'launch_metadata'" in _guard_result(root, spec, source, text)


@pytest.mark.parametrize("decorator", [
    '@triton.heuristics({"EVEN": lambda a: a["K"] % 16 == 0, "NB": lambda a: max(1, triton.cdiv(a["K"], 64))})\n',
    '@triton.autotune(configs=[triton.Config({}, num_warps=4), triton.Config({}, num_warps=8)], key=["M"])\n',
])
def test_editable_triton_launch_metadata_decorators_remain_allowed(tmp_path, decorator):
    root, spec = _copy_task(tmp_path, "triton2triton/vllm/triton_bmm")
    source = root / "source/triton_bmm.py"
    text = source.read_text()
    assert _guard_result(root, spec, source, text.replace(BMM_ANCHOR, decorator + BMM_ANCHOR, 1)) is None


def test_editable_config_factory_cannot_add_host_callbacks(tmp_path):
    relative = "triton2triton/rocmbench/hard/multreduce_matmul_dot_kernel"
    root, spec = _copy_task(tmp_path, relative)
    source = root / "multreduce_matmul_dot_kernel.py"
    text = source.read_text()
    start = text.index("triton.Config(", text.index("def get_triton_dot_autotune_configs")) + len("triton.Config(")
    hooked = text[:start] + "pre_hook=None, kwargs=" + text[start:]
    assert "Unapproved definition-time factory" in _guard_result(root, spec, source, hooked)
    tuned = text.replace("num_warps_range: list[int] = [1, 2]", "num_warps_range: list[int] = [1, 2, 4]", 1)
    assert tuned != text and _guard_result(root, spec, source, tuned) is None


def _runtime(tmp_path, task):
    root = tmp_path / task.name
    shutil.copytree(task, root)
    (root / "_aka_benchmark.py").write_text((ROOT / "src/tools/perf/aka_benchmark.py").read_text())
    spec = importlib.util.spec_from_file_location(f"runtime_{task.name}", root / "task_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return root, module


def _candidate(root, body):
    path = root / "kernel.py"
    path.write_text(FLYDSL + body)
    return path


def test_closure_runtime_copies_are_identical():
    assert len(FLYDSL_CLOSURE_TASKS) == 111
    for name in ("candidate_source_closure", "check_candidate_control_plane"):
        bodies = set()
        for task in FLYDSL_CLOSURE_TASKS:
            source = (task / "task_runtime.py").read_text()
            node = next(n for n in ast.parse(source).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
            bodies.add(ast.get_source_segment(source, node))
        assert len(bodies) == 1, name


EVALUATOR_STATE_ROUTES = {
    "timing helper import": "import _aka_benchmark\n_aka_benchmark._event_elapsed_ms = lambda s, e: 0.0\n",
    "timing helper from-import": "from _aka_benchmark import _event_elapsed_ms\n",
    "module registry through os": "import os\nloaded = os.sys.modules\n",
    "os alias to sys": "import os\nx = os\nx.sys.modules\n",
    "sys from os": "from os import sys as s\ns.modules\n",
    "builtins import": "import builtins\nbuiltins.__import__('sys').modules\n",
    "caller frame globals": "import inspect\ninspect.currentframe().f_back.f_globals['TOL'] = 1\n",
    "function globals": "def h():\n    pass\nh.__globals__['TOL'] = 1\n",
    "dispatch mode disable": "import torch\nwith torch.utils._python_dispatch._disable_current_modes():\n    pass\n",
    "dispatch mode import": "from torch.utils._python_dispatch import _disable_current_modes\n",
    "event timer mutation": "import torch\ntorch.cuda.Event.elapsed_time = lambda s, e: 0.0\n",
    "event timer setattr": "import torch\nsetattr(torch.cuda.Event, 'elapsed_time', None)\n",
    "reassigned sys alias": "import sys\nlog = sys.stdout\nlog = sys.stderr\nlog = sys\nlog.modules\n",
}


@pytest.mark.parametrize("task", FLYDSL_CLOSURE_TASKS, ids=lambda p: p.name)
def test_flydsl_candidates_cannot_reach_evaluator_state(tmp_path, task):
    root, runtime = _runtime(tmp_path, task)
    for label, body in EVALUATOR_STATE_ROUTES.items():
        with pytest.raises(ValueError):
            runtime.check_dependencies([_candidate(root, body)], final_language=True)
            pytest.fail(label)


@pytest.mark.parametrize("task", FLYDSL_CLOSURE_TASKS, ids=lambda p: p.name)
def test_flydsl_ordinary_candidate_python_remains_allowed(tmp_path, task):
    root, runtime = _runtime(tmp_path, task)
    for body in ("import sys\nlimit = sys.maxsize\n",
                 "import sys\nlog = sys.stdout\nlog = sys.stderr\nlog.write('x')\n",
                 "import torch\ny = torch.stack([torch.zeros(1)])\nout = torch.empty(2)\nout[0] = 1\n",
                 "import os\nvalue = os.environ.get('X')\npath = os.path.join('a', 'b')\n"):
        runtime.check_dependencies([_candidate(root, body)], final_language=True)


@pytest.mark.parametrize("task", FLYDSL_CLOSURE_TASKS[:1] + FLYDSL_CLOSURE_TASKS[-1:], ids=lambda p: p.name)
def test_reassigned_sys_alias_analysis_is_linear(tmp_path, task):
    root, runtime = _runtime(tmp_path, task)
    body = "import sys\n" + "".join(f"log = sys.stdout\nlog = sys.stderr\nx{i} = 1 + 2 * 3\n"
                                    for i in range(1500))
    started = time.perf_counter()
    runtime.check_dependencies([_candidate(root, body)], final_language=True)
    assert time.perf_counter() - started < 10


@pytest.mark.parametrize("task", FLYDSL_CLOSURE_TASKS, ids=lambda p: p.name)
def test_same_name_module_precedes_namespace_directory(tmp_path, task):
    root, runtime = _runtime(tmp_path, task)
    (root / "device_helper.py").write_text("import flydsl\nTILE = 32\n")
    (root / "device_helper").mkdir()
    (root / "device_helper/TILE.py").write_text("import aiter\n")
    candidate = _candidate(root, "from device_helper import TILE\n")
    runtime.check_dependencies([candidate], final_language=True)
    assert root / "device_helper.py" in runtime.candidate_source_closure([candidate])
    # A regular package still wins, exactly as in Python's path finder.
    (root / "device_helper/__init__.py").write_text("")
    with pytest.raises(ValueError, match="aiter"):
        runtime.check_dependencies([candidate], final_language=True)


@pytest.mark.parametrize("label", sorted(EVALUATOR_STATE_ROUTES))
def test_each_evaluator_state_route_is_rejected_by_every_flydsl_runtime(tmp_path, label):
    accepted = []
    for index, task in enumerate(FLYDSL_CLOSURE_TASKS):
        root, runtime = _runtime(tmp_path / str(index), task)
        try:
            runtime.check_dependencies([_candidate(root, EVALUATOR_STATE_ROUTES[label])], final_language=True)
        except ValueError:
            continue
        accepted.append(task.relative_to(ROOT / "tasks").as_posix())
    assert not accepted, (label, accepted)


@pytest.mark.parametrize("task", FLYDSL_CLOSURE_TASKS, ids=lambda p: p.name)
def test_shipped_initial_sources_pass_the_evaluator_state_checks(task):
    # Bundled helpers are part of the import closure, so the shipped initial
    # implementation must satisfy the same checks as a final candidate.
    spec = importlib.util.spec_from_file_location(f"shipped_{task.name}", task / "task_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    cfg = runtime.config()
    if cfg["baseline"]["kind"] == "provided":
        pytest.skip("provided baselines are not candidate sources")
    runtime.check_dependencies(runtime.candidate_files(cfg),
                               final_language=cfg["candidate"].get("initial_language") == "flydsl")


COMPAT_META = sorted(path for family in ("flydsl2flydsl", "torch2flydsl", "triton2flydsl")
                     for path in (ROOT / "tasks" / family).rglob("flydsl_compat/meta.py"))


def test_vendored_caller_location_selects_the_original_frame(monkeypatch):
    class Location:
        def __init__(self, *value):
            self.value = value

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        @staticmethod
        def unknown():
            return Location("unknown")

        @staticmethod
        def file(*args):
            return Location("file", *args)

        @staticmethod
        def name(label, childLoc=None):
            return Location("name", label, childLoc.value)

    # Only the MLIR Location API is stubbed; frame selection is pure Python.
    mlir = types.ModuleType("flydsl._mlir")
    mlir.ir = types.SimpleNamespace(Location=Location, Value=type("Value", (), {}))
    monkeypatch.setitem(sys.modules, "flydsl", types.ModuleType("flydsl"))
    monkeypatch.setitem(sys.modules, "flydsl._mlir", mlir)
    # Main added an immutable native compatibility variant with column-level
    # locations. It is outside the line-only migration exercised below; pin its
    # original bytes rather than requiring unrelated APIs to be identical.
    native = ROOT / "tasks/torch2flydsl/moe_a8w4_kernel/flydsl_compat/meta.py"
    assert native in COMPAT_META
    assert hashlib.sha256(native.read_bytes()).hexdigest() == (
        "a54bb32cbdc7a99d05e5aec617fb6c71f500f50286d44c0c0a8f1359cc49e92e"
    )
    line_meta = [path for path in COMPAT_META if path != native]
    assert len(line_meta) == 13 and len({path.read_bytes() for path in line_meta}) == 1
    source = line_meta[0].read_text()
    assert "import inspect" not in source and ".f_back" not in source

    def load(name, text):
        module = types.ModuleType(name)
        exec(compile(text, str(COMPAT_META[0]), "exec"), module.__dict__)
        return module

    current = load("compat_meta_current", source)

    def original_location(depth=1):
        # Independent reference: the original frame walk, with the declared
        # portable line-level column policy applied to both implementations.
        frame = inspect.currentframe()
        for _ in range(depth + 1):
            if frame is not None:
                frame = frame.f_back
        if frame is None:
            return Location.unknown()
        info = inspect.getframeinfo(frame)
        line = info.positions.lineno if hasattr(info, "positions") else info.lineno
        label = " ".join(ln.strip() for ln in info.code_context) if info.code_context else info.function
        return Location.name(label, childLoc=Location.file(info.filename, line, 0))

    def site(depth):
        return current._caller_location(depth=depth).value, original_location(depth=depth).value

    for depth in (1, 2, 3, 10_000):
        selected, expected = site(depth)
        assert selected == expected
    assert site(10_000)[0] == ("unknown",)
    line = inspect.currentframe().f_lineno + 1
    selected, _ = site(1)
    assert selected[0] == "name" and selected[1] == "selected, _ = site(1)"
    assert selected[2][1:3] == (__file__, line)
    # traced_op still records its caller's location and runs the operation.
    calls = []
    op = current.traced_op(lambda *args, **kwargs: calls.append((args, kwargs)) or "ok")
    assert op(1, flag=True) == "ok" and calls == [((1,), {"flag": True})]


def test_gemm_a8w8_audits_actual_performance_calls(tmp_path):
    task = ROOT / "tasks/triton2flydsl/aiter/gemm_a8w8"
    actions = (task / "scripts/task_actions.py").read_text()
    performance = next(n for n in ast.parse(actions).body
                       if isinstance(n, ast.FunctionDef) and n.name == "performance")
    assert "audit_candidate_calls(h, performance=True)" in ast.get_source_segment(actions, performance)
    checks = (task / "scripts/candidate_checks.py").read_text()
    sibling = (ROOT / "tasks/triton2flydsl/sglang/lightning_attn/scripts/candidate_checks.py").read_text()
    historical = reviewed_repair_source(checks, path=task / "scripts/candidate_checks.py")
    assert historical == sibling.replace("h.load_module", "h._load_source")
    assert "def _load_source" in (task / "test_kernel_harness.py").read_text()
    root = tmp_path / "gemm_a8w8"
    shutil.copytree(task, root)
    spec = importlib.util.spec_from_file_location("gemm_a8w8_runtime", root / "task_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    for body in ("import torch\nout = torch._scaled_mm(a, b)\n", "import _aka_benchmark\n"):
        with pytest.raises(ValueError):
            runtime.check_dependencies([_candidate(root, body)], final_language=True)
