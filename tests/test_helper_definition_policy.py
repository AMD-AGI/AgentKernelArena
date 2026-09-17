"""Definition-time hooks cannot hide inside excluded implementation helpers."""
from pathlib import Path
import ast
import shutil

import pytest
import yaml

from src.harness_guard import describe_workspace_harness, snapshot_workspace_harness, verify_workspace_harness
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
IMPORTS = '''import triton
import triton as tr
from triton import jit as compile_kernel
import pytest
import pytest as pt
from pytest import fixture as fi
'''


def fixture_workspace(tmp_path):
    root = tmp_path / "task"; root.mkdir()
    source = root / "kernel.py"
    source.write_text(IMPORTS + "\n@triton.jit\ndef kernel(x): return x\n"
                      "def reference(): return 7\n"
                      "def get_configs(): return [triton.Config({'BLOCK': 128})]\n")
    (root / "runner.py").write_text("# protected runner\n")
    config = {"schema_version": 2, "candidate": {"language": "triton", "editable": [{
        "path": "kernel.py", "scope": "symbols", "symbols": ["kernel"], "allow_new_helpers": True,
    }], "entrypoints": [{"file": "kernel.py", "kind": "function", "symbol": "kernel"}]},
        "evaluation": {"runner": ["python3", "runner.py"]}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    return root, source, snapshot_workspace_harness(root)


@pytest.mark.parametrize("declaration", [
    "def helper(x): return x + 1\n",
    "async def helper(x): return x\n",
    "@triton.jit\ndef helper(x): return x\n",
    "@tr.jit\ndef helper(x): return x\n",
    "@compile_kernel\ndef helper(x): return x\n",
    "@triton.jit(do_not_specialize=['n'])\ndef helper(x, n): return x\n",
    "@triton.autotune(configs=[triton.Config({'BLOCK': 128}, num_warps=4)], key=['n'])\n"
    "@triton.jit\ndef helper(x, n): return x\n",
    "@triton.heuristics({'EVEN': lambda args: args['n'] % 128 == 0})\n"
    "@triton.jit\ndef helper(x, n): return x\n",
    "@triton.autotune(configs=get_configs(), key=['n'])\n@triton.jit\ndef helper(x, n): return x\n",
    "@triton.autotune(configs=[triton.Config({'BLOCK': n}) for n in range(32, 129, 32)], key=['n'])\n"
    "@triton.jit\ndef helper(x, n): return x\n",
    "@triton.heuristics({'EVEN': lambda args: args.get('n', 0) % 128 == 0})\n"
    "@triton.jit\ndef helper(x, n): return x\n",
    "class Tile:\n    size = 128\n    @staticmethod\n    def extent(): return 128\n",
    "class Tile(object):\n    def __init__(self, x=128): self.x = x\n"
    "    @property\n    def size(self): return self.x\n",
    "class Attributes:\n    def __getattr__(self, name): raise AttributeError(name)\n"
    "    def __dir__(self): return ['size']\n",
])
def test_normal_implementation_helpers_remain_editable(tmp_path, declaration):
    root, source, snapshot = fixture_workspace(tmp_path)
    source.write_text(source.read_text() + declaration)
    verify_workspace_harness(snapshot)
    info = describe_workspace_harness(root, snapshot=snapshot)["protected_path_policies"]["kernel.py"]
    assert info["definition_policy"] == "compiler_decorators_no_test_hooks_v1"
    assert info["allowed_compiler_decorators"] == ["triton.autotune", "triton.heuristics", "triton.jit"]


@pytest.mark.parametrize("declaration", [
    "@pytest.fixture(autouse=True)\ndef helper(monkeypatch): pass\n",
    "@pt.fixture(autouse=True)\ndef helper(monkeypatch): pass\n",
    "@fi(autouse=True)\ndef helper(monkeypatch): pass\n",
    "@getattr(pytest, 'fixture')(autouse=True)\ndef helper(): pass\n",
    "@globals()['fi'](autouse=True)\ndef helper(): pass\n",
    "def register(fn): return pytest.fixture(autouse=True)(fn)\n@register\ndef helper(): pass\n",
    "@triton.jit\n@pytest.fixture(autouse=True)\ndef helper(): pass\n",
    "@triton.jit(launch_metadata=pytest.fixture(autouse=True))\ndef helper(): pass\n",
    "def install(): return pytest.fixture(autouse=True)(lambda: None)\n"
    "@triton.jit(launch_metadata=install())\ndef helper(): pass\n",
    "@triton.jit(launch_metadata=(lambda: None)())\ndef helper(): pass\n",
    "def range(*args): return pytest.fixture(autouse=True)(lambda: None)\n"
    "@triton.autotune(configs=range(3), key=['n'])\n@triton.jit\ndef helper(x, n): return x\n",
    "def test_added_case(): pass\n",
    "def pytest_runtest_setup(item): pass\n",
    "def setup_module(module): pass\n",
    "def teardown_function(function): pass\n",
    "class TestAdded:\n    def test_skip(self): pass\n",
    "def helper(x=pytest.fixture(autouse=True)(lambda: None)): pass\n",
    "def helper(x: pytest.fixture(autouse=True)(lambda: None)): pass\n",
    "def helper() -> pytest.fixture(autouse=True)(lambda: None): pass\n",
    "class Helper:\n    pytest.fixture(autouse=True)(lambda: None)\n",
    "class Helper:\n    x = pytest.fixture(autouse=True)(lambda: None)\n",
    "class Helper(metaclass=Factory): pass\n",
    "class Helper(Factory()): pass\n",
    "class Helper:\n    @pytest.fixture(autouse=True)\n    def patch(self): pass\n",
    "def staticmethod(fn): return pytest.fixture(autouse=True)(fn)\n"
    "class Helper:\n    @staticmethod\n    def patch(): pass\n",
    "class object:\n    def __init_subclass__(cls): pytest.fixture(autouse=True)(lambda: None)\n"
    "class Helper(object): pass\n",
    "class Helper:\n    def staticmethod(fn): return pytest.fixture(autouse=True)(fn)\n"
    "    @staticmethod\n    def patch(): pass\n",
])
def test_test_environment_hooks_cannot_be_new_helpers(tmp_path, declaration):
    _, source, snapshot = fixture_workspace(tmp_path)
    source.write_text(source.read_text() + declaration)
    with pytest.raises(RuntimeError, match="Protected test/harness policy rejected"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("task", [
    "instruction2triton/rocmbench/moe_gemm", "triton2triton/rocmbench/hard/moe_gemm",
])
@pytest.mark.parametrize("name,args,return_statement", [
    ("__getattr__", "name", "raise AttributeError(name)"),
    ("__dir__", "", "return list(globals())"),
])
def test_actual_tasks_reject_automatic_module_inspection_hooks(tmp_path, task, name, args, return_statement):
    workspace = tmp_path / "task"
    shutil.copytree(ROOT / "tasks" / task, workspace)
    materialize_perf_helpers_in_workspace(workspace)
    snapshot = snapshot_workspace_harness(workspace)
    source = workspace / "moe_gemm.py"
    source.write_text(source.read_text() + f'''
def {name}({args}):
    import _arena_reference
    _arena_reference.compare = lambda *args, **kwargs: None
    {return_statement}
''')
    with pytest.raises(RuntimeError, match="automatic introspection hook"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("declaration", [
    "@pytest.fixture(autouse=True)\n@triton.jit\ndef kernel(x): return x\n",
    "@triton.jit\ndef kernel(x=pytest.fixture(autouse=True)(lambda: None)): return x\n",
])
def test_existing_editable_target_cannot_receive_a_fixture_header(tmp_path, declaration):
    _, source, snapshot = fixture_workspace(tmp_path)
    source.write_text(IMPORTS + declaration + "def reference(): return 7\n")
    with pytest.raises(RuntimeError, match="Protected test/harness policy rejected"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("outer", ["host_launcher", "host_launcher()", "getattr(host_launcher, '__call__')"])
def test_existing_jit_target_cannot_be_replaced_by_a_host_decorator(tmp_path, outer):
    root, source, _ = fixture_workspace(tmp_path)
    wrapper = "\ndef protected_wrapper(x): return kernel[(1,)](x)\n"
    original = source.read_text() + wrapper
    source.write_text(original)
    snapshot = snapshot_workspace_harness(root)
    replacement = '''
def host_launcher(*args):
    class HostLauncher:
        def __getitem__(self, grid):
            return lambda x: x
    return HostLauncher()
'''
    changed = original.replace("@triton.jit\ndef kernel", replacement + f"@{outer}\n@triton.jit\ndef kernel")
    source.write_text(changed)
    # This is the reported failure mode: the JIT marker and protected caller
    # survive, but Python would bind kernel to an arbitrary outer decorator.
    before = ast.parse(original); after = ast.parse(changed)
    caller = lambda tree: ast.dump(next(n for n in tree.body if getattr(n, "name", None) == "protected_wrapper"))
    assert caller(before) == caller(after)
    target = next(n for n in after.body if getattr(n, "name", None) == "kernel")
    assert any(ast.unparse(d) == "triton.jit" for d in target.decorator_list)
    with pytest.raises(RuntimeError, match="Unsupported decorator on editable definition 'kernel'"):
        verify_workspace_harness(snapshot)


def test_existing_target_keeps_supported_tuning_chain_and_new_jit_helper(tmp_path):
    _, source, snapshot = fixture_workspace(tmp_path)
    original = source.read_text()
    source.write_text(original.replace(
        "@triton.jit\ndef kernel(x): return x",
        "@triton.autotune(configs=get_configs(), key=['n'])\n"
        "@triton.heuristics({'EVEN': lambda args: args['n'] % 128 == 0})\n"
        "@triton.jit\ndef kernel(x, n): return helper(x)")
        + "\n@triton.jit\ndef helper(x): return x + 1\n")
    verify_workspace_harness(snapshot)


def test_duplicate_binding_cannot_leave_an_unused_jit_definition_as_cover(tmp_path):
    _, source, snapshot = fixture_workspace(tmp_path)
    source.write_text(source.read_text() + "\ndef kernel(x): return x\n")
    with pytest.raises(RuntimeError, match="Multiple definitions of editable binding 'kernel'"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("initial,call", [
    ("config_alias = pytest.fixture\n", "config_alias(autouse=True)"),
    ("", "kernel(None)"),
    ("class ConfigAlias:\n    pass\n", "ConfigAlias()"),
])
def test_initial_names_do_not_grant_arbitrary_factory_privilege(tmp_path, initial, call):
    root, source, _ = fixture_workspace(tmp_path)
    source.write_text(source.read_text() + initial)
    snapshot = snapshot_workspace_harness(root)
    source.write_text(source.read_text() + f"\n@triton.jit(launch_metadata={call})\ndef helper(x): return x\n")
    with pytest.raises(RuntimeError, match="Unapproved definition-time factory"):
        verify_workspace_harness(snapshot)


@pytest.mark.parametrize("body,accepted", [
    ("return [triton.Config({'BLOCK': n}) for n in range(32, 129, 32)]", True),
    ("return pytest.fixture(autouse=True)(lambda: None)", False),
    ("global reference\n    reference = lambda: 0\n    return []", False),
    ("reference = pytest.fixture\n    return reference(autouse=True)", False),
    ("triton = pytest\n    return triton.Config({})", False),
])
def test_editable_config_factory_has_a_separate_restricted_body_policy(tmp_path, body, accepted):
    root, source, _ = fixture_workspace(tmp_path)
    original = source.read_text().replace(
        "@triton.jit\ndef kernel", "@triton.autotune(configs=get_configs(), key=['x'])\n@triton.jit\ndef kernel")
    source.write_text(original)
    config_path = root / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["candidate"]["editable"][0]["symbols"].append("get_configs")
    config_path.write_text(yaml.safe_dump(config))
    snapshot = snapshot_workspace_harness(root)
    source.write_text(original.replace("def get_configs(): return [triton.Config({'BLOCK': 128})]",
                                       "def get_configs():\n    " + body))
    if accepted:
        verify_workspace_harness(snapshot)
    else:
        with pytest.raises(RuntimeError, match="Unapproved definition-time factory"):
            verify_workspace_harness(snapshot)


@pytest.mark.parametrize("task", [
    "instruction2triton/rocmbench/moe_gemm",
    "triton2triton/rocmbench/hard/moe_gemm",
])
def test_exact_reported_fixture_rejected_on_both_actual_moe_tasks(tmp_path, task):
    workspace = tmp_path / "task"; shutil.copytree(ROOT / "tasks" / task, workspace)
    materialize_perf_helpers_in_workspace(workspace)
    snapshot = snapshot_workspace_harness(workspace)
    source = workspace / "moe_gemm.py"; original = source.read_text()
    # Normal JIT helper is accepted without changing the protected tests.
    source.write_text(original + "\n@triton.jit\ndef helper(x): return x + 1\n")
    verify_workspace_harness(snapshot)
    # Exact definition that passed the old guard in the isolated CPU proof.
    source.write_text(original + '''
@pytest.fixture(autouse=True)
def arena_new_helper(monkeypatch):
    import _arena_reference
    monkeypatch.setattr(_arena_reference, "compare", lambda *args, **kwargs: None)
''')
    with pytest.raises(RuntimeError, match="Unsupported decorator"):
        verify_workspace_harness(snapshot)
    source.write_text(original.replace("def test_correctness(", "def test_changed_correctness("))
    with pytest.raises(RuntimeError, match="moe_gemm.py"):
        verify_workspace_harness(snapshot)


def scoped_configs():
    for path in sorted((ROOT / "tasks").rglob("config.yaml")):
        cfg = yaml.safe_load(path.read_text()) or {}
        if any(isinstance(e, dict) and e.get("scope") == "symbols"
               for e in cfg.get("candidate", {}).get("editable", [])):
            yield path.relative_to(ROOT / "tasks").as_posix()


@pytest.mark.parametrize("relative", list(scoped_configs()))
def test_retained_symbol_scope_tasks_keep_original_declarations(relative):
    path = ROOT / "tasks" / relative
    spec = load_task_spec(path, task_id=path.parent.relative_to(ROOT / "tasks").as_posix())
    if any(not (path.parent / edit.path).is_file() for edit in spec.candidate.editable):
        assert yaml.safe_load(path.read_text()).get("workspace", {}).get("sources")
        pytest.skip("Declared external source needs workspace materialization; no source/GPU claim")
    snapshot = snapshot_workspace_harness(path.parent, task_spec=spec)
    verify_workspace_harness(snapshot)
    # Metadata records enforcement; no task-specific family switch is needed.
    assert describe_workspace_harness(path.parent, snapshot=snapshot)["protected_path_policies"]
