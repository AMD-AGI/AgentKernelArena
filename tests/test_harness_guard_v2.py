from dataclasses import replace
import logging
import shutil

import pytest
import yaml

from src.harness_guard import describe_workspace_harness, snapshot_workspace_harness, verify_workspace_harness


def workspace(tmp_path, *, symbols=False, helpers=False, tree=False):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "source").mkdir()
    source = root / "source/kernel.py"
    source.write_text("import math\nTOLERANCE = 0.01\ndef compute(): return 1\ndef reference(): return 1\n")
    edit = "source/kernel.py"
    if tree:
        edit = {"path": "source", "scope": "tree"}
    elif symbols:
        edit = {"path": "source/kernel.py", "scope": "symbols", "symbols": ["compute"], "allow_new_helpers": helpers}
    config = {"schema_version": 2, "candidate": {"language": "triton", "editable": [edit]},
              "evaluation": {"runner": ["python3", "evaluate.py"]}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "evaluate.py").write_text("# protected custom evaluation runner\n")
    (root / "inputs.bin").write_bytes(b"task-specific inputs")
    return root, source


def test_file_edits_leave_all_non_candidate_task_material_protected(tmp_path):
    root, source = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text("def compute(): return 2\n")
    verify_workspace_harness(before)
    (root / "inputs.bin").write_bytes(b"easier case")
    with pytest.raises(RuntimeError, match="inputs.bin"):
        verify_workspace_harness(before)


def test_read_only_guard_preserves_additions_but_evaluation_keeps_logged_cleanup(tmp_path, caplog):
    root, _ = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    extra = root / "debug_test.py"
    extra.write_text("# agent's diagnostic\n")
    with pytest.raises(RuntimeError, match="Added protected harness files require review"):
        verify_workspace_harness(before, discard_added=False)
    assert extra.read_text() == "# agent's diagnostic\n"

    verify_workspace_harness(before, logger=logging.getLogger(__name__))
    assert not extra.exists()
    assert "debug_test.py" in caplog.text


def test_symbol_edits_only_mask_named_implementation_and_permitted_new_helpers(tmp_path):
    root, source = workspace(tmp_path, symbols=True, helpers=True)
    before = snapshot_workspace_harness(root)
    source.write_text(source.read_text().replace("def compute(): return 1", "def compute(): return helper()")
                      + "def helper(): return 1\n")
    verify_workspace_harness(before)
    source.write_text(source.read_text() + "def reference(): return 999\n")
    with pytest.raises(RuntimeError, match="source/kernel.py"):
        verify_workspace_harness(before)


@pytest.mark.parametrize("change", [
    lambda text: text.replace("import math", "import random as math"),
    lambda text: text.replace("TOLERANCE = 0.01", "TOLERANCE = 1"),
    lambda text: text.replace("def reference(): return 1", "def reference(): return 2"),
    lambda text: text + "def new_helper(): return 2\n",
])
def test_symbol_scope_does_not_allow_import_reference_or_unpermitted_helper_edits(tmp_path, change):
    root, source = workspace(tmp_path, symbols=True)
    before = snapshot_workspace_harness(root)
    source.write_text(change(source.read_text()))
    with pytest.raises(RuntimeError, match="source/kernel.py"):
        verify_workspace_harness(before)


def test_tree_edit_does_not_override_harness_file_protection(tmp_path):
    root, source = workspace(tmp_path, tree=True)
    harness = root / "source/check_harness.py"
    harness.write_text("# trusted evaluation\n")
    before = snapshot_workspace_harness(root)
    source.write_text("optimized")
    verify_workspace_harness(before)
    harness.write_text("# weaker evaluation\n")
    with pytest.raises(RuntimeError, match="check_harness.py"):
        verify_workspace_harness(before)


@pytest.mark.parametrize("patch", [
    "def compute():\n    globals()['math'] = None\n    return 1\n",
    "def compute():\n    math.sqrt = abs\n    return 1\n",
    "def compute():\n    import sys\n    return sys._getframe(1).f_code\n",
    "def compute():\n    return reference.__code__\n",
    "def compute():\n    alias = {'m': math}['m']\n    alias.sqrt.__doc__ = ''\n    return 1\n",
    "def compute():\n    setattr(math, 'pi', 3)\n    return 1\n",
])
def test_a_candidate_may_not_add_runtime_patching(tmp_path, patch):
    # A delivered refk candidate swapped the module's torch for a proxy and
    # patched Tensor.contiguous, so the protected launcher ran its fused kernel.
    root, source = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(source.read_text().replace("def compute(): return 1\n", patch))
    with pytest.raises(RuntimeError, match="adds runtime patching"):
        verify_workspace_harness(before)


def test_runtime_patching_the_task_already_had_stays_allowed(tmp_path):
    root, source = workspace(tmp_path, symbols=True, helpers=True)
    source.write_text(source.read_text() + "def setup(obj):\n    setattr(obj, 'ready', True)\n")
    before = snapshot_workspace_harness(root)
    original = source.read_text()
    source.write_text(original.replace("def compute(): return 1", "def compute(): return 2"))
    verify_workspace_harness(before)
    source.write_text(original.replace("def compute(): return 1", "def compute(): return helper()")
                      + "def helper():\n    setattr(math, 'pi', 3)\n    return 1\n")
    with pytest.raises(RuntimeError, match=r"adds runtime patching \(calls setattr\(\)\)"):
        verify_workspace_harness(before)
    # A snapshot saved before sites were recorded cannot tell added ones apart.
    verify_workspace_harness(replace(before, initial_patching=None))


@pytest.mark.parametrize("statement", [
    "import builtins; builtins.setattr(math, 'pi', 3)",
    "import builtins as b; b.delattr(math, 'pi')",
    "from builtins import setattr as change; change(math, 'pi', 3)",
    "from builtins import exec as run; run('import math; math.pi = 3')",
    "import builtins; builtins.exec('import math; math.pi = 3')",
    "import builtins; builtins.eval('1 + 2')",
    "import builtins; builtins.globals()",
    "import builtins; builtins.vars(math)",
    "import builtins; builtins.__import__('math')",
])
def test_qualified_and_aliased_reflection_is_rejected_by_real_guard(tmp_path, statement):
    root, source = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(source.read_text().replace('def compute(): return 1',
                      'def compute():\n    ' + statement + '\n    return 1'))
    with pytest.raises(RuntimeError, match='adds runtime patching'):
        verify_workspace_harness(before)


def test_existing_reflective_call_has_one_identity_across_spellings(tmp_path):
    root, source = workspace(tmp_path)
    source.write_text(source.read_text().replace('def compute(): return 1',
                      "def compute():\n    setattr(math, 'pi', 3)\n    return 1"))
    before = snapshot_workspace_harness(root)
    source.write_text(source.read_text().replace("setattr(math, 'pi', 3)",
                      "__import__('builtins').setattr(math, 'pi', 3)"))
    with pytest.raises(RuntimeError, match='runtime patching'):
        verify_workspace_harness(before)
    source.write_text("import math\nimport builtins as b\n"
                      "def compute():\n    b.setattr(math, 'pi', 3)\n    return 1\n")
    verify_workspace_harness(before)


def test_effective_guard_describes_source_policy_and_old_snapshot_limit(tmp_path):
    root, _ = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    policy = describe_workspace_harness(root, snapshot=before)['candidate_source_policy']
    assert policy['enabled'] is True
    assert policy['cxx_custom_members'] == 'allowed_without_library_receiver_identity'
    assert describe_workspace_harness(root, snapshot=replace(before, initial_patching=None))[
        'candidate_source_policy']['enabled'] is False


HIP_KERNEL = """#include <torch/extension.h>
// A reference could call at::matmul(a, b); this file does not.
torch::Tensor forward(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty({a.size(0), b.size(1)}, a.options());
    auto scale = at::bmm(a, b);
    launch_gemm(a.contiguous(), b.contiguous(), out);
    return out;
}
"""


def hip_workspace(tmp_path):
    root = tmp_path / "workspace"
    (root / "hip").mkdir(parents=True)
    source = root / "hip/kernel.hip"
    source.write_text(HIP_KERNEL)
    config = {"schema_version": 2, "candidate": {"language": "hip", "editable": ["hip/kernel.hip"]},
              "evaluation": {"runner": ["python3", "evaluate.py"]}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "evaluate.py").write_text("# protected custom evaluation runner\n")
    return root, source


@pytest.mark.parametrize("delegation", [
    "    return at::matmul(a, b * 2);\n",
    "    return a.mm(b) * 2;\n",
    "    hipblasSgemm(handle, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha, x, m, y, k, &beta, z, m);\n",
    "    return torch::softmax(a, -1);\n",
    "    return at::mv(a, b.squeeze(1)).unsqueeze(1);\n",
    "    return torch::nn::functional::cross_entropy(a, b);\n",
    "    return at::nll_loss(at::log_softmax(a, 1), b);\n",
    "    return a / (a * a).mean(1, true).add(1e-5).sqrt();\n",
    "    return at::rms_norm(a, {64}, {}, 1e-5);\n",
])
def test_a_hip_candidate_may_not_add_library_operator_calls(tmp_path, delegation):
    root, source = hip_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(HIP_KERNEL.replace("    return out;\n", delegation))
    with pytest.raises(RuntimeError, match="adds library operator calls"):
        verify_workspace_harness(before)


def test_library_calls_the_hip_task_already_had_stay_allowed(tmp_path):
    root, source = hip_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(HIP_KERNEL.replace("launch_gemm(", "// at::matmul is slower here\n    launch_tiled_gemm("))
    verify_workspace_harness(before)
    source.write_text(HIP_KERNEL.replace("at::bmm(a, b);", "at::bmm(a, b);\n    auto again = at::bmm(b, a);"))
    with pytest.raises(RuntimeError, match=r"adds library operator calls \(calls at::bmm\)"):
        verify_workspace_harness(before)


@pytest.mark.parametrize('prefix', [
    'const char* label = "https://local"; ',
    'const char* label = "/* not a comment */"; ',
    'const char* label = "escaped \\\" // still a string"; ',
    'const char* label = R"tag(// quotes " and /* markers)tag"; ',
    "char quote = '\"'; ",
    "constexpr int block = 1'024; ",
    "constexpr int block = 0xAB'CD'EF; ",
])
def test_literal_contents_do_not_hide_following_library_calls(tmp_path, prefix):
    root, source = hip_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(HIP_KERNEL.replace('    return out;\n',
                      '    ' + prefix + 'return at::matmul(a, b);\n'))
    with pytest.raises(RuntimeError, match='adds library operator calls'):
        verify_workspace_harness(before)


@pytest.mark.parametrize('call', ['at :: matmul(a, b)', 'at::mat' + '\\\n' + 'mul(a, b)',
                                  'at /* boundary */ :: matmul(a, b)'])
def test_cpp_spacing_comments_and_line_splicing_do_not_hide_library_calls(tmp_path, call):
    root, source = hip_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(HIP_KERNEL.replace('    return out;\n', '    return ' + call + ';\n'))
    with pytest.raises(RuntimeError, match='adds library operator calls'):
        verify_workspace_harness(before)


def test_library_names_in_literals_and_comments_are_not_calls(tmp_path):
    root, source = hip_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(HIP_KERNEL + '\nconst char* text = "rocblas_sgemm(a, b)";\n'
                      'const char* raw = R"tag(at::matmul(a, b); // ignored)tag";\n'
                      '/* torch::mm(a, b); */\n// at::matmul(a, b);\n')
    verify_workspace_harness(before)


@pytest.mark.parametrize('name', ['sum', 'mean', 'dot', 'norm', 'mm'])
def test_custom_cpp_members_are_not_library_operators(tmp_path, name):
    root, source = hip_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    # Reuse the Tensor parameter's name in a separate custom-type function.
    source.write_text(HIP_KERNEL + f'''
struct Accumulator {{
    float x, y;
    float {name}() const {{ return x + y; }}
}};
float own_operation(Accumulator a) {{ return a.{name}(); }}
''')
    verify_workspace_harness(before)


def test_custom_cpp_receiver_can_shadow_a_tensor_in_a_nested_scope(tmp_path):
    root, source = hip_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(HIP_KERNEL + '''
struct Accumulator { float sum() const { return 1; } };
void own_operation(at::Tensor a) { { Accumulator a; a.sum(); } }
struct Tensor { float sum() const { return 1; } };
float custom_tensor(Tensor a) { return a.sum(); }
''')
    verify_workspace_harness(before)


@pytest.mark.parametrize('expression', [
    'auto copy = a.contiguous(); return copy.sum();',
    'auto copy = torch::empty_like(a); return copy.mean();',
    'return (a * a).sum();',
])
def test_inferred_tensor_members_still_refuse_library_delegation(tmp_path, expression):
    root, source = hip_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(HIP_KERNEL.replace('    return out;\n', '    ' + expression + '\n'))
    with pytest.raises(RuntimeError, match='adds library operator calls'):
        verify_workspace_harness(before)


TRITON_TASK = """import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=-float("inf"))
    tl.store(y_ptr + offs, tl.softmax(x), mask=offs < n)


def softmax(x):
    y = torch.empty_like(x)
    softmax_kernel[(1,)](x, y, x.numel(), BLOCK=1024)
    return y


def test_softmax():
    x = torch.randn(1000)
    torch.testing.assert_close(softmax(x), torch.softmax(x, -1))
"""


def triton_workspace(tmp_path, language="triton"):
    root = tmp_path / "workspace"
    root.mkdir(parents=True)
    source = root / "softmax.py"
    source.write_text(TRITON_TASK)
    edit = {"path": "softmax.py", "scope": "symbols", "symbols": ["softmax_kernel", "softmax"],
            "allow_new_helpers": True}
    config = {"schema_version": 2, "candidate": {"language": language, "editable": [edit]},
              "evaluation": {"runner": ["python3", "evaluate.py"]}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "evaluate.py").write_text("# protected custom evaluation runner\n")
    return root, source


@pytest.mark.parametrize("delegation", [
    "    return torch.softmax(x, -1)\n",
    "    return x.float().softmax(-1)\n",
    "    import torch.nn.functional as F\n    return F.scaled_dot_product_attention(x, x, x)\n",
    "    return x @ x.T\n",
    "    return getattr(torch, 'mat' + 'mul')(x, x)\n",
    "    return torch.ops.aten._softmax(x, -1, False)\n",
])
def test_a_triton_launcher_may_not_add_library_operator_calls(tmp_path, delegation):
    root, source = triton_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(TRITON_TASK.replace("    return y\n", delegation, 1))
    with pytest.raises(RuntimeError, match="adds library operator calls"):
        verify_workspace_harness(before)


def test_triton_launch_choices_and_the_tasks_own_reference_stay_allowed(tmp_path):
    root, source = triton_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(TRITON_TASK.replace("softmax_kernel[(1,)](x, y, x.numel(), BLOCK=1024)",
                                          "softmax_kernel[(1,)](x, y, x.numel(), BLOCK=2048, num_warps=8)"))
    verify_workspace_harness(before)
    # Only Triton candidates are measured this way; other Python DSLs keep the runtime-patching policy.
    root, source = triton_workspace(tmp_path / "flydsl", language="flydsl")
    before = snapshot_workspace_harness(root)
    source.write_text(TRITON_TASK.replace("    return y\n", "    return torch.softmax(x, -1)\n", 1))
    verify_workspace_harness(before)


def test_new_task_config_cannot_redefine_the_protected_boundary(tmp_path):
    root, source = workspace(tmp_path, symbols=True)
    before = snapshot_workspace_harness(root)
    config = yaml.safe_load((root / "config.yaml").read_text())
    config["candidate"]["editable"] = ["source/kernel.py"]
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    source.write_text("def reference(): return 'anything passes'\n")
    with pytest.raises(RuntimeError, match="config.yaml"):
        verify_workspace_harness(before)


def test_protected_symlink_escape_is_rejected_even_if_bytes_match(tmp_path):
    root, _ = workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    original = root / "inputs.bin"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(RuntimeError, match="escaped workspace"):
        verify_workspace_harness(before)


def test_missing_shipped_file_is_not_omitted_from_snapshot(tmp_path):
    root, _ = workspace(tmp_path)
    package = tmp_path / "task"
    shutil.copytree(root, package)
    (root / "inputs.bin").unlink()
    with pytest.raises(RuntimeError, match="inputs.bin"):
        snapshot_workspace_harness(root, task_root=package)


def test_validator_description_uses_v2_scope(tmp_path):
    root, _ = workspace(tmp_path, symbols=True)
    info = describe_workspace_harness(root)
    assert info["editable_entrypoint_targets"] == {"source/kernel.py": ["compute"]}
    assert "evaluate.py" in info["protected_paths"]


@pytest.mark.parametrize("name", ["yaml.py", "usercustomize.py"])
def test_new_import_shadow_files_follow_described_guard_policy(tmp_path, name):
    root, _ = workspace(tmp_path, symbols=True)
    before = snapshot_workspace_harness(root)
    info = describe_workspace_harness(root, snapshot=before)
    assert "scripts" in info["new_file_policy"]["protected_directories"]
    extra = root / "scripts" / name
    extra.parent.mkdir()
    extra.write_text("raise RuntimeError('unexpected import')\n")
    with pytest.raises(RuntimeError, match="Added protected harness files"):
        verify_workspace_harness(before, discard_added=False)
    assert extra.exists()
    verify_workspace_harness(before)
    assert not extra.exists()



LAUNCH_SOURCE = ("import math\n"
                 "def _launch_config(n):\n    return {'BLOCK': 64, 'num_warps': 4}\n"
                 "def launch(n):\n    cfg = _launch_config(n)\n    return cfg['BLOCK'] * n\n")
LAUNCH_RETURN = "    return {'BLOCK': 64, 'num_warps': 4}\n"


def launch_workspace(tmp_path, *, declared=True):
    root = tmp_path / "workspace"
    (root / "source").mkdir(parents=True)
    source = root / "source/kernel.py"
    source.write_text(LAUNCH_SOURCE)
    edit = {"path": "source/kernel.py", "scope": "symbols", "symbols": ["_launch_config"], "allow_new_helpers": True}
    if declared:
        edit["configuration_symbols"] = ["_launch_config"]
    config = {"schema_version": 2, "candidate": {"language": "triton", "editable": [edit]},
              "evaluation": {"runner": ["python3", "evaluate.py"]}}
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    (root / "evaluate.py").write_text("# protected custom evaluation runner\n")
    return root, source


def test_declared_configuration_symbol_may_be_retuned(tmp_path):
    root, source = launch_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    assert describe_workspace_harness(root, snapshot=before)["protected_path_policies"]["source/kernel.py"][
        "configuration_symbols"] == ["_launch_config"]
    tuned = ("    if n >= 4096:\n        return {'BLOCK': 256, 'num_warps': 8}\n"
             "    block = 128\n    return {'BLOCK': block, 'num_warps': 4}\n")
    source.write_text(source.read_text().replace(LAUNCH_RETURN, tuned))
    verify_workspace_harness(before)


HOST_CODE = [
    "    math.sqrt(n)\n",
    "    import os\n",
    "    for _ in range(2):\n        pass\n",
    "    launch.cache = 1\n",
    "    globals()['TOLERANCE'] = 1\n",
    "    with open('x') as f:\n        pass\n",
]


@pytest.mark.parametrize("host", HOST_CODE)
def test_declared_configuration_symbol_cannot_run_host_code(tmp_path, host):
    root, source = launch_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(source.read_text().replace(LAUNCH_RETURN, host + LAUNCH_RETURN))
    with pytest.raises(RuntimeError, match="only builds configuration data"):
        verify_workspace_harness(before)


@pytest.mark.parametrize("change", [
    lambda text: text.replace("def _launch_config(n):", "@staticmethod\ndef _launch_config(n):"),
    lambda text: text.replace("def _launch_config(n):\n" + LAUNCH_RETURN, ""),
])
def test_declared_configuration_symbol_must_remain_a_plain_function(tmp_path, change):
    root, source = launch_workspace(tmp_path)
    before = snapshot_workspace_harness(root)
    source.write_text(change(source.read_text()))
    with pytest.raises(RuntimeError, match="source/kernel.py|kernel.py"):
        verify_workspace_harness(before)


def test_undeclared_editable_functions_keep_the_existing_definition_policy(tmp_path):
    # Python implementation tasks let protected code call editable host code.
    root, source = launch_workspace(tmp_path, declared=False)
    before = snapshot_workspace_harness(root)
    source.write_text(source.read_text().replace(LAUNCH_RETURN, HOST_CODE[0] + LAUNCH_RETURN))
    verify_workspace_harness(before)


@pytest.mark.parametrize("edit,message", [
    ({"symbols": ["compute"], "configuration_symbols": ["other"]}, "listed in symbols"),
    ({"symbols": ["compute"], "configuration_symbols": "compute"}, "listed in symbols"),
    ({"symbols": ["compute"], "configuration_symbols": [{}]}, "listed in symbols"),
    ({"symbols": ["compute"], "configuration_symbols": [[]]}, "listed in symbols"),
    ({"symbols": ["compute"], "configuration_symbols": ["compute"], "entry": True}, "entrypoint"),
    ({"scope": "file", "configuration_symbols": []}, "symbol scope"),
])
def test_configuration_symbols_schema(tmp_path, edit, message):
    from src.task_spec import TaskConfigError, load_task_spec
    entry = edit.pop("entry", False)
    raw = {"path": "source/kernel.py", "scope": "symbols", **edit}
    candidate = {"language": "triton", "initial_state": "implemented", "initial_language": "triton",
                 "editable": [raw]}
    if entry:
        candidate["entrypoints"] = [{"file": "source/kernel.py", "kind": "function", "symbol": "compute"}]
    config = {"schema_version": 2, "description": "d", "candidate": candidate,
              "evaluation": {"runner": ["python3", "evaluate.py"]}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(TaskConfigError, match=message):
        load_task_spec(path, task_id="t")


@pytest.mark.parametrize('trusted', ['_aka_benchmark.py', 'evaluate.py', 'scripts/reference.py'])
@pytest.mark.parametrize('discard_added', [False, True])
def test_new_package_cannot_shadow_a_frozen_python_module(tmp_path, trusted, discard_added):
    from importlib.machinery import PathFinder
    root, _ = workspace(tmp_path)
    module = root/trusted;module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text('VALUE = "trusted"\n')
    snapshot = snapshot_workspace_harness(root)
    package=module.with_suffix('');package.mkdir()
    initializer=package/'__init__.py'
    initializer.write_text('raise RuntimeError("must never execute")\n')
    # This is real Python import resolution, without executing the new code.
    found=PathFinder.find_spec(module.stem, [str(module.parent)])
    assert found.origin == str(initializer)
    with pytest.raises(RuntimeError, match='shadow protected modules'):
        verify_workspace_harness(snapshot, discard_added=discard_added)
    assert initializer.is_file() and module.read_text()=='VALUE = "trusted"\n'


@pytest.mark.parametrize('package', [False, True])
def test_new_native_extension_cannot_shadow_a_frozen_python_module(tmp_path, package):
    from importlib.machinery import EXTENSION_SUFFIXES, PathFinder
    root,_=workspace(tmp_path);module=root/'evaluate.py'
    snapshot=snapshot_workspace_harness(root)
    if package:
        directory=module.with_suffix('');directory.mkdir()
        extension=directory/('__init__'+EXTENSION_SUFFIXES[0])
    else:extension=module.with_suffix(EXTENSION_SUFFIXES[0])
    extension.write_bytes(b'not loaded')
    found=PathFinder.find_spec('evaluate',[str(root)])
    assert found.origin==str(extension)
    with pytest.raises(RuntimeError,match='shadow protected modules'):verify_workspace_harness(snapshot)
    assert extension.read_bytes()==b'not loaded'


def test_namespace_directory_does_not_replace_a_frozen_module(tmp_path):
    from importlib.machinery import PathFinder
    root,_=workspace(tmp_path);directory=root/'evaluate';directory.mkdir()
    snapshot=snapshot_workspace_harness(root)
    helper=directory/'helper.py';helper.write_text('VALUE=1\n')
    assert PathFinder.find_spec('evaluate',[str(root)]).origin==str(root/'evaluate.py')
    verify_workspace_harness(snapshot, discard_added=False)
    assert helper.is_file()


def test_an_original_frozen_package_remains_usable(tmp_path):
    root,_=workspace(tmp_path);directory=root/'evaluate';directory.mkdir()
    (directory/'__init__.py').write_text('VALUE=1\n')
    snapshot=snapshot_workspace_harness(root)
    verify_workspace_harness(snapshot,discard_added=False)
