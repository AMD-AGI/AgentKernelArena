"""Bind a private Triton source leg to the frozen native wrapper body."""
import ast
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import uuid

from source_guard import validate_sources


def raw_wrapper_tree(source):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "gemm_afp4wfp4_"]
    if len(functions) != 1:
        raise ValueError("native wrapper definition is missing or duplicated")
    function = functions[0]
    expected = ast.parse("@torch_compile_guard(gen_fake=gemm_afp4wfp4_fake_tensor)\ndef f(): pass").body[0].decorator_list
    if [ast.dump(n) for n in function.decorator_list] != [ast.dump(n) for n in expected]:
        raise ValueError("native custom-op decorator context changed")
    function.decorator_list = []
    return tree


def bind_globals(wrapper, kernel):
    wrapper._triton_gemm_afp4wfp4_kernel = kernel._gemm_afp4wfp4_kernel
    wrapper._get_config = kernel._get_config
    if wrapper.gemm_afp4wfp4_.__globals__ is not wrapper.__dict__:
        raise ValueError("wrapper still routes through a process-global custom op")
    return wrapper.gemm_afp4wfp4


def load_leg(root, leg, *, use_splitk_bf16=False):
    root = Path(root).resolve()
    if leg not in {"candidate", "reference"}:
        raise ValueError("unknown source leg")
    if type(use_splitk_bf16) is not bool:
        raise ValueError("split-K partial dtype control must be boolean")
    provenance = json.loads((root / "SOURCE-PROVENANCE.json").read_text())
    validate_sources(root, root)
    wrapper_path = root / "ut/native/wrapper.py"
    if hashlib.sha256(wrapper_path.read_bytes()).hexdigest() != provenance["sources"]["ut/native/wrapper.py"]["sha256"]:
        raise ValueError("frozen native wrapper changed")
    for name, expected in provenance["runtime_dependencies"].items():
        module = importlib.import_module(name)
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError("pinned runtime dependency changed: " + name)
    source = root / ("source/kernel.py" if leg == "candidate" else "ut/reference/kernel.py")
    name = "_aka_minimax_fp4_" + leg + "_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, source)
    kernel = importlib.util.module_from_spec(spec)
    sys.modules[name] = kernel
    spec.loader.exec_module(kernel)
    wrapper = types.ModuleType(name + "_wrapper")
    wrapper.__file__ = str(wrapper_path)
    sys.modules[wrapper.__name__] = wrapper
    exec(compile(raw_wrapper_tree(wrapper_path.read_text()), str(wrapper_path), "exec"), wrapper.__dict__)
    wrapper._USE_GEMM_SPLITK_BF16 = use_splitk_bf16
    call = bind_globals(wrapper, kernel)
    return call, {"source_leg": leg, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "private_module": name, "wrapper_registration_removed": True,
                  "use_splitk_bf16": use_splitk_bf16,
                  "gpu_source_binding_validated": False}
