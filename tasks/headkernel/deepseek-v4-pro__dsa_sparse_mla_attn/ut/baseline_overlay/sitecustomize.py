# Auto-generated reversible overlay (e2e_workflow). Drop this dir from PYTHONPATH to revert.
import json, os, sys, importlib, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAN = os.path.join(_HERE, "_overlay_manifest.json")


def _geak_rocm_tool():
    """True for ROCm/HIP command-line tools that merely INHERIT this overlay's PYTHONPATH.

    Applying the overlay imports the patched module (torch / tilelang / aiter). On ROCm those
    imports shell out to `rocm_agent_enumerator` for arch detection -- a *python* script, which
    therefore re-runs this sitecustomize, re-imports tilelang, and shells out again: an unbounded
    fork chain (observed +1 process every ~7s, 70+ deep) that hangs the leg subprocess or the
    server launcher with no error. Those tools never contain the op seam, so the overlay is a no-op
    for them, and CAPTURE_* is dropped from their env so their own children skip it too.
    """
    _a0 = (sys.argv[0] if sys.argv else "") or ""
    if not _a0 or _a0 == "-c":
        return False
    try:
        _real = os.path.realpath(_a0)
    except Exception:
        return False
    _base = os.path.basename(_real).lower()
    if _base.startswith(("rocm", "roc-", "roc_", "hip", "amd-smi", "amdgpu")):
        return True
    return _real.startswith("/opt/rocm") and not _base.startswith("python")


if _geak_rocm_tool():
    for _v in ("CAPTURE_TARGET", "CAPTURE_OUT"):
        os.environ.pop(_v, None)
    sys.stderr.write("[overlay] ROCm tool process (%s): overlay skipped\n"
                     % os.path.basename(sys.argv[0]))
    _m = {"modules": [], "rebinds": [], "markers": [], "captures": []}
else:
    try:
        with open(_MAN) as _fh:
            _m = json.load(_fh)
    except Exception as _e:
        _m = {"modules": [], "rebinds": [], "markers": [], "captures": []}

# (a) inject patched submodules under their dotted names BEFORE anything imports them.
for _e in _m.get("modules", []):
    try:
        _dotted, _file = _e["module"], os.path.join(_HERE, _e["file"])
        _spec = importlib.util.spec_from_file_location(_dotted, _file)
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_dotted] = _mod
        _spec.loader.exec_module(_mod)
        # bind as attribute on the parent so both `from a.b import c` and `import a.b; a.b.c` see the patch.
        if "." in _dotted:
            _parent, _child = _dotted.rsplit(".", 1)
            try:
                setattr(importlib.import_module(_parent), _child, _mod)
            except Exception:
                pass
        sys.stderr.write("[overlay] injected module %s <- %s\n" % (_dotted, _file))
    except Exception as _ex:
        sys.stderr.write("[overlay] module inject FAILED %r: %r\n" % (_e, _ex))

# (b) rebind single attributes (monkeypatch).
for _e in _m.get("rebinds", []):
    try:
        _modname, _attr = _e["target"].split(":")
        _t = importlib.import_module(_modname)
        _impl = importlib.import_module(_e["impl_module"])
        setattr(_t, _attr, getattr(_impl, _e["impl_attr"]))
        sys.stderr.write("[overlay] rebound %s -> %s.%s\n" % (_e["target"], _e["impl_module"], _e["impl_attr"]))
    except Exception as _ex:
        sys.stderr.write("[overlay] rebind FAILED %r: %r\n" % (_e, _ex))

# (c) capture hooks (shape/IO oracle recording) go on FIRST, so the capture wrapper is the innermost
# stand-in and is already bound before any marker install imports a module that does
# `from <capture target module> import <attr>` (which would otherwise alias the un-captured function).
for _e in _m.get("captures", []):
    try:
        import capture_shapes
        capture_shapes.install(_e["target"], _e["out"], int(_e.get("max", 5)))
    except Exception as _ex:
        sys.stderr.write("[overlay] capture install FAILED %r: %r\n" % (_e, _ex))

# (d) marker-only hooks used to compare every candidate seam in one trace.
for _e in _m.get("markers", []):
    try:
        import seam_trace
        seam_trace.install(_e["target"])
    except Exception as _ex:
        sys.stderr.write("[overlay] seam marker install FAILED %r: %r\n" % (_e, _ex))
