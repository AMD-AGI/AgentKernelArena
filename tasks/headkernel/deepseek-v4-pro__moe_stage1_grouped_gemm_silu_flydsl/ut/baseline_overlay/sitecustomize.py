# Auto-generated reversible overlay (e2e_workflow). Drop this dir from PYTHONPATH to revert.
import json, os, sys, importlib, importlib.util

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAN = os.path.join(_HERE, "_overlay_manifest.json")

# --- REENTRANCY GUARD (ROCm fork-bomb fix) --------------------------------------------------------
# Installing the overlay imports the target package (e.g. aiter -> flydsl), and those packages probe
# the GPU by SHELLING OUT to a python helper (`rocm_agent_enumerator`). That helper inherits this
# overlay on PYTHONPATH, so its sitecustomize re-runs the install, which probes again, forever: an
# unbounded process chain that never boots the server. The env marker below is set for the duration
# of the install ONLY, so any python subprocess spawned *while installing* skips the overlay, while
# workers the server spawns AFTERWARDS (the processes we actually want hooked) still install.
_GUARD = "_GEAK_OVERLAY_INSTALLING"
_GEAK_REENTRANT = os.environ.get(_GUARD) == "1"


def _geak_clear_guard():
    if not _GEAK_REENTRANT:
        os.environ.pop(_GUARD, None)


if _GEAK_REENTRANT:
    _m = {"modules": [], "rebinds": [], "markers": [], "captures": []}
else:
    os.environ[_GUARD] = "1"
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

# Install is done: drop the reentrancy marker so processes the SERVER spawns from here on (TP/DP
# workers, the very processes we want hooked) install the overlay normally.
_geak_clear_guard()
