"""Load the pinned module under an isolated name and attest real target launches."""
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys

from served_contract import sha
from evaluation_contract import canonical, require
from source_guard import validate_sources


class KernelProbe:
    def __init__(self, kernel, name):
        self.kernel, self.name = kernel, name
        self.launches = []

    def __getattr__(self, name):
        return getattr(self.kernel, name)

    def __getitem__(self, grid):
        call = self.kernel[grid]
        def launch(*args, **kwargs):
            compiled = call(*args, **kwargs)
            metadata = getattr(compiled, "metadata", None)
            self.launches.append({"kernel": self.name,
                "grid": list(grid) if isinstance(grid, (tuple, list)) else None,
                "constexpr": {name: value for name, value in kwargs.items()
                              if value is None or type(value) in (str, int, float, bool)},
                "num_warps": getattr(metadata, "num_warps", None),
                "num_stages": getattr(metadata, "num_stages", None),
                "shared": getattr(metadata, "shared", None)})
            return compiled
        return launch


class Operator:
    def __init__(self, root, definition, *, reference=False):
        import torch
        root = Path(root)
        if os.environ.get("TRITON_INTERPRET") not in (None, "", "0"):
            raise RuntimeError("native Triton compilation is required")
        validate_sources(root, root)
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError("the pinned ROCm runtime and MI355X are required")
        if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx950":
            raise RuntimeError("the observed gfx950 architecture is required")
        installed = importlib.import_module(definition["module"])
        if sha(installed.__file__) != definition["source_sha256"]:
            raise RuntimeError("installed runtime source differs from pinned SG520 source")
        for module, expected in definition["runtime_helpers"].items():
            if sha(importlib.import_module(module).__file__) != expected:
                raise RuntimeError("runtime helper source changed: " + module)
        path = root / definition["reference_file" if reference else "source_file"]
        self.source_hash = sha(path)
        module_name = definition["module"].rsplit(".", 1)[0] + "._aka_" + ("oracle_" if reference else "candidate_") + self.source_hash[:16]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.function = getattr(module, definition["wrapper"])
        if Path(inspect.unwrap(self.function).__code__.co_filename).resolve() != path.resolve():
            raise RuntimeError("public wrapper resolved outside the selected task source")
        self.probes = []
        self.tuners = {}
        self.expected_launches = None
        for name in definition["kernels"]:
            kernel = getattr(module, name)
            leaf = kernel
            for _ in range(10):
                if hasattr(leaf, "__code__"):
                    break
                leaf = getattr(leaf, "fn", None)
            if leaf is None or not hasattr(leaf, "__code__") or Path(leaf.__code__.co_filename).resolve() != path.resolve():
                raise RuntimeError("Triton kernel is not bound to the selected source: " + name)
            probe = KernelProbe(kernel, name)
            setattr(module, name, probe)
            self.probes.append(probe)
            tuner = kernel
            for _ in range(10):
                if hasattr(tuner, "configs"):
                    self.tuners[name] = (tuner, tuple(tuner.configs))
                    break
                tuner = getattr(tuner, "fn", None)
                if tuner is None:
                    break

    def select_launch_contract(self, launches):
        """Select an observed original config in this isolated task module.

        The original host wrapper and its complete config definitions remain
        frozen. Each case replays its recorded warps/stages in both legs;
        compilation shared-memory usage remains an optimization result.
        """
        require(len(launches) == len(self.probes), "wrong selected launch count")
        expected = {row["kernel"]: row for row in launches}
        require(set(expected) == {p.name for p in self.probes}, "wrong selected native kernel")
        for probe in self.probes:
            options = expected[probe.name]["compiled"]
            require(probe.name in self.tuners, "recorded launch has no original native config selector")
            tuner, configs = self.tuners[probe.name]
            matches = [config for config in configs
                       if config.num_warps == options["num_warps"] and config.num_stages == options["num_stages"]]
            require(len(matches) == 1, "recorded compiler config is missing or ambiguous in frozen source")
            tuner.configs = matches
            tuner.cache.clear()
        self.expected_launches = expected

    def __call__(self, args):
        before = sum(len(p.launches) for p in self.probes)
        result = self.function(**args)
        if sum(len(p.launches) for p in self.probes) != before + 1:
            raise RuntimeError("wrapper did not execute exactly one declared GPU kernel")
        if self.expected_launches is not None:
            for probe in self.probes:
                actual, expected = probe.launches[-1], self.expected_launches[probe.name]
                require(canonical(actual["grid"]) == canonical(expected["grid"])
                        and canonical(actual["constexpr"]) == canonical(expected["constexpr"])
                        and actual["num_warps"] == expected["compiled"]["num_warps"]
                        and actual["num_stages"] == expected["compiled"]["num_stages"],
                        "actual selected native launch differs from its captured case contract")
        return result

    def proof(self):
        return {"source_sha256": self.source_hash,
                "engaged_kernels": [p.name for p in self.probes if p.launches],
                "last_launches": [p.launches[-1] for p in self.probes if p.launches],
                "production_namespace_rebound": False}
