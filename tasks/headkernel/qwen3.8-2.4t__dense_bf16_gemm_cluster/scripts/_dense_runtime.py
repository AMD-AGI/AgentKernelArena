"""Protected image-native GEMM routing and independent numerical checks."""
from __future__ import annotations

import importlib
import math
from pathlib import Path

from _routing_contract import case_params, file_sha, require

BACKENDS = {"hipblaslt": "hipb_gemm", "torch": "torch_gemm", "asm": "asm_gemm",
            "flydsl": "flydsl_gemm", "triton": "triton_gemm"}


def route_config(spec, route):
    require(route in ("recorded", "torch", "triton"), "unapproved dense route")
    if route == "recorded":
        return {"libtype": spec["expected_backend"], "solidx": int(spec["expected_solidx"]),
                "splitK": int(spec["expected_split_k"]), "kernelName": spec["expected_kernel"]}
    return {"libtype": route, "solidx": 0, "splitK": 0, "kernelName": "native"}


class DenseCase:
    """Only a checked route name is supplied by the editable source policy.

    Image-owned backend functions receive the frozen public GEMM argument
    contract. No candidate Python module or callback is imported.
    """

    def __init__(self, torch, meta, spec, context, seed, *, device="cuda", native=None):
        self.torch, self.meta, self.spec = torch, meta, spec
        self.context = context
        self.params = case_params(meta, spec)
        self.route = context["routes"][spec["sig"]]
        self.config = route_config(spec, self.route)
        self.native = importlib.import_module("aiter.tuned_gemm") if native is None else native
        self.native_path = Path(self.native.__file__).resolve()
        require(file_sha(self.native_path) == context["native_source_sha256"], "live SG520 native dispatcher source differs")
        backend = self.config["libtype"]
        require(backend in BACKENDS, "recorded backend is outside the frozen native interface")
        name = BACKENDS[backend]
        self.function = getattr(self.native, name)
        require(self.native.solMap[backend] is self.function
                and self.function.__globals__ is self.native.__dict__
                and Path(self.function.__code__.co_filename).resolve() == self.native_path,
                "image backend does not bind its actual source body")
        self.reference = torch.nn.functional.linear
        self.args = {
            "A": torch.empty_strided((spec["m"], spec["k"]), tuple(spec["a_stride"]), dtype=torch.bfloat16, device=device),
            "B": torch.empty_strided((spec["n"], spec["k"]), tuple(spec["b_stride"]), dtype=torch.bfloat16, device=device),
        }
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.input_seed = seed
        self.cpu_inputs = None
        self.generated = 0
        self.calls = 0
        self.previous_output = None

    def fresh_inputs(self):
        torch = self.torch
        self.cpu_inputs = {name: torch.randn(tuple(value.shape), generator=self.generator,
                            device="cpu", dtype=torch.bfloat16).mul_(0.125)
                           for name, value in self.args.items()}
        for name, tensor in self.args.items():
            tensor.copy_(self.cpu_inputs[name])
        self.generated += 1

    def invoke(self):
        backend = self.config["libtype"]
        require(self.native.solMap[backend] is self.function
                and getattr(self.native, BACKENDS[backend]) is self.function,
                "native backend binding changed")
        output = self.function(self.args["A"], self.args["B"], self.config["solidx"],
                               None, self.torch.bfloat16, None, None, None, False,
                               config=dict(self.config))
        require(tuple(output.shape) == tuple(self.params["output_shape"])
                and output.dtype == self.torch.bfloat16 and output.device == self.args["A"].device,
                "native GEMM output ABI differs")
        pointers = {value.untyped_storage().data_ptr() for value in self.args.values()}
        require(output.untyped_storage().data_ptr() not in pointers, "native output aliases an input")
        if self.previous_output is not None:
            require(output.untyped_storage().data_ptr() != self.previous_output.untyped_storage().data_ptr(),
                    "native host calls reuse a persistent output buffer")
        self.previous_output = output
        self.calls += 1
        return output

    def snapshot_and_reference(self, output):
        """Own observations on CPU before constructing the reference output."""
        torch = self.torch
        actual = output.detach().cpu().clone()
        observed = {name: value.detach().cpu().clone() for name, value in self.args.items()}
        for name in self.args:
            require(torch.equal(observed[name].view(torch.uint8), self.cpu_inputs[name].view(torch.uint8)),
                    "timed native GEMM mutated input " + name)
        require(torch.isfinite(actual.float()).all().item(), "timed native output is not finite")
        # Reference operands are reconstructed from protected CPU truth. No
        # candidate-visible expected output exists during its execution.
        device = self.args["A"].device
        reference_args = {name: value.to(device=device, copy=True) for name, value in self.cpu_inputs.items()}
        expected_gpu = self.reference(reference_args["A"], reference_args["B"], bias=None)
        require(expected_gpu.untyped_storage().data_ptr() != output.untyped_storage().data_ptr(),
                "reference output aliases candidate output")
        expected = expected_gpu.detach().cpu().clone()
        del expected_gpu, reference_args
        return actual, expected

    def binding(self):
        return {"native_source_sha256": self.context["native_source_sha256"],
                "backend": self.config["libtype"], "function": BACKENDS[self.config["libtype"]],
                "native_config": dict(self.config), "source_file": str(self.native_path),
                "python_invocations": self.calls, "editable_python_executed": False}


class ReplayState:
    def __init__(self, case, compare):
        self.case, self.compare = case, compare
        self.torch = case.torch
        self.tol = float(case.spec.get("tol", case.meta["tol"]))
        self.outputs = None
        self.needs_validation = False
        self.capture_pending = False
        self.checked = 0
        self.worst_error = 0.0

    def prepare(self):
        self.validate()
        self.capture_pending = False
        self.case.fresh_inputs()
        if self.outputs is not None:
            self.outputs.fill_(float("nan"))
        self.needs_validation = True

    def run(self):
        self.outputs = self.case.invoke()
        self.capture_pending = self.torch.cuda.is_current_stream_capturing()
        return self.outputs

    def validate(self):
        if self.outputs is None or self.capture_pending or not self.needs_validation:
            return
        actual, expected = self.case.snapshot_and_reference(self.outputs)
        require(actual.device.type == expected.device.type == "cpu", "expected output escaped CPU ownership")
        ok, error = self.compare(actual, expected, self.tol)
        require(ok and math.isfinite(error), f"timed dense output failed tolerance {self.tol}: {error}")
        self.worst_error = max(self.worst_error, error)
        self.checked += 1
        self.needs_validation = False

    def validation(self):
        return {"checked_invocations": self.checked, "fresh_input_sets": self.case.generated,
                "input_seed": self.case.input_seed, "expected_device": "cpu", "inputs_unchanged": True,
                "reference_after_candidate_snapshot": True, "tolerance": self.tol,
                "max_rel_err": self.worst_error, "outputs_poisoned_before_replay": True,
                "inputs_varied": ["A", "B"], "native_binding": self.case.binding()}


class TimedGraph:
    def _bind(self, rerun, outputs=None):
        self._rerun, self.outputs = rerun, outputs

    def rerun(self):
        return self._rerun()
