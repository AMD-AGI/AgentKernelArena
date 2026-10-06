"""Protected FP4 replay state: CPU truth, complete storage, actual native launch."""
from binding import load_leg
from evaluation_contract import observe_case, require, strict_json
from fixture_codec import file_sha, restore, safe_file, views
from reference import tensor_reference


class ReferenceCalibrationError(AssertionError):
    pass


def compare(actual, expected, policy):
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("output shape/dtype differs")
    a, b = actual.double(), expected.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise AssertionError("unwritten/nonfinite output or reference")
    floor = b.square().mean().sqrt().clamp_min(1e-30)
    ratio = (a - b).abs() / torch.maximum(b.abs(), floor)
    if torch.any(ratio > policy["tolerance"]):
        raise AssertionError("independent packed-value oracle mismatch: " + str(float(ratio.max())))


class CheckedKernel:
    def __init__(self, original, expected):
        self.original, self.expected, self.launches = original, expected, []

    def __getattr__(self, name):
        return getattr(self.original, name)

    def __getitem__(self, grid):
        names = ("M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_ck",
                 "stride_cm", "stride_cn", "stride_asm", "stride_ask", "stride_bsn", "stride_bsk")
        call = self.original[grid]
        def invoke(*args, **kwargs):
            values = {**dict(zip(names, args[5:])), **kwargs}
            require(all(values.get(k) == v and type(values.get(k)) is type(v) for k, v in self.expected.items()),
                    "native launch ABI/config differs from capture")
            result = call(*args, **kwargs)
            self.launches.append({"kernel_name": str(getattr(result, "name", "")),
                                  "kernel_hash": str(getattr(result, "hash", ""))})
            return result
        return invoke


class FP4Case:
    def __init__(self, root, dataset, case, policy, *, leg="candidate", candidate_workspace=None,
                 defer_candidate_check=False, diagnostic=False):
        import torch
        self.torch, self.case, self.policy = torch, case, policy
        path = safe_file(dataset, case["fixture"]["path"])
        require(file_sha(path) == case["fixture"]["sha256"], "fixture identity changed")
        self.fixture = strict_json(path.read_text())
        self.base_raw, base = restore(path.parent, self.fixture, "inputs", allow_diagnostic=diagnostic)
        _, captured = restore(path.parent, self.fixture, "outputs", allow_diagnostic=diagnostic)
        self.base_expected = tensor_reference(base, case["scalars"])
        try:
            compare(captured["result"], self.base_expected, policy)
        except AssertionError as error:
            raise ReferenceCalibrationError("captured output disagrees with independent FP4 reference") from error
        self.raw = {alias: value.to("cuda") for alias, value in self.base_raw.items()}
        self.inputs = views(self.raw, self.fixture["inputs"])
        controls = case["scalars"]
        self.result = None if controls["skip_reduce"] and controls["resolved_config"]["NUM_KSPLIT"] > 1 else self.inputs["y"]
        meta = self.fixture["outputs"]["result"]
        self.output_bytes = torch.zeros(meta["storage_nbytes"], dtype=torch.bool)
        indices = torch.tensor(meta["storage_offset"], dtype=torch.int64)
        for size, stride in zip(meta["shape"], meta["stride"]):
            indices = indices.unsqueeze(-1) + torch.arange(size) * stride
        offsets = indices.reshape(-1) * meta["element_size"]
        for byte in range(meta["element_size"]):
            self.output_bytes[offsets + byte] = True
        self.call, self.proof = load_leg(root, leg, use_splitk_bf16=controls["use_splitk_bf16"],
                                          candidate_workspace=candidate_workspace)
        namespace = self.call.__globals__
        self.probe = CheckedKernel(namespace["_triton_gemm_afp4wfp4_kernel"], controls["native_arguments"])
        namespace["_triton_gemm_afp4wfp4_kernel"] = self.probe
        self.graph = None
        self.truth = None
        self.calls = 0
        self.reset(0)
        self.initialize()
        self.invoke()
        try:
            self.verify(self.truth)
        except AssertionError:
            if not defer_candidate_check:
                raise

    def invoke(self):
        torch, controls = self.torch, self.case["scalars"]
        dtype = None if controls["dtype"] is None else getattr(torch, controls["dtype"].removeprefix("torch."))
        self.result = self.call(self.inputs["x"], self.inputs["w"], self.inputs["x_scales"], self.inputs["w_scales"],
                                dtype, self.inputs["y"], dict(controls["resolved_config"]), controls["skip_reduce"])
        self.calls += 1
        require(len(self.probe.launches) == self.calls, "wrapper did not launch submitted kernel exactly once")

    def reset(self, seed):
        torch = self.torch
        cpu_raw = {alias: value.clone() for alias, value in self.base_raw.items()}
        cpu = views(cpu_raw, self.fixture["inputs"])
        m = cpu["x"].shape[0]
        generator = torch.Generator(device="cpu").manual_seed(seed)
        order = torch.randperm(m, generator=generator) if seed else torch.arange(m)
        flips = torch.randint(0, 2, (m,), generator=generator, dtype=torch.uint8) if seed else torch.zeros(m, dtype=torch.uint8)
        cpu["x"].copy_(cpu["x"].index_select(0, order) ^ (flips[:, None] * 0x88))
        cpu["x_scales"][:m].copy_(cpu["x_scales"][:m].index_select(0, order))
        row_dim = 1 if self.base_expected.ndim == 3 else 0
        expected = self.base_expected.index_select(row_dim, order)
        sign = 1 - 2 * flips.to(torch.int32)
        expected = expected * (sign[None, :, None] if row_dim else sign[:, None])
        expected = expected.to(self.base_expected.dtype)
        for alias, value in cpu_raw.items():
            self.raw[alias].copy_(value)
        self.truth = (cpu_raw, expected)
        return self.truth

    def initialize(self):
        if self.result is not None:
            self.result.fill_(float("nan"))
        output_y = self.inputs["y"]
        if output_y is not None and not (self.case["scalars"]["skip_reduce"] and self.case["scalars"]["resolved_config"]["NUM_KSPLIT"] > 1):
            output_y.fill_(float("nan"))

    def observe(self):
        require(self.result is not None, "native output was not produced")
        return observe_case(self.case, {**{k: v for k, v in self.inputs.items() if v is not None}, "result": self.result}, self.case["scalars"])

    def verify(self, truth):
        require(truth is self.truth, "stale replay truth")
        torch = self.torch
        torch.cuda.synchronize()
        self.observe()
        actual = self.result.detach().to("cpu", copy=True)
        result_storage = self.result.untyped_storage()
        immutable = {self.fixture["inputs"][name]["alias"] for name in ("x", "w", "x_scales", "w_scales")}
        y = self.inputs["y"]
        expected_alias = self.fixture["outputs"]["result"]["alias"]
        if y is not None:
            y_alias = self.fixture["inputs"]["y"]["alias"]
            if (result_storage.data_ptr() == y.untyped_storage().data_ptr()) != (expected_alias == y_alias):
                raise AssertionError("native output alias differs from capture")
            if y_alias != expected_alias:
                immutable.add(y_alias)
            elif not torch.equal(self.raw[y_alias].to("cpu")[~self.output_bytes], truth[0][y_alias][~self.output_bytes]):
                raise AssertionError("candidate wrote outside the defined output view")
        if result_storage.nbytes() != self.fixture["outputs"]["result"]["storage_nbytes"]:
            raise AssertionError("native output physical storage differs")
        for alias in immutable:
            if result_storage.data_ptr() == self.raw[alias].untyped_storage().data_ptr():
                raise AssertionError("output aliases immutable input")
            if not torch.equal(self.raw[alias].to("cpu"), truth[0][alias]):
                raise AssertionError("candidate mutated complete input storage")
        compare(actual, truth[1], self.policy)
        self.truth = None
        return True

    def capture_graph(self):
        torch = self.torch
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.invoke()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.invoke()
        self.proof["graph_captured"] = True

    def replay(self):
        if self.graph is None:
            self.invoke()
        else:
            self.graph.replay()

    def check_once(self, seed):
        truth = self.reset(seed)
        self.initialize()
        self.replay()
        if self.graph is not None:
            self.proof["graph_replayed"] = True
        return self.verify(truth)

    def measure(self, replay):
        begin, end = self.torch.cuda.Event(enable_timing=True), self.torch.cuda.Event(enable_timing=True)
        begin.record()
        replay()
        end.record()
        end.synchronize()
        return float(begin.elapsed_time(end))
