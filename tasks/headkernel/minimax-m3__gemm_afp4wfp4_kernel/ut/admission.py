"""Convert a verified actual capture ABI into the protected portable contract."""
from capture_contract import validate_abi, validate_launch
from evaluation_contract import canonical, require

POLICY = {"method": "cuda_graph", "warmup_iterations": 10, "benchmark_iterations": 100,
          "correctness_seeds": [0, 1, 2], "negative_controls": ["no_op", "wrong_output"],
          "refresh_inputs": "each_replay", "initialize_outputs": "each_replay",
          "validate_outputs": "each_replay"}
RUNTIME_ARGS = {"M", "N", "K", "stride_am", "stride_ak", "stride_bk", "stride_bn", "stride_ck",
                "stride_cm", "stride_cn", "stride_asm", "stride_ask", "stride_bsn", "stride_bsk"}


def oracle_policy(policy):
    require(set(policy) == {"metric", "tolerance", "basis"}, "explicit reviewed oracle policy required")
    require(policy["metric"] == "mixed_rms" and type(policy["tolerance"]) in (int, float)
            and 0 < policy["tolerance"] <= 0.02 and isinstance(policy["basis"], str) and bool(policy["basis"]),
            "oracle policy requires a justified mixed-RMS bound no larger than 0.02")
    return policy


def case_from_fixture(fixture, rank_counts, fixture_ref):
    controls = fixture["controls"]
    require(controls["packing"] == "e2m1_low_nibble_first" and controls["scale_format"] == "e8m0_group32_unshuffled",
            "captured FP4 packing contract differs")
    abi = validate_abi(fixture["inputs"], controls)
    require(len(controls["launches"]) == 1, "exactly one original GEMM launch is required")
    launch = controls["launches"][0]
    require(launch["kernel"] == "_gemm_afp4wfp4_kernel", "wrong native device kernel")
    validate_launch(abi, launch)
    arguments = launch["arguments"]
    require(RUNTIME_ARGS <= set(arguments), "native scalar stride controls are missing")
    # Derive these strides from the captured views; w is transposed by the wrapper.
    inputs = fixture["inputs"]
    expected = {"stride_am": inputs["x"]["stride"][0], "stride_ak": inputs["x"]["stride"][1],
                "stride_bk": inputs["w"]["stride"][1], "stride_bn": inputs["w"]["stride"][0],
                "stride_asm": inputs["x_scales"]["stride"][0], "stride_ask": inputs["x_scales"]["stride"][1],
                "stride_bsn": inputs["w_scales"]["stride"][0], "stride_bsk": inputs["w_scales"]["stride"][1]}
    require(all(arguments[k] == v for k, v in expected.items()), "launch strides do not address captured operands")
    aliases = [inputs[name]["alias"] for name in ("x", "w", "x_scales", "w_scales")]
    require(len(set(aliases)) == 4, "overlapping packed operands require an explicit reference extension")
    if inputs["y"] is not None:
        require(inputs["y"]["alias"] not in aliases, "output overlaps an immutable packed input")
    result = fixture["outputs"]["result"]
    require(result["alias"] not in aliases, "result aliases an immutable packed input")
    partial = controls["skip_reduce"] and arguments["NUM_KSPLIT"] > 1
    expected_shape = [arguments["NUM_KSPLIT"], abi["M"], abi["N"]] if partial else [abi["M"], abi["N"]]
    require(result["shape"] == expected_shape, "captured output shape disagrees with reduction controls")
    y = inputs["y"]
    if y is not None:
        require(y["shape"] == [abi["M"], abi["N"]], "preallocated output shape differs")
        require((result["alias"] == y["alias"]) is (not partial), "output alias disagrees with reduction controls")
    output_stride = result["stride"] if arguments["NUM_KSPLIT"] == 1 else [abi["M"] * abi["N"], abi["N"], 1]
    expected_output = {"stride_ck": 0, "stride_cm": output_stride[0], "stride_cn": output_stride[1]} if arguments["NUM_KSPLIT"] == 1 else dict(zip(("stride_ck", "stride_cm", "stride_cn"), output_stride))
    require(all(arguments[k] == v for k, v in expected_output.items()), "captured partial/output strides disagree with wrapper")
    tensors = {}
    for name, meta in {**inputs, "result": result}.items():
        if meta is not None:
            tensors[name] = {"role": "output" if name == "result" or name == "y" and not partial else "input",
                             "shape": meta["shape"], "strides": meta["stride"], "storage_offset": meta["storage_offset"],
                             "dtype": meta["dtype"].removeprefix("torch."), "device_type": "cuda"}
    scalars = {key: controls[key] for key in ("dtype", "config_requested", "skip_reduce", "use_splitk_bf16", "packing", "scale_format")}
    scalars.update(resolved_config={k: v for k, v in arguments.items() if k not in RUNTIME_ARGS},
                   output_dtype=result["dtype"], native_arguments=arguments, native_grid=launch["grid"])
    require(rank_counts and all(type(v) is int and v > 0 for v in rank_counts.values()), "actual case counts required")
    return {"case_id": fixture["case_key"], "occurrences": sum(rank_counts.values()), "calls_per_sample": 1,
            "tensors": tensors, "scalars": scalars, "fixture": fixture_ref,
            "occurrences_by_rank": rank_counts, "production_stage": fixture["served"]["stage"],
            "production_mode": fixture["origin"].removeprefix("served_"),
            "storage_contract": {"inputs": inputs, "outputs": fixture["outputs"]}}
