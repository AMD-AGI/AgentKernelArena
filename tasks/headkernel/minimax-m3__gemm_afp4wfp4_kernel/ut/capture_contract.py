"""Metadata admission for the packed, unshuffled FP4 GEMM boundary."""


def validate_abi(tensors, controls):
    for name in ("x", "w", "x_scales", "w_scales"):
        tensor = tensors[name]
        if tensor["dtype"].removeprefix("torch.") != "uint8" or len(tensor["shape"]) != 2:
            raise ValueError(name + " must be a two-dimensional uint8 view")
        strides = tensor.get("stride", tensor.get("strides"))
        if strides is None or len(strides) != 2 or any(type(n) is not int or n <= 0 for n in tensor["shape"] + strides):
            raise ValueError(name + " must have positive physical dimensions and strides")
        if tensor.get("storage_offset", 0) < 0:
            raise ValueError("negative storage offset")
    m, k_bytes = tensors["x"]["shape"]
    n, w_k = tensors["w"]["shape"]
    if w_k != k_bytes or k_bytes % 16:
        raise ValueError("packed K must match and contain whole 32-value scale groups")
    for name, rows in (("x_scales", m), ("w_scales", n)):
        shape = tensors[name]["shape"]
        if shape[0] < rows or shape[1] < k_bytes // 16:
            raise ValueError("scale view cannot address every live K group")
    if type(controls["skip_reduce"]) is not bool or type(controls["use_splitk_bf16"]) is not bool:
        raise ValueError("reduction controls must be explicit booleans")
    dtype = controls["dtype"]
    if dtype is None:
        dtype = (tensors.get("y") or {}).get("dtype", "")
    if not isinstance(dtype, str) or dtype.removeprefix("torch.") not in ("bfloat16", "float16"):
        raise ValueError("unsupported wrapper output dtype")
    return {"M": m, "N": n, "K_packed_bytes": k_bytes, "K_logical_values": 2 * k_bytes}


def validate_launch(abi, launch):
    for key, expected in (("M", abi["M"]), ("N", abi["N"]), ("K", abi["K_packed_bytes"])):
        if launch["arguments"].get(key) != expected:
            raise ValueError("native launch disagrees with operand ABI: " + key)
    for key in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M", "NUM_KSPLIT", "SPLITK_BLOCK_SIZE"):
        value = launch["arguments"].get(key)
        if type(value) is not int or value <= 0:
            raise ValueError("missing/invalid resolved native launch control: " + key)
