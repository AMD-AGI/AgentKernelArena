"""Storage size and arithmetic precision of the dtypes a v3 task can declare."""
from __future__ import annotations

from ..task_contract import DTYPES

BYTES_PER_ELEMENT = {
    "float64": 8, "int64": 8,
    "float32": 4, "int32": 4,
    "bfloat16": 2, "float16": 2, "int16": 2,
    "float8_e4m3fn": 1, "float8_e4m3fnuz": 1, "float8_e5m2": 1, "float8_e5m2fnuz": 1,
    "float8_e8m0fnu": 1, "int8": 1, "uint8": 1, "bool": 1,
    # Two fp4 values per stored element; shapes count stored elements.
    "float4_e2m1fn_x2": 1,
}
assert set(BYTES_PER_ELEMENT) == set(DTYPES), "every contract dtype needs a storage size"

# Bits per value when a tensor is a matrix-multiply operand, before any packing rule.
_OPERAND_BITS = {
    "float64": 64, "float32": 32, "int32": 32, "int64": 64,
    "bfloat16": 16, "float16": 16, "int16": 16,
    "float8_e4m3fn": 8, "float8_e4m3fnuz": 8, "float8_e5m2": 8, "float8_e5m2fnuz": 8,
    "int8": 8, "uint8": 8,
    "float4_e2m1fn_x2": 4,
}


def tensor_bytes(shape: tuple[int, ...], dtype: str) -> int:
    count = 1
    for extent in shape:
        count *= extent
    return count * BYTES_PER_ELEMENT[dtype]


def operand_bits(dtype: str, packed_values_per_element: int = 1) -> int:
    """Bits per logical value of a matrix operand stored ``packed_values_per_element`` to an element."""
    return _OPERAND_BITS[dtype] // packed_values_per_element


def matrix_peak_key(bits: int, *, integer: bool = False, half: bool = False) -> str:
    """The datasheet matrix peak for a product whose lowest-precision operand has ``bits`` bits per value.

    4- and 6-bit operands use the MX fp4 rate, the fastest one, so the floor stays a lower bound
    even where the hardware has no native format of that width. ``half`` selects the fp16 peak
    over the bf16 one for 16-bit operands that are all float16.
    """
    if bits <= 6:
        return "matrix_fp4"
    if bits <= 8:
        return "matrix_int8" if integer else "matrix_fp8"
    if bits <= 16:
        return "matrix_fp16" if half else "matrix_bf16"
    if bits <= 32:
        return "matrix_fp32"
    return "matrix_fp64"
