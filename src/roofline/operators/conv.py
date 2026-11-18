"""Direct convolutions (Tier 2).

FLOPs = 2 * N * C_out * C_in_per_group * (taps that fall inside the input, summed over the outputs).
This is the direct count (TraceLens's ``CONV.flops_func``), less the taps that land in zero padding:
those multiply zeros, so no kernel has to evaluate them.

Geometry:
- A ``kernel_size`` axis means a valid, stride-1 convolution (every such reference was checked).
- Otherwise the kernel extent comes from the declared weight layout (``WEIGHT_LAYOUTS``); the
  layouts are the ones the task descriptions spell out, and a test checks each description against
  the table. Stride and padding follow from the input and output extents.
- A weight layout that is not in the table keeps the old bound of one tap per output, which holds
  for any geometry, and says so in its notes.

Winograd. A weight passed in the Winograd F(4x4, 3x3) domain makes the operator 36 products per
4x4 output tile (6 per 4 outputs on each axis), which is what the task's baseline computes. For
float32 tasks no other Winograd variant is assumed: with the products in float32, F(4x4, 3x3) and
F(6x6, 3x3) reach about 121 dB and 115 dB SQNR, below the 124-130 dB those tasks require.

16-bit tasks are checked against their baseline's error (``baseline_relative``), not a fixed SQNR,
so that argument does not carry over, and each 3-tap, stride-1 axis is counted at F(4, 3)'s 6
multiplies per 4 outputs: 2.25 products per output in 2-D instead of 9. That is below every
Winograd variant that could pass. One 16-bit product per transformed element costs about 96 dB
of SQNR against float32 products (F(4x4, 3x3) would reach about 25 dB, far below a bf16 direct
convolution), and larger tiles are worse, so a passing variant needs three part products:
3 * (1 + 2/m)^2 products per output for tile size m, more than 3 for every m (in 3-D,
3 * (1 + 2/m)^3 against 3.375, for every tile under 50).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..dtypes import operand_bits
from ..interface import CaseShapes
from . import Estimate, OperatorModel, register
from .gemm import add_product, mm

# Winograd F(4, 3): 6 multiplies for every 4 outputs of a 3-tap, stride-1 axis.
WINOGRAD_F43_MULTIPLIES_PER_OUTPUT = Fraction(6, 4)

# Declared weight layout (shape expressions) -> (kernel extent per spatial axis, Winograd-domain weights).
WEIGHT_LAYOUTS: dict[tuple[str, ...], tuple[int, bool]] = {
    ("out_channels", "in_channels"): (1, False),            # pointwise filters
    ("out_channels", "9", "in_channels"): (3, False),       # 3x3 taps in front of the channels
    ("out_channels", "kernel_taps_padded"): (7, False),     # 7x7 x C_in taps, zero-padded columns
    ("36", "out_channels", "in_channels"): (3, True),       # 3x3 filters in the F(4x4, 3x3) domain
    ("k", "27 * c"): (3, False),                            # 3x3x3 taps, input channels fastest
}


@dataclass(frozen=True)
class Conv:
    batch: str
    in_per_group: str
    out_channels: str
    x: str
    weight: str
    output: str
    # Input spatial axes, in the order of the output's spatial extents.
    spatial: tuple[str, ...] = ()
    # Floor-mode pooling straight after the convolution (no normalization over the whole map in
    # between): per spatial axis, the conv positions actually used, ``pool * pooled``.
    used: tuple[str, ...] = ()


def in_bounds_taps(extent: int, out: int, kernel: int, stride: int, padding: int) -> int:
    """Taps of all ``out`` positions of one axis that read a real input element."""
    return sum(1 for position in range(out) for tap in range(kernel)
               if 0 <= position * stride + tap - padding < extent)


def stride_and_padding(extent: int, out: int, kernel: int) -> tuple[int, int] | None:
    """A stride and padding for which ``extent`` gives ``out`` positions.

    Small extents can admit several; the one with the fewest in-bounds taps is returned, so the count
    stays a lower bound whichever the task uses.
    """
    best = None
    for stride in range(1, extent + 1):
        for padding in range(kernel):
            if (extent + 2 * padding - kernel) // stride + 1 == out:
                taps = in_bounds_taps(extent, out, kernel, stride, padding)
                if best is None or taps < best[0]:
                    best = (taps, stride, padding)
    return None if best is None else best[1:]


def _output_extents(shapes: CaseShapes, spec: Conv) -> tuple[int, ...]:
    output = shapes.tensor(spec.output)
    named = [extent for extent, expression in zip(output.shape, output.expressions)
             if str(expression) not in (spec.batch, spec.out_channels)]
    if len(named) == len(spec.spatial):
        return tuple(named)
    return tuple(output.shape[-len(spec.spatial):])  # channels-first


def conv_flops(shapes: CaseShapes, spec: Conv) -> tuple[int, list[str]]:
    batch, cin, cout = (shapes.value(e) for e in (spec.batch, spec.in_per_group, spec.out_channels))
    if spec.spatial and "kernel_size" in shapes.symbols:
        kernel, winograd_domain = shapes.symbols["kernel_size"], False
        extents = [shapes.symbols[axis] for axis in spec.spatial]
        outs = ([shapes.value(e) for e in spec.used] if spec.used
                else [extent - kernel + 1 for extent in extents])
        geometry = [(1, 0)] * len(extents)  # valid, stride 1
    else:
        layout = WEIGHT_LAYOUTS.get(tuple(str(e) for e in shapes.tensor(spec.weight).expressions))
        extents = [shapes.symbols[axis] for axis in spec.spatial]
        outs = list(_output_extents(shapes, spec)) if spec.spatial else []
        geometry = ([stride_and_padding(extent, out, layout[0]) for extent, out in zip(extents, outs)]
                    if layout else [None])
        if layout is None or not spec.spatial or any(g is None for g in geometry):
            output = shapes.tensor(spec.output)
            elements = 1
            for extent in output.shape:
                elements *= extent
            return (2 * cin * elements, ["kernel_geometry_undeclared_one_tap_counted"])
        kernel, winograd_domain = layout
    sixteen_bit = all(operand_bits(shapes.tensor(name).dtype) == 16 for name in (spec.x, spec.weight))
    work, winograd_counted = Fraction(1), False
    for extent, out, (stride, padding) in zip(extents, outs, geometry):
        taps = in_bounds_taps(extent, out, kernel, stride, padding)
        if winograd_domain:
            work *= out * WINOGRAD_F43_MULTIPLIES_PER_OUTPUT
        elif sixteen_bit and kernel == 3 and stride == 1 and out * WINOGRAD_F43_MULTIPLIES_PER_OUTPUT < taps:
            work *= out * WINOGRAD_F43_MULTIPLIES_PER_OUTPUT
            winograd_counted = True
        else:
            work *= taps
    notes = (["winograd_f4x3_domain_weights"] if winograd_domain
             else ["winograd_f4x3_counted: 16-bit 3x3 stride-1"] if winograd_counted else [])
    return int(2 * batch * cout * cin * work), notes


def conv_model(spec: Conv):
    def estimate(shapes: CaseShapes) -> Estimate:
        flops, notes = conv_flops(shapes, spec)
        result = Estimate(notes=notes)
        add_product(shapes, result, mm("1", "1", "1", spec.x, spec.weight), flops)
        if spec.used and "kernel_size" in shapes.symbols:
            # Input rows beyond the last used window are never read.
            kernel, x = shapes.symbols["kernel_size"], shapes.tensor(spec.x)
            fraction = Fraction(1)
            for index, axis in enumerate(spec.spatial):
                extent = shapes.symbols[axis]
                fraction *= Fraction(min(extent, shapes.value(spec.used[index]) + kernel - 1), extent)
            result.bytes_override = (int(x.nbytes * fraction)
                                     + sum(t.nbytes for t in shapes.inputs if t.name != spec.x)
                                     + sum(t.nbytes for t in shapes.outputs))
            result.notes.append("floor_mode_pool_trailing_positions_excluded")
        return result
    return estimate


NCHW = ("height", "width")
SPECS = {
    "conv2d": Conv("batch", "in_channels", "out_channels", "x", "weight", "out", NCHW),
    "conv2d_act_batchnorm": Conv("batch", "in_channels", "out_channels", "x", "conv_weight", "out", NCHW),
    "conv2d_batchnorm_scale": Conv("batch", "in_channels", "out_channels", "x", "conv_weight", "out", NCHW),
    "conv2d_depthwise": Conv("batch", "1", "channels", "x", "weight", "out", NCHW),
    "conv2d_groupnorm_scale_maxpool_clamp": Conv("batch", "in_channels", "out_channels", "x", "conv_weight",
                                                 "out", NCHW),
    "conv2d_instancenorm_div": Conv("batch", "in_channels", "out_channels", "x", "conv_weight", "out", NCHW),
    "conv2d_sub_tanh_sub_avgpool": Conv("batch", "in_channels", "out_channels", "x", "conv_weight", "out", NCHW,
                                        ("pool_size * pooled_height", "pool_size * pooled_width")),
    "conv2d_tanh_scale_bias_maxpool": Conv("batch", "in_channels", "out_channels", "x", "conv_weight", "out",
                                           NCHW, ("pool_size * pooled_height", "pool_size * pooled_width")),
    "conv3d": Conv("n", "c", "k", "x", "weight", "out", ("d", "h", "w")),
    # Softmax is over channels at each position, so trailing positions stay independent.
    "conv3d_softmax_maxpool2": Conv("batch", "in_channels", "out_channels", "x", "weight", "out",
                                    ("depth", "height", "width"),
                                    ("4 * pooled_depth", "4 * pooled_height", "4 * pooled_width")),
}
for _name, _spec in SPECS.items():
    register(_name, OperatorModel(f"convolution/{_name}", 1, conv_model(_spec)))
