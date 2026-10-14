"""Blackwell MXFP8 block-scaled linear for the H3 DiT.

Weights and activations use E4M3 values with one E8M0 scale per 32 values along K
in cuBLASLt's ``SWIZZLE_32_4_4`` layout. Subset of the Sol-H3 implementation
(models/minimax_h3/Sol-H3/h3_runtime/mxfp8.py), which follows the Apache-2.0
MiniMax-H3 swizzle/exponent selection at yujincheng08/minimax-h3-deploy@bae3c859.
Only the weight and activation quantize path is carried over; the fused
RMSNorm/SwiGLU producers are not used by this runtime.
"""
import torch
import triton
import triton.language as tl
from torch.nn.functional import ScalingType, SwizzleType, scaled_mm

_E4M3 = torch.float8_e4m3fn
_E8M0 = torch.float8_e8m0fnu


@triton.jit
def _mx_e8m0_from_amax(amax):
    bits = amax.to(tl.int32, bitcast=True)
    e0 = ((bits >> 23) & 0xFF) - 135
    threshold = (((e0 + 135) << 23) | 0x600000).to(tl.float32, bitcast=True)
    exponent = e0 + (amax > threshold).to(tl.int32)
    exponent = tl.maximum(exponent, -127)
    inverse = ((127 - exponent) << 23).to(tl.float32, bitcast=True)
    return exponent + 127, inverse


@triton.jit
def _mx_scale_offsets(row, column, column_blocks):
    tile = (row // 128) * column_blocks + (column // 4)
    return (
        tile * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (column % 4)
    )


@triton.jit
def _mxfp8_quant_kernel(
    input_ptr,
    quantized_ptr,
    scale_ptr,
    rows,
    k,
    num_groups,
    column_blocks,
    input_row_stride,
    BLOCK_ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
):
    row_program = tl.program_id(0)
    group_program = tl.program_id(1)
    row = row_program * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    group = group_program * GROUPS + tl.arange(0, GROUPS)
    column = group_program * (GROUPS * 32) + tl.arange(0, GROUPS * 32)
    row_mask = row < rows
    mask = row_mask[:, None] & (column < k)[None, :]
    value = tl.load(
        input_ptr
        + row[:, None].to(tl.int64) * input_row_stride
        + column[None, :],
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values = tl.reshape(value, [BLOCK_ROWS, GROUPS, 32])
    amax = tl.max(tl.abs(values), axis=2)
    scale_byte, inverse = _mx_e8m0_from_amax(amax)
    quantized = tl.reshape(values * inverse[:, :, None], [BLOCK_ROWS, GROUPS * 32])
    tl.store(
        quantized_ptr + row[:, None].to(tl.int64) * k + column[None, :],
        quantized.to(tl.float8e4nv),
        mask=mask,
    )
    scale_mask = row_mask[:, None] & (group < num_groups)[None, :]
    tl.store(
        scale_ptr + _mx_scale_offsets(row[:, None], group[None, :], column_blocks),
        scale_byte.to(tl.uint8),
        mask=scale_mask,
    )


def _scale_numel(rows, k):
    num_groups = k // 32
    return -(-rows // 128) * 128 * (-(-num_groups // 4) * 4)


def mxfp8_quantize_swizzled(input_):
    """Quantize a row-major BF16 ``[rows, K]`` tensor for block-scaled GEMM."""
    if input_.ndim != 2 or input_.stride(-1) != 1 or input_.dtype != torch.bfloat16:
        raise ValueError("expected a row-major BF16 [rows, K] tensor")
    rows, k = input_.shape
    if k % 32 != 0:
        raise ValueError(f"MXFP8 requires K % 32 == 0, got K={k}")
    quantized = torch.empty(rows, k, dtype=_E4M3, device=input_.device)
    scale = torch.zeros(_scale_numel(rows, k), dtype=torch.uint8, device=input_.device)
    if rows == 0:
        return quantized, scale
    num_groups = k // 32
    column_blocks = -(-num_groups // 4)
    block_rows, groups = 32, 8
    grid = (triton.cdiv(rows, block_rows), triton.cdiv(num_groups, groups))
    _mxfp8_quant_kernel[grid](
        input_, quantized, scale, rows, k, num_groups, column_blocks,
        input_.stride(0), BLOCK_ROWS=block_rows, GROUPS=groups, num_warps=4)
    return quantized, scale


class MXFP8Linear(torch.nn.Module):
    """Bias-free block-scaled MXFP8 linear built from a BF16 ``[N, K]`` weight."""

    def __init__(self, weight):
        super().__init__()
        if not isinstance(weight, torch.Tensor) or type(weight) not in (torch.Tensor, torch.nn.Parameter):
            raise TypeError(f"MXFP8Linear needs a plain tensor weight, got {type(weight).__name__}")
        weight = weight.detach()
        if weight.dtype != torch.bfloat16:
            raise ValueError(f"MXFP8Linear expects a BF16 weight, got {weight.dtype}")
        if weight.shape[0] % 16 != 0:
            raise ValueError(f"MXFP8 requires N % 16 == 0, got N={weight.shape[0]}")
        self.out_features, self.in_features = weight.shape
        quantized, scale = mxfp8_quantize_swizzled(weight.contiguous())
        self.register_buffer("weight", quantized, persistent=False)
        self.register_buffer("weight_scale", scale.view(_E8M0), persistent=False)

    @property
    def storage_bytes(self):
        return self.weight.nbytes + self.weight_scale.nbytes

    def forward(self, input_):
        leading_shape = input_.shape[:-1]
        flat = input_.reshape(-1, input_.shape[-1]).contiguous()
        activation, scale = mxfp8_quantize_swizzled(flat)
        output = scaled_mm(
            activation,
            self.weight.t(),
            scale_a=scale.view(_E8M0),
            scale_recipe_a=ScalingType.BlockWise1x32,
            swizzle_a=SwizzleType.SWIZZLE_32_4_4,
            scale_b=self.weight_scale,
            scale_recipe_b=ScalingType.BlockWise1x32,
            swizzle_b=SwizzleType.SWIZZLE_32_4_4,
            output_dtype=torch.bfloat16,
        )
        return output.view(*leading_shape, self.out_features)
