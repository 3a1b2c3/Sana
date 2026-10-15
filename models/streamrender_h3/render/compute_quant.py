"""Opt-in MXFP8 compute profile for the H3 DiT (single Blackwell GPU, e.g. GB300).

Replaces the attention (qkv_proj, out_proj) and FFN (fc1, fc2) linears of the
blocks in ``[first_block, last_block]`` with block-scaled MXFP8 GEMMs. Boundary
blocks, AdaLN projections, token refiners, VAE and text/audio encoders stay BF16,
matching the validated Sol-H3 profile (blocks 2-46 of 50). Installation is
one-way: the BF16 weights are released so memory drops with the compute.
"""
import gc
import os
from dataclasses import dataclass

import torch

from .mxfp8 import (
    MXFP8Linear,
    fused_residual_gate_rmsnorm_modulate_mxfp8,
    fused_rmsnorm_modulate_mxfp8,
    fused_silu_mul_mxfp8,
)

ENV_MODE = "STREAMRENDER_COMPUTE_QUANT"
ENV_BLOCKS = "STREAMRENDER_QUANT_BLOCKS"
ENV_FUSED = "STREAMRENDER_MXFP8_FUSED"
DEFAULT_FIRST_BLOCK = 2
DEFAULT_LAST_BLOCK = 46
LINEARS = (("attn", "qkv_proj"), ("attn", "out_proj"), ("mlp", "fc1"), ("mlp", "fc2"))


@dataclass(frozen=True)
class ComputeQuantizationReport:
    first_block: int
    last_block: int
    quantized_linears: int
    original_bytes: int
    quantized_bytes: int


class MXFP8Projection(torch.nn.Module):
    """MXFP8 GEMM with the DiT's ``(output, output_bias)`` linear return contract."""

    def __init__(self, source):
        super().__init__()
        if source.bias is not None:
            raise ValueError("MXFP8 projections must be bias-free")
        self.linear = MXFP8Linear(source.weight)

    def forward(self, x):
        """``x`` is a BF16 tensor or an ``MXActivation`` already quantized by a fused producer."""
        return self.linear(x), None


def validate_platform():
    if not torch.cuda.is_available():
        raise RuntimeError("MXFP8 compute quantization requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        raise RuntimeError(
            "MXFP8 compute quantization is enabled only for the SM100-family "
            f"(GB200/GB300) path; detected SM{capability[0]}{capability[1]}")
    missing = [name for name in ("ScalingType", "SwizzleType", "scaled_mm")
               if not hasattr(torch.nn.functional, name)]
    if not hasattr(torch, "float8_e8m0fnu"):
        missing.append("torch.float8_e8m0fnu")
    if missing:
        raise RuntimeError("The installed PyTorch build lacks MXFP8 support: " + ", ".join(missing))


def fused_enabled(mode):
    """Whether the fused producers are requested; they only exist on top of the MXFP8 GEMMs."""
    value = os.environ.get(ENV_FUSED, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{ENV_FUSED} must be 0 or 1, got {value!r}")
    if value == "1" and mode != "mxfp8":
        raise ValueError(f"{ENV_FUSED}=1 requires {ENV_MODE}=mxfp8")
    return value == "1"


def block_range():
    value = os.environ.get(ENV_BLOCKS, "")
    if not value:
        return DEFAULT_FIRST_BLOCK, DEFAULT_LAST_BLOCK
    first, _, last = value.partition("-")
    return int(first), int(last)


def install(dit, first_block=DEFAULT_FIRST_BLOCK, last_block=DEFAULT_LAST_BLOCK):
    """Quantize the linears of blocks ``first_block..last_block`` (inclusive)."""
    validate_platform()
    blocks = dit.blocks
    if not 0 <= first_block <= last_block < len(blocks):
        raise ValueError(f"Invalid MXFP8 block range [{first_block}, {last_block}] for {len(blocks)} blocks")
    original_bytes = quantized_bytes = linears = 0
    for index in range(first_block, last_block + 1):
        for parent_name, name in LINEARS:
            parent = getattr(blocks[index], parent_name)
            source = getattr(parent, name)
            projection = MXFP8Projection(source)
            original_bytes += source.weight.numel() * source.weight.element_size()
            quantized_bytes += projection.linear.storage_bytes
            linears += 1
            setattr(parent, name, projection)
            del source, projection
    gc.collect()
    torch.cuda.empty_cache()
    return ComputeQuantizationReport(first_block, last_block, linears, original_bytes, quantized_bytes)


def _fused_block_forward(block):
    """The causal block forward with each elementwise chain fused into the next GEMM's quantize.

    Same sequence as ``CausalMiniMaxH3DiTBlock.forward``: norm1+modulate -> attention ->
    gated residual -> norm2+modulate -> MLP -> gated residual. The producers keep the math in
    FP32 and round once, where the eager chain rounds to BF16 after every op, so outputs differ
    from the eager path by BF16 rounding, on top of the MXFP8 quantization itself.
    """
    def forward(x, *, adaln_input, combined_indices, rope_cache, adaln_params=None, **attn_kwargs):
        if adaln_params is None:
            adaln_params = block.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params
        h = fused_rmsnorm_modulate_mxfp8(
            x, block.norm1.weight, scale_msa, shift_msa, combined_indices, block.norm1.eps)
        h = block.attn(h, rope_cache=rope_cache, **attn_kwargs)
        x, h = fused_residual_gate_rmsnorm_modulate_mxfp8(
            x, h, gate_msa, block.norm2.weight, scale_mlp, shift_mlp, combined_indices, block.norm2.eps)
        up, _ = block.mlp.fc1(h)
        h, _ = block.mlp.fc2(fused_silu_mul_mxfp8(up))
        return (x + gate_mlp.index_select(0, combined_indices) * h).to(torch.bfloat16)
    return forward


def install_fused_producers(dit, first_block=DEFAULT_FIRST_BLOCK, last_block=DEFAULT_LAST_BLOCK):
    """Patch blocks ``first_block..last_block`` to emit MXFP8 activations straight from the fused chains.

    Call after ``install()`` with the same range; every patched block must already hold MXFP8
    qkv_proj, fc1 and fc2, because the producers' output is only consumable by those GEMMs.
    """
    blocks = dit.blocks
    if not 0 <= first_block <= last_block < len(blocks):
        raise ValueError(f"Invalid fused block range [{first_block}, {last_block}] for {len(blocks)} blocks")
    for index in range(first_block, last_block + 1):
        block = blocks[index]
        consumers = (block.attn.qkv_proj, block.mlp.fc1, block.mlp.fc2)
        if not all(isinstance(layer, MXFP8Projection) for layer in consumers):
            raise RuntimeError(f"Block {index} is not MXFP8-quantized; run install() over the same range first")
        if getattr(block, "adaln_condition_head", None) is not None:
            raise RuntimeError(f"Block {index} has an AdaLN condition head, which the fused forward does not apply")
        block.forward = _fused_block_forward(block)
    return last_block - first_block + 1
