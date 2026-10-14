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

from .mxfp8 import MXFP8Linear

ENV_MODE = "STREAMRENDER_COMPUTE_QUANT"
ENV_BLOCKS = "STREAMRENDER_QUANT_BLOCKS"
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
