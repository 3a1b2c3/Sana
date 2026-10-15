"""GPU checks for the MXFP8 fused producers. Skipped unless an SM100-family GPU and Triton are present.

Run on the DGX from the streamrender_h3 directory: python -m unittest tests.test_mxfp8_producers -v

Tolerances: E4M3 keeps 3 mantissa bits, so quantizing a value changes it by up to 2^-4 relative and
about 3.6% RMS; the bounds below leave margin over that and are judged on whole-tensor relative error.
"""
import unittest

import torch
import torch.nn.functional as F

try:
    from render import mxfp8
    HAVE_MXFP8 = True
except ImportError:
    HAVE_MXFP8 = False

SM100 = HAVE_MXFP8 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10
QUANT_BOUND = 0.05
ROUNDING_BOUND = 0.01


def relative_error(value, reference):
    return ((value.float() - reference.float()).norm() / reference.float().norm()).item()


def modulation_tables(rows, columns, device):
    """Shift/scale/gate chunks of one [rows, 6 * columns] tensor, as adaln_proj produces them."""
    packed = torch.randn(rows, 6 * columns, device=device, dtype=torch.bfloat16) * 0.1
    return packed.chunk(6, dim=-1)


def rmsnorm_modulate_reference(x, weight, scale, shift, index, eps):
    x = x.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.float()
    return normed * (1.0 + scale.float()[index]) + shift.float()[index]


@unittest.skipUnless(SM100, "needs an SM100-family GPU and Triton")
class ProducerTests(unittest.TestCase):
    rows, columns, eps = 300, 4096, 1e-6

    def setUp(self):
        torch.manual_seed(0)
        self.device = "cuda"
        self.x = torch.randn(self.rows, self.columns, device=self.device, dtype=torch.bfloat16)
        self.weight = torch.randn(self.columns, device=self.device, dtype=torch.bfloat16).abs() + 0.5
        self.index = torch.randint(0, 8, (self.rows,), device=self.device)
        self.shift, self.scale, self.gate, self.shift2, self.scale2, self.gate2 = modulation_tables(
            8, self.columns, self.device)

    def test_rmsnorm_modulate_matches_reference(self):
        activation = mxfp8.fused_rmsnorm_modulate_mxfp8(
            self.x, self.weight, self.scale, self.shift, self.index, self.eps)
        reference = rmsnorm_modulate_reference(self.x, self.weight, self.scale, self.shift, self.index, self.eps)
        self.assertEqual(tuple(activation.shape), (self.rows, self.columns))
        self.assertLess(relative_error(activation.dequantize(), reference), QUANT_BOUND)

    def test_residual_gate_rmsnorm_modulate_matches_reference(self):
        branch = torch.randn_like(self.x)
        hidden, activation = mxfp8.fused_residual_gate_rmsnorm_modulate_mxfp8(
            self.x, branch, self.gate, self.weight, self.scale, self.shift, self.index, self.eps)
        reference_hidden = self.x.float() + self.gate.float()[self.index] * branch.float()
        self.assertEqual(hidden.dtype, torch.bfloat16)
        self.assertLess(relative_error(hidden, reference_hidden), ROUNDING_BOUND)
        reference = rmsnorm_modulate_reference(
            reference_hidden, self.weight, self.scale, self.shift, self.index, self.eps)
        self.assertLess(relative_error(activation.dequantize(), reference), QUANT_BOUND)

    def test_silu_mul_uses_first_half_as_gate(self):
        width = 3072
        wide = torch.randn(self.rows, 2 * width, device=self.device, dtype=torch.bfloat16)
        first, second = wide.float().chunk(2, dim=-1)
        correct = F.silu(first) * second
        swapped = F.silu(second) * first
        produced = mxfp8.fused_silu_mul_mxfp8(wide).dequantize()
        self.assertEqual(tuple(produced.shape), (self.rows, width))
        self.assertLess(relative_error(produced, correct), QUANT_BOUND)
        self.assertGreater(relative_error(produced, swapped), 10 * relative_error(produced, correct))

    def test_linear_accepts_prequantized_activation(self):
        weight = torch.randn(1024, self.columns, device=self.device, dtype=torch.bfloat16) * 0.02
        linear = mxfp8.MXFP8Linear(weight)
        quantized, scale = mxfp8.mxfp8_quantize_swizzled(self.x)
        prequantized = mxfp8.MXActivation(quantized, scale, self.x.shape)
        self.assertTrue(torch.equal(linear(self.x), linear(prequantized)))
        self.assertLess(relative_error(linear(prequantized), F.linear(self.x, weight)), 2 * QUANT_BOUND)


if __name__ == "__main__":
    unittest.main()
