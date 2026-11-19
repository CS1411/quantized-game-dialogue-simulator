"""Milestone 4: post-training quantization, written out by hand (the 'Shrink a Neural Network with Quantisation' recipe).

Affine mapping: x ~ scale * (q - zero_point), with q an integer in [qmin, qmax].
  scale = (hi - lo) / (qmax - qmin)       zero_point = qmin - round(lo / scale)

Weights: quantized once after training, symmetric (zero_point 0), one scale per output row ("per channel").
Activations (the input of each Linear):
  - DYNAMIC: range taken from the live tensor on every call (so it adapts to the input, at some runtime cost);
  - STATIC:  range learned once by observers on calibration data, then frozen.
The matrix product itself runs on integers: sum(xq * wq) - zero_point * sum(wq), rescaled by scale_x * scale_w. The integers
are carried in float32 tensors, which is exact here because every partial sum stays below 2**24; tests check it against a
plain integer computation.
"""
import copy
import warnings

import torch
import torch.nn as nn

ACT_MIN, ACT_MAX = -128, 127  # 8-bit activations


def affine_params(lo: torch.Tensor, hi: torch.Tensor, qmin: int = ACT_MIN, qmax: int = ACT_MAX):
    "scale and zero_point so that [lo, hi] (widened to include 0, so zero is exactly representable) maps onto [qmin, qmax]."
    lo, hi = torch.clamp(lo, max=0.0), torch.clamp(hi, min=0.0)
    scale = torch.clamp((hi - lo) / (qmax - qmin), min=1e-8)
    zero_point = torch.clamp(qmin - torch.round(lo / scale), qmin, qmax)
    return scale, zero_point


def quantize(x, scale, zero_point, qmin: int = ACT_MIN, qmax: int = ACT_MAX):
    return torch.clamp(torch.round(x / scale + zero_point), qmin, qmax)


def dequantize(q, scale, zero_point):
    return (q - zero_point) * scale


def quantize_weight(w: torch.Tensor, bits: int = 8):
    "Symmetric per-output-row quantization of a weight matrix. Returns (int8 tensor holding values in +-(2**(bits-1)-1), scale per row)."
    qmax = 2 ** (bits - 1) - 1
    scale = torch.clamp(w.abs().amax(dim=1) / qmax, min=1e-8)
    return torch.clamp(torch.round(w / scale[:, None]), -qmax, qmax).to(torch.int8), scale


class QuantLinear(nn.Module):
    """A Linear layer holding int weights. act_scale/act_zero None means dynamic activation quantization, else static."""

    def __init__(self, linear: nn.Linear, bits: int = 8, act_scale=None, act_zero=None):
        super().__init__()
        self.bits = bits
        wq, ws = quantize_weight(linear.weight.data, bits)
        self.register_buffer("w_q", wq)
        self.register_buffer("w_scale", ws)
        self.register_buffer("w_sum", wq.float().sum(dim=1))
        self.register_buffer("bias", None if linear.bias is None else linear.bias.data.clone())
        static = act_scale is not None
        self.register_buffer("act_scale", act_scale.clone() if static else None)
        self.register_buffer("act_zero", act_zero.clone() if static else None)

    @property
    def is_static(self) -> bool:
        return self.act_scale is not None

    def forward(self, x):
        if self.is_static:
            scale, zero = self.act_scale, self.act_zero
        else:
            scale, zero = affine_params(x.min(), x.max())
        xq = quantize(x, scale, zero)
        acc = xq @ self.w_q.float().T - zero * self.w_sum          # the integer matrix product
        y = acc * (scale * self.w_scale)
        return y if self.bias is None else y + self.bias


class MinMaxObserver:
    "Remembers the lowest and highest value a layer input took during calibration."

    def __init__(self):
        self.lo, self.hi = None, None

    def __call__(self, module, inputs):
        x = inputs[0].detach()
        lo, hi = x.min(), x.max()
        self.lo = lo if self.lo is None else torch.minimum(self.lo, lo)
        self.hi = hi if self.hi is None else torch.maximum(self.hi, hi)


def _linears(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]


def _replace(model: nn.Module, name: str, new: nn.Module) -> None:
    parent = model
    *path, leaf = name.split(".")
    for p in path:
        parent = getattr(parent, p)
    setattr(parent, leaf, new)


@torch.no_grad()
def calibrate(model: nn.Module, batches) -> dict[str, tuple]:
    "Run calibration batches through the float model with observers on every Linear input; return {layer: (scale, zero_point)}."
    model.eval()
    observers = {n: MinMaxObserver() for n, _ in _linears(model)}
    handles = [m.register_forward_pre_hook(observers[n]) for n, m in _linears(model)]
    for x in batches:
        model(x)
    for h in handles:
        h.remove()
    return {n: affine_params(o.lo, o.hi) for n, o in observers.items()}


@torch.no_grad()
def quantize_model(model: nn.Module, mode: str = "dynamic", bits: int = 8, calibration=None) -> nn.Module:
    """A quantized copy of model: every nn.Linear becomes a QuantLinear. mode is 'dynamic' or 'static' (needs calibration batches)."""
    assert mode in ("dynamic", "static")
    params = calibrate(model, calibration) if mode == "static" else {}
    q = copy.deepcopy(model).eval()
    for name, lin in _linears(q):
        scale, zero = params.get(name, (None, None))
        _replace(q, name, QuantLinear(lin, bits, scale, zero))
    return q


def torch_dynamic(model: nn.Module) -> nn.Module:
    "PyTorch's own dynamic int8 quantization of the Linear layers, as a reference for the hand-written version."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.quantization.quantize_dynamic(copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8)
