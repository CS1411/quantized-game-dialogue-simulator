"""Milestones 5 and 6: model size, an operation count, and a simulated inference speed on constrained devices.

SIZE. Counted from the tensors: fp32 parameters take 4 bytes; a QuantLinear takes bits/8 bytes per weight (a packed estimate:
the tensors in memory hold one int8 per weight) plus a float scale per row and the float bias. A real file size is also taken
by saving the state dict (this is the unpacked int8 size for every bit-width).

OPERATIONS. For ONE generated token with a key/value cache and a context of T tokens, per layer:
  weight matmuls (Linear): d*3d + d*d + d*4d + 4d*d = 12 d^2 multiply-accumulates   (these are what quantization turns into int)
  attention products (q.k and att.v): 2 * T * d multiply-accumulates                 (activation x activation: stay float)
plus the output head d*V. Layer norms, softmax, GELU and embeddings are element-wise and not counted as MACs.

SIMULATED SPEED. A roofline: time per token = max(compute time, memory time), where compute time = float MACs / float rate +
integer MACs / integer rate, and memory time = bytes of weights read / bandwidth (at batch 1 the whole model is read for every
token). The device numbers below are ASSUMPTIONS chosen to look like a phone-class and a microcontroller-class device; they are
inputs of the simulation, not measurements. Change them in DEVICES to model other hardware.
"""
import io
import statistics
import time

import torch
import torch.nn as nn

from app.gpt import evaluate
from app.quant import QuantLinear

# macs_per_s: float32 multiply-accumulates per second; int8_speedup: integer rate relative to float;
# bandwidth: bytes/second from memory; ram: bytes available for the model
DEVICES = {
    "phone-class CPU": {"macs_per_s": 4e9, "int8_speedup": 4.0, "bandwidth": 8e9, "ram": 64 * 2**20},
    "microcontroller-class": {"macs_per_s": 5e7, "int8_speedup": 4.0, "bandwidth": 2e8, "ram": 2**20},
}


def size_bytes(model: nn.Module) -> dict:
    "Bytes of the model's tensors: {'total', 'weights' (the Linear weight matrices), 'other'}."
    weights = other = 0.0
    quant_names = set()
    for name, m in model.named_modules():
        if isinstance(m, QuantLinear):
            weights += m.w_q.numel() * m.bits / 8
            other += m.w_scale.numel() * 4 + (0 if m.bias is None else m.bias.numel() * 4) + (8 if m.is_static else 0)
            quant_names.add(name)
        elif isinstance(m, torch.ao.nn.quantized.dynamic.Linear):  # PyTorch's own dynamic int8 Linear
            w, b = m._weight_bias()
            weights += w.numel()                                  # one int8 byte per weight
            other += (w.q_per_channel_scales().numel() * 4 if w.qscheme() == torch.per_channel_affine else 4) \
                + (0 if b is None else b.numel() * 4)
            quant_names.add(name)
        elif isinstance(m, nn.Linear):
            weights += m.weight.numel() * 4
            other += 0 if m.bias is None else m.bias.numel() * 4
            quant_names.add(name)
    for name, m in model.named_modules():
        if name not in quant_names and not any(name.startswith(q + ".") for q in quant_names):
            other += sum(p.numel() * 4 for p in m.parameters(recurse=False))
    return {"weights": weights, "other": other, "total": weights + other}


def file_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def token_ops(config, context: int) -> dict:
    "Multiply-accumulates to generate one token with a KV cache at the given context length."
    d, layers = config.n_embd, config.n_layer
    linear = layers * 12 * d * d + d * config.vocab_size
    attention = layers * 2 * context * d
    return {"linear_macs": linear, "attention_macs": attention, "total_macs": linear + attention}


def simulate(config, context: int, weight_bytes: float, quantized: bool, device: dict) -> dict:
    "Roofline estimate of one token on a device. quantized=True runs the weight matmuls at the integer rate."
    ops = token_ops(config, context)
    int_rate = device["macs_per_s"] * device["int8_speedup"]
    compute = (ops["attention_macs"] / device["macs_per_s"]
               + ops["linear_macs"] / (int_rate if quantized else device["macs_per_s"]))
    memory = weight_bytes / device["bandwidth"]
    t = max(compute, memory)
    return {"compute_s": compute, "memory_s": memory, "token_s": t, "tokens_per_s": 1 / t,
            "bound": "memory" if memory > compute else "compute", "fits_in_ram": weight_bytes <= device["ram"]}


def agreement(model: nn.Module, reference: nn.Module, data: torch.Tensor, block: int, batches: int = 10) -> dict:
    "How close a quantized model's predictions are to the float model's, on fixed windows of data."
    gen = torch.Generator().manual_seed(7)
    same, diffs, total = 0, [], 0
    with torch.no_grad():
        for _ in range(batches):
            ix = torch.randint(len(data) - block - 1, (16,), generator=gen)
            x = torch.stack([data[i:i + block] for i in ix])
            a, b = model(x)[0], reference(x)[0]
            same += (a.argmax(-1) == b.argmax(-1)).sum().item()
            total += a.shape[0] * a.shape[1]
            diffs.append((a - b).abs().max().item())
    return {"top1_agreement": same / total, "max_logit_diff": max(diffs)}


def quality(model: nn.Module, val: torch.Tensor, block: int) -> dict:
    loss = evaluate(model, val, block)
    return {"val_loss": loss, "perplexity": float(torch.exp(torch.tensor(loss)))}


def wall_clock(model: nn.Module, context: int = 128, repeats: int = 15, threads: int = 1) -> float:
    "Median seconds for one forward pass over `context` characters, on one CPU thread (a stand-in for a small device)."
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        x = torch.randint(0, model.config.vocab_size if hasattr(model, "config") else 40, (1, context))
        with torch.no_grad():
            model(x)
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                model(x)
                times.append(time.perf_counter() - t0)
        return statistics.median(times)
    finally:
        torch.set_num_threads(previous)
