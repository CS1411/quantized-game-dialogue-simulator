import pytest
import torch

from app.gpt import GPT, Config
from app.measure import DEVICES, file_bytes, size_bytes, simulate, token_ops, wall_clock
from app.quant import quantize_model

CFG = Config(vocab_size=40, block_size=64, n_layer=2, n_head=2, n_embd=32)


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return GPT(CFG).eval()


def test_float_size_is_four_bytes_per_parameter(model):
    assert size_bytes(model)["total"] == 4 * sum(p.numel() for p in model.parameters())


def test_int8_weights_are_a_quarter_and_lower_bit_widths_shrink_further(model):
    fp32 = size_bytes(model)
    int8, int4 = size_bytes(quantize_model(model, bits=8)), size_bytes(quantize_model(model, bits=4))
    assert int8["weights"] == fp32["weights"] / 4 and int4["weights"] == fp32["weights"] / 8
    assert int8["total"] < fp32["total"] and int4["total"] < int8["total"]
    assert file_bytes(quantize_model(model)) < file_bytes(model)


def test_operation_counts_for_a_known_config():
    ops = token_ops(CFG, context=10)
    assert ops["linear_macs"] == 2 * 12 * 32 * 32 + 32 * 40
    assert ops["attention_macs"] == 2 * 2 * 10 * 32
    assert ops["total_macs"] == ops["linear_macs"] + ops["attention_macs"]


def test_quantized_is_faster_when_compute_bound_and_never_slower(model):
    dev = {**DEVICES["microcontroller-class"], "bandwidth": 1e15}
    f = simulate(CFG, 64, 1e5, False, dev)
    q = simulate(CFG, 64, 1e5, True, dev)
    assert f["bound"] == q["bound"] == "compute" and q["tokens_per_s"] > f["tokens_per_s"]
    mem = {**DEVICES["phone-class CPU"], "macs_per_s": 1e15}
    assert simulate(CFG, 64, 4e6, True, mem)["bound"] == "memory"
    assert simulate(CFG, 64, 1e7, False, {**dev, "ram": 1e6})["fits_in_ram"] is False


def test_wall_clock_returns_a_positive_time(model):
    assert wall_clock(model, context=16, repeats=3) > 0


def test_pytorch_dynamic_model_size_counts_its_packed_int8_weights(model):
    from app.quant import torch_dynamic
    ours, theirs = size_bytes(quantize_model(model)), size_bytes(torch_dynamic(model))
    assert theirs["weights"] == ours["weights"]                 # one int8 byte per weight either way
    assert abs(theirs["total"] - ours["total"]) < 0.1 * ours["total"]
