import numpy as np
import pytest
import torch
import torch.nn as nn

from app.corpus import CharTokenizer, build_corpus, clean, split
from app.gpt import GPT, Config, train
from app.quant import (QuantLinear, affine_params, calibrate, dequantize, quantize, quantize_model, quantize_weight,
                       torch_dynamic)


def test_affine_round_trip_error_is_at_most_half_a_step():
    x = torch.linspace(-1.3, 2.1, 1000)
    scale, zero = affine_params(x.min(), x.max())
    err = (dequantize(quantize(x, scale, zero), scale, zero) - x).abs().max()
    assert err <= scale / 2 + 1e-6
    assert dequantize(quantize(torch.tensor(0.0), scale, zero), scale, zero) == 0   # zero is exactly representable


def test_weight_quantization_is_per_row_and_within_half_a_step():
    w = torch.randn(16, 64) * torch.linspace(0.1, 3, 16)[:, None]
    for bits in (8, 4):
        q, scale = quantize_weight(w, bits)
        assert q.abs().max() <= 2 ** (bits - 1) - 1 and q.dtype == torch.int8
        assert ((q.float() * scale[:, None] - w).abs() <= scale[:, None] / 2 + 1e-6).all()


def test_quantlinear_integer_product_matches_plain_integer_arithmetic():
    torch.manual_seed(0)
    lin = nn.Linear(512, 8)
    x = torch.randn(3, 512) * 2
    q = QuantLinear(lin)
    scale, zero = affine_params(x.min(), x.max())
    xq = quantize(x, scale, zero).numpy().astype(np.int64)
    acc = xq @ q.w_q.numpy().astype(np.int64).T - int(zero) * q.w_q.numpy().astype(np.int64).sum(1)
    expected = acc * (scale.item() * q.w_scale.numpy()) + lin.bias.detach().numpy()
    assert np.allclose(q(x).detach().numpy(), expected, rtol=1e-5, atol=1e-5)
    assert (q(x) - lin(x)).abs().max() < 0.1 * lin(x).abs().max()


def test_static_uses_frozen_calibrated_ranges():
    lin = nn.Sequential(nn.Linear(16, 16))
    data = [torch.randn(4, 16) for _ in range(5)]
    ranges = calibrate(lin, data)
    assert ranges["0"][0] > 0
    q = quantize_model(lin, "static", calibration=data)
    assert q[0].is_static and torch.equal(q[0].act_scale, ranges["0"][0])
    far = torch.randn(4, 16) * 100                      # outside the calibrated range: clipped, not rescaled
    assert not torch.allclose(q[0](far), lin(far), rtol=0.1)
    assert not quantize_model(lin, "dynamic")[0].is_static


@pytest.fixture(scope="module")
def trained():
    text = clean(build_corpus(150))
    tok = CharTokenizer(text)
    tr, va = split(tok.encode(text))
    torch.manual_seed(0)
    model = GPT(Config(tok.vocab_size, 32, 2, 2, 32))
    train(model, tr, va, steps=80, batch=16, lr=3e-3, warmup=5, eval_every=80, log=lambda *_: None)
    return model.eval(), torch.tensor(va)


def test_quantized_gpt_stays_close_to_the_float_model(trained):
    model, va = trained
    x = torch.stack([va[i:i + 32] for i in range(0, 256, 32)])
    ref, _ = model(x)
    calib = [torch.stack([va[i:i + 32] for i in range(j, j + 128, 32)]) for j in range(0, 256, 128)]
    for q in (quantize_model(model, "dynamic"), quantize_model(model, "static", calibration=calib), torch_dynamic(model)):
        out, _ = q(x)
        assert (out.argmax(-1) == ref.argmax(-1)).float().mean() > 0.9
        assert (out - ref).abs().max() < 0.25 * ref.abs().max()
    assert not any(isinstance(m, nn.Linear) for m in quantize_model(model).modules())
