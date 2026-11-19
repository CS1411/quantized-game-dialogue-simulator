import json

import torch

from app import gpt
from app.cli import main
from app.corpus import CharTokenizer, build_corpus, clean


def make_tiny_model(tmp_path):
    text = clean(build_corpus(150))
    corpus = tmp_path / "dialogue.txt"
    corpus.write_text(text, encoding="utf-8")
    tok = CharTokenizer(text)
    torch.manual_seed(0)
    model = gpt.GPT(gpt.Config(tok.vocab_size, 32, 2, 2, 32))
    path = str(tmp_path / "gpt.pt")
    gpt.save(model, tok.to_json(), path)
    return path, str(corpus)


def test_generate_command_prints_dialogue(tmp_path, capsys):
    path, _ = make_tiny_model(tmp_path)
    main(["generate", "--model", path, "--prompt", "Brenna:", "--length", "30"])
    assert capsys.readouterr().out.startswith("Brenna:")


def test_quantize_command_writes_results(tmp_path, capsys):
    path, corpus = make_tiny_model(tmp_path)
    out = tmp_path / "out"
    main(["quantize", "--model", path, "--corpus", corpus, "--out", str(out), "--no-sweep", "--repeats", "2",
          "--calibration-batches", "2", "--sample-length", "20"])
    result = json.loads((out / "results.json").read_text())
    names = [v["name"] for v in result["variants"]]
    assert names == ["fp32 (original)", "int8 dynamic (PyTorch)", "int8 dynamic (ours)", "int8 static (ours)"]
    fp32, ours = result["variants"][0], result["variants"][2]
    assert ours["size_bytes"] < fp32["size_bytes"] and ours["size_ratio"] > 1.5
    assert (out / "results.md").read_text().startswith("# Results")
