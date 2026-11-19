"""Command line: python -m app.cli corpus | train | generate | quantize | report"""
import argparse
import json
from pathlib import Path

import torch

from app import gpt
from app.corpus import CharTokenizer, build_corpus, clean, split
from app.measure import DEVICES, agreement, file_bytes, quality, simulate, size_bytes, token_ops, wall_clock
from app.quant import quantize_model, torch_dynamic

NL = chr(10)
MODEL = "model/gpt.pt"
CORPUS = "data/dialogue.txt"


def _corpus(path: str = CORPUS) -> str:
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(clean(build_corpus(1500, seed=0)), encoding="utf-8")
    return p.read_text(encoding="utf-8")


def cmd_train(args) -> None:
    text = _corpus(args.corpus)
    tok = CharTokenizer(text)
    train_ids, val_ids = split(tok.encode(text))
    torch.manual_seed(0)
    model = gpt.GPT(gpt.Config(vocab_size=tok.vocab_size))
    print(f"{len(text):,} characters, vocabulary {tok.vocab_size}, {sum(p.numel() for p in model.parameters()):,} parameters")
    history = gpt.train(model, train_ids, val_ids, steps=args.steps, batch=args.batch, lr=args.lr, eval_every=args.eval_every)
    Path(MODEL).parent.mkdir(parents=True, exist_ok=True)
    gpt.save(model, tok.to_json(), MODEL)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/train_log.json").write_text(json.dumps(history, indent=1))
    print(f"saved {MODEL}")


def cmd_generate(args) -> None:
    model, tok_json = gpt.load(args.model)
    tok = CharTokenizer.from_json(tok_json)
    prompt = args.prompt.replace("\\n", "\n")
    ids = gpt.generate(model, tok.encode(prompt), args.length, temperature=args.temperature, seed=args.seed)
    print(tok.decode(ids))


def _variants(model, calibration, sweep: bool):
    "(name, model) pairs: the float model, PyTorch's int8 dynamic, and the hand-written ones."
    out = [("fp32 (original)", model, 32), ("int8 dynamic (PyTorch)", torch_dynamic(model), 8),
           ("int8 dynamic (ours)", quantize_model(model, "dynamic", 8), 8),
           ("int8 static (ours)", quantize_model(model, "static", 8, calibration), 8)]
    if sweep:
        out += [(f"int{b} weights, dynamic", quantize_model(model, "dynamic", b), b) for b in (6, 4, 3, 2)]
        out += [("int4 weights, static", quantize_model(model, "static", 4, calibration), 4)]
    return out


def cmd_quantize(args) -> None:
    model, tok_json = gpt.load(args.model)
    tok = CharTokenizer.from_json(tok_json)
    text = _corpus(args.corpus)
    train_ids, val_ids = split(tok.encode(text))
    val, block, ctx = torch.tensor(val_ids), model.config.block_size, model.config.block_size
    gen = torch.Generator().manual_seed(1)
    calibration = [gpt.get_batch(torch.tensor(train_ids), 16, block, gen)[0] for _ in range(args.calibration_batches)]
    base_size = size_bytes(model)["total"]
    prompt = tok.encode("Brenna:")
    rows = []
    for name, m, bits in _variants(model, calibration, not args.no_sweep):
        size = size_bytes(m)
        row = {"name": name, "bits": bits, **quality(m, val, block), **agreement(m, model, val, block),
               "size_bytes": size["total"], "weight_bytes": size["weights"], "file_bytes": file_bytes(m),
               "size_ratio": base_size / size["total"], "wall_ms": 1000 * wall_clock(m, ctx, repeats=args.repeats),
               "sample": tok.decode(gpt.generate(m, prompt, args.sample_length, seed=0)),
               "simulated": {d: simulate(model.config, ctx, size["total"], bits < 32, dev) for d, dev in DEVICES.items()}}
        rows.append(row)
        print(f"{name:26s} val {row['val_loss']:.3f}  size {row['size_bytes'] / 1024:7.1f} KB ({row['size_ratio']:.2f}x smaller)  "
              f"agree {row['top1_agreement']:.1%}  wall {row['wall_ms']:.1f} ms")
    result = {"config": vars(model.config) if hasattr(model.config, "__dict__") else {}, "context": ctx,
              "ops_per_token": token_ops(model.config, ctx), "devices": DEVICES, "variants": rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=1))
    (out / "results.md").write_text(render_results(result))
    print(f"wrote {out}/results.json and {out}/results.md")


def render_results(result: dict) -> str:
    rows, ops, ctx = result["variants"], result["ops_per_token"], result["context"]
    base = rows[0]
    lines = ["# Results", "",
             f"Model: {result['config']}. Context {ctx} characters. All numbers come from `python -m app.cli quantize`.", "",
             "## Size and quality", "",
             "| Variant | Size (KB) | Smaller by | Weights (KB) | Saved file (KB) | Val loss | Perplexity | Top-1 agrees with fp32 | Max logit diff |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['size_bytes'] / 1024:.1f} | {r['size_ratio']:.2f}x | {r['weight_bytes'] / 1024:.1f} | "
                     f"{r['file_bytes'] / 1024:.1f} | {r['val_loss']:.3f} | {r['perplexity']:.3f} | {r['top1_agreement']:.1%} | {r['max_logit_diff']:.2f} |")
    lines += ["", "Size is counted from the tensors; bit-widths below 8 are a packed estimate (the saved file keeps one int8 per weight).", "",
              "## Operations per generated token (KV cache, context %d)" % ctx, "",
              f"- Weight matmuls: {ops['linear_macs']:,} multiply-accumulates (integer after quantization)",
              f"- Attention products: {ops['attention_macs']:,} multiply-accumulates (stay float)", "",
              "## Simulated speed on assumed devices", "",
              "Roofline model: time per token is the larger of compute time and the time to read all weights from memory. "
              "Device speeds are assumptions (see `DEVICES` in `app/measure.py`), not measurements.", ""]
    for dev, spec in result["devices"].items():
        lines += [f"### {dev}", f"Assumed: {spec['macs_per_s']:.1e} float MAC/s, integer {spec['int8_speedup']:.0f}x faster, "
                  f"{spec['bandwidth']:.1e} B/s memory, {spec['ram'] / 2**20:.0f} MB for the model.", "",
                  "| Variant | Tokens/s | Limited by | Fits in memory | Speed-up vs fp32 |", "|---|---|---|---|---|"]
        for r in rows:
            s, b = r["simulated"][dev], base["simulated"][dev]
            lines.append(f"| {r['name']} | {s['tokens_per_s']:.1f} | {s['bound']} | {'yes' if s['fits_in_ram'] else 'NO'} | {s['tokens_per_s'] / b['tokens_per_s']:.2f}x |")
        lines.append("")
    lines += ["## Measured wall-clock on this machine (one thread, one forward pass over the context)", "",
              "| Variant | Milliseconds |", "|---|---|"] + [f"| {r['name']} | {r['wall_ms']:.1f} |" for r in rows]
    lines += ["", "The hand-written quantized layers emulate integer arithmetic in float tensors, so they are not expected to be faster here; "
              "PyTorch's own dynamic layers use real integer kernels. The simulated table is the estimate for hardware with integer units.", "",
              "## Same prompt, same seed, each variant", ""]
    for r in rows:
        lines += [f"### {r['name']}", "```", r["sample"], "```", ""]
    return NL.join(lines) + NL


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="quantized-game-dialogue-simulator", description="Tiny game-dialogue GPT, quantized.")
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train the GPT on the generated dialogue corpus")
    t.add_argument("--corpus", default=CORPUS)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch", type=int, default=48)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--eval-every", type=int, default=250)
    g = sub.add_parser("generate", help="sample dialogue from the trained model")
    g.add_argument("--model", default=MODEL)
    g.add_argument("--prompt", default="Brenna:")
    g.add_argument("--length", type=int, default=400)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--seed", type=int, default=0)
    q = sub.add_parser("quantize", help="quantize the model, measure size, quality and simulated speed")
    q.add_argument("--model", default=MODEL)
    q.add_argument("--corpus", default=CORPUS)
    q.add_argument("--out", default="reports")
    q.add_argument("--calibration-batches", type=int, default=8)
    q.add_argument("--repeats", type=int, default=15)
    q.add_argument("--sample-length", type=int, default=300)
    q.add_argument("--no-sweep", action="store_true", help="skip the lower bit-width sweep")
    args = parser.parse_args(argv)
    {"train": cmd_train, "generate": cmd_generate, "quantize": cmd_quantize}[args.command](args)


if __name__ == "__main__":
    main()
