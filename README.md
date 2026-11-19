# Quantized Game Dialogue Simulator

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/8rjg8qhkbp/badge.svg)](https://ask.thili.ai/b/8rjg8qhkbp)

Train a small GPT on game dialogue, shrink it with post-training quantization, and simulate what that does to memory footprint and generation speed on resource-constrained devices. Uses the techniques of the askThili drills *Build GPT from Scratch* and *Shrink a Neural Network with Quantisation*.

## What you are building

**Goal.** A small quantized GPT for in-game dialogue, plus a simulation of its memory footprint and generation speed on constrained devices.

**Done when.** A small GPT is trained on a custom game dialogue corpus, dynamic and static post-training quantization are applied, and the effect on model size and simulated inference speed is measured.

## Milestones

- [x] **1. Prepare a small game dialogue text corpus.** `app/corpus.py`: original dialogue generator, cleaning, character tokenizer
- [x] **2. Implement a basic GPT architecture and train it on the corpus.** `app/gpt.py`: 4 layers, 822,272 parameters, trained 2,000 steps ([log](reports/train_output.txt))
- [x] **3. Generate sample dialogue from the trained GPT.** `python -m app.cli generate`, samples in [reports/sample_dialogue.txt](reports/sample_dialogue.txt)
- [x] **4. Apply dynamic and static post-training quantization to the model.** `app/quant.py`, written by hand and compared with PyTorch's `quantize_dynamic`
- [x] **5. Measure and compare the original and quantized model sizes.** `app/measure.py`, table below
- [x] **6. Simulate inference speed by comparing operations on original versus quantized weights.** operation count plus a roofline simulation for two assumed devices; full tables in [reports/results.md](reports/results.md)

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.cli generate                 # uses the trained model in model/gpt.pt (3 MB, committed)
python -m app.cli quantize                 # all variants: size, quality, simulated and measured speed -> reports/
python -m app.cli train                    # retrain from scratch (about 7 minutes on a laptop CPU)
pytest
```

## Results

Corpus: 565,019 characters (1,500 generated conversations, 49-character vocabulary). Validation loss of the float model: 0.123 nats per character.

| Variant | Size | Smaller by | Val loss | Top-1 agrees with fp32 |
|---|---|---|---|---|
| fp32 (original) | 3,212 KB | 1.00x | 0.123 | 100% |
| int8 dynamic (PyTorch) | 890 KB | 3.61x | 0.124 | 99.6% |
| int8 dynamic (ours) | 908 KB | 3.54x | 0.123 | 99.8% |
| int8 static (ours) | 908 KB | 3.54x | 0.123 | 99.8% |
| int6 weights, dynamic | 714 KB | 4.50x | 0.124 | 99.6% |
| int4 weights, dynamic | 521 KB | 6.17x | 0.125 | 98.6% |
| int3 weights, dynamic | 424 KB | 7.58x | 0.149 | 96.1% |
| int2 weights, dynamic | 327 KB | 9.82x | 4.036 | 28.6% |

- The weight matrices shrink exactly 4x at int8; the whole model shrinks 3.5x because embeddings, layer norms and biases stay float.
- int8 dynamic and static are indistinguishable in quality here; static freezes activation ranges from 8 calibration batches.
- Quality holds down to int4 weights, starts to degrade at int3 (the generated text gets typos such as "How ald and the") and collapses at int2 (letter noise). Samples for every variant are in [reports/results.md](reports/results.md).
- **Simulated speed** (assumed devices, a roofline of compute time versus time to read the weights): per generated token the model needs 792,704 weight multiply-accumulates (integer after quantization) and 131,072 attention multiply-accumulates (stay float). On a phone-class CPU int8 is about 3.5x faster because decoding at batch 1 is memory bound; on a microcontroller-class device int8 is 2.8x faster and, at 908 KB, fits in 1 MB where the 3.2 MB float model does not.
- **Measured speed on this laptop is not faster**: one forward pass over 128 characters takes 5.1 ms in float, 5.3 ms with PyTorch's int8 layers and 7-8 ms with my hand-written layers, which emulate integer arithmetic in float tensors. A model this small does not hit the memory or integer-unit limits that quantization helps with on this machine, so the simulated numbers are a statement about the assumed hardware, not about this laptop.

## How it works

Affine mapping `x ~ scale * (q - zero_point)`. Weights are quantized once, symmetric, one scale per output row. Activations are quantized per layer input: dynamic quantization reads the range from the live tensor on every call; static quantization learns it once from calibration data with min/max observers and freezes it. The matrix product runs on integers (`sum(xq * wq) - zero_point * sum(wq)`) and is rescaled by `scale_x * scale_w`; `tests/test_quant.py` checks this against plain int64 arithmetic.

## Limits

- The corpus is template-generated and highly regular (perplexity 1.13), so quantization looks easier than it would for a larger model trained on real, varied dialogue.
- The device speeds in `DEVICES` are assumptions, not measurements of real hardware; bit-widths below 8 are packed-size estimates (nothing is bit-packed in memory); attention products and layer norms stay float.
- Only the Linear weights are quantized (below 8 bits only the weights; activations stay 8 bit); the embedding tables are not quantized.
- No quantization-aware training; this project is post-training only.

## Honest development notes

- Built with AI assistance (Claude) in a single working session; the commit timeline does not reflect days of work.
- Milestones 5 and 6 were developed together and share one commit.
- A test caught a bug in my size counting: PyTorch's own int8 layers were not counted, which made them look 33x smaller. It was fixed and a test now checks that both implementations count one byte per weight.
- The training run (2,000 steps, batch 48) and all measurements come from one CPU machine; numbers vary slightly between runs.
