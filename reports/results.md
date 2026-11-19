# Results

Model: {'vocab_size': 49, 'block_size': 128, 'n_layer': 4, 'n_head': 4, 'n_embd': 128}. Context 128 characters. All numbers come from `python -m app.cli quantize`.

## Size and quality

| Variant | Size (KB) | Smaller by | Weights (KB) | Saved file (KB) | Val loss | Perplexity | Top-1 agrees with fp32 | Max logit diff |
|---|---|---|---|---|---|---|---|---|
| fp32 (original) | 3212.0 | 1.00x | 3096.5 | 3228.4 | 0.123 | 1.131 | 100.0% | 0.00 |
| int8 dynamic (PyTorch) | 889.7 | 3.61x | 774.1 | 918.0 | 0.124 | 1.132 | 99.6% | 3.17 |
| int8 dynamic (ours) | 907.8 | 3.54x | 774.1 | 952.1 | 0.123 | 1.131 | 99.8% | 1.03 |
| int8 static (ours) | 907.9 | 3.54x | 774.1 | 961.5 | 0.123 | 1.131 | 99.8% | 1.01 |
| int6 weights, dynamic | 714.3 | 4.50x | 580.6 | 952.1 | 0.124 | 1.132 | 99.6% | 1.54 |
| int4 weights, dynamic | 520.8 | 6.17x | 387.1 | 952.1 | 0.125 | 1.134 | 98.6% | 4.63 |
| int3 weights, dynamic | 424.0 | 7.58x | 290.3 | 952.1 | 0.149 | 1.160 | 96.1% | 14.85 |
| int2 weights, dynamic | 327.2 | 9.82x | 193.5 | 952.1 | 4.036 | 56.587 | 28.6% | 30.55 |
| int4 weights, static | 520.9 | 6.17x | 387.1 | 961.5 | 0.125 | 1.134 | 98.5% | 5.21 |

Size is counted from the tensors; bit-widths below 8 are a packed estimate (the saved file keeps one int8 per weight).

## Operations per generated token (KV cache, context 128)

- Weight matmuls: 792,704 multiply-accumulates (integer after quantization)
- Attention products: 131,072 multiply-accumulates (stay float)

## Simulated speed on assumed devices

Roofline model: time per token is the larger of compute time and the time to read all weights from memory. Device speeds are assumptions (see `DEVICES` in `app/measure.py`), not measurements.

### phone-class CPU
Assumed: 4.0e+09 float MAC/s, integer 4x faster, 8.0e+09 B/s memory, 64 MB for the model.

| Variant | Tokens/s | Limited by | Fits in memory | Speed-up vs fp32 |
|---|---|---|---|---|
| fp32 (original) | 2432.3 | memory | yes | 1.00x |
| int8 dynamic (PyTorch) | 8781.1 | memory | yes | 3.61x |
| int8 dynamic (ours) | 8605.8 | memory | yes | 3.54x |
| int8 static (ours) | 8604.6 | memory | yes | 3.54x |
| int6 weights, dynamic | 10937.5 | memory | yes | 4.50x |
| int4 weights, dynamic | 12148.9 | compute | yes | 4.99x |
| int3 weights, dynamic | 12148.9 | compute | yes | 4.99x |
| int2 weights, dynamic | 12148.9 | compute | yes | 4.99x |
| int4 weights, static | 12148.9 | compute | yes | 4.99x |

### microcontroller-class
Assumed: 5.0e+07 float MAC/s, integer 4x faster, 2.0e+08 B/s memory, 1 MB for the model.

| Variant | Tokens/s | Limited by | Fits in memory | Speed-up vs fp32 |
|---|---|---|---|---|
| fp32 (original) | 54.1 | compute | NO | 1.00x |
| int8 dynamic (PyTorch) | 151.9 | compute | yes | 2.81x |
| int8 dynamic (ours) | 151.9 | compute | yes | 2.81x |
| int8 static (ours) | 151.9 | compute | yes | 2.81x |
| int6 weights, dynamic | 151.9 | compute | yes | 2.81x |
| int4 weights, dynamic | 151.9 | compute | yes | 2.81x |
| int3 weights, dynamic | 151.9 | compute | yes | 2.81x |
| int2 weights, dynamic | 151.9 | compute | yes | 2.81x |
| int4 weights, static | 151.9 | compute | yes | 2.81x |

## Measured wall-clock on this machine (one thread, one forward pass over the context)

| Variant | Milliseconds |
|---|---|
| fp32 (original) | 5.1 |
| int8 dynamic (PyTorch) | 5.3 |
| int8 dynamic (ours) | 7.8 |
| int8 static (ours) | 7.0 |
| int6 weights, dynamic | 7.9 |
| int4 weights, dynamic | 7.8 |
| int3 weights, dynamic | 7.8 |
| int2 weights, dynamic | 7.8 |
| int4 weights, static | 6.6 |

The hand-written quantized layers emulate integer arithmetic in float tensors, so they are not expected to be faster here; PyTorch's own dynamic layers use real integer kernels. The simulated table is the estimate for hardware with integer units.

## Same prompt, same seed, each variant

### fp32 (original)
```
Brenna: Welcome to my forge, traveler. A fine hunting bow will cost you fifteen gold.
You: I can take care of the bandits.
Brenna: It is not safe. Not with a cold wind and the marsh trolls about.
Brenna: Do not forget your amber charm.

Old Tam: They say cave spiders walk the road to Gull Point after dark.
```

### int8 dynamic (PyTorch)
```
Brenna: Welcome to my forge, traveler. A fine hunting bow will cost you fifteen gold.
You: I can take care of the bandits.
Brenna: It is not safe. Not with a cold wind and the marsh trolls about.
Brenna: Do not forget your amber charm.

Old Tam: They say cave spiders walk the road to Gull Point after dark.
```

### int8 dynamic (ours)
```
Brenna: Welcome to my forge, traveler. A fine hunting bow will cost you fifteen gold.
You: I can take care of the bandits.
Brenna: It is not safe. Not with a cold wind and the marsh trolls about.
Brenna: Do not forget your amber charm.

Old Tam: They say cave spiders walk the road to Gull Point after dark.
```

### int8 static (ours)
```
Brenna: Welcome to my forge, traveler. A fine hunting bow will cost you fifteen gold.
You: I can take care of the bandits.
Brenna: It is not safe. Not with a cold wind and the marsh trolls about.
Brenna: Do not forget your amber charm.

Old Tam: They say cave spiders walk the road to Gull Point after dark.
```

### int6 weights, dynamic
```
Brenna: Welcome to my forge, traveler. A fine hunting bow will cost you fifteen gold.
You: I can take care of the bandits.
Brenna: It is not safe. Not with a cold wind and the marsh trolls about.
You: I can take care of the bandits.
Brenna: The cave spiders near Ash Road keep breaking my best blades. Bring
```

### int4 weights, dynamic
```
Brenna: Welcome to my forge, traveler. A fine hunting bow will cost you fifteen gold.
You: I can take care of the bandits.
Brenna: It is not safe. Not with a cold wind and the marsh trolls about.
You: I can take care of the bandits.
Brenna: The cave spiders near Ash Road keep breaking my best blades. Bring
```

### int3 weights, dynamic
```
Brenna: Welcome to my forge, traveler. A fine healing potion will cost you ten gold.
You: How ald and the and the stless skeletons about.
You: I will do it. What is the rewand?
Brenna: It is not safeen tor me to Thornwick Keep.
You: I will do it. What is the reward?
Brenna: It is not safeever rin fom the b
```

### int2 weights, dynamic
```
Brenna: tlf mewme wnhewllf ren rewmank ow wfl reswis K wow Bftfte te r re dofofof mee rey Rowavesiw wa ofof tlco m of me the yofou my eKp me se lp me me se Rfk!  eollon tuy RyoP ea Rsesk Rom k k! theeof  tusshe the tre the folcl m me   fofou  thunhe the g g Bunhe founhe m?
 the the s the thuny Bre P me The
```

### int4 weights, static
```
Brenna: Welcome to my forge, traveler. A fine hunting bow will cost you fifteen gold.
You: I can take care of the bandits.
Brenna: It is not safe. Not with a cold wind and the marsh trolls about.
You: I can take care of the bandits.
Brenna: The cave spiders near Ash Road keep breaking my best blades. Bring
```

