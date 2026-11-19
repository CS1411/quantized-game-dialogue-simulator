"""Milestones 2 and 3: a small GPT (the 'Build GPT from Scratch' recipe), training, and sampling dialogue.

Every weight matrix lives in an nn.Linear, so quantization can swap those layers later without touching the architecture.
"""
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 49
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.n_head = c.n_head
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.register_buffer("mask", torch.tril(torch.ones(c.block_size, c.block_size)).bool(), persistent=False)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_head, d // self.n_head).transpose(1, 2) for z in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])
        att = att.masked_fill(~self.mask[:t, :t], float("-inf")).softmax(dim=-1)
        return self.proj((att @ v).transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.attn = Attention(c)
        self.fc = nn.Linear(c.n_embd, 4 * c.n_embd)
        self.out = nn.Linear(4 * c.n_embd, c.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.out(F.gelu(self.fc(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.pos = nn.Embedding(c.block_size, c.n_embd)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.n_layer))
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return logits, loss


def get_batch(data: torch.Tensor, batch: int, block: int, gen: torch.Generator):
    ix = torch.randint(len(data) - block - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix])
    y = torch.stack([data[i + 1:i + block + 1] for i in ix])
    return x, y


@torch.no_grad()
def evaluate(model: nn.Module, data: torch.Tensor, block: int, batches: int = 20, batch: int = 32, seed: int = 123) -> float:
    "Mean cross-entropy (nats per character) on fixed random windows of data; the same windows for every model compared."
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    return sum(model(*get_batch(data, batch, block, gen))[1].item() for _ in range(batches)) / batches


def train(model: GPT, train_ids: list[int], val_ids: list[int], steps: int = 3000, batch: int = 64, lr: float = 3e-3,
          warmup: int = 100, eval_every: int = 500, seed: int = 0, log=print) -> list[dict]:
    """AdamW with warmup then cosine decay. Returns [{"step", "train_loss", "val_loss", "seconds"}]."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    train_t, val_t = torch.tensor(train_ids), torch.tensor(val_ids)
    block = model.config.block_size
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    history, start = [], time.time()
    for step in range(1, steps + 1):
        scale = step / warmup if step <= warmup else 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup)))
        for g in opt.param_groups:
            g["lr"] = lr * max(scale, 0.05)
        model.train()
        _, loss = model(*get_batch(train_t, batch, block, gen))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == steps:
            row = {"step": step, "train_loss": loss.item(), "val_loss": evaluate(model, val_t, block), "seconds": time.time() - start}
            history.append(row)
            log(f"step {step:5d}  train {row['train_loss']:.3f}  val {row['val_loss']:.3f}  {row['seconds']:.0f}s")
    return history


@torch.no_grad()
def generate(model: nn.Module, prompt_ids: list[int], n_new: int, temperature: float = 0.8, seed: int = 0, block: int | None = None) -> list[int]:
    "Sample n_new tokens after the prompt. Same seed, same text."
    model.eval()
    block = block or model.config.block_size
    gen = torch.Generator().manual_seed(seed)
    ids = torch.tensor([prompt_ids])
    for _ in range(n_new):
        logits, _ = model(ids[:, -block:])
        probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        ids = torch.cat([ids, torch.multinomial(probs, 1, generator=gen)], dim=1)
    return ids[0].tolist()


def save(model: GPT, tokenizer_json: str, path: str) -> None:
    torch.save({"config": asdict(model.config), "state": model.state_dict(), "tokenizer": tokenizer_json}, path)


def load(path: str) -> tuple[GPT, str]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    model = GPT(Config(**blob["config"]))
    model.load_state_dict(blob["state"])
    return model.eval(), blob["tokenizer"]
