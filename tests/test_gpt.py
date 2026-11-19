import torch

from app.corpus import CharTokenizer, build_corpus, clean, split
from app.gpt import GPT, Config, evaluate, generate, load, save, train

TINY = Config(vocab_size=49, block_size=32, n_layer=2, n_head=2, n_embd=32)


def tiny_data():
    text = clean(build_corpus(120))
    tok = CharTokenizer(text)
    return tok, *split(tok.encode(text))


def test_forward_shapes_and_untrained_loss_is_near_chance():
    tok, train_ids, _ = tiny_data()
    model = GPT(Config(tok.vocab_size, 32, 2, 2, 32))
    x = torch.tensor([train_ids[:32]])
    logits, loss = model(x, x)
    assert logits.shape == (1, 32, tok.vocab_size)
    assert abs(loss.item() - torch.log(torch.tensor(float(tok.vocab_size))).item()) < 0.3


def test_training_lowers_validation_loss():
    tok, tr, va = tiny_data()
    torch.manual_seed(0)
    model = GPT(Config(tok.vocab_size, 32, 2, 2, 32))
    before = evaluate(model, torch.tensor(va), 32, batches=4)
    train(model, tr, va, steps=60, batch=16, lr=3e-3, warmup=5, eval_every=60, log=lambda *_: None)
    assert evaluate(model, torch.tensor(va), 32, batches=4) < before - 0.5


def test_generation_is_seeded_and_save_load_round_trips(tmp_path):
    tok, tr, va = tiny_data()
    model = GPT(Config(tok.vocab_size, 32, 2, 2, 32))
    prompt = tok.encode("Brenna:")
    a = generate(model, prompt, 20, seed=1)
    assert a == generate(model, prompt, 20, seed=1) and len(a) == len(prompt) + 20
    path = str(tmp_path / "m.pt")
    save(model, tok.to_json(), path)
    loaded, tok_json = load(path)
    assert generate(loaded, prompt, 20, seed=1) == a and CharTokenizer.from_json(tok_json).chars == tok.chars
