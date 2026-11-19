from app.corpus import CharTokenizer, build_corpus, clean, split


def test_corpus_is_reproducible_and_has_the_dialogue_shape():
    a = build_corpus(50, seed=1)
    assert a == build_corpus(50, seed=1) and a != build_corpus(50, seed=2)
    scenes = a.strip().split("\n\n")
    assert len(scenes) == 50
    for scene in scenes:
        lines = scene.split("\n")
        assert lines[0].split(":")[0] != "You" and all(": " in line for line in lines)
        assert "{" not in scene and "}" not in scene          # every slot was filled


def test_clean_normalises_typography_and_whitespace():
    assert clean("Hello’s   “world” — ok\r\n\n\n\nBye\x00 ") == "Hello's \"world\" - ok\n\nBye\n"


def test_tokenizer_round_trips_and_json_round_trips():
    text = build_corpus(20)
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text
    assert CharTokenizer.from_json(tok.to_json()).chars == tok.chars
    assert 30 < tok.vocab_size < 100


def test_split_keeps_order_and_sizes():
    train, val = split(list(range(100)), 0.1)
    assert len(train) == 90 and train[-1] + 1 == val[0] == 90
