"""Milestone 1: a small game dialogue corpus, cleaned and tokenised.

The corpus is original: short conversations between the player ("You") and non-player characters of a fantasy town,
assembled from hand-written line templates with random fill-ins, seeded so the same seed gives the same text.
Cleaning normalises typography and whitespace; tokenising is character level (a vocabulary of about 70 symbols keeps the
model tiny, which is the point of quantizing it afterwards).
"""
import json
import random
import re
import unicodedata

ITEMS = ["iron sword", "oak shield", "healing potion", "silver ring", "hunting bow", "lantern", "map of the marsh", "bag of seeds", "amber charm", "steel helm"]
PLACES = ["Old Mill", "Ravenhollow", "Sunken Chapel", "Gull Point", "Ash Road", "Thornwick Keep", "Glass Caves", "Marrow Bridge"]
FOES = ["wolves", "bandits", "marsh trolls", "restless skeletons", "road thieves", "cave spiders"]
GOLD = ["five", "ten", "fifteen", "twenty", "thirty", "fifty"]
WEATHER = ["rain", "fog", "a cold wind", "heavy snow", "thunder"]

# speaker -> opening lines and reply lines (slots: {item} {place} {foe} {gold} {weather})
NPCS = {
    "Brenna": ["Welcome to my forge, traveler. A fine {item} will cost you {gold} gold.",
               "Mind the sparks. I can mend a {item} while you wait.",
               "The {foe} near {place} keep breaking my best blades. Bring me news of them."],
    "Old Tam": ["Sit down by the fire and tell me where you are headed. {place} is no place to go in {weather}.",
                "I have kept this inn for forty winters. A bed is {gold} gold and the stew is free.",
                "They say {foe} walk the road to {place} after dark."],
    "Captain Vael": ["Halt. The gate is closed until the {foe} are driven from {place}.",
                     "Show me your {item} and I will let you pass. Hold it high where I can see it.",
                     "We are short of men. Clear {place} and the guard will pay you {gold} gold."],
    "Sister Mara": ["Peace be with you. The chapel at {place} needs a {item} blessed before winter.",
                    "I will heal your wounds for {gold} gold, or for a promise to light a candle.",
                    "Do not speak of the {foe} here. Speak of {place}, and of hope."],
    "Pip the Merchant": ["Fresh stock! A {item} for only {gold} gold, and I never lie about {item} prices.",
                         "Buy two, get a lantern free. Well, almost free, if you carry it for me to {place}.",
                         "Business is slow with the {foe} on the road. Everything is half price in {weather}."],
    "Elder Rowan": ["The old stories speak of a {item} buried below {place}. Few return from there.",
                    "I am too old to fight the {foe}, but I remember every path to {place}.",
                    "Young one, patience is a better weapon than any {item}."],
}
PLAYER = [
    "What do you have for sale?", "Tell me about {place}.", "I can take care of the {foe}.",
    "How much for the {item}?", "I am only passing through.", "Is it safe to travel in {weather}?",
    "I will do it. What is the reward?", "Do you know anything about the {foe}?",
]
CLOSERS = [
    "Good luck out there.", "Come back alive, and I will have {gold} gold waiting.", "Safe roads, friend.",
    "Do not forget your {item}.", "May the road be kind to you.", "Whatever you do, avoid {place} after dark.",
]
REPLIES = [
    "Aye, I have heard of it. The {foe} gather near {place}, so keep your {item} close.",
    "That depends. Can you pay {gold} gold?", "Then you are braver than most. {place} lies two days north.",
    "It is not safe. Not with {weather} and the {foe} about.", "A fair price is {gold} gold, and not a coin less.",
    "The reward is {gold} gold and my thanks. Bring me the {item} from {place}.",
]


def _fill(text: str, rng: random.Random) -> str:
    return text.format(item=rng.choice(ITEMS), place=rng.choice(PLACES), foe=rng.choice(FOES),
                       gold=rng.choice(GOLD), weather=rng.choice(WEATHER))


def make_scene(rng: random.Random) -> str:
    npc = rng.choice(list(NPCS))
    lines = [f"{npc}: {_fill(rng.choice(NPCS[npc]), rng)}"]
    for _ in range(rng.randint(1, 3)):
        lines.append(f"You: {_fill(rng.choice(PLAYER), rng)}")
        lines.append(f"{npc}: {_fill(rng.choice(REPLIES + NPCS[npc]), rng)}")
    lines.append(f"{npc}: {_fill(rng.choice(CLOSERS), rng)}")
    return "\n".join(lines)


def build_corpus(n_scenes: int = 1500, seed: int = 0) -> str:
    rng = random.Random(seed)
    return "\n\n".join(make_scene(rng) for _ in range(n_scenes)) + "\n"


def clean(text: str) -> str:
    """Normalise typography (curly quotes, dashes), drop control characters, collapse runs of spaces and blank lines."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = "".join(c for c in text if c == "\n" or c >= " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


class CharTokenizer:
    "Character-level tokenizer: every distinct character is one token."

    def __init__(self, chars: str):
        self.chars = "".join(sorted(set(chars)))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    def to_json(self) -> str:
        return json.dumps({"chars": self.chars})

    @classmethod
    def from_json(cls, text: str) -> "CharTokenizer":
        return cls(json.loads(text)["chars"])


def split(ids: list[int], val_fraction: float = 0.1) -> tuple[list[int], list[int]]:
    cut = int(len(ids) * (1 - val_fraction))
    return ids[:cut], ids[cut:]
