"""Day 11 — Text generation and perplexity.

Day 10 built a distribution over continuations. Two things follow, and they
are the two halves of how language models are actually used and judged.

**Generation** samples from that distribution repeatedly, feeding each
choice back in as context. Every autoregressive text generator works this
way; the difference between an n-gram model and a modern one is entirely in
how ``P(next | context)`` is estimated, not in this loop.

**Perplexity** measures how surprised the model is by text it has not seen.
It is the standard intrinsic metric for language models, and the standard
way to be misled about one — this day measures both what it tells you and
what it hides.
"""

from __future__ import annotations

import math
import random
import re
import sys
import unicodedata

BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"


# reused from day01
def _require_str(text) -> None:
    """Shared type guard so every function enforces the same contract."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")


# reused from day03
def _require_tokens(tokens) -> None:
    """Reject a bare string where a token list is expected."""
    if isinstance(tokens, str):
        raise TypeError("expected a list of tokens, got str")


# reused from day01
def tokenize_words(text: str) -> list[str]:
    """Word-level regex tokenization: words group, punctuation separates."""
    _require_str(text)
    text = text.replace("’", "'")  # curly apostrophe -> ASCII
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


# reused from day03 onward
def normalize(text: str) -> list[str]:
    """NFC, casefold, tokenize, drop standalone punctuation."""
    _require_str(text)
    text = unicodedata.normalize("NFC", text)
    tokens = tokenize_words(text.casefold())
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


# reused from day04
def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Every window of n adjacent tokens, as tuples."""
    _require_tokens(tokens)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    tokens = list(tokens)
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# reused from day04/day10
def pad_sequence(tokens: list[str], n: int, bos: str = BOS, eos: str = EOS) -> list[str]:
    """Wrap a sequence in n-1 start markers and one end marker."""
    _require_tokens(tokens)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    return [bos] * (n - 1) + list(tokens) + [eos]


# reused from day10
def replace_rare(sentences: list[list[str]], min_count: int = 2, unk: str = UNK) -> list[list[str]]:
    """Fold tokens occurring fewer than min_count times into <unk>."""
    counts: dict[str, int] = {}
    for sentence in sentences:
        if isinstance(sentence, str):
            raise TypeError("sentences must be token lists, not str")
        for token in sentence:
            counts[token] = counts.get(token, 0) + 1
    return [[t if counts[t] >= min_count else unk for t in s] for s in sentences]


# reused from day10
class NgramModel:
    """An n-gram language model with add-alpha smoothing."""

    def __init__(self, n: int = 2, alpha: float = 1.0) -> None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int, got {type(n).__name__}")
        if n < 1:
            raise ValueError(f"n must be at least 1, got {n}")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        self.n = n
        self.alpha = alpha
        self.vocabulary: set[str] = set()
        self.context_counts: dict[tuple[str, ...], int] = {}
        self.ngram_counts: dict[tuple[str, ...], int] = {}

    @property
    def fitted(self) -> bool:
        return bool(self.vocabulary)

    def fit(self, sentences: list[list[str]]) -> None:
        if isinstance(sentences, str):
            raise TypeError("expected a list of sentences, got str")
        self.vocabulary = set()
        self.context_counts = {}
        self.ngram_counts = {}
        for sentence in sentences:
            if isinstance(sentence, str):
                raise TypeError("sentences must be token lists, not str")
            padded = pad_sequence(sentence, self.n)
            self.vocabulary.update(sentence)
            for gram in ngrams(padded, self.n):
                context = gram[:-1]
                self.ngram_counts[gram] = self.ngram_counts.get(gram, 0) + 1
                self.context_counts[context] = self.context_counts.get(context, 0) + 1
        if self.vocabulary:
            self.vocabulary.add(EOS)

    def probability(self, token: str, context: tuple[str, ...] = ()) -> float:
        if not self.fitted:
            raise RuntimeError("model is not fitted; call fit() first")
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        gram_count = self.ngram_counts.get(context + (token,), 0)
        context_count = self.context_counts.get(context, 0)
        denominator = context_count + self.alpha * len(self.vocabulary)
        if denominator == 0:
            return 0.0
        return (gram_count + self.alpha) / denominator

    def log_probability(self, sentence: list[str]) -> float:
        _require_tokens(sentence)
        if not self.fitted:
            raise RuntimeError("model is not fitted; call fit() first")
        total = 0.0
        for gram in ngrams(pad_sequence(sentence, self.n), self.n):
            probability = self.probability(gram[-1], gram[:-1])
            if probability <= 0:
                return float("-inf")
            total += math.log(probability)
        return total

    def continuations(self, context: tuple[str, ...] = ()) -> dict[str, float]:
        if not self.fitted:
            raise RuntimeError("model is not fitted; call fit() first")
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        return {token: self.probability(token, context) for token in sorted(self.vocabulary)}


def apply_temperature(
    distribution: dict[str, float],
    temperature: float = 1.0,
) -> dict[str, float]:
    """Sharpen or flatten a distribution, then renormalize.

    Each probability is raised to ``1 / temperature``. Below 1 the peaks
    grow and the tail collapses; above 1 everything moves toward uniform.
    Temperature is not a probability — it is a knob on how much the model
    is allowed to hedge.

    >>> sharp = apply_temperature({"a": 0.6, "b": 0.4}, 0.5)
    >>> round(sharp["a"], 3)
    0.692
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    weights = {token: p ** (1.0 / temperature) for token, p in distribution.items()}
    total = sum(weights.values())
    if total == 0:
        return dict(distribution)
    return {token: weight / total for token, weight in weights.items()}


def top_k_filter(distribution: dict[str, float], k: int) -> dict[str, float]:
    """Keep the ``k`` most likely tokens and renormalize.

    Truncation matters because a smoothed model spreads a real share of its
    mass over thousands of implausible continuations (Day 10 measured
    57.5% for one context). Sampling from the full distribution keeps
    drawing from that tail; top-k refuses to.

    >>> {t: round(p, 3) for t, p in top_k_filter({"a": 0.5, "b": 0.3, "c": 0.2}, 2).items()}
    {'a': 0.625, 'b': 0.375}
    """
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    ranked = sorted(distribution.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    total = sum(p for _, p in ranked)
    if total == 0:
        return dict(ranked)
    return {token: p / total for token, p in ranked}


def sample_next(
    model: NgramModel,
    context: tuple[str, ...],
    rng: random.Random,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
) -> str:
    """Draw one token from the model's distribution over continuations."""
    distribution = model.continuations(context)
    if greedy:
        return min(distribution, key=lambda t: (-distribution[t], t))
    if top_k is not None:
        distribution = top_k_filter(distribution, top_k)
    distribution = apply_temperature(distribution, temperature)

    tokens = sorted(distribution)  # sorted so a given seed is reproducible
    weights = [distribution[t] for t in tokens]
    return rng.choices(tokens, weights=weights, k=1)[0]


def generate(
    model: NgramModel,
    max_tokens: int = 20,
    seed: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    greedy: bool = False,
) -> list[str]:
    """Generate a sentence, one sampled token at a time.

    Starts from the all-``<s>`` context and stops at ``</s>`` or
    ``max_tokens``, whichever comes first. The stop condition is only
    reachable because ``</s>`` is in the vocabulary (Day 10) — without it
    nothing would ever end.

    Sampling uses an explicit ``random.Random``, so a seed makes output
    exactly reproducible. Tests depend on that; so does being able to
    discuss a specific generated sentence.

    >>> model = NgramModel(2, alpha=0.0)
    >>> model.fit([["the", "cat", "sat"]])
    >>> generate(model, seed=1, greedy=True)
    ['the', 'cat', 'sat']
    """
    if not model.fitted:
        raise RuntimeError("model is not fitted; call fit() first")
    if max_tokens < 0:
        raise ValueError(f"max_tokens must be non-negative, got {max_tokens}")

    rng = random.Random(seed)
    context = [BOS] * (model.n - 1)
    produced: list[str] = []
    for _ in range(max_tokens):
        token = sample_next(model, tuple(context), rng, temperature, top_k, greedy)
        if token == EOS:
            break
        produced.append(token)
        context.append(token)
    return produced


def perplexity(model: NgramModel, sentences: list[list[str]]) -> float:
    """Per-token perplexity: ``exp(-1/N · Σ log P)``.

    Perplexity is the model's average branching factor — a perplexity of
    30 means it is about as uncertain as it would be choosing uniformly
    among 30 words at each step. Lower is more confident.

    ``N`` counts predicted tokens, which includes each ``</s>`` (the model
    really is predicting it) and excludes the ``<s>`` padding (it never
    is). Getting that count wrong silently shifts every number, and it is
    the usual reason two implementations disagree.

    Returns ``inf`` if the model calls any sentence impossible — an
    unsmoothed model on unseen text is infinitely perplexed, which is the
    correct answer rather than an error.

    >>> model = NgramModel(1, alpha=1.0)
    >>> model.fit([["a", "b"]])
    >>> round(perplexity(model, [["a", "b"]]), 3)
    3.0
    """
    if not model.fitted:
        raise RuntimeError("model is not fitted; call fit() first")
    if isinstance(sentences, str):
        raise TypeError("expected a list of sentences, got str")
    if not sentences:
        raise ValueError("cannot measure perplexity on an empty collection")

    total_log = 0.0
    total_tokens = 0
    for sentence in sentences:
        log_probability = model.log_probability(sentence)
        if log_probability == float("-inf"):
            return float("inf")
        total_log += log_probability
        total_tokens += len(sentence) + 1  # + </s>
    if total_tokens == 0:
        raise ValueError("no tokens to score")
    return math.exp(-total_log / total_tokens)


# reused from day10
def build_corpus() -> list[list[str]]:
    """A deterministic synthetic corpus with known distributional structure."""
    people = [("king", "his"), ("queen", "her"), ("man", "his"),
              ("woman", "her"), ("boy", "his"), ("girl", "her")]
    animals = ["cat", "dog", "bird", "fox"]
    places = ["garden", "forest", "village", "market"]
    foods = ["bread", "soup", "rice", "fish"]
    motions = ["ran", "walked", "wandered"]

    sentences = []
    # Rotated slices, so words of one category are not perfectly
    # interchangeable. With every template applied to every word, same-category
    # rows come out bit-identical and every similarity below is exactly 1.0.
    for i, (person, pronoun) in enumerate(people):
        for place in places[i % 2:] + places[:i % 2]:
            sentences.append(f"the {person} walked to the {place} with {pronoun} friend")
        for place in places[:2 + i % 3]:
            sentences.append(f"the {person} bought bread at the {place}")
        for food in foods[i % 4:] + foods[:i % 4]:
            sentences.append(f"the {person} ate the {food} for {pronoun} dinner")
    for i, animal in enumerate(animals):
        for place in places[i % 3:] + places[:i % 3]:
            for motion in motions[i % 2:] + motions[:i % 2]:
                sentences.append(f"the {animal} {motion} through the {place}")
        for food in foods[:2 + i % 3]:
            sentences.append(f"the {animal} ate the {food} quickly")
    for person, _ in people:
        for animal in animals:
            sentences.append(f"the {person} saw the {animal} near the river")

    # One context of its own per word, repeated so it survives min_count=2.
    # The his/her pairs are deliberate: they are the only systematic
    # difference between king/queen and between man/woman, which is what
    # Day 15's analogy arithmetic has to work with.
    for sentence in (
        "the cat slept on the warm windowsill",
        "the dog barked at the passing cart",
        "the bird sang in the tall tree",
        "the fox hid behind the stone wall",
        "the king ruled the kingdom from his throne",
        "the queen ruled the kingdom from her throne",
        "the man carried his heavy sack",
        "the woman carried her heavy sack",
        "the boy played with his wooden toy",
        "the girl played with her wooden toy",
        "the garden was full of bright flowers",
        "the forest was dark and very deep",
        "the village had a small stone church",
        "the market was loud on market day",
        "the bread was fresh from the oven",
        "the soup was hot and rather salty",
        "the rice was cooked with great care",
        "the fish was caught that same morning",
    ):
        sentences.extend([sentence] * 3)

    # A deliberate hapax tail. Real corpora are mostly rare words (Day 3
    # measured ~60% of types occurring once); a purely templated corpus has
    # none, and then <unk> has nothing to train on.
    for oddity in ("wizard", "cobbler", "lantern", "harbour",
                   "thistle", "kettle", "meadow", "cobweb"):
        sentences.append(f"the {oddity} stood by the old wall")

    for subject in ("고양이가", "개가"):
        for place in ("마당을", "숲을"):
            sentences.append(f"{subject} {place}천천히 지나갔다")
        sentences.append(f"{subject} 밥을 빠르게 먹었다")

    return [sentence.split() for sentence in sentences]


# reused from day10
def split_corpus(sentences, held_out: int = 20):
    """Split off a held-out set by taking every k-th sentence."""
    if held_out < 0:
        raise ValueError(f"held_out must be non-negative, got {held_out}")
    if held_out >= len(sentences):
        raise ValueError(f"held_out={held_out} leaves no training data")
    if held_out == 0:
        return list(sentences), []
    step = len(sentences) / held_out
    picked = {int(i * step) for i in range(held_out)}
    train = [s for i, s in enumerate(sentences) if i not in picked]
    test = [s for i, s in enumerate(sentences) if i in picked]
    return train, test


if __name__ == "__main__":
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    train, test = split_corpus(build_corpus(), held_out=20)
    train = replace_rare(train, min_count=2)

    print("generation at four orders (same seed, alpha=0.1):")
    for n in (1, 2, 3, 5):
        model = NgramModel(n, alpha=0.1)
        model.fit(train)
        print(f"  n={n}: {' '.join(generate(model, max_tokens=14, seed=7))}")
    print("  n=1 has no context at all. n=2 is locally fluent and globally aimless.")
    print("  n=5 is no better than n=3 - and the reason is smoothing, not the order:")

    print("\nthe same n=5 model, smoothed and unsmoothed:")
    for alpha in (0.1, 0.0):
        model = NgramModel(5, alpha=alpha)
        model.fit(train)
        produced = generate(model, max_tokens=14, seed=7)
        verbatim = "verbatim training sentence" if produced in train else "not in the training data"
        print(f"  alpha={alpha:<5} {' '.join(produced)}")
        print(f"  {'':<12}-> {verbatim}")
    print("  smoothing spreads mass over continuations the context never had, so one")
    print("  step off a seen path drops the model into near-uniform noise. Unsmoothed,")
    print("  it can only walk paths it saw - which is fluent because it is memorized.")

    bigram = NgramModel(2, alpha=0.1)
    bigram.fit(train)

    print("\ntemperature, on the same model and seed:")
    for temperature in (0.3, 1.0, 2.0):
        text = " ".join(generate(bigram, max_tokens=12, seed=3, temperature=temperature))
        print(f"  T={temperature:<4} {text}")

    print("\ngreedy decoding takes the locally likeliest token every time:")
    print(f"  {' '.join(generate(bigram, max_tokens=14, greedy=True))}")
    print("  no seed is needed, and it stops early - the likeliest continuation of the")
    print("  likeliest first word soon turns out to be </s>. Locally optimal, globally short.")

    print("\ntop-k refuses the smoothed tail:")
    for k in (1, 3, 10):
        print(f"  k={k:<3} {' '.join(generate(bigram, max_tokens=12, seed=5, top_k=k))}")

    print("\nperplexity by model order (lower = less surprised):")
    print(f"  {'n':>2}{'train':>12}{'held-out':>12}{'ratio':>9}")
    for n in (1, 2, 3, 4, 5):
        model = NgramModel(n, alpha=0.1)
        model.fit(train)
        on_train = perplexity(model, train)
        on_test = perplexity(model, test)
        print(f"  {n:>2}{on_train:>12.1f}{on_test:>12.1f}{on_test / on_train:>9.2f}")
    print("  held-out perplexity bottoms out at n=3 and climbs after. The ratio never")
    print("  stops climbing: the model fits data it has seen better than data it has not,")
    print("  by a widening margin. That is memorization, measured.")

    print("\nperplexity by smoothing constant (bigram, held-out):")
    for alpha in (1.0, 0.1, 0.01, 0.0):
        model = NgramModel(2, alpha=alpha)
        model.fit(train)
        value = perplexity(model, test)
        note = "  <- MLE: one unseen word makes it infinite" if value == float("inf") else ""
        print(f"  alpha={alpha:<6} {value:>10.1f}{note}")
