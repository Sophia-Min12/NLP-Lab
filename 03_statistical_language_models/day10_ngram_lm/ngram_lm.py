"""Day 10 — N-gram language model with Laplace smoothing.

Day 9 estimated ``P(term | class)`` and threw word order away. A **language
model** estimates ``P(term | the terms before it)`` and keeps it. Same
counting, one change of conditioning variable, and the result can score how
likely a sentence is — which is what spelling correction, speech
recognition, machine translation and every autoregressive text generator
are built on.

The counting is Day 4's n-grams; the arithmetic is one division. What makes
this a whole day is the **zero**. Maximum likelihood assigns probability 0
to any continuation it never saw, and Day 4 measured how common that is:
by n=5 essentially every window in a corpus occurs exactly once. A single
zero makes an entire sentence impossible, so a model that has never seen
one ordinary phrase will rank a perfectly fluent sentence as strictly
impossible.

Laplace smoothing fixes the zero, and this day also measures **what it
costs** — which is a great deal more than it is usually given credit for.
"""

from __future__ import annotations

import math
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
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    tokens = list(tokens)
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


# reused from day04
def pad_sequence(tokens: list[str], n: int, bos: str = BOS, eos: str = EOS) -> list[str]:
    """Wrap a sequence in n-1 start markers and one end marker."""
    _require_tokens(tokens)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    return [bos] * (n - 1) + list(tokens) + [eos]


def replace_rare(
    sentences: list[list[str]],
    min_count: int = 2,
    unk: str = UNK,
) -> list[list[str]]:
    """Fold tokens occurring fewer than ``min_count`` times into ``<unk>``.

    Without this the model has no way to score a word it has never seen —
    every unseen token would be a hard zero, or would silently borrow
    smoothing mass that was never budgeted for it. Training the ``<unk>``
    symbol on the corpus's own rare words gives it a real probability
    estimated from real data.

    This is the Day 3 hapax tail being put to work rather than discarded.

    >>> replace_rare([["a", "a", "b"]], min_count=2)
    [['a', 'a', '<unk>']]
    """
    counts: dict[str, int] = {}
    for sentence in sentences:
        if isinstance(sentence, str):
            raise TypeError("sentences must be token lists, not str")
        for token in sentence:
            counts[token] = counts.get(token, 0) + 1
    return [
        [token if counts[token] >= min_count else unk for token in sentence]
        for sentence in sentences
    ]


class NgramModel:
    """An n-gram language model with add-alpha smoothing.

    ``alpha=0`` gives the maximum-likelihood estimate, zeros and all.
    ``alpha=1`` is Laplace (add-one). Anything between is "add-k", which
    exists because add-one is usually far too aggressive — see
    ``smoothing_cost``.

    >>> model = NgramModel(2)
    >>> model.fit([["the", "cat", "sat"], ["the", "cat", "ran"]])
    >>> round(model.probability("cat", ("the",)), 3)
    0.429
    """

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
        """Count padded n-grams and their contexts.

        Each sentence is padded separately, so ``<s>`` contexts describe
        genuine sentence starts rather than whatever followed the previous
        sentence. The vocabulary includes ``</s>`` — ending is an event the
        model must be able to predict — but not ``<s>``, which is only ever
        conditioned on, never generated.
        """
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
        """``P(token | context)``, add-alpha smoothed.

        The context is truncated to the ``n-1`` most recent tokens, so a
        caller may pass a whole history and let the model take what its
        order allows.
        """
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
        """Total ``log P(sentence)``, or ``-inf`` if any step is impossible.

        Unsmoothed models return ``-inf`` routinely. That is not a bug in
        the arithmetic — it is the model sincerely reporting that it
        considers the sentence impossible, which is exactly the behaviour
        smoothing exists to soften.
        """
        _require_tokens(sentence)
        if not self.fitted:
            raise RuntimeError("model is not fitted; call fit() first")

        padded = pad_sequence(sentence, self.n)
        total = 0.0
        for gram in ngrams(padded, self.n):
            probability = self.probability(gram[-1], gram[:-1])
            if probability <= 0:
                return float("-inf")
            total += math.log(probability)
        return total

    def continuations(self, context: tuple[str, ...] = ()) -> dict[str, float]:
        """``P(token | context)`` for every token in the vocabulary.

        Sums to 1 whenever the model is smoothed, which is what makes it a
        distribution rather than a table of scores. Day 11 samples from it.
        """
        if not self.fitted:
            raise RuntimeError("model is not fitted; call fit() first")
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        return {token: self.probability(token, context) for token in sorted(self.vocabulary)}

    def smoothing_cost(self, context: tuple[str, ...]) -> tuple[float, float]:
        """How much probability mass smoothing moves off the observed data.

        Returns ``(observed_mass, unseen_mass)`` for one context: the share
        the smoothed model gives to continuations that were actually seen,
        and the share it hands to everything else. Under maximum likelihood
        the split is 1.0 / 0.0 by definition.

        This is the number that makes add-one look bad.
        """
        if not self.fitted:
            raise RuntimeError("model is not fitted; call fit() first")
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()
        seen = {
            gram[-1] for gram in self.ngram_counts
            if gram[:-1] == context
        }
        observed = sum(self.probability(token, context) for token in seen)
        return observed, 1.0 - observed


def build_corpus() -> list[list[str]]:
    """A deterministic synthetic corpus with known distributional structure.

    Every sentence comes from a template, so the co-occurrence structure is
    **built in, not discovered**: animals share contexts with animals,
    people with people, and the possessive tracks the person. That makes it
    useful for showing what these methods do when the structure is there
    (Days 13-15), and useless as evidence that they would find it in real
    text. The READMEs say so wherever the distinction matters.

    Real text was the alternative, and it is not available here: Level 3 is
    standard-library only, so there is nothing to download it with.

    >>> len(build_corpus()) > 100
    True
    """
    people = [("king", "his"), ("queen", "her"), ("man", "his"),
              ("woman", "her"), ("boy", "his"), ("girl", "her")]
    animals = ["cat", "dog", "bird", "fox"]
    places = ["garden", "forest", "village", "market"]
    foods = ["bread", "soup", "rice", "fish"]
    motions = ["ran", "walked", "wandered"]

    sentences = []
    for person, pronoun in people:
        for place in places:
            sentences.append(f"the {person} walked to the {place} with {pronoun} friend")
            sentences.append(f"the {person} bought bread at the {place}")
        for food in foods:
            sentences.append(f"the {person} ate the {food} for {pronoun} dinner")
    for animal in animals:
        for place in places:
            for motion in motions:
                sentences.append(f"the {animal} {motion} through the {place}")
        for food in foods:
            sentences.append(f"the {animal} ate the {food} quickly")
    for person, _ in people:
        for animal in animals:
            sentences.append(f"the {person} saw the {animal} near the river")

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


def split_corpus(
    sentences: list[list[str]],
    held_out: int = 20,
) -> tuple[list[list[str]], list[list[str]]]:
    """Split off a held-out set by taking every k-th sentence.

    Deterministic, and it draws from across the corpus rather than from one
    end — the templates are generated in blocks, so a tail slice would hand
    the model an entirely unseen sentence type and measure the wrong thing.

    >>> train, test = split_corpus([[str(i)] for i in range(10)], held_out=2)
    >>> len(train), len(test)
    (8, 2)
    """
    if held_out < 0:
        raise ValueError(f"held_out must be non-negative, got {held_out}")
    if held_out >= len(sentences):
        raise ValueError(
            f"held_out={held_out} leaves no training data from {len(sentences)} sentences"
        )
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

    corpus = build_corpus()
    train, test = split_corpus(corpus, held_out=20)
    train = replace_rare(train, min_count=2)
    tokens = sum(len(s) for s in train)
    print(f"corpus: {len(corpus)} sentences, {tokens} training tokens, "
          f"{len(set(t for s in train for t in s))} types\n")

    bigram = NgramModel(2, alpha=1.0)
    bigram.fit(train)

    print("what follows 'the' (top 6):")
    ranked = sorted(bigram.continuations(("the",)).items(), key=lambda kv: (-kv[1], kv[0]))
    for token, probability in ranked[:6]:
        print(f"  P({token:<9}| the) = {probability:.4f}")
    print(f"  distribution sums to {sum(bigram.continuations(('the',)).values()):.6f}")

    print("\nthe zero problem, as a function of n (held-out sentences called impossible):")
    print(f"  {'n':>2}{'MLE':>18}{'Laplace':>12}")
    for n in (2, 3, 4, 5):
        mle = NgramModel(n, alpha=0.0)
        mle.fit(train)
        smoothed = NgramModel(n, alpha=1.0)
        smoothed.fit(train)
        zeros = sum(1 for s in test if mle.log_probability(s) == float("-inf"))
        kept = sum(1 for s in test if smoothed.log_probability(s) == float("-inf"))
        print(f"  {n:>2}{zeros:>13}/{len(test)}{kept:>9}/{len(test)}")
    print("  n=2 and n=3: the single zero is an out-of-vocabulary *word*")
    print("  n=4 and n=5: known words in sequences never seen - Day 4's sparsity curve,")
    print("               arriving as impossibility rather than as a percentage")

    print("\nwhat Laplace costs - probability mass moved off the observed data:")
    print(f"  {'context':<14}{'seen':>6}{'count':>7}{'observed':>11}{'unseen':>9}")
    for context in (("the",), ("king",), ("ate",), ("wandered",)):
        if context not in bigram.context_counts:
            continue
        seen = len({g[-1] for g in bigram.ngram_counts if g[:-1] == context})
        observed, unseen = bigram.smoothing_cost(context)
        print(f"  {context[0]:<14}{seen:>6}{bigram.context_counts[context]:>7}"
              f"{observed:>10.1%}{unseen:>9.1%}")

    print("\n  add-one gives the vocabulary's unseen continuations more mass than")
    print("  the data it actually observed. add-k with a small k is the usual retreat:")
    for alpha in (1.0, 0.1, 0.01):
        model = NgramModel(2, alpha=alpha)
        model.fit(train)
        observed, unseen = model.smoothing_cost(("ate",))
        print(f"    alpha={alpha:<6} observed {observed:>6.1%}   unseen {unseen:>6.1%}")

    print("\n<unk> lets the model score a word it has never seen:")
    unseen_sentence = normalize("the wizard ate the bread")
    folded = [t if t in bigram.vocabulary else UNK for t in unseen_sentence]
    print(f"  {' '.join(unseen_sentence)}")
    print(f"  -> {' '.join(folded)}")
    print(f"  log P = {bigram.log_probability(folded):.2f}  (finite, so it is scoreable)")
