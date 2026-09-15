"""Day 4 — N-gram extraction.

Day 3 counted words in isolation. A bag of words cannot tell "the dog bit
the man" from "the man bit the dog": same counts, opposite meaning. An
**n-gram** is the cheapest possible repair — a window of *n* adjacent
tokens, slid one position at a time, so that a little local word order
survives into the counts.

This is the representation Day 10 turns into a probability model
(``P(word | previous n-1 words)``), and the one Day 12's BPE operates on at
the character level. It is also where the data-sparsity problem of Level 3
first becomes visible: raising *n* by one multiplies the number of possible
sequences by the vocabulary size, while the corpus stays exactly as big.

Character n-grams get equal billing here rather than a footnote. For Korean
they are not a curiosity but the practical workaround, for the reason
Days 1-3 kept running into: word-level units carry particles.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

BOS = "<s>"
EOS = "</s>"


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


def _require_n(n) -> None:
    """N-gram size must be a positive integer.

    ``bool`` is excluded explicitly: ``True`` is an ``int`` in Python and
    would silently be read as ``n=1``.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")


# reused from day01
def tokenize_words(text: str) -> list[str]:
    """Word-level regex tokenization: words group, punctuation separates."""
    _require_str(text)
    text = text.replace("’", "'")  # curly apostrophe -> ASCII
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


# reused from day02/day03, minus the stopword step
def normalize(text: str) -> list[str]:
    """NFC, casefold, tokenize, drop standalone punctuation.

    >>> normalize("The dog, THE Dog!")
    ['the', 'dog', 'the', 'dog']
    """
    _require_str(text)
    text = unicodedata.normalize("NFC", text)
    tokens = tokenize_words(text.casefold())
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


# reused from day03
def word_frequencies(tokens: list) -> dict:
    """Count how often each item occurs, in first-appearance order."""
    _require_tokens(tokens)
    freqs: dict = {}
    for token in tokens:
        freqs[token] = freqs.get(token, 0) + 1
    return freqs


# reused from day03
def rank_words(freqs: dict) -> list[tuple]:
    """Order items by count, most frequent first; ties break by the item."""
    return sorted(freqs.items(), key=lambda item: (-item[1], item[0]))


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Every window of ``n`` adjacent tokens, as tuples.

    Tuples, not lists or joined strings: an n-gram has to be hashable to be
    a dictionary key, and joining on a space would make ``("new", "york")``
    indistinguishable from the single token ``"new york"``.

    A sequence of ``len(tokens)`` yields ``len(tokens) - n + 1`` n-grams,
    and an empty list when the text is shorter than the window.

    >>> ngrams(["a", "b", "c"], 2)
    [('a', 'b'), ('b', 'c')]
    >>> ngrams(["a", "b"], 5)
    []
    """
    _require_tokens(tokens)
    _require_n(n)
    tokens = list(tokens)
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def bigrams(tokens: list[str]) -> list[tuple[str, ...]]:
    """Adjacent pairs.

    >>> bigrams(["the", "dog", "barked"])
    [('the', 'dog'), ('dog', 'barked')]
    """
    return ngrams(tokens, 2)


def trigrams(tokens: list[str]) -> list[tuple[str, ...]]:
    """Adjacent triples.

    >>> trigrams(["the", "dog", "barked", "loudly"])
    [('the', 'dog', 'barked'), ('dog', 'barked', 'loudly')]
    """
    return ngrams(tokens, 3)


def pad_sequence(tokens: list[str], n: int, bos: str = BOS, eos: str = EOS) -> list[str]:
    """Wrap a sequence in ``n - 1`` start markers and one end marker.

    Without padding, the first word of a sentence never appears as the
    *target* of an n-gram, so a model built on these counts has no way to
    ask what tends to start a sentence — or to end one. Day 10 needs both.

    ``n - 1`` start markers, because that is the amount of left context an
    n-gram model conditions on.

    >>> pad_sequence(["dogs", "bark"], 2)
    ['<s>', 'dogs', 'bark', '</s>']
    >>> pad_sequence(["dogs", "bark"], 3)
    ['<s>', '<s>', 'dogs', 'bark', '</s>']
    """
    _require_tokens(tokens)
    _require_n(n)
    return [bos] * (n - 1) + list(tokens) + [eos]


def ngram_frequencies(
    tokens: list[str],
    n: int,
    pad: bool = False,
) -> dict[tuple[str, ...], int]:
    """Count n-grams, optionally padding the sequence boundaries first.

    >>> ngram_frequencies(["a", "b", "a", "b"], 2)
    {('a', 'b'): 2, ('b', 'a'): 1}
    """
    _require_tokens(tokens)
    _require_n(n)
    if pad:
        tokens = pad_sequence(tokens, n)
    return word_frequencies(ngrams(tokens, n))


def top_ngrams(
    freqs: dict[tuple[str, ...], int],
    k: int = 10,
) -> list[tuple[tuple[str, ...], int]]:
    """The ``k`` most frequent n-grams, ties broken by the n-gram itself.

    >>> top_ngrams({("b", "c"): 1, ("a", "b"): 3}, 1)
    [(('a', 'b'), 3)]
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    return rank_words(freqs)[:k]


def char_ngrams(text: str, n: int, strip_spaces: bool = False) -> list[str]:
    """Character windows over raw text, as strings.

    The units are characters, so nothing depends on getting word boundaries
    right. That is why this is the standard fallback for Korean, Chinese
    and Japanese retrieval: ``고양이가`` and ``고양이를`` share the character
    trigram ``고양이``, which word-level counting never sees.

    Text is NFC-normalized first — over decomposed Hangul the windows would
    slice syllables into jamo (Day 2).

    >>> char_ngrams("cat", 2)
    ['ca', 'at']
    >>> char_ngrams("a b", 2, strip_spaces=True)
    ['ab']
    """
    _require_str(text)
    _require_n(n)
    text = unicodedata.normalize("NFC", text)
    if strip_spaces:
        text = "".join(text.split())
    return [text[i:i + n] for i in range(len(text) - n + 1)]


def sparsity(freqs: dict) -> float:
    """Share of distinct n-grams that were seen exactly once.

    The number that makes the case for smoothing: it climbs steeply with
    ``n``, and everything it counts is an event a maximum-likelihood model
    would assign a probability estimated from a single observation — while
    every *unseen* n-gram gets probability zero. Returns 0.0 for an empty
    table.

    >>> sparsity({("a", "b"): 1, ("b", "c"): 3})
    0.5
    """
    if not freqs:
        return 0.0
    return sum(1 for count in freqs.values() if count == 1) / len(freqs)


SAMPLE = (
    "The dog bit the man and the man bit the dog. "
    "The dog barked at the man, and the man barked at the dog. "
    "개가 사람을 물었고 사람이 개를 물었다. "
    "개가 사람에게 짖었고 사람이 개에게 짖었다."
)


def _demo_corpus() -> tuple[str, str]:
    """Return ``(label, text)``: this day's README if present, else SAMPLE."""
    readme = Path(__file__).with_name("README.md")
    try:
        return readme.name, readme.read_text(encoding="utf-8")
    except OSError:
        return "SAMPLE", SAMPLE


if __name__ == "__main__":
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    label, text = _demo_corpus()
    tokens = normalize(text)

    print(f"corpus : {label}  ({len(tokens)} tokens, {len(set(tokens))} types)\n")

    print("bag of words cannot tell these apart, bigrams can:")
    a, b = normalize("the dog bit the man"), normalize("the man bit the dog")
    print("  word counts equal :", word_frequencies(a) == word_frequencies(b))
    print("  bigrams equal     :", ngram_frequencies(a, 2) == ngram_frequencies(b, 2))

    print("\nhow the counts explode as n grows:")
    print(f"  {'n':>2}  {'n-grams':>8} {'distinct':>9} {'seen once':>10}")
    for n in (1, 2, 3, 4, 5):
        freqs = ngram_frequencies(tokens, n)
        print(f"  {n:>2}  {len(ngrams(tokens, n)):>8} {len(freqs):>9} {sparsity(freqs):>9.0%}")

    # Deliberately SAMPLE, not the corpus above: the claim is about ordinary
    # prose. A technical README's top bigrams are its own jargon ("n grams",
    # "python m"), which is a fact about this file rather than about language.
    print("\nmost frequent bigrams in SAMPLE (function words dominate - Day 3's Zipf head):")
    for gram, count in top_ngrams(ngram_frequencies(normalize(SAMPLE), 2), 8):
        print(f"  {count:>4}  {' '.join(gram)}")

    print("\npadding, so first and last words can be modelled (Day 10):")
    print("  ", pad_sequence(["dogs", "bark"], 3))
    print("  ", ngrams(pad_sequence(["dogs", "bark"], 2), 2))

    print("\ncharacter n-grams find the Korean stem that word n-grams miss:")
    korean = "고양이가 고양이를 고양이에게"
    print("  word types      :", sorted(set(normalize(korean))))
    trigram_counts = word_frequencies(char_ngrams(korean, 3, strip_spaces=True))
    print("  char trigram 고양이:", trigram_counts.get("고양이"), "occurrences")
