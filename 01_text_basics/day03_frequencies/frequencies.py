"""Day 3 — Word frequencies and Zipf's law.

Days 1 and 2 produced a clean token stream. Counting it is the last step
before text becomes numbers: a frequency table *is* the unnormalized
bag-of-words vector Day 5 builds, and the document frequencies Day 6 needs
for TF-IDF.

Counting also exposes the single most important statistical property of
natural language. Rank the words of any corpus by frequency and the count of
the word at rank *r* falls off roughly as ``C / r``: the 2nd word appears
about half as often as the 1st, the 10th about a tenth as often. This is
**Zipf's law**, and nearly everything downstream — vocabulary size, index
size, smoothing, subword tokenization — is a response to it.

Stopword removal is deliberately *not* applied here. The stopwords of Day 2
are simply the head of the Zipf curve, and that is much easier to see when
they are still in the table.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata
from pathlib import Path


# reused from day01
def _require_str(text) -> None:
    """Shared type guard so every function enforces the same contract."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")


def _require_tokens(tokens) -> None:
    """Reject a bare string where a token list is expected.

    ``"hello"`` is iterable, so without this guard it would silently be
    counted character by character.
    """
    if isinstance(tokens, str):
        raise TypeError("expected a list of tokens, got str")


# reused from day01
def tokenize_words(text: str) -> list[str]:
    """Word-level regex tokenization: words group, punctuation separates."""
    _require_str(text)
    text = text.replace("’", "'")  # curly apostrophe -> ASCII
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


# reused from day02, minus the stopword step (see the module docstring)
def normalize(text: str) -> list[str]:
    """NFC, casefold, tokenize, drop standalone punctuation.

    >>> normalize("The cat, THE Cat!")
    ['the', 'cat', 'the', 'cat']
    """
    _require_str(text)
    text = unicodedata.normalize("NFC", text)
    tokens = tokenize_words(text.casefold())
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


def word_frequencies(tokens: list[str]) -> dict[str, int]:
    """Count how often each token occurs.

    Written as an explicit loop rather than ``collections.Counter`` because
    the counting dictionary is the primitive worth seeing once. ``Counter``
    is the same thing and is what production code should use.

    Insertion order is first-appearance order, which keeps output
    reproducible.

    >>> word_frequencies(["the", "cat", "the"])
    {'the': 2, 'cat': 1}
    """
    _require_tokens(tokens)
    freqs: dict[str, int] = {}
    for token in tokens:
        freqs[token] = freqs.get(token, 0) + 1
    return freqs


def rank_words(freqs: dict[str, int]) -> list[tuple[str, int]]:
    """Order words by count, most frequent first.

    Ties break alphabetically. Without that second key the order would
    depend on insertion order, and every rank-based number below would
    wobble between runs.

    >>> rank_words({"b": 1, "a": 1, "c": 3})
    [('c', 3), ('a', 1), ('b', 1)]
    """
    return sorted(freqs.items(), key=lambda item: (-item[1], item[0]))


def top_n(freqs: dict[str, int], n: int = 10) -> list[tuple[str, int]]:
    """The ``n`` most frequent words.

    >>> top_n({"a": 5, "b": 2, "c": 9}, 2)
    [('c', 9), ('a', 5)]
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    return rank_words(freqs)[:n]


def hapax_legomena(freqs: dict[str, int]) -> list[str]:
    """Words that occur exactly once, in ranked order.

    Typically half or more of the vocabulary of a corpus, and a larger
    share the shorter the text — this day's README runs about 60%. They are
    the long flat tail of the Zipf curve, and the reason unsmoothed
    probability models assign zero to so much of the language (Day 10).

    >>> hapax_legomena({"the": 9, "zebra": 1, "aardvark": 1})
    ['aardvark', 'zebra']
    """
    return [word for word, count in rank_words(freqs) if count == 1]


def type_token_ratio(tokens: list[str]) -> float:
    """Vocabulary size divided by corpus size; 0.0 for empty input.

    A rough measure of lexical variety, but it falls as a text gets longer
    (new tokens keep repeating old types), so it only compares texts of
    similar length.

    >>> type_token_ratio(["a", "b", "a", "c"])
    0.75
    """
    _require_tokens(tokens)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def coverage(freqs: dict[str, int], k: int) -> float:
    """Share of all tokens accounted for by the ``k`` most frequent types.

    The headline Zipf number: in English the top 10 types typically cover a
    quarter of a corpus. Returns 0.0 when there is nothing to count.

    >>> coverage({"the": 7, "cat": 2, "sat": 1}, 1)
    0.7
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    total = sum(freqs.values())
    if total == 0:
        return 0.0
    return sum(count for _, count in top_n(freqs, k)) / total


def zipf_table(freqs: dict[str, int], n: int = 10) -> list[tuple[int, str, int, float]]:
    """Compare observed counts against the ``C / rank`` Zipf prediction.

    ``C`` is taken as the count of the rank-1 word, so row 1 always matches
    exactly and the interesting question is how well later rows track it.
    Each row is ``(rank, word, observed, predicted)``.

    >>> zipf_table({"a": 10, "b": 6, "c": 2}, 3)
    [(1, 'a', 10, 10.0), (2, 'b', 6, 5.0), (3, 'c', 2, 3.3333333333333335)]
    """
    ranked = top_n(freqs, n)
    if not ranked:
        return []
    top_count = ranked[0][1]
    return [
        (rank, word, count, top_count / rank)
        for rank, (word, count) in enumerate(ranked, start=1)
    ]


def zipf_exponent(freqs: dict[str, int], max_rank: int | None = None) -> float:
    """Fit ``log(count) = log(C) - s * log(rank)`` and return ``s``.

    A plain least-squares slope over the log-log points, computed by hand
    since NumPy does not arrive until Level 4.

    ``max_rank`` limits the fit to the head of the distribution. Most of a
    corpus's *types* sit in the flat tail of count 1, where ``log(count)``
    is 0 for thousands of ranks; fitting across all of them drags the slope
    below the true value. A synthetic head that fits at 1.003 alone drops
    to 0.91 once 3000 hapax are appended, and recovers to 1.003 when the
    fit is capped at the head. A test pins this.

    Corpus size dominates both effects, though. The often-quoted exponent
    of ~1.0 is asymptotic: it needs large corpora. A thousand-token text
    fits near 0.7 whatever rank window is used, and no amount of
    rank-trimming will talk it up to 1.0.

    >>> round(zipf_exponent({"a": 100, "b": 50, "c": 33, "d": 25}), 3)
    1.003
    >>> round(zipf_exponent({"a": 100, "b": 50, "c": 33, "d": 25}, max_rank=2), 3)
    1.0
    """
    if max_rank is not None and max_rank < 2:
        raise ValueError(f"max_rank must be at least 2, got {max_rank}")

    ranked = rank_words(freqs)
    if max_rank is not None:
        ranked = ranked[:max_rank]
    if len(ranked) < 2:
        raise ValueError("need at least two distinct words to fit an exponent")

    xs = [math.log(rank) for rank in range(1, len(ranked) + 1)]
    ys = [math.log(count) for _, count in ranked]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("need at least two distinct ranks to fit an exponent")
    return -(numerator / denominator)


SAMPLE = (
    "The cat sat on the mat. The cat then sat on the hat, and the hat "
    "was not a mat. A dog saw the cat on the hat and the dog barked. "
    "고양이가 매트 위에 앉았다. 고양이가 다시 모자 위에 앉았다. "
    "개가 고양이를 보았고 개가 짖었다."
)


def _demo_corpus() -> tuple[str, str]:
    """Return ``(label, text)``: this day's README if present, else SAMPLE.

    Using the README keeps the folder self-contained while still giving the
    demo a few thousand words of real prose — short toy strings are too
    small for Zipf's law to be visible.
    """
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
    freqs = word_frequencies(tokens)

    print(f"corpus            : {label}")
    print(f"tokens (running)  : {len(tokens)}")
    print(f"types (distinct)  : {len(freqs)}")
    print(f"type-token ratio  : {type_token_ratio(tokens):.3f}")
    print(f"hapax legomena    : {len(hapax_legomena(freqs))} "
          f"({len(hapax_legomena(freqs)) / max(len(freqs), 1):.0%} of the vocabulary)")

    print("\nZipf table (predicted = count of rank 1, divided by rank):")
    print(f"  {'rank':>4}  {'word':<14} {'observed':>8} {'predicted':>10}")
    for rank, word, observed, predicted in zipf_table(freqs, 12):
        print(f"  {rank:>4}  {word:<14} {observed:>8} {predicted:>10.1f}")

    print("\nfitted Zipf exponent (the quoted ~1.0 is asymptotic, for large corpora):")
    print(f"  all {len(freqs):>3} ranks : {zipf_exponent(freqs):.3f}")
    for head in (25, 50, 100):
        if len(freqs) > head:
            print(f"  top {head:>3} ranks : {zipf_exponent(freqs, max_rank=head):.3f}")
    print("  this corpus is too small to reach 1.0 — trimming ranks will not fix that")

    print("\ncoverage — share of all tokens held by the top k types:")
    for k in (1, 5, 10, 25, 50, 100):
        print(f"  top {k:>3}: {coverage(freqs, k):.1%}")

    print("\nthe stopword list of Day 2 is just the head of this curve:")
    print("  ", [word for word, _ in top_n(freqs, 10)])
