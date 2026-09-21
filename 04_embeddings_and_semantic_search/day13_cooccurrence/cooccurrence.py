"""Day 13 — Co-occurrence matrix and PPMI.

Day 7 found the wall: ``word segmentation`` scored exactly 0.000 against a
document about tokenization. No amount of reweighting closes that gap,
because the two strings genuinely never coincide. Counting *words* cannot
help; the information is not in the representation.

The distributional hypothesis says where to look instead: **a word is
characterised by the company it keeps**. Two words are similar if they
appear in similar contexts, even when they never appear together. So stop
counting how often a word occurs and start counting **what occurs near it**.

That gives a word-by-word matrix instead of Day 5's document-by-word one.
Raw counts are useless there for the same reason they were useless in
Day 6 — ``the`` sits next to everything — and the fix has the same shape,
but a sharper form: **PMI** asks whether two words co-occur *more than
chance would predict*, which is exactly the question raw counts cannot
answer.

NumPy arrives here, as Level 4 allows. Everything below is still a count
and a logarithm.
"""

from __future__ import annotations

import re
import sys
import unicodedata

import numpy as np


# reused from day01
def _require_str(text) -> None:
    """Shared type guard so every function enforces the same contract."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")


# reused from day05
def _require_documents(documents) -> None:
    """Reject a single document where a list of documents is expected."""
    if isinstance(documents, str):
        raise TypeError("expected a list of documents, got str")
    for i, document in enumerate(documents):
        if isinstance(document, str):
            raise TypeError(f"document {i} is a str; documents must be token lists")


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


# reused from day05
def build_vocabulary(documents: list[list[str]], min_count: int = 1) -> dict[str, int]:
    """Map each term occurring at least ``min_count`` times to an index.

    Sorted alphabetically before numbering, for the Day 5 reason: the index
    assignment is a contract shared by every row and column.

    >>> build_vocabulary([["b", "a"], ["a"]], min_count=2)
    {'a': 0}
    """
    _require_documents(documents)
    if min_count < 1:
        raise ValueError(f"min_count must be at least 1, got {min_count}")
    counts: dict[str, int] = {}
    for document in documents:
        for token in document:
            counts[token] = counts.get(token, 0) + 1
    kept = sorted(t for t, c in counts.items() if c >= min_count)
    return {term: index for index, term in enumerate(kept)}


def cooccurrence_matrix(
    sentences: list[list[str]],
    vocabulary: dict[str, int],
    window: int = 2,
    symmetric: bool = True,
) -> np.ndarray:
    """Count how often each pair of words appears within ``window`` tokens.

    The window is the whole model. Narrow windows (1-2) pick up syntactic
    relations — what can grammatically follow what. Wide windows (5-10)
    blur into topical association. Neither is correct; they answer
    different questions, and the choice shows up directly in what the
    embeddings of Day 14 consider "similar".

    Counting stays inside a sentence: a window is not allowed to run off
    the end into the next one, for the same reason Day 10 padded sentences
    separately.

    ``symmetric`` counts both directions, so the matrix equals its own
    transpose and "x near y" means the same as "y near x".

    >>> vocab = {"a": 0, "b": 1}
    >>> cooccurrence_matrix([["a", "b"]], vocab, window=1)
    array([[0., 1.],
           [1., 0.]])
    """
    _require_documents(sentences)
    if window < 1:
        raise ValueError(f"window must be at least 1, got {window}")

    size = len(vocabulary)
    matrix = np.zeros((size, size), dtype=float)
    for sentence in sentences:
        indices = [vocabulary.get(token) for token in sentence]
        for position, centre in enumerate(indices):
            if centre is None:
                continue
            start = max(0, position - window)
            stop = min(len(indices), position + window + 1)
            for other in range(start, stop):
                if other == position:
                    continue
                context = indices[other]
                if context is None:
                    continue
                matrix[centre, context] += 1.0
                if not symmetric and other < position:
                    matrix[centre, context] -= 1.0
    return matrix


def pmi_matrix(counts: np.ndarray, positive: bool = True, epsilon: float = 0.0) -> np.ndarray:
    """Pointwise mutual information, optionally clipped at zero (PPMI).

    ``PMI(x, y) = log( P(x, y) / (P(x) · P(y)) )`` — how much more often
    two words co-occur than they would if they were independent. A word
    like ``the`` sits next to everything, so its high raw counts are
    exactly what independence already predicts, and its PMI collapses
    toward zero. Nothing had to be told that ``the`` is a stopword.

    **Positive** PMI clips negatives away. Negative PMI claims two words
    co-occur *less* than chance, and in any realistic corpus that estimate
    is dominated by sampling noise — most pairs simply never co-occur, and
    "never" is not evidence of repulsion. Clipping also keeps the matrix
    non-negative and very sparse, which is what Day 14's factorization
    wants.

    Pairs that never co-occur give ``log 0``; they are set to zero rather
    than ``-inf``, which is the same decision as clipping and is applied
    before it so ``positive=False`` stays finite too.

    >>> counts = np.array([[0.0, 2.0], [2.0, 0.0]])
    >>> pmi_matrix(counts).round(3)
    array([[0.   , 0.693],
           [0.693, 0.   ]])
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {counts.shape}")
    if epsilon < 0:
        raise ValueError(f"epsilon must be non-negative, got {epsilon}")

    counts = counts + epsilon
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts)

    joint = counts / total
    row = joint.sum(axis=1, keepdims=True)
    column = joint.sum(axis=0, keepdims=True)
    expected = row * column

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(expected > 0, joint / expected, 0.0)
        pmi = np.where(ratio > 0, np.log(np.where(ratio > 0, ratio, 1.0)), 0.0)
    return np.maximum(pmi, 0.0) if positive else pmi


def top_associations(
    matrix: np.ndarray,
    vocabulary: dict[str, int],
    word: str,
    k: int = 5,
) -> list[tuple[str, float]]:
    """The ``k`` words most associated with ``word`` under ``matrix``.

    Ties break alphabetically so the output is reproducible.
    """
    if word not in vocabulary:
        raise KeyError(f"{word!r} is not in the vocabulary")
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    terms = sorted(vocabulary, key=vocabulary.get)
    row = matrix[vocabulary[word]]
    pairs = [(term, float(row[i])) for i, term in enumerate(terms) if term != word]
    return sorted(pairs, key=lambda pair: (-pair[1], pair[0]))[:k]


def sparsity(matrix: np.ndarray) -> float:
    """Share of entries that are exactly zero."""
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return 0.0
    return float(np.count_nonzero(matrix == 0) / matrix.size)


# reused from day10
def build_corpus() -> list[list[str]]:
    """A deterministic synthetic corpus with known distributional structure.

    Worth repeating here, because this is the day where it starts to
    matter: the structure below is **built in, not discovered**. Animals
    share contexts with animals and people with people because the
    templates say so. What follows demonstrates that the method finds
    structure that exists — not that this structure would emerge from a
    comparable amount of real text. It would not; real corpora for this
    are measured in millions of tokens.
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
    for oddity in ("wizard", "cobbler", "lantern", "harbour",
                   "thistle", "kettle", "meadow", "cobweb"):
        sentences.append(f"the {oddity} stood by the old wall")
    for subject in ("고양이가", "개가"):
        for place in ("마당을", "숲을"):
            sentences.append(f"{subject} {place}천천히 지나갔다")
        sentences.append(f"{subject} 밥을 빠르게 먹었다")
    return [sentence.split() for sentence in sentences]


if __name__ == "__main__":
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    corpus = build_corpus()
    vocabulary = build_vocabulary(corpus, min_count=2)
    counts = cooccurrence_matrix(corpus, vocabulary, window=2)
    ppmi = pmi_matrix(counts)

    print(f"{len(corpus)} sentences, {len(vocabulary)} word types")
    print(f"co-occurrence matrix: {counts.shape[0]}x{counts.shape[1]}, "
          f"{sparsity(counts):.0%} zero")
    print(f"PPMI matrix:          {ppmi.shape[0]}x{ppmi.shape[1]}, "
          f"{sparsity(ppmi):.0%} zero  (clipping adds sparsity)\n")

    print("raw counts are dominated by whatever is common:")
    for word in ("cat", "king", "ate"):
        picked = ", ".join(f"{t}({c:.0f})" for t, c in top_associations(counts, vocabulary, word, 4))
        print(f"  {word:<7} {picked}")

    print("\nPPMI asks 'more than chance?' and the answer changes:")
    for word in ("cat", "king", "ate"):
        picked = ", ".join(f"{t}({v:.2f})" for t, v in top_associations(ppmi, vocabulary, word, 4))
        print(f"  {word:<7} {picked}")

    print("\n'the' demotes itself - no stopword list required:")
    print(f"  {'word':<7}{'raw total':>11}{'partners':>10}{'max PPMI':>10}{'mean PPMI*':>12}")
    for word in ("the", "ate", "cat", "king", "fish"):
        row = ppmi[vocabulary[word]]
        nonzero = row[row > 0]
        print(f"  {word:<7}{counts[vocabulary[word]].sum():>11.0f}"
              f"{np.count_nonzero(counts[vocabulary[word]]):>10}"
              f"{row.max():>10.2f}{nonzero.mean() if nonzero.size else 0.0:>12.2f}")
    print("  * averaged over the words it actually co-occurs with, not over the zeros")
    print("  'the' leads on raw count by 14x and comes last on PPMI strength. It has the")
    print("  most partners and the weakest ties to any of them - which is what being a")
    print("  function word *is*. 'fish' is rare and sits in few contexts, so it scores")
    print("  highest. Day 2's hand-written list and Day 6's IDF both fall out of counting.")

    print("\nthe window is the model:")
    for window in (1, 2, 5):
        windowed = pmi_matrix(cooccurrence_matrix(corpus, vocabulary, window=window))
        picked = ", ".join(t for t, _ in top_associations(windowed, vocabulary, "cat", 4))
        print(f"  window={window}: cat ~ {picked}")
    print("  narrow windows find what can grammatically sit next to a word;")
    print("  wide ones blur toward topic. Neither is more correct than the other.")

    print("\nwords that never co-occur can still be similar - the Day 7 gap:")
    rows = {w: ppmi[vocabulary[w]] for w in ("cat", "dog", "king", "queen")}
    print(f"  cat and dog co-occur {counts[vocabulary['cat'], vocabulary['dog']]:.0f} times,")
    print("  yet their PPMI rows overlap on the contexts they share:")
    for a, b in (("cat", "dog"), ("king", "queen"), ("cat", "king")):
        shared = np.count_nonzero((rows[a] > 0) & (rows[b] > 0))
        print(f"    {a:<6} vs {b:<6}: {shared} shared context words")
    print("  that overlap is what Day 14 compresses into a vector.")
