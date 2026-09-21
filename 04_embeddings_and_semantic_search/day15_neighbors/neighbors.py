"""Day 15 — Nearest neighbours and word analogies.

Day 14 turned every word into a short dense vector. This day asks what
those vectors actually know, using the two questions the embedding
literature is built on: **who is nearest**, and **does the famous analogy
arithmetic work**.

``king - man + woman ≈ queen`` is the most repeated result in the field.
It does work here. It also works for the wrong reason, and this day
measures that rather than reporting the headline — because the corpus was
*built* with the structure the arithmetic recovers, and because a baseline
that does no arithmetic at all gets most of the same answers.

Both findings are real. Neither is a reason to distrust embeddings; they
are reasons to distrust *analogy accuracy as a metric*, which is the
conclusion the literature eventually reached too.
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


# reused from day13
def build_vocabulary(documents: list[list[str]], min_count: int = 1) -> dict[str, int]:
    """Map each term occurring at least min_count times to an index."""
    _require_documents(documents)
    if min_count < 1:
        raise ValueError(f"min_count must be at least 1, got {min_count}")
    counts: dict[str, int] = {}
    for document in documents:
        for token in document:
            counts[token] = counts.get(token, 0) + 1
    kept = sorted(t for t, c in counts.items() if c >= min_count)
    return {term: index for index, term in enumerate(kept)}


# reused from day13
def cooccurrence_matrix(
    sentences: list[list[str]],
    vocabulary: dict[str, int],
    window: int = 2,
) -> np.ndarray:
    """Count how often each pair of words appears within window tokens."""
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
            for other in range(max(0, position - window), min(len(indices), position + window + 1)):
                if other == position:
                    continue
                context = indices[other]
                if context is not None:
                    matrix[centre, context] += 1.0
    return matrix


# reused from day13
def pmi_matrix(counts: np.ndarray, positive: bool = True) -> np.ndarray:
    """Pointwise mutual information, optionally clipped at zero (PPMI)."""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or counts.shape[0] != counts.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {counts.shape}")
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts)
    joint = counts / total
    expected = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(expected > 0, joint / expected, 0.0)
        pmi = np.where(ratio > 0, np.log(np.where(ratio > 0, ratio, 1.0)), 0.0)
    return np.maximum(pmi, 0.0) if positive else pmi


# reused from day14
def svd_embeddings(matrix: np.ndarray, k: int = 24, weighting: str = "sqrt") -> np.ndarray:
    """One dense k-dimensional vector per row of matrix."""
    if weighting not in {"sqrt", "full", "none"}:
        raise ValueError(f"unknown weighting {weighting!r}")
    matrix = np.asarray(matrix, dtype=float)
    limit = min(matrix.shape)
    if not 1 <= k <= limit:
        raise ValueError(f"k must be between 1 and {limit}, got {k}")
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    u, s = u[:, :k], s[:k]
    if weighting == "none":
        return u
    return u * (np.sqrt(s) if weighting == "sqrt" else s)


# reused from day14
def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, leaving zero rows alone."""
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


class WordVectors:
    """A vocabulary plus one unit-length vector per word.

    Rows are normalized once at construction, so every similarity below is
    a dot product and every neighbour search is one matrix-vector multiply.

    >>> vectors = WordVectors({"a": 0, "b": 1}, np.array([[1.0, 0.0], [0.0, 1.0]]))
    >>> vectors.similarity("a", "b")
    0.0
    """

    def __init__(self, vocabulary: dict[str, int], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=float)
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D array, got shape {vectors.shape}")
        if len(vocabulary) != vectors.shape[0]:
            raise ValueError(
                f"{len(vocabulary)} words but {vectors.shape[0]} vectors; they must match"
            )
        self.vocabulary = dict(vocabulary)
        self.terms = sorted(vocabulary, key=vocabulary.get)
        self.vectors = unit_rows(vectors)

    def __len__(self) -> int:
        return len(self.vocabulary)

    def __contains__(self, word: object) -> bool:
        return word in self.vocabulary

    def vector(self, word: str) -> np.ndarray:
        if word not in self.vocabulary:
            raise KeyError(f"{word!r} is not in the vocabulary")
        return self.vectors[self.vocabulary[word]]

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity, which for unit rows is just a dot product."""
        return float(self.vector(a) @ self.vector(b))

    def _rank(self, target: np.ndarray, exclude: set[str], k: int) -> list[tuple[str, float]]:
        """Rank every word against a target direction, best first."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        norm = np.linalg.norm(target)
        if norm == 0:
            return []
        scores = self.vectors @ (target / norm)
        order = sorted(
            range(len(self.terms)),
            key=lambda i: (-scores[i], self.terms[i]),
        )
        picked = [(self.terms[i], float(scores[i])) for i in order if self.terms[i] not in exclude]
        return picked[:k]

    def most_similar(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        """The ``k`` nearest words, excluding the word itself."""
        return self._rank(self.vector(word), {word}, k)

    def analogy(
        self,
        a: str,
        b: str,
        c: str,
        k: int = 3,
        exclude_inputs: bool = True,
    ) -> list[tuple[str, float]]:
        """``a : b :: c : ?`` — rank words near ``b - a + c``.

        ``exclude_inputs`` drops ``a``, ``b`` and ``c`` from the results.
        This is standard practice and it is not a neutral choice: without
        it the top answer is almost always one of the inputs, because
        ``b - a + c`` lands closest to the vector that contributed most to
        it. Excluding them is what makes the task *look* solvable, and it
        is worth knowing that the convention is doing that work.

        >>> vocabulary = {"man": 0, "woman": 1, "king": 2, "queen": 3}
        >>> vectors = WordVectors(vocabulary, np.array([[1.0, 0.0, 0.0],
        ...                                             [0.0, 1.0, 0.0],
        ...                                             [1.0, 0.0, 1.0],
        ...                                             [0.0, 1.0, 1.0]]))
        >>> vectors.analogy("man", "king", "woman", k=1)[0][0]
        'queen'
        """
        target = self.vector(b) - self.vector(a) + self.vector(c)
        exclude = {a, b, c} if exclude_inputs else set()
        return self._rank(target, exclude, k)

    def nearest_to_input(self, a: str, b: str, c: str, which: str = "c") -> str:
        """Baseline: the nearest neighbour of one input, ignoring arithmetic.

        If this answers an analogy as well as ``analogy()`` does, then the
        vector arithmetic contributed nothing to that case — the answer was
        simply the closest word to one of the inputs. Comparing against
        this is the cheapest available defence against believing a result
        that a much dumber method also produces.
        """
        if which not in {"a", "b", "c"}:
            raise ValueError(f"which must be 'a', 'b' or 'c', got {which!r}")
        source = {"a": a, "b": b, "c": c}[which]
        ranked = self._rank(self.vector(source), {a, b, c}, 1)
        return ranked[0][0] if ranked else ""


def evaluate_analogies(
    vectors: WordVectors,
    cases: list[tuple[str, str, str, str]],
) -> dict[str, float]:
    """Accuracy of the arithmetic against two do-nothing baselines.

    Returns shares in ``[0, 1]`` for the arithmetic and for answering with
    the nearest neighbour of ``b`` or of ``c``.
    """
    if not cases:
        raise ValueError("cannot evaluate an empty case list")
    usable = [case for case in cases if all(word in vectors for word in case[:3])]
    if not usable:
        raise ValueError("no case has all of its input words in the vocabulary")

    correct = {"arithmetic": 0, "nearest_b": 0, "nearest_c": 0}
    agreement = 0
    for a, b, c, expected in usable:
        predicted = vectors.analogy(a, b, c, k=1)
        answer = predicted[0][0] if predicted else ""
        baseline_b = vectors.nearest_to_input(a, b, c, "b")
        baseline_c = vectors.nearest_to_input(a, b, c, "c")
        correct["arithmetic"] += answer == expected
        correct["nearest_b"] += baseline_b == expected
        correct["nearest_c"] += baseline_c == expected
        agreement += answer in {baseline_b, baseline_c}

    total = len(usable)
    return {
        "cases": float(total),
        "arithmetic": correct["arithmetic"] / total,
        "nearest_b": correct["nearest_b"] / total,
        "nearest_c": correct["nearest_c"] / total,
        "agrees_with_a_baseline": agreement / total,
    }


#: ``(a, b, c, expected)`` — the gender relation the corpus was built to
#: contain. That it is built in is the point, not a disclaimer.
ANALOGY_CASES = [
    ("man", "king", "woman", "queen"),
    ("king", "queen", "man", "woman"),
    ("woman", "queen", "man", "king"),
    ("his", "her", "king", "queen"),
    ("boy", "girl", "king", "queen"),
    ("king", "queen", "boy", "girl"),
]


# reused from day10/day13/day14
def build_corpus() -> list[list[str]]:
    """A deterministic synthetic corpus with known distributional structure."""
    people = [("king", "his"), ("queen", "her"), ("man", "his"),
              ("woman", "her"), ("boy", "his"), ("girl", "her")]
    animals = ["cat", "dog", "bird", "fox"]
    places = ["garden", "forest", "village", "market"]
    foods = ["bread", "soup", "rice", "fish"]
    motions = ["ran", "walked", "wandered"]

    sentences = []
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
    for oddity in ("wizard", "cobbler", "lantern", "harbour",
                   "thistle", "kettle", "meadow", "cobweb"):
        sentences.append(f"the {oddity} stood by the old wall")
    for subject in ("고양이가", "개가"):
        for place in ("마당을", "숲을"):
            sentences.append(f"{subject} {place}천천히 지나갔다")
        sentences.append(f"{subject} 밥을 빠르게 먹었다")
    return [sentence.split() for sentence in sentences]


def load_vectors(k: int = 24, window: int = 2, min_count: int = 2) -> WordVectors:
    """The whole Level 4 pipeline in one call: count, PPMI, factorize."""
    corpus = build_corpus()
    vocabulary = build_vocabulary(corpus, min_count=min_count)
    ppmi = pmi_matrix(cooccurrence_matrix(corpus, vocabulary, window=window))
    return WordVectors(vocabulary, svd_embeddings(ppmi, k=k))


if __name__ == "__main__":
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    vectors = load_vectors(k=24)
    print(f"{len(vectors)} words, {vectors.vectors.shape[1]} dimensions\n")

    print("nearest neighbours:")
    for word in ("cat", "king", "bread", "garden", "the"):
        picked = ", ".join(f"{t}({s:.2f})" for t, s in vectors.most_similar(word, 4))
        print(f"  {word:<8} {picked}")

    print("\nthe famous one:")
    for a, b, c, expected in ANALOGY_CASES[:3]:
        answer = vectors.analogy(a, b, c, k=3)
        got = ", ".join(f"{t}({s:.2f})" for t, s in answer)
        mark = "ok  " if answer and answer[0][0] == expected else "MISS"
        print(f"  {mark} {a} : {b} :: {c} : ?  ->  {got}   (expected {expected})")

    print("\nBUT - what happens without excluding the input words:")
    for a, b, c, expected in ANALOGY_CASES[:3]:
        raw = vectors.analogy(a, b, c, k=3, exclude_inputs=False)
        print(f"  {a} : {b} :: {c} : ?  ->  {', '.join(t for t, _ in raw)}")
    print("  the arithmetic lands nearest the inputs that built it. The convention of")
    print("  excluding them is not neutral - it is what makes the task look solvable.")

    print("\nAND - a baseline that does no arithmetic at all:")
    print(f"  {'analogy':<26}{'arithmetic':>12}{'nn(b)':>9}{'nn(c)':>9}{'want':>9}")
    for a, b, c, expected in ANALOGY_CASES:
        answer = vectors.analogy(a, b, c, k=1)
        print(f"  {a} : {b} :: {c} : ?{'':<6}"
              f"{(answer[0][0] if answer else '-'):>12}"
              f"{vectors.nearest_to_input(a, b, c, 'b'):>9}"
              f"{vectors.nearest_to_input(a, b, c, 'c'):>9}{expected:>9}")

    scores = evaluate_analogies(vectors, ANALOGY_CASES)
    print(f"\n  over {int(scores['cases'])} cases:")
    print(f"    arithmetic          {scores['arithmetic']:.0%}")
    print(f"    nearest neighbour of b  {scores['nearest_b']:.0%}")
    print(f"    nearest neighbour of c  {scores['nearest_c']:.0%}")
    print(f"    arithmetic agreed with a baseline in {scores['agrees_with_a_baseline']:.0%} of cases")

    print("\nwhy king:queen is nearly free:")
    print(f"  similarity(king, queen) = {vectors.similarity('king', 'queen'):.3f}")
    print("  they are almost the same vector, so 'queen' is what you get by asking for")
    print("  anything near 'king' at all. The arithmetic is not tested by this pair.")

    print("\nand the corpus was built with exactly this structure:")
    print("    'the king ruled the kingdom from his throne'")
    print("    'the queen ruled the kingdom from her throne'")
    print("  his/her is the only systematic difference between them. The analogy")
    print("  recovers a relation that was put there on purpose. That is a working")
    print("  demonstration of the arithmetic, not evidence about real language.")
