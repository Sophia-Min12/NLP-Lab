"""Day 14 — SVD word embeddings.

Day 13 produced a PPMI row per word: 100 numbers, 94% of them zero. That is
already a word vector, and it has two problems.

It is **fragile**. Two words can be similar and share no *exact* context —
if one occurs beside ``walked`` and the other beside ``strolled``, the
sparse rows overlap nowhere and similarity reads as zero. The very failure
Day 7 hit, one level down.

It is **large**. One dimension per vocabulary word means 50,000 dimensions
on a real corpus, nearly all zero, for every word.

The singular value decomposition fixes both at once. It factorizes the
matrix into orthogonal directions ordered by how much variance each
explains, and keeping the first ``k`` gives a short dense vector per word.
Words that occurred in *related* contexts end up close even when no exact
context is shared, because the retained directions capture the pattern
rather than the individual cells.

This is latent semantic analysis, and it is the direct ancestor of
word2vec and GloVe — both of which were later shown to be implicitly
factorizing a matrix very much like this one.
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


def truncated_svd(matrix: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factorize and keep the ``k`` strongest directions.

    Returns ``(U_k, S_k, Vt_k)`` with ``U_k`` of shape ``(n, k)``.

    NumPy's ``svd`` already returns singular values in descending order, so
    truncating is a slice. The sign of a singular vector is arbitrary —
    ``(-u, -v)`` describes the same direction as ``(u, v)`` — so never
    compare raw coordinates across runs or libraries; compare distances and
    angles, which are sign-invariant.

    >>> u, s, vt = truncated_svd(np.eye(3), 2)
    >>> u.shape, s.shape, vt.shape
    ((3, 2), (2,), (2, 3))
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {matrix.shape}")
    limit = min(matrix.shape)
    if not 1 <= k <= limit:
        raise ValueError(f"k must be between 1 and {limit}, got {k}")

    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :k], s[:k], vt[:k, :]


def svd_embeddings(matrix: np.ndarray, k: int = 8, weighting: str = "sqrt") -> np.ndarray:
    """One dense ``k``-dimensional vector per row of ``matrix``.

    ``weighting`` decides how much the singular values scale the axes:

    - ``sqrt`` — ``U · sqrt(S)``, the usual choice. It splits the scaling
      symmetrically between the word and context sides, which keeps strong
      directions dominant without letting the first one swamp everything.
    - ``full`` — ``U · S``. The leading direction dominates; on a PPMI
      matrix that direction largely encodes overall word frequency, which
      is rarely what similarity should be about.
    - ``none`` — ``U`` alone. Every direction counts equally, including the
      last ones, which on a small corpus are mostly noise.

    >>> vectors = svd_embeddings(np.eye(4), k=2)
    >>> vectors.shape
    (4, 2)
    """
    if weighting not in {"sqrt", "full", "none"}:
        raise ValueError(f"unknown weighting {weighting!r}; expected sqrt, full or none")
    u, s, _ = truncated_svd(matrix, k)
    if weighting == "none":
        return u
    scale = np.sqrt(s) if weighting == "sqrt" else s
    return u * scale


def explained_variance_ratio(matrix: np.ndarray) -> np.ndarray:
    """Share of squared magnitude carried by each singular direction.

    The cumulative sum answers "how many dimensions do I actually need",
    and on a small corpus the answer is usually "fewer than you think" —
    the tail directions are fitting noise.
    """
    matrix = np.asarray(matrix, dtype=float)
    _, s, _ = np.linalg.svd(matrix, full_matrices=False)
    squares = s ** 2
    total = squares.sum()
    if total == 0:
        return np.zeros_like(squares)
    return squares / total


def reconstruction_error(matrix: np.ndarray, k: int) -> float:
    """Relative Frobenius error of the best rank-``k`` approximation.

    0.0 means perfect reconstruction. This is what is *lost* by
    compressing — and losing some of it is the point, since the discarded
    directions are mostly sampling noise on a corpus this size.
    """
    matrix = np.asarray(matrix, dtype=float)
    u, s, vt = truncated_svd(matrix, k)
    approximation = (u * s) @ vt
    denominator = np.linalg.norm(matrix)
    if denominator == 0:
        return 0.0
    return float(np.linalg.norm(matrix - approximation) / denominator)


def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, leaving zero rows alone.

    Doing this once turns every later cosine similarity into a dot
    product, which is what makes Day 15's neighbour search a single matrix
    multiply.
    """
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """All pairwise cosine similarities at once.

    Day 7 compared two vectors at a time in pure Python. Normalizing the
    rows and multiplying by the transpose does every pair in one call —
    the same arithmetic, rearranged.
    """
    normalized = unit_rows(vectors)
    return normalized @ normalized.T


# reused from day10/day13
def build_corpus() -> list[list[str]]:
    """A deterministic synthetic corpus with known distributional structure.

    The structure is built in, not discovered. See the Day 13 README.
    """
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


if __name__ == "__main__":
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    corpus = build_corpus()
    vocabulary = build_vocabulary(corpus, min_count=2)
    terms = sorted(vocabulary, key=vocabulary.get)
    ppmi = pmi_matrix(cooccurrence_matrix(corpus, vocabulary, window=2))

    print(f"PPMI matrix: {ppmi.shape[0]}x{ppmi.shape[1]}, "
          f"{np.count_nonzero(ppmi == 0) / ppmi.size:.0%} zero\n")

    ratios = explained_variance_ratio(ppmi)
    cumulative = np.cumsum(ratios)
    print("how much each direction explains:")
    print(f"  {'k':>3}{'this one':>11}{'cumulative':>13}{'reconstruction':>16}")
    for k in (1, 2, 4, 8, 16, 32):
        print(f"  {k:>3}{ratios[k - 1]:>10.1%}{cumulative[k - 1]:>13.1%}"
              f"{reconstruction_error(ppmi, k):>16.3f}")
    print(f"  the full matrix has {len(ratios)} directions; the tail is mostly noise")

    vectors = svd_embeddings(ppmi, k=8)
    print(f"\ncompressed: {ppmi.shape[1]} sparse dimensions -> {vectors.shape[1]} dense ones")
    print(f"  'cat' as a vector: {np.round(vectors[vocabulary['cat']], 3)}")

    similarity = cosine_similarity_matrix(vectors)
    print("\nsimilarity in the compressed space:")
    for a, b in (("cat", "dog"), ("king", "queen"), ("cat", "king"),
                 ("bread", "soup"), ("cat", "bread")):
        print(f"  {a:<7} vs {b:<8} {similarity[vocabulary[a], vocabulary[b]]:+.3f}")

    print("\nwhy compress at all - sparse rows miss what dense ones catch:")
    sparse_similarity = cosine_similarity_matrix(ppmi)
    print(f"  {'pair':<18}{'sparse PPMI':>13}{'SVD k=8':>10}")
    for a, b in (("cat", "dog"), ("king", "queen"), ("garden", "forest")):
        print(f"  {a + ' / ' + b:<18}{sparse_similarity[vocabulary[a], vocabulary[b]]:>13.3f}"
              f"{similarity[vocabulary[a], vocabulary[b]]:>10.3f}")

    print("\nthe weighting changes what 'similar' means:")
    for weighting in ("sqrt", "full", "none"):
        trial = cosine_similarity_matrix(svd_embeddings(ppmi, k=8, weighting=weighting))
        print(f"  {weighting:<6} cat/dog {trial[vocabulary['cat'], vocabulary['dog']]:+.3f}"
              f"   cat/the {trial[vocabulary['cat'], vocabulary['the']]:+.3f}")

    print("\nchoosing k, honestly:")
    for k in (2, 4, 8, 16, 32):
        trial = cosine_similarity_matrix(svd_embeddings(ppmi, k=k))
        same = trial[vocabulary["cat"], vocabulary["dog"]]
        across = trial[vocabulary["cat"], vocabulary["bread"]]
        print(f"  k={k:<3} cat/dog {same:+.3f}   cat/bread {across:+.3f}   gap {same - across:+.3f}")
    print("  too few dimensions and everything collapses together; too many and the")
    print("  noise directions come back. With 49 words there is not much room to move.")
