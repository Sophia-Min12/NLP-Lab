"""Day 7 — Cosine similarity and document ranking.

Days 5 and 6 turned documents into weighted vectors. Comparing them is now
a geometry question, and the answer is not the obvious one.

Euclidean distance asks how far apart two vectors are, which makes a long
document about soup look unlike a short document about soup — the long one
simply has bigger numbers in every coordinate. **Cosine similarity** asks
about the angle instead, which is a question about *proportions*: what share
of this document's weight sits on each term. Doubling a document's length
leaves its direction unchanged, so the two soup documents finally match.

Ranking search results is then just: weight the query with the collection's
own IDF, compare it to every document, sort. That is a complete retrieval
system, and Day 8 changes only how the candidates are found — not the score.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata


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


# reused from day05
def _require_documents(documents) -> None:
    """Reject a single document where a list of documents is expected."""
    if isinstance(documents, str):
        raise TypeError("expected a list of documents, got str")
    for i, document in enumerate(documents):
        if isinstance(document, str):
            raise TypeError(
                f"document {i} is a str; documents must be token lists "
                "(call normalize() on each one first)"
            )


def _require_same_length(a, b) -> None:
    """Two vectors can only be compared inside one vocabulary."""
    if len(a) != len(b):
        raise ValueError(
            f"vectors have different lengths ({len(a)} and {len(b)}); "
            "they must share one vocabulary"
        )


# reused from day01
def tokenize_words(text: str) -> list[str]:
    """Word-level regex tokenization: words group, punctuation separates."""
    _require_str(text)
    text = text.replace("’", "'")  # curly apostrophe -> ASCII
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


# reused from day03/day04/day05/day06
def normalize(text: str) -> list[str]:
    """NFC, casefold, tokenize, drop standalone punctuation."""
    _require_str(text)
    text = unicodedata.normalize("NFC", text)
    tokens = tokenize_words(text.casefold())
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


# reused from day05
def document_frequencies(documents: list[list[str]]) -> dict[str, int]:
    """How many documents contain each term."""
    _require_documents(documents)
    freqs: dict[str, int] = {}
    for document in documents:
        for term in sorted(set(document)):
            freqs[term] = freqs.get(term, 0) + 1
    return freqs


# reused from day05
def build_vocabulary(documents: list[list[str]], min_df: int = 1, max_df: float = 1.0) -> dict[str, int]:
    """Map each surviving term to a column index, alphabetically."""
    _require_documents(documents)
    if min_df < 1:
        raise ValueError(f"min_df must be at least 1, got {min_df}")
    if not 0 < max_df <= 1:
        raise ValueError(f"max_df must be in (0, 1], got {max_df}")
    documents = list(documents)
    freqs = document_frequencies(documents)
    ceiling = max_df * len(documents)
    kept = [term for term, df in freqs.items() if min_df <= df <= ceiling]
    return {term: index for index, term in enumerate(sorted(kept))}


# reused from day05
def bag_of_words(tokens: list[str], vocabulary: dict[str, int]) -> list[int]:
    """Count each vocabulary term in one document."""
    _require_tokens(tokens)
    vector = [0] * len(vocabulary)
    for token in tokens:
        index = vocabulary.get(token)
        if index is not None:
            vector[index] += 1
    return vector


# reused from day06
def term_frequency(counts: list[int], scheme: str = "relative") -> list[float]:
    """Rescale one document's raw counts."""
    if scheme == "raw":
        return [float(count) for count in counts]
    if scheme == "boolean":
        return [1.0 if count else 0.0 for count in counts]
    if scheme == "log":
        return [1.0 + math.log(count) if count else 0.0 for count in counts]
    if scheme != "relative":
        raise ValueError(f"unknown tf scheme {scheme!r}")
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [count / total for count in counts]


# reused from day06
def inverse_document_frequency(
    documents: list[list[str]],
    vocabulary: dict[str, int],
) -> list[float]:
    """Smooth IDF: ``log((1 + N) / (1 + df)) + 1``, safe on unseen terms."""
    _require_documents(documents)
    documents = list(documents)
    n = len(documents)
    freqs = document_frequencies(documents)
    weights = [0.0] * len(vocabulary)
    for term, index in vocabulary.items():
        weights[index] = math.log((1 + n) / (1 + freqs.get(term, 0))) + 1
    return weights


# reused from day06
def tfidf_vector(
    tokens: list[str],
    vocabulary: dict[str, int],
    idf: list[float],
    scheme: str = "relative",
) -> list[float]:
    """One document as TF x IDF, element by element."""
    if len(idf) != len(vocabulary):
        raise ValueError(
            f"idf has {len(idf)} weights but the vocabulary has {len(vocabulary)} terms"
        )
    tf = term_frequency(bag_of_words(tokens, vocabulary), scheme)
    return [t * weight for t, weight in zip(tf, idf)]


def dot(a: list[float], b: list[float]) -> float:
    """Sum of element-wise products.

    Only terms present in *both* vectors contribute, since every other
    product has a zero in it. That single fact is what makes the inverted
    index of Day 8 possible.

    >>> dot([1.0, 2.0], [3.0, 4.0])
    11.0
    """
    _require_same_length(a, b)
    return float(sum(x * y for x, y in zip(a, b)))


def norm(vector: list[float]) -> float:
    """Euclidean length of a vector.

    >>> norm([3.0, 4.0])
    5.0
    """
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of the angle between two vectors, in ``[0, 1]`` for counts.

    Dividing by both lengths is what removes document size from the
    comparison: only the *direction* survives, which is the mix of terms
    rather than the amount of them.

    A zero vector has no direction, so the similarity is defined as 0.0
    rather than raising. That case is routine, not exotic — it is what an
    all-out-of-vocabulary query looks like.

    The result is *mathematically* in ``[0, 1]`` for non-negative vectors,
    but the division makes it only approximately so in floating point:
    parallel vectors can come back as 0.9999999999999998, and occasionally
    a hair above 1.0. Nothing here clamps the value — the formula stays a
    formula — so compare with a tolerance rather than ``== 1.0``.

    >>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
    1.0
    >>> cosine_similarity([1.0, 0.0], [0.0, 1.0])
    0.0
    >>> round(cosine_similarity([1.0, 1.0], [2.0, 2.0]), 12)
    1.0
    """
    _require_same_length(a, b)
    magnitude = norm(a) * norm(b)
    if magnitude == 0:
        return 0.0
    return dot(a, b) / magnitude


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """Straight-line distance, kept for contrast with cosine.

    >>> euclidean_distance([0.0, 0.0], [3.0, 4.0])
    5.0
    """
    _require_same_length(a, b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def rank_documents(
    query: list[str],
    matrix: list[list[float]],
    vocabulary: dict[str, int],
    idf: list[float],
    k: int | None = None,
) -> list[tuple[int, float]]:
    """Score every document against a query and sort, best first.

    The query is weighted with the collection's own ``idf`` — the weights
    that produced ``matrix`` — because fitting new ones on a one-document
    query would make every term equally rare (Day 6).

    Ties break on document id, so results are reproducible. Returns
    ``(document_id, score)`` pairs; ``k`` truncates to the top k.

    >>> vocab = {"a": 0, "b": 1}
    >>> matrix = [[1.0, 0.0], [0.0, 1.0]]
    >>> rank_documents(["a"], matrix, vocab, [1.0, 1.0])
    [(0, 1.0), (1, 0.0)]
    """
    _require_tokens(query)
    if k is not None and k < 0:
        raise ValueError(f"k must be non-negative, got {k}")

    query_vector = tfidf_vector(query, vocabulary, idf)
    scored = [
        (document_id, cosine_similarity(query_vector, row))
        for document_id, row in enumerate(matrix)
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored if k is None else scored[:k]


# reused from day05
DOCUMENTS = [
    "Tokenization splits text into tokens. Every NLP pipeline starts with tokenization.",
    "Normalization folds case and strips punctuation before the tokens are counted.",
    "A search engine ranks documents by how well the documents match the query.",
    "The soup needs more salt. Taste the soup again before serving the soup.",
    "Fry the onions in butter until soft, then add the garlic and the tomatoes.",
    "Rain is expected tomorrow, with strong wind along the coast.",
    "자연어 처리는 토큰화에서 시작된다. 토큰화는 문장을 토큰으로 나눈다.",
    "검색 엔진은 질의와 문서의 유사도를 계산하여 문서를 정렬한다.",
]


if __name__ == "__main__":
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    documents = [normalize(text) for text in DOCUMENTS]
    vocabulary = build_vocabulary(documents)
    idf = inverse_document_frequency(documents, vocabulary)
    matrix = [tfidf_vector(d, vocabulary, idf) for d in documents]

    print("why cosine and not distance - same topic, different lengths:")
    short, long = normalize("soup salt"), normalize("soup salt " * 20)
    pair_vocabulary = build_vocabulary([short, long])
    a = [float(v) for v in bag_of_words(short, pair_vocabulary)]
    b = [float(v) for v in bag_of_words(long, pair_vocabulary)]
    print(f"  short vector      : {a}")
    print(f"  long vector       : {b}")
    print(f"  euclidean distance: {euclidean_distance(a, b):.3f}  <- looks unrelated")
    print(f"  cosine similarity : {cosine_similarity(a, b):.3f}  <- identical direction")

    for query_text in ("tokenization tokens", "soup salt", "search query documents", "검색 문서"):
        print(f"\nquery: {query_text!r}")
        for document_id, score in rank_documents(normalize(query_text), matrix, vocabulary, idf, k=3):
            marker = "  " if score > 0 else " x"
            print(f" {marker} {score:.3f}  doc {document_id}: {DOCUMENTS[document_id][:58]}")

    print("\nevery document against document 0:")
    for document_id, row in enumerate(matrix):
        print(f"  doc {document_id}: {cosine_similarity(matrix[0], row):.3f}")

    print("\nthe lexical ceiling - no shared term means no score, whatever the meaning:")
    for query_text in ("tokenization", "word segmentation"):
        ranked = rank_documents(normalize(query_text), matrix, vocabulary, idf, k=1)
        document_id, score = ranked[0]
        print(f"  {query_text!r:<22} -> best {score:.3f} (doc {document_id})")
    print("  'word segmentation' means the same thing as 'tokenization' and scores nothing.")
