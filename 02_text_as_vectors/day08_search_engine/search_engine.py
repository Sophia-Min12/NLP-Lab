"""Day 8 — Inverted index mini search engine.

Day 7 ranked documents correctly and wastefully. It scored every document
in the collection against every query, and the demo output showed most of
those scores were exactly ``0.000``.

That waste is not incidental, it is predictable. A dot product only adds
terms present in *both* vectors, so a document sharing no term with the
query contributes nothing and could have been skipped without looking at
it. The **inverted index** is the data structure that makes skipping them
possible: instead of *document → terms*, store *term → documents*. A query
then names exactly the documents worth scoring.

This closes Level 2. The scores are identical to Day 7's — the index changes
only which documents get visited, never the answer — and a test asserts
exactly that, because a faster search that quietly ranks differently is not
a faster search.
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


# reused from day01
def tokenize_words(text: str) -> list[str]:
    """Word-level regex tokenization: words group, punctuation separates."""
    _require_str(text)
    text = text.replace("’", "'")  # curly apostrophe -> ASCII
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


# reused from day03/day04/day05/day06/day07
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


# reused from day06/day07
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


# reused from day06/day07
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


# reused from day07
def norm(vector: list[float]) -> float:
    """Euclidean length of a vector."""
    return math.sqrt(sum(value * value for value in vector))


# reused from day07
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine of the angle between two equal-length vectors."""
    if len(a) != len(b):
        raise ValueError(f"vectors have different lengths ({len(a)} and {len(b)})")
    magnitude = norm(a) * norm(b)
    if magnitude == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / magnitude


def build_index(documents: list[list[str]]) -> dict[str, dict[int, int]]:
    """Invert the collection: ``term -> {document_id: count}``.

    Day 5's matrix maps document → terms and stores a cell for every
    term the document does *not* contain. This stores only what exists, and
    answers the question a query actually asks — *which documents contain
    this term* — without scanning anything.

    The per-document counts ride along because scoring needs them anyway;
    a pure boolean index would store only the ids.

    >>> build_index([["a", "b", "a"], ["b"]])
    {'a': {0: 2}, 'b': {0: 1, 1: 1}}
    """
    _require_documents(documents)
    index: dict[str, dict[int, int]] = {}
    for document_id, document in enumerate(documents):
        for term in document:
            postings = index.setdefault(term, {})
            postings[document_id] = postings.get(document_id, 0) + 1
    return index


def candidates(query: list[str], index: dict[str, dict[int, int]]) -> set[int]:
    """Documents containing at least one query term — an OR over postings.

    Everything outside this set is guaranteed to score exactly zero, so
    skipping it costs no accuracy at all. That guarantee is the entire
    justification for the index, and a test checks it directly.

    >>> sorted(candidates(["a", "z"], {"a": {0: 1}, "b": {1: 1}}))
    [0]
    """
    _require_tokens(query)
    found: set[int] = set()
    for term in query:
        found.update(index.get(term, {}))
    return found


def boolean_and(query: list[str], index: dict[str, dict[int, int]]) -> set[int]:
    """Documents containing *every* query term.

    The strictest classical retrieval model, and a useful contrast: it is
    unranked, so it cannot say which of two matches is better, and it goes
    empty the moment one term is missing. An empty query matches nothing
    here rather than everything — the reading that is never a surprise.

    >>> sorted(boolean_and(["a", "b"], {"a": {0: 1, 1: 1}, "b": {1: 1}}))
    [1]
    """
    _require_tokens(query)
    if not query:
        return set()
    sets = [set(index.get(term, {})) for term in query]
    return set.intersection(*sets)


class SearchEngine:
    """Days 5-7 assembled behind one ``search`` call.

    Indexing happens once, in ``__init__``: the vocabulary, the IDF
    weights, the document vectors and the inverted index are all fitted on
    the collection and then reused for every query. This split is the point
    — a search engine does expensive work once and cheap work per query.

    >>> engine = SearchEngine(["the cat sat", "the dog barked"])
    >>> engine.search("cat")[0][0]
    0
    >>> engine.search("nothing here at all")
    []
    """

    def __init__(self, texts: list[str], min_df: int = 1, max_df: float = 1.0) -> None:
        if isinstance(texts, str):
            raise TypeError("expected a list of documents, got str")
        self.texts = list(texts)
        self.documents = [normalize(text) for text in self.texts]
        self.vocabulary = build_vocabulary(self.documents, min_df=min_df, max_df=max_df)
        self.idf = inverse_document_frequency(self.documents, self.vocabulary)
        self.vectors = [tfidf_vector(d, self.vocabulary, self.idf) for d in self.documents]
        self.index = build_index(self.documents)

    def __len__(self) -> int:
        return len(self.documents)

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Rank documents against a query string, best first.

        Only documents the index names as candidates are scored, and only
        results scoring above zero are returned — a search box showing
        every document in the collection at score 0.000 is showing noise.
        Ties break on document id.
        """
        _require_str(query)
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        tokens = normalize(query)
        query_vector = tfidf_vector(tokens, self.vocabulary, self.idf)
        scored = [
            (document_id, cosine_similarity(query_vector, self.vectors[document_id]))
            for document_id in candidates(tokens, self.index)
        ]
        hits = [(document_id, score) for document_id, score in scored if score > 0]
        hits.sort(key=lambda pair: (-pair[1], pair[0]))
        return hits[:k]

    def scored_document_count(self, query: str) -> int:
        """How many documents ``search`` would actually score — the saving."""
        _require_str(query)
        return len(candidates(normalize(query), self.index))

    def explain(self, query: str, document_id: int) -> list[tuple[str, float]]:
        """Which query terms contributed to one document's score, heaviest first.

        Retrieval that cannot say *why* is hard to debug and harder to
        trust; this is the smallest useful version of that.
        """
        _require_str(query)
        if not 0 <= document_id < len(self.documents):
            raise IndexError(f"no document {document_id}; the collection has {len(self.documents)}")

        query_vector = tfidf_vector(normalize(query), self.vocabulary, self.idf)
        document_vector = self.vectors[document_id]
        magnitude = norm(query_vector) * norm(document_vector)
        contributions = []
        for term in sorted(set(normalize(query))):
            index = self.vocabulary.get(term)
            if index is None:
                continue
            product = query_vector[index] * document_vector[index]
            if product > 0:
                contributions.append((term, product / magnitude if magnitude else 0.0))
        return sorted(contributions, key=lambda pair: (-pair[1], pair[0]))


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

    engine = SearchEngine(DOCUMENTS)
    print(f"indexed {len(engine)} documents, {len(engine.vocabulary)} terms, "
          f"{len(engine.index)} postings lists\n")

    print("the index, for three terms:")
    for term in ("tokenization", "the", "soup"):
        print(f"  {term:<14} -> {engine.index[term]}")

    for query in ("tokenization tokens", "soup salt", "search query documents", "검색 문서"):
        print(f"\nquery: {query!r}")
        print(f"  scored {engine.scored_document_count(query)} of {len(engine)} documents")
        for document_id, score in engine.search(query, k=3):
            print(f"    {score:.3f}  doc {document_id}: {engine.texts[document_id][:54]}")

    print("\nwork avoided, per query:")
    for query in ("tokenization", "soup", "the", "quantum helicopter"):
        scored = engine.scored_document_count(query)
        print(f"  {query!r:<22} scores {scored}/{len(engine)} "
              f"-> skips {(1 - scored / len(engine)):.0%}")

    print("\nwhy did doc 2 match 'search query documents'?")
    for term, contribution in engine.explain("search query documents", 2):
        print(f"  {term:<12} {contribution:.3f}")

    print("\nboolean AND is strict, and unranked:")
    for query in ("soup salt", "soup tokenization"):
        matched = sorted(boolean_and(normalize(query), engine.index))
        print(f"  {query!r:<20} -> {matched if matched else 'no document has every term'}")

    print("\nranked search returns nothing rather than a page of zeros:")
    print("  ", engine.search("quantum helicopter"))
