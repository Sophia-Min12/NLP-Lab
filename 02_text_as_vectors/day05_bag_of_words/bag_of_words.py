"""Day 5 — Bag-of-words and vocabulary building.

Level 1 ended with text as countable units. Turning those counts into
*vectors* is what lets a machine compare two documents at all: once a
document is a list of numbers, similarity becomes geometry (Day 7) and
retrieval becomes arithmetic (Day 8).

The move itself is small. Fix an ordering of the vocabulary, then represent
each document by how many times it uses each term. The subtle part is the
word **fix**: two documents are only comparable if position 7 means the same
term in both. That shared ordering is the vocabulary, and building it is a
separate step from vectorizing precisely because it has to be shared.

What is lost is word order — "the dog bit the man" and "the man bit the dog"
collapse to the same vector, which is the failure Day 4 already diagnosed.
An n-gram vocabulary is the standard patch, and it works here unchanged:
these functions never assume a term is a single word.
"""

from __future__ import annotations

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


def _require_documents(documents) -> None:
    """Reject a single document where a list of documents is expected.

    ``["the", "cat"]`` is a perfectly good document *and* a plausible list
    of two documents, so the mistake is silent without this guard.
    """
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


# reused from day03/day04, stopwords deliberately kept (see the README)
def normalize(text: str) -> list[str]:
    """NFC, casefold, tokenize, drop standalone punctuation.

    >>> normalize("The Cat, THE cat!")
    ['the', 'cat', 'the', 'cat']
    """
    _require_str(text)
    text = unicodedata.normalize("NFC", text)
    tokens = tokenize_words(text.casefold())
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


def document_frequencies(documents: list[list[str]]) -> dict[str, int]:
    """How many documents contain each term — not how often it occurs.

    A term appearing nine times in one document has a document frequency of
    1. The distinction is the whole basis of IDF on Day 6.

    >>> document_frequencies([["a", "a"], ["a", "b"]])
    {'a': 2, 'b': 1}
    """
    _require_documents(documents)
    freqs: dict[str, int] = {}
    for document in documents:
        for term in sorted(set(document)):
            freqs[term] = freqs.get(term, 0) + 1
    return freqs


def build_vocabulary(
    documents: list[list[str]],
    min_df: int = 1,
    max_df: float = 1.0,
) -> dict[str, int]:
    """Map each surviving term to a column index.

    Terms are sorted alphabetically before being numbered. Any deterministic
    order would do; what matters is that it does not depend on which
    document happened to come first, because the index assignment is a
    contract every vector in the collection has to honour.

    ``min_df`` is an absolute document count and ``max_df`` a proportion of
    the collection — the two filters people actually reach for. ``min_df=2``
    drops terms seen in only one document (typos, hapax, proper nouns);
    ``max_df=0.5`` drops terms seen in more than half of them, which is a
    corpus-derived stopword list rather than a hand-written one.

    >>> build_vocabulary([["a", "b"], ["b", "c"]])
    {'a': 0, 'b': 1, 'c': 2}
    >>> build_vocabulary([["a", "b"], ["b", "c"]], min_df=2)
    {'b': 0}
    """
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


def bag_of_words(tokens: list[str], vocabulary: dict[str, int]) -> list[int]:
    """Count each vocabulary term in one document.

    The result has one slot per vocabulary term, in vocabulary order, so
    every document in a collection comes out the same length. Terms outside
    the vocabulary are dropped silently — that is what a fixed vocabulary
    means — so measure the damage with ``oov_rate`` rather than assuming
    there is none.

    >>> bag_of_words(["a", "a", "c"], {"a": 0, "b": 1, "c": 2})
    [2, 0, 1]
    """
    _require_tokens(tokens)
    vector = [0] * len(vocabulary)
    for token in tokens:
        index = vocabulary.get(token)
        if index is not None:
            vector[index] += 1
    return vector


def document_term_matrix(
    documents: list[list[str]],
    vocabulary: dict[str, int],
) -> list[list[int]]:
    """One row per document, one column per vocabulary term.

    >>> document_term_matrix([["a"], ["b", "b"]], {"a": 0, "b": 1})
    [[1, 0], [0, 2]]
    """
    _require_documents(documents)
    return [bag_of_words(document, vocabulary) for document in documents]


def oov_rate(tokens: list[str], vocabulary: dict[str, int]) -> float:
    """Share of a document's tokens that the vocabulary cannot represent.

    Counted over running tokens, not distinct ones: dropping a rare word
    once costs less than dropping a common word ten times. Returns 0.0 for
    an empty document.

    >>> oov_rate(["a", "z"], {"a": 0})
    0.5
    """
    _require_tokens(tokens)
    if not tokens:
        return 0.0
    missing = sum(1 for token in tokens if token not in vocabulary)
    return missing / len(tokens)


def matrix_sparsity(matrix: list[list[int]]) -> float:
    """Share of matrix cells that are zero.

    Routinely above 99% on real collections, because any one document uses
    a tiny fraction of the vocabulary. Storing and multiplying all those
    zeros is precisely what the inverted index of Day 8 avoids. Returns 0.0
    for an empty matrix.

    >>> matrix_sparsity([[1, 0], [0, 0]])
    0.75
    """
    cells = sum(len(row) for row in matrix)
    if cells == 0:
        return 0.0
    zeros = sum(1 for row in matrix for value in row if value == 0)
    return zeros / cells


def to_sparse(vector: list[int]) -> dict[int, int]:
    """Keep only the non-zero entries, as ``{column: count}``.

    The same information in space proportional to what the document
    actually uses rather than to the vocabulary size.

    >>> to_sparse([2, 0, 1])
    {0: 2, 2: 1}
    """
    return {index: value for index, value in enumerate(vector) if value != 0}


#: A small multi-topic collection, carried through Days 5-8 so the vectors,
#: weights, rankings and index are all built over the same documents.
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
    matrix = document_term_matrix(documents, vocabulary)

    print(f"documents      : {len(documents)}")
    print(f"running tokens : {sum(len(d) for d in documents)}")
    print(f"vocabulary     : {len(vocabulary)} terms -> {len(vocabulary)} dimensions")
    print(f"matrix         : {len(matrix)} x {len(vocabulary)} "
          f"= {len(matrix) * len(vocabulary)} cells, "
          f"{matrix_sparsity(matrix):.1%} of them zero")

    print("\none document as a vector:")
    print(f"  text   : {DOCUMENTS[3]}")
    print(f"  dense  : {matrix[3]}")
    print(f"  sparse : {to_sparse(matrix[3])}")
    print("  sparse, with terms spelled out:")
    terms = sorted(vocabulary, key=vocabulary.get)
    for index, count in to_sparse(matrix[3]).items():
        print(f"    {terms[index]:<12} {count}")

    print("\nword order is gone - these two share a vector:")
    pair = [normalize("the dog bit the man"), normalize("the man bit the dog")]
    pair_vocabulary = build_vocabulary(pair)
    rows = document_term_matrix(pair, pair_vocabulary)
    print(f"  {rows[0]}")
    print(f"  {rows[1]}")
    print(f"  identical: {rows[0] == rows[1]}")

    print("\ndocument-frequency filters shrink the vocabulary:")
    for label, kwargs in (
        ("no filter", {}),
        ("min_df=2  (drop terms in only one document)", {"min_df": 2}),
        ("max_df=0.5 (drop terms in over half of them)", {"max_df": 0.5}),
    ):
        size = len(build_vocabulary(documents, **kwargs))
        print(f"  {label:<44} {size:>4} terms")

    print("\nwhat max_df=0.5 removes is a corpus-derived stopword list:")
    wide = set(build_vocabulary(documents)) - set(build_vocabulary(documents, max_df=0.5))
    print("  ", sorted(wide))

    print("\nout-of-vocabulary rate against a min_df=2 vocabulary:")
    narrow = build_vocabulary(documents, min_df=2)
    for i in (0, 3, 6):
        print(f"  doc {i}: {oov_rate(documents[i], narrow):.0%} of tokens dropped")
