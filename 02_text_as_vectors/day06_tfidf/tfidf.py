"""Day 6 — TF-IDF from scratch.

Day 5's raw counts make ``the`` the loudest number in every document, which
is exactly backwards: a term that appears everywhere cannot tell any two
documents apart. TF-IDF is the standard repair, and it is two ideas
multiplied together.

**TF** — term frequency — asks how much this document is about the term.
**IDF** — inverse document frequency — asks how much knowing the term
narrows the collection down. A term in every document narrows nothing, and
its IDF is zero, so the product is zero and the term drops out on its own.

That last sentence is the day's real content. Day 2 removed stopwords from a
hand-written list and Day 5 cut them with a ``max_df`` threshold; TF-IDF
needs neither, because a continuous weight derived from the collection
demotes ubiquitous terms automatically and by exactly how ubiquitous they
are. It works for the reason Day 3 measured: word frequencies are wildly
uneven, so "rare" is genuinely informative.
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


# reused from day03/day04/day05
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
def build_vocabulary(
    documents: list[list[str]],
    min_df: int = 1,
    max_df: float = 1.0,
) -> dict[str, int]:
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


TF_SCHEMES = ("raw", "relative", "log", "boolean")


def term_frequency(counts: list[int], scheme: str = "relative") -> list[float]:
    """Rescale one document's raw counts.

    The schemes answer different questions, and the choice matters more
    than it looks:

    - ``raw`` — the counts unchanged. A 1000-word document beats a 50-word
      one on every term simply for being longer.
    - ``relative`` — count / document length. Removes that length bias,
      which is why it is the default.
    - ``log`` — ``1 + log(count)``, zero left at zero. Assumes the 10th
      occurrence of a term adds less than the 2nd, which is usually true.
    - ``boolean`` — presence only. Useful when repetition says nothing,
      as in short titles.

    >>> term_frequency([2, 0, 1], scheme="raw")
    [2.0, 0.0, 1.0]
    >>> term_frequency([3, 1], scheme="relative")
    [0.75, 0.25]
    >>> term_frequency([4, 0], scheme="boolean")
    [1.0, 0.0]
    """
    if scheme not in TF_SCHEMES:
        raise ValueError(f"unknown tf scheme {scheme!r}; expected one of {TF_SCHEMES}")

    if scheme == "raw":
        return [float(count) for count in counts]
    if scheme == "boolean":
        return [1.0 if count else 0.0 for count in counts]
    if scheme == "log":
        return [1.0 + math.log(count) if count else 0.0 for count in counts]

    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [count / total for count in counts]


IDF_SCHEMES = ("smooth", "classic", "probabilistic")


def inverse_document_frequency(
    documents: list[list[str]],
    vocabulary: dict[str, int],
    scheme: str = "smooth",
) -> list[float]:
    """One weight per vocabulary term, in vocabulary order.

    - ``smooth`` — ``log((1 + N) / (1 + df)) + 1``. Both offsets pretend
      one extra document contains every term, so a term present in all *N*
      documents still gets a small positive weight instead of vanishing,
      and an unseen term cannot divide by zero. The trailing ``+ 1`` keeps
      every weight positive. This is the default, and what scikit-learn
      uses.
    - ``classic`` — ``log(N / df)``. The textbook form. A term in every
      document gets exactly 0 and is erased; a term in none divides by
      zero, so it is only safe over a vocabulary built from these same
      documents.
    - ``probabilistic`` — ``log((N - df) / df)``. Falls *negative* for
      terms in more than half the collection, actively penalising them
      rather than merely ignoring them.

    >>> vocab = {"common": 0, "rare": 1}
    >>> docs = [["common", "rare"], ["common"], ["common"]]
    >>> [round(w, 3) for w in inverse_document_frequency(docs, vocab, "classic")]
    [0.0, 1.099]
    """
    if scheme not in IDF_SCHEMES:
        raise ValueError(f"unknown idf scheme {scheme!r}; expected one of {IDF_SCHEMES}")
    _require_documents(documents)

    documents = list(documents)
    n = len(documents)
    freqs = document_frequencies(documents)

    weights = [0.0] * len(vocabulary)
    for term, index in vocabulary.items():
        df = freqs.get(term, 0)
        if scheme == "smooth":
            weights[index] = math.log((1 + n) / (1 + df)) + 1
        elif scheme == "classic":
            if df == 0:
                raise ValueError(
                    f"term {term!r} appears in no document; the 'classic' scheme "
                    "divides by zero on it — use 'smooth' for an open vocabulary"
                )
            weights[index] = math.log(n / df)
        else:  # probabilistic
            if df == 0 or df == n:
                raise ValueError(
                    f"term {term!r} has df={df} of {n} documents; the "
                    "'probabilistic' scheme is undefined there — use 'smooth'"
                )
            weights[index] = math.log((n - df) / df)
    return weights


def tfidf_vector(
    tokens: list[str],
    vocabulary: dict[str, int],
    idf: list[float],
    scheme: str = "relative",
) -> list[float]:
    """One document as TF × IDF, element by element.

    >>> tfidf_vector(["a", "a"], {"a": 0, "b": 1}, [2.0, 5.0])
    [2.0, 0.0]
    """
    if len(idf) != len(vocabulary):
        raise ValueError(
            f"idf has {len(idf)} weights but the vocabulary has {len(vocabulary)} terms"
        )
    tf = term_frequency(bag_of_words(tokens, vocabulary), scheme)
    return [t * weight for t, weight in zip(tf, idf)]


def tfidf_matrix(
    documents: list[list[str]],
    vocabulary: dict[str, int],
    scheme: str = "relative",
    idf_scheme: str = "smooth",
) -> tuple[list[list[float]], list[float]]:
    """Weight a whole collection. Returns ``(matrix, idf)``.

    The IDF weights come back too because they are fitted on *this*
    collection and a query has to be weighted with the same ones to be
    comparable (Day 7). Recomputing them from a one-document query would
    make every term equally rare.
    """
    _require_documents(documents)
    documents = list(documents)
    idf = inverse_document_frequency(documents, vocabulary, idf_scheme)
    matrix = [tfidf_vector(document, vocabulary, idf, scheme) for document in documents]
    return matrix, idf


def top_terms(
    vector: list[float],
    vocabulary: dict[str, int],
    k: int = 5,
) -> list[tuple[str, float]]:
    """The ``k`` heaviest terms in one weighted vector.

    Ties break alphabetically, so the output is reproducible.

    >>> top_terms([0.5, 0.9], {"a": 0, "b": 1}, 1)
    [('b', 0.9)]
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    terms = sorted(vocabulary, key=vocabulary.get)
    pairs = [(term, vector[i]) for term, i in zip(terms, range(len(vector)))]
    return sorted(pairs, key=lambda pair: (-pair[1], pair[0]))[:k]


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
    matrix, idf = tfidf_matrix(documents, vocabulary)
    terms = sorted(vocabulary, key=vocabulary.get)

    print(f"{len(documents)} documents, {len(vocabulary)} terms\n")

    print("IDF gives ubiquitous terms the smallest weight:")
    dfs = document_frequencies(documents)
    ranked = sorted(vocabulary, key=lambda t: (idf[vocabulary[t]], t))
    for term in ranked[:3] + ranked[-3:]:
        print(f"  {term:<14} df={dfs[term]}/{len(documents)}  idf={idf[vocabulary[term]]:.3f}")

    print("\nraw counts vs TF-IDF for document 3:")
    print(f"  text: {DOCUMENTS[3]}")
    counts = bag_of_words(documents[3], vocabulary)
    by_count = sorted(zip(terms, counts), key=lambda p: (-p[1], p[0]))[:4]
    print("  loudest by raw count :", [f"{t} ({c})" for t, c in by_count])
    print("  loudest by TF-IDF    :",
          [f"{t} ({w:.3f})" for t, w in top_terms(matrix[3], vocabulary, 4)])

    print("\n'the' under each IDF scheme (it is in 5 of 8 documents):")
    for scheme in IDF_SCHEMES:
        weights = inverse_document_frequency(documents, vocabulary, scheme)
        print(f"  {scheme:<14} {weights[vocabulary['the']]:+.3f}")

    print("\nclassic IDF erases a term present in every document:")
    every = [["x", "a"], ["x", "b"], ["x", "c"]]
    every_vocabulary = build_vocabulary(every)
    classic = inverse_document_frequency(every, every_vocabulary, "classic")
    smooth = inverse_document_frequency(every, every_vocabulary, "smooth")
    print(f"  classic idf('x') = {classic[every_vocabulary['x']]:.3f}  -> the term vanishes")
    print(f"  smooth  idf('x') = {smooth[every_vocabulary['x']]:.3f}  -> demoted, still present")

    print("\nthe TF scheme changes which document looks longest:")
    for scheme in TF_SCHEMES:
        weights = [sum(term_frequency(bag_of_words(d, vocabulary), scheme)) for d in documents]
        longest = max(range(len(weights)), key=lambda i: weights[i])
        print(f"  {scheme:<10} heaviest document = {longest}  (total weight {weights[longest]:.2f})")

    print("\ntop TF-IDF terms per document (smooth idf):")
    for i, vector in enumerate(matrix):
        picked = ", ".join(f"{t}" for t, _ in top_terms(vector, vocabulary, 3))
        print(f"  doc {i}: {picked}")

    # Honest check on the headline claim. On a collection this small the
    # smooth scheme's +1 floor leaves "the" competitive; classic, which can
    # reach 0, suppresses it properly. TF-IDF replaces a stopword list only
    # when there are enough documents to spread the weights out.
    classic_matrix, _ = tfidf_matrix(documents, vocabulary, idf_scheme="classic")
    print("\ndoes TF-IDF really retire the stopword list? not at N=8:")
    for i in (2, 4):
        smooth_top = [t for t, _ in top_terms(matrix[i], vocabulary, 3)]
        classic_top = [t for t, _ in top_terms(classic_matrix[i], vocabulary, 3)]
        print(f"  doc {i}  smooth : {smooth_top}")
        print(f"         classic: {classic_top}")
    spread = max(idf) / min(idf)
    print(f"  smooth idf spans {min(idf):.2f}-{max(idf):.2f} (ratio {spread:.1f}x) "
          "- too narrow to demote a term appearing 3 times")
