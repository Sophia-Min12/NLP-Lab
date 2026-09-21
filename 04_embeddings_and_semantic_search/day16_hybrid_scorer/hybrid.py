"""Day 16 — Capstone I: hybrid ranking scorer.

Two representations now exist, and each fails exactly where the other
works.

**Lexical** (Days 5-8) is precise. TF-IDF over an inverted index knows that
a document containing your exact word is a match, and it can say which
term earned the score. It returns **exactly zero** when the words do not
coincide, however close the meaning — Day 7 measured that with ``word
segmentation`` scoring 0.000 against a document about tokenization.

**Semantic** (Days 13-15) crosses that gap. ``cat`` and ``dog`` never
co-occur and still come out neighbours. It is also vague: it will happily
return something loosely related when an exact match exists, and Day 15
showed how easily it can look better than it is.

Combining them is not a compromise, it is the point. Lexical precision
where the words line up, semantic recall where they do not. That is the
hybrid retrieval RAG systems are built on, and this is the smallest
honest version of it.

The whole difficulty is **combining two scores that are not on the same
scale**, which is where most of this file's care goes.
"""

from __future__ import annotations

import math
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


# reused from day05
def document_frequencies(documents: list[list[str]]) -> dict[str, int]:
    """How many documents contain each term."""
    _require_documents(documents)
    freqs: dict[str, int] = {}
    for document in documents:
        for term in sorted(set(document)):
            freqs[term] = freqs.get(term, 0) + 1
    return freqs


# reused from day13
def cooccurrence_matrix(sentences, vocabulary, window: int = 2) -> np.ndarray:
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
def svd_embeddings(matrix: np.ndarray, k: int = 24) -> np.ndarray:
    """One dense k-dimensional vector per row, scaled by sqrt(S)."""
    matrix = np.asarray(matrix, dtype=float)
    limit = min(matrix.shape)
    if not 1 <= k <= limit:
        raise ValueError(f"k must be between 1 and {limit}, got {k}")
    u, s, _ = np.linalg.svd(matrix, full_matrices=False)
    return u[:, :k] * np.sqrt(s[:k])


# reused from day14
def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, leaving zero rows alone."""
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def min_max(scores: np.ndarray) -> np.ndarray:
    """Rescale to ``[0, 1]``; an all-equal input becomes all zeros.

    Combining two score sets without this is the classic mistake. Lexical
    cosine and embedding cosine occupy different ranges on any given
    query, so a raw weighted sum silently lets whichever happens to be
    larger dominate — and ``alpha`` then does not mean what it says.

    All-equal input maps to zeros rather than ones: if a signal cannot
    distinguish any document, it should contribute nothing rather than a
    constant that survives the weighting.

    >>> min_max(np.array([1.0, 3.0, 2.0]))
    array([0. , 1. , 0.5])
    >>> min_max(np.array([2.0, 2.0]))
    array([0., 0.])
    """
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    low, high = float(scores.min()), float(scores.max())
    if high - low <= 0:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def reciprocal_rank_fusion(rankings: list[list[int]], k: float = 60.0) -> dict[int, float]:
    """Combine ranked id lists by ``Σ 1 / (k + rank)``.

    The alternative to normalizing scores: throw the scores away and keep
    only the order. Nothing then needs to be commensurable, which makes it
    robust when the two scorers have wildly different distributions — the
    reason it is a common default in production retrieval.

    The cost is real: a document that one scorer ranks first by an
    enormous margin counts exactly the same as one it barely preferred.

    >>> sorted(reciprocal_rank_fusion([[1, 2], [2, 1]], k=1.0).items())
    [(1, 0.8333333333333333), (2, 0.8333333333333333)]
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[document_id] = scores.get(document_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridSearcher:
    """Lexical TF-IDF and semantic embeddings over one document collection.

    Both signals are fitted on the same texts, so every document word has
    a vector and the two halves are genuinely comparable.

    >>> searcher = HybridSearcher(["the cat sat", "the dog sat", "the cat ran"])
    >>> searcher.search("cat", alpha=1.0)[0][0] in (0, 2)
    True
    """

    def __init__(
        self,
        texts: list[str],
        embedding_corpus: list[list[str]] | None = None,
        k: int = 24,
        window: int = 2,
        min_count: int = 2,
    ):
        """``embedding_corpus`` trains the word vectors; the documents are
        searched.

        Separating them is not a convenience, it is how this actually
        works. A document collection is usually far too small to learn
        distributional vectors from -- twelve short documents here, where
        ``queen`` occurs exactly once and would be dropped by any sensible
        ``min_count``. Real systems pretrain vectors on a large corpus and
        apply them to whatever they are asked to search. Passing ``None``
        trains on the documents themselves, which is the degenerate case
        and is shown failing in the demo.
        """
        if isinstance(texts, str):
            raise TypeError("expected a list of documents, got str")
        if not texts:
            raise ValueError("cannot build a searcher over no documents")

        self.texts = list(texts)
        self.documents = [normalize(text) for text in self.texts]
        corpus = self.documents if embedding_corpus is None else embedding_corpus
        _require_documents(corpus)

        # --- lexical half (Days 5-8) ---
        self.vocabulary = build_vocabulary(self.documents, min_count=1)
        frequencies = document_frequencies(self.documents)
        total = len(self.documents)
        self.idf = np.zeros(len(self.vocabulary))
        for term, index in self.vocabulary.items():
            self.idf[index] = math.log((1 + total) / (1 + frequencies.get(term, 0))) + 1
        self.lexical_vectors = unit_rows(
            np.array([self._tfidf(document) for document in self.documents])
        )
        self.index: dict[str, set[int]] = {}
        for document_id, document in enumerate(self.documents):
            for term in document:
                self.index.setdefault(term, set()).add(document_id)

        # --- semantic half (Days 13-15) ---
        self.embedding_vocabulary = build_vocabulary(corpus, min_count=min_count)
        ppmi = pmi_matrix(cooccurrence_matrix(corpus, self.embedding_vocabulary, window))
        usable = min(k, min(ppmi.shape)) if len(self.embedding_vocabulary) else 1
        self.word_vectors = unit_rows(svd_embeddings(ppmi, k=usable))
        self.semantic_vectors = unit_rows(
            np.array([self._embed(document) for document in self.documents])
        )

    def __len__(self) -> int:
        return len(self.documents)

    def _tfidf(self, tokens: list[str]) -> np.ndarray:
        counts = np.zeros(len(self.vocabulary))
        for token in tokens:
            index = self.vocabulary.get(token)
            if index is not None:
                counts[index] += 1.0
        total = counts.sum()
        if total > 0:
            counts /= total
        return counts * self.idf

    def _embed(self, tokens: list[str]) -> np.ndarray:
        """IDF-weighted mean of the word vectors present.

        A plain mean would let ``the`` dominate every document vector, for
        the reason Day 6 gave. Weighting by IDF makes the rare, contentful
        words steer the direction — the same fix, applied to a mean
        instead of to a coordinate.
        """
        dimensions = self.word_vectors.shape[1]
        accumulated = np.zeros(dimensions)
        weight_total = 0.0
        for token in tokens:
            index = self.embedding_vocabulary.get(token)
            if index is None:
                continue
            weight = self.idf[self.vocabulary[token]] if token in self.vocabulary else 1.0
            accumulated += weight * self.word_vectors[index]
            weight_total += weight
        return accumulated / weight_total if weight_total else accumulated

    def lexical_scores(self, query: str) -> np.ndarray:
        """Cosine between the TF-IDF query vector and every document."""
        _require_str(query)
        vector = self._tfidf(normalize(query))
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.zeros(len(self.documents))
        return self.lexical_vectors @ (vector / norm)

    def semantic_scores(self, query: str) -> np.ndarray:
        """Cosine between the embedded query and every document."""
        _require_str(query)
        vector = self._embed(normalize(query))
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.zeros(len(self.documents))
        return self.semantic_vectors @ (vector / norm)

    def search(
        self,
        query: str,
        alpha: float = 0.5,
        k: int = 5,
        fusion: str = "linear",
    ) -> list[tuple[int, float]]:
        """Rank documents. ``alpha=1`` is pure lexical, ``0`` pure semantic.

        ``fusion="linear"`` min-max normalizes each signal and takes a
        weighted sum. ``fusion="rrf"`` ignores the scores and combines the
        two orderings instead, which makes ``alpha`` meaningless and is
        reported as such.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        if fusion not in {"linear", "rrf"}:
            raise ValueError(f"unknown fusion {fusion!r}; expected linear or rrf")

        lexical = self.lexical_scores(query)
        semantic = self.semantic_scores(query)

        if fusion == "rrf":
            order = lambda s: sorted(range(len(s)), key=lambda i: (-s[i], i))
            fused = reciprocal_rank_fusion([order(lexical), order(semantic)])
            combined = np.array([fused.get(i, 0.0) for i in range(len(self.documents))])
        else:
            combined = alpha * min_max(lexical) + (1 - alpha) * min_max(semantic)

        ranked = sorted(range(len(combined)), key=lambda i: (-combined[i], i))
        return [(i, float(combined[i])) for i in ranked[:k]]

    def explain(self, query: str, document_id: int) -> dict[str, float]:
        """The two raw signals for one document, before any normalization."""
        _require_str(query)
        if not 0 <= document_id < len(self.documents):
            raise IndexError(f"no document {document_id}")
        return {
            "lexical": float(self.lexical_scores(query)[document_id]),
            "semantic": float(self.semantic_scores(query)[document_id]),
        }


def recall_at_k(
    searcher: HybridSearcher,
    cases: list[tuple[str, set[int]]],
    alpha: float,
    k: int = 3,
) -> float:
    """Share of queries whose relevant document appears in the top ``k``."""
    if not cases:
        raise ValueError("cannot evaluate an empty case list")
    hits = 0
    for query, relevant in cases:
        retrieved = {document_id for document_id, _ in searcher.search(query, alpha=alpha, k=k)}
        hits += bool(retrieved & relevant)
    return hits / len(cases)


# reused from day10/day13/day14/day15
def build_corpus() -> list[list[str]]:
    """The synthetic corpus, used here to *pretrain* the word vectors.

    Its structure is built in, not discovered (Day 13). Every word in
    DOCUMENTS below appears in it, which is what lets the semantic half
    say anything at all about a twelve-document collection.
    """
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


#: Short documents with a deliberate vocabulary gap: the queries below use
#: words that the relevant documents do not contain.
DOCUMENTS = [
    "the cat slept on the warm windowsill all afternoon",
    "the cat ran through the garden chasing a leaf",
    "the dog barked at the passing cart in the village",
    "the dog ran through the forest after a stick",
    "the bird sang in the tall tree beside the river",
    "the fox hid behind the stone wall near the meadow",
    "the king ruled the kingdom from his golden throne",
    "the queen ruled the kingdom from her golden throne",
    "the man carried his heavy sack to the busy market",
    "the woman carried her heavy sack to the busy market",
    "the bread was fresh and warm from the stone oven",
    "the soup was hot and rather salty for my taste",
]

#: ``(query, relevant document ids)`` where the relevant document does
#: **not** contain the query word - the gap lexical search cannot cross.
GAP_QUERIES = [
    ("dog", {0, 1}),        # cat documents
    ("cat", {2, 3}),        # dog documents
    ("queen", {6}),         # the king document
    ("king", {7}),          # the queen document
    ("woman", {8}),         # the man document
    ("soup", {10}),         # the bread document
]

#: Queries whose exact word is in the target. Included so the evaluation is
#: not rigged: a set made only of GAP_QUERIES measures nothing except "can
#: the semantic half do what the lexical half cannot", and the answer to
#: that was never in doubt.
EXACT_QUERIES = [
    ("windowsill", {0}),
    ("cart", {2}),
    ("stick", {3}),
    ("throne", {6, 7}),
    ("oven", {10}),
    ("salty", {11}),
]

EVALUATION = GAP_QUERIES + EXACT_QUERIES


if __name__ == "__main__":
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    corpus = build_corpus()
    searcher = HybridSearcher(DOCUMENTS, embedding_corpus=corpus)
    naive = HybridSearcher(DOCUMENTS)  # vectors trained on the documents alone
    print(f"{len(searcher)} documents, {len(searcher.vocabulary)} terms, "
          f"{searcher.word_vectors.shape[1]} embedding dimensions\n")

    print("the two signals are not on the same scale:")
    for query in ("queen", "dog"):
        lexical = searcher.lexical_scores(query)
        semantic = searcher.semantic_scores(query)
        print(f"  {query!r:<8} lexical  {lexical.min():+.3f} .. {lexical.max():+.3f}")
        print(f"  {'':<8} semantic {semantic.min():+.3f} .. {semantic.max():+.3f}")
    print("  a raw weighted sum would let whichever range is wider win, and alpha")
    print("  would not mean what it says. min-max first, then weight.")

    print("\nquery 'queen' - the relevant document never says 'queen':")
    print(f"  target doc 6: {DOCUMENTS[6]}")
    for name, alpha in (("lexical only", 1.0), ("hybrid", 0.5), ("semantic only", 0.0)):
        ranked = searcher.search("queen", alpha=alpha, k=3)
        shown = ", ".join(f"{i}({s:.2f})" for i, s in ranked)
        print(f"  {name:<14} {shown}")
    raw = searcher.explain("queen", 6)
    print(f"  raw signals for doc 6: lexical {raw['lexical']:.3f}, "
          f"semantic {raw['semantic']:.3f}")

    print("\nrecall@3 by alpha, split by query type:")
    print(f"  {'alpha':>6}{'gap':>8}{'exact':>8}{'overall':>10}")
    for alpha in (1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0):
        gap = recall_at_k(searcher, GAP_QUERIES, alpha=alpha, k=3)
        exact = recall_at_k(searcher, EXACT_QUERIES, alpha=alpha, k=3)
        overall = recall_at_k(searcher, EVALUATION, alpha=alpha, k=3)
        label = ("  <- pure lexical" if alpha == 1.0
                 else "  <- pure semantic" if alpha == 0.0 else "")
        print(f"  {alpha:>6.1f}{gap:>8.0%}{exact:>8.0%}{overall:>10.0%}{label}")
    print("  each extreme fails on the half the other handles, and any blend beats")
    print("  both. That is the case for hybrid retrieval, in one table.")
    print(f"  caveat: {len(EVALUATION)} queries over {len(searcher)} documents, and")
    print(f"  recall@3 by chance alone is {3 / len(searcher):.0%}. Directionally right,")
    print("  statistically worth very little.")

    print("\nand what happens without a pretraining corpus:")
    for alpha in (0.5, 0.0):
        print(f"  alpha={alpha}  vectors from the 12 documents only: "
              f"{recall_at_k(naive, EVALUATION, alpha=alpha, k=3):.0%}")
    print("  twelve short documents cannot support distributional vectors - 'queen'")
    print("  occurs once and is dropped by min_count. The semantic half needs a")
    print("  corpus of its own, which is why real systems pretrain.")

    print("\nreciprocal rank fusion, which needs no normalization at all:")
    fused = recall_at_k(searcher, [(q, r) for q, r in EVALUATION], alpha=0.5, k=3)
    rrf_hits = sum(
        bool({i for i, _ in searcher.search(q, fusion="rrf", k=3)} & r)
        for q, r in EVALUATION
    )
    print(f"  linear (alpha=0.5)  {fused:.0%}")
    print(f"  rrf                 {rrf_hits / len(EVALUATION):.0%}")

    print("\nwhere each signal earns its place:")
    for query, relevant in (("queen", {6}), ("cat", {2, 3}), ("windowsill", {0})):
        lexical_rank = searcher.search(query, alpha=1.0, k=3)
        semantic_rank = searcher.search(query, alpha=0.0, k=3)
        hybrid_rank = searcher.search(query, alpha=0.5, k=3)
        def hit(ranked):
            return "hit " if {i for i, _ in ranked} & relevant else "miss"
        print(f"  {query!r:<13} lexical {hit(lexical_rank)}  "
              f"semantic {hit(semantic_rank)}  hybrid {hit(hybrid_rank)}")
