"""Day 17 — Capstone II: bilingual search, a CLI, and the honest account.

Seventeen days of primitives, assembled into one command-line tool that
searches a bilingual collection — and used to settle the question the
whole curriculum has been deferring: **can any of this find a Korean
document by its stem?**

The answer is yes, by two routes, and the route everything pointed at
turns out to need an amendment. Day 12's BPE gives the stem back as a
unit, exactly as promised — but only if the end-of-word marker is
dropped, and that marker was one of Day 12's deliberate design decisions.
A choice that is right for one task is wrong for another; the README
works through it.

Run ``python capstone.py --help`` for the interface.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import unicodedata

import numpy as np

END = "</w>"


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


# reused from day04
def char_ngrams(text: str, n: int = 2) -> list[str]:
    """Character windows over the text, spaces removed."""
    _require_str(text)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    joined = "".join(unicodedata.normalize("NFC", text.casefold()).split())
    return [joined[i:i + n] for i in range(len(joined) - n + 1)]


# reused from day12
def word_counts(corpus: list[str]) -> dict[str, int]:
    """Count whole words across a corpus of raw strings."""
    if isinstance(corpus, str):
        raise TypeError("expected a list of strings, got str")
    counts: dict[str, int] = {}
    for text in corpus:
        for word in normalize(text):
            counts[word] = counts.get(word, 0) + 1
    return counts


# reused from day12
def merge_symbols(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Replace every occurrence of pair with the joined symbol, left to right."""
    merged: list[str] = []
    i = 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            merged.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return tuple(merged)


def train_bpe(
    corpus: list[str],
    num_merges: int = 200,
    boundary: bool = True,
) -> list[tuple[str, str]]:
    """Day 12's trainer, with the end-of-word marker made optional.

    ``boundary=True`` is Day 12 exactly: each word ends in ``</w>``, which
    keeps a word-final piece distinct from the same letters mid-word.

    ``boundary=False`` drops it, and for *retrieval* that turns out to
    matter more than the distinction it preserves. See the README.
    """
    if num_merges < 0:
        raise ValueError(f"num_merges must be non-negative, got {num_merges}")

    suffix = (END,) if boundary else ()
    splits = {tuple(word) + suffix: count for word, count in word_counts(corpus).items()}
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        counts: dict[tuple[str, str], int] = {}
        for symbols, weight in splits.items():
            for pair in zip(symbols, symbols[1:]):
                counts[pair] = counts.get(pair, 0) + weight
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        if counts[best] < 2:
            break
        splits = {merge_symbols(s, best): w for s, w in splits.items()}
        merges.append(best)
    return merges


def encode_word(word: str, merges: list[tuple[str, str]], boundary: bool = True) -> list[str]:
    """Replay merges over one word, matching how they were trained."""
    _require_str(word)
    symbols = tuple(word) + ((END,) if boundary else ())
    for pair in merges:
        symbols = merge_symbols(symbols, pair)
    return list(symbols)


class Analyzer:
    """Turns text into index terms. The one choice this capstone is about.

    - ``word``    — Days 1-3. One term per whitespace-and-punctuation word.
    - ``subword`` — Day 12's BPE, with ``boundary`` controlling ``</w>``.
    - ``char``    — Day 4's character n-grams, the classic CJK fallback.

    >>> Analyzer("word").terms("The cat sat")
    ['the', 'cat', 'sat']
    >>> Analyzer("char", n=2).terms("cat")
    ['ca', 'at']
    """

    KINDS = ("word", "subword", "char")

    def __init__(
        self,
        kind: str = "word",
        corpus: list[str] | None = None,
        num_merges: int = 200,
        boundary: bool = False,
        n: int = 2,
    ) -> None:
        if kind not in self.KINDS:
            raise ValueError(f"unknown analyzer {kind!r}; expected one of {self.KINDS}")
        if kind == "subword" and not corpus:
            raise ValueError("the subword analyzer needs a corpus to learn merges from")
        self.kind = kind
        self.boundary = boundary
        self.n = n
        self.merges = train_bpe(corpus, num_merges, boundary) if kind == "subword" else []

    def terms(self, text: str) -> list[str]:
        _require_str(text)
        if self.kind == "word":
            return normalize(text)
        if self.kind == "char":
            return char_ngrams(text, self.n)
        return [
            piece
            for word in normalize(text)
            for piece in encode_word(word, self.merges, self.boundary)
        ]


# reused from day14
def unit_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, leaving zero rows alone."""
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


# reused from day16
def min_max(scores: np.ndarray) -> np.ndarray:
    """Rescale to [0, 1]; an all-equal input becomes all zeros."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return scores
    low, high = float(scores.min()), float(scores.max())
    if high - low <= 0:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


class Engine:
    """TF-IDF retrieval over whatever terms the analyzer produces.

    The analyzer is the only moving part. Everything downstream — the
    vocabulary contract of Day 5, the IDF of Day 6, the cosine of Day 7 —
    is indifferent to whether a term is a word, a subword or a character
    bigram, which is exactly why swapping the units is a one-line change.

    >>> engine = Engine(["the cat sat", "the dog ran"], Analyzer("word"))
    >>> engine.search("cat")[0][0]
    0
    """

    def __init__(self, texts: list[str], analyzer: Analyzer) -> None:
        if isinstance(texts, str):
            raise TypeError("expected a list of documents, got str")
        if not texts:
            raise ValueError("cannot build an engine over no documents")
        self.texts = list(texts)
        self.analyzer = analyzer
        self.documents = [analyzer.terms(text) for text in self.texts]

        vocabulary: dict[str, int] = {}
        for term in sorted({t for document in self.documents for t in document}):
            vocabulary[term] = len(vocabulary)
        self.vocabulary = vocabulary

        total = len(self.documents)
        frequencies: dict[str, int] = {}
        for document in self.documents:
            for term in set(document):
                frequencies[term] = frequencies.get(term, 0) + 1
        self.idf = np.zeros(len(vocabulary))
        for term, index in vocabulary.items():
            self.idf[index] = math.log((1 + total) / (1 + frequencies.get(term, 0))) + 1

        self.vectors = unit_rows(np.array([self._vector(d) for d in self.documents]))

    def __len__(self) -> int:
        return len(self.documents)

    def _vector(self, terms: list[str]) -> np.ndarray:
        counts = np.zeros(len(self.vocabulary))
        for term in terms:
            index = self.vocabulary.get(term)
            if index is not None:
                counts[index] += 1.0
        total = counts.sum()
        if total > 0:
            counts /= total
        return counts * self.idf

    def scores(self, query: str) -> np.ndarray:
        _require_str(query)
        vector = self._vector(self.analyzer.terms(query))
        norm = np.linalg.norm(vector)
        if norm == 0:
            return np.zeros(len(self.documents))
        return self.vectors @ (vector / norm)

    def search(self, query: str, k: int = 3) -> list[tuple[int, float]]:
        """Documents scoring above zero, best first."""
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        scores = self.scores(query)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [(i, float(scores[i])) for i in ranked if scores[i] > 0][:k]

    def shared_terms(self, query: str, document_id: int) -> list[str]:
        """Which analyzer terms the query and a document have in common."""
        if not 0 <= document_id < len(self.documents):
            raise IndexError(f"no document {document_id}")
        return sorted(set(self.analyzer.terms(query)) & set(self.documents[document_id]))


#: A small bilingual collection. The Korean documents never contain a bare
#: stem - every occurrence carries a particle, which is the whole problem.
DOCUMENTS = [
    "검색 엔진은 질의와 문서의 유사도를 계산하여 문서를 정렬한다",
    "토큰화는 문장을 토큰으로 나누고 단어를 정규화한다",
    "고양이가 마당을 천천히 지나갔고 강아지가 뒤를 따랐다",
    "학생이 학교에서 선생에게 질문을 했고 선생은 답을 주었다",
    "the search engine ranks documents by their similarity to the query",
    "tokenization splits a sentence into tokens and normalizes each word",
    "the cat wandered slowly through the garden and the dog followed",
    "the student asked the teacher a question and the teacher answered",
]

#: Pretraining text for the subword analyzer: each stem appears with many
#: particles, each combination once. That keeps (stem, particle) pair
#: counts at 1 so the stem survives as a unit instead of being re-absorbed.
BPE_CORPUS = [
    " ".join(stem + particle for particle in
             ("가", "를", "에", "의", "는", "와", "도", "만", "에서", "으로"))
    for stem in ("문서", "검색", "질의", "토큰", "문장", "단어", "고양이",
                 "강아지", "학생", "선생", "학교", "마당", "유사도", "정렬")
] + [
    "tokenize tokenizer tokenization tokenized tokenizing tokens",
    "search searching searcher searched searches",
    "document documents documented documenting",
    "normalize normalizer normalization normalized",
    "similar similarity similarly",
]

#: ``(query, relevant document ids)``. Every Korean query is a bare stem
#: that appears in the target only with a particle attached.
EVALUATION = [
    ("문서", {0}),
    ("검색", {0}),
    ("질의", {0}),
    ("토큰", {1}),
    ("단어", {1}),
    ("고양이", {2}),
    ("강아지", {2}),
    ("학생", {3}),
    ("선생", {3}),
    ("search", {4}),
    ("tokenization", {5}),
    ("garden", {6}),
]


def build_analyzers() -> dict[str, Analyzer]:
    """The four strategies this capstone compares."""
    return {
        "word": Analyzer("word"),
        "subword(+</w>)": Analyzer("subword", BPE_CORPUS, boundary=True),
        "subword": Analyzer("subword", BPE_CORPUS, boundary=False),
        "char2": Analyzer("char", n=2),
    }


def recall_at_k(engine: Engine, cases=EVALUATION, k: int = 3) -> float:
    """Share of queries whose relevant document lands in the top k."""
    if not cases:
        raise ValueError("cannot evaluate an empty case list")
    hits = 0
    for query, relevant in cases:
        hits += bool({i for i, _ in engine.search(query, k=k)} & relevant)
    return hits / len(cases)


def _print(*parts: str) -> None:
    print(*parts)


def command_search(args) -> int:
    analyzers = build_analyzers()
    if args.analyzer not in analyzers:
        _print(f"unknown analyzer {args.analyzer!r}; choose from {', '.join(analyzers)}")
        return 2
    engine = Engine(DOCUMENTS, analyzers[args.analyzer])
    results = engine.search(args.query, k=args.k)
    if not results:
        _print(f"no match for {args.query!r} using the {args.analyzer} analyzer")
        return 0
    for document_id, score in results:
        _print(f"  {score:.3f}  [{document_id}] {DOCUMENTS[document_id]}")
        if args.explain:
            shared = engine.shared_terms(args.query, document_id)
            _print(f"          matched on: {', '.join(shared) if shared else '(nothing)'}")
    return 0


def command_compare(args) -> int:
    engines = {name: Engine(DOCUMENTS, a) for name, a in build_analyzers().items()}
    _print(f"query: {args.query!r}\n")
    for name, engine in engines.items():
        results = engine.search(args.query, k=args.k)
        if results:
            best, score = results[0]
            shared = engine.shared_terms(args.query, best)
            _print(f"  {name:<16} {score:.3f}  [{best}] {DOCUMENTS[best][:44]}")
            _print(f"  {'':<16} via {', '.join(shared[:6]) if shared else '(nothing)'}")
        else:
            _print(f"  {name:<16} no match")
    return 0


def command_tokenize(args) -> int:
    for name, analyzer in build_analyzers().items():
        _print(f"  {name:<16} {analyzer.terms(args.text)}")
    return 0


def command_evaluate(args) -> int:
    _print(f"recall@{args.k} over {len(EVALUATION)} queries "
           f"({sum(1 for q, _ in EVALUATION if any('가' <= c <= '힣' for c in q))} Korean, "
           f"{sum(1 for q, _ in EVALUATION if q.isascii())} English)\n")
    _print(f"  {'analyzer':<16}{'overall':>9}{'Korean':>9}{'English':>9}{'terms':>8}")
    korean = [(q, r) for q, r in EVALUATION if not q.isascii()]
    english = [(q, r) for q, r in EVALUATION if q.isascii()]
    for name, analyzer in build_analyzers().items():
        engine = Engine(DOCUMENTS, analyzer)
        _print(f"  {name:<16}{recall_at_k(engine, EVALUATION, args.k):>8.0%}"
               f"{recall_at_k(engine, korean, args.k):>9.0%}"
               f"{recall_at_k(engine, english, args.k):>9.0%}"
               f"{len(engine.vocabulary):>8}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="capstone",
        description="Bilingual search over the NLP-Lab primitives.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="search the collection")
    search.add_argument("query")
    search.add_argument("-a", "--analyzer", default="subword",
                        help="word, subword, subword(+</w>) or char2")
    search.add_argument("-k", type=int, default=3)
    search.add_argument("-e", "--explain", action="store_true",
                        help="show which terms matched")
    search.set_defaults(func=command_search)

    compare = subparsers.add_parser("compare", help="run one query through every analyzer")
    compare.add_argument("query")
    compare.add_argument("-k", type=int, default=1)
    compare.set_defaults(func=command_compare)

    tokenize = subparsers.add_parser("tokenize", help="show how each analyzer splits text")
    tokenize.add_argument("text")
    tokenize.set_defaults(func=command_tokenize)

    evaluate = subparsers.add_parser("evaluate", help="recall@k for every analyzer")
    evaluate.add_argument("-k", type=int, default=3)
    evaluate.set_defaults(func=command_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
