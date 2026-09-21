"""Day 12 — Byte-Pair Encoding tokenizer.

Every unresolved problem in this curriculum traces back to one decision made
on Day 1: that the unit of text is the word. Day 5 watched the vocabulary
explode, Day 6 watched a stem's evidence split across its inflected forms,
Day 7 watched a query miss a document that plainly contained the word, and
Day 10 had no way at all to score a word it had never seen.

**Byte-Pair Encoding** stops treating words as atoms. It starts from
characters and repeatedly merges the most frequent adjacent pair, so the
units are *learned from the corpus* rather than fixed in advance. Frequent
words end up as single symbols; rare words decompose into pieces that are
themselves frequent.

Two properties follow, and they are why every GPT-style model uses a member
of this family:

- **There is no out-of-vocabulary.** Any string at all encodes into
  subwords, falling back to single characters in the worst case.
- **Related forms share pieces.** ``고양이가`` and ``고양이를`` both contain
  ``고양이``, so evidence about the stem finally accumulates in one place.

BPE is a *compression* algorithm applied to tokenization, and it knows
nothing about morphology. Where its pieces line up with meaningful units,
that is frequency agreeing with linguistics, not the algorithm
understanding anything.
"""

from __future__ import annotations

import re
import sys
import unicodedata

END = "</w>"


# reused from day01
def _require_str(text) -> None:
    """Shared type guard so every function enforces the same contract."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")


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


def word_counts(corpus: list[str]) -> dict[str, int]:
    """Count whole words across a corpus of raw strings.

    BPE is trained on *types weighted by frequency*, not on running text:
    merging inside ``the`` once is the same decision however many times
    ``the`` occurs, so long as the count is carried along.

    >>> word_counts(["a cat", "a dog"])
    {'a': 2, 'cat': 1, 'dog': 1}
    """
    if isinstance(corpus, str):
        raise TypeError("expected a list of strings, got str")
    counts: dict[str, int] = {}
    for text in corpus:
        for word in normalize(text):
            counts[word] = counts.get(word, 0) + 1
    return counts


def initial_splits(counts: dict[str, int]) -> dict[tuple[str, ...], int]:
    """Split every word into characters plus an end-of-word marker.

    The ``</w>`` marker does two jobs. It stops merges running across word
    boundaries, and it keeps a word-final piece distinct from the same
    letters mid-word — ``er</w>`` in *faster* is a suffix, ``er`` in
    *ergonomic* is not.

    >>> initial_splits({"at": 2})
    {('a', 't', '</w>'): 2}
    """
    return {tuple(word) + (END,): count for word, count in counts.items()}


def pair_counts(splits: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    """Count adjacent symbol pairs, weighted by word frequency.

    >>> pair_counts({("a", "t", "</w>"): 2})
    {('a', 't'): 2, ('t', '</w>'): 2}
    """
    counts: dict[tuple[str, str], int] = {}
    for symbols, weight in splits.items():
        for pair in zip(symbols, symbols[1:]):
            counts[pair] = counts.get(pair, 0) + weight
    return counts


def merge_symbols(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Replace every occurrence of ``pair`` with the joined symbol.

    Scanning left to right means overlapping occurrences are taken
    greedily and consistently — ``aaa`` with pair ``(a, a)`` becomes
    ``aa a``, never ``a aa``. Training and encoding share this function, so
    they cannot drift apart.

    >>> merge_symbols(("a", "t", "</w>"), ("a", "t"))
    ('at', '</w>')
    """
    if len(pair) != 2:
        raise ValueError(f"pair must have two symbols, got {len(pair)}")
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
    num_merges: int = 50,
) -> tuple[list[tuple[str, str]], set[str]]:
    """Learn merge rules. Returns ``(merges, symbol_vocabulary)``.

    Each round counts every adjacent pair and merges the most frequent one.
    Ties break on the pair itself so training is reproducible — without
    that, two runs on the same corpus could produce different tokenizers.

    Training stops early if no pair occurs more than once: merging a
    one-off pair adds a symbol that will never be useful again.

    >>> merges, vocabulary = train_bpe(["low low lower"], num_merges=3)
    >>> merges[0]
    ('l', 'o')
    """
    if num_merges < 0:
        raise ValueError(f"num_merges must be non-negative, got {num_merges}")

    splits = initial_splits(word_counts(corpus))
    vocabulary = {symbol for symbols in splits for symbol in symbols}
    merges: list[tuple[str, str]] = []

    for _ in range(num_merges):
        counts = pair_counts(splits)
        if not counts:
            break
        best = min(counts, key=lambda pair: (-counts[pair], pair))
        if counts[best] < 2:
            break
        splits = {merge_symbols(symbols, best): weight for symbols, weight in splits.items()}
        merges.append(best)
        vocabulary.add(best[0] + best[1])

    return merges, vocabulary


def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Encode one word by replaying the merges in the order they were learned.

    Order matters: a later merge may depend on a symbol an earlier one
    created. Replaying them in learned order is what makes encoding agree
    with training.

    >>> merges, _ = train_bpe(["low low lower lowest"], num_merges=6)
    >>> encode_word("low", merges)
    ['low</w>']
    """
    _require_str(word)
    symbols = tuple(word) + (END,)
    for pair in merges:
        symbols = merge_symbols(symbols, pair)
    return list(symbols)


def encode(text: str, merges: list[tuple[str, str]]) -> list[str]:
    """Normalize, split into words, and encode each one into subwords."""
    _require_str(text)
    return [piece for word in normalize(text) for piece in encode_word(word, merges)]


class BPETokenizer:
    """A trained tokenizer: merges plus the symbol vocabulary.

    >>> tokenizer = BPETokenizer(["low low lower lowest"], num_merges=6)
    >>> tokenizer.encode_word("low")
    ['low</w>']
    """

    def __init__(self, corpus: list[str], num_merges: int = 50) -> None:
        self.num_merges = num_merges
        self.merges, self.vocabulary = train_bpe(corpus, num_merges)

    def __len__(self) -> int:
        return len(self.vocabulary)

    def encode_word(self, word: str) -> list[str]:
        return encode_word(word, self.merges)

    def encode(self, text: str) -> list[str]:
        return encode(text, self.merges)

    def pieces_per_word(self, words: list[str]) -> float:
        """Average subwords per word — the compression the merges bought.

        Starts near ``len(word) + 1`` with no merges and falls toward 1.0
        as frequent words become single symbols.
        """
        if not words:
            return 0.0
        return sum(len(self.encode_word(w)) for w in words) / len(words)

    def shares_subword(self, a: str, b: str) -> set[str]:
        """Subwords two words have in common — the Korean payoff, measured."""
        return set(self.encode_word(a)) & set(self.encode_word(b))


#: Built so that both halves of the lab's problem are visible: English
#: words sharing a stem across suffixes, and Korean stems sharing across
#: particles. Repetition is deliberate — BPE merges by frequency, so a
#: pattern seen once teaches it nothing.
CORPUS = [
    "tokenize tokenizer tokenization tokenized tokenizing tokens token",
    "normalize normalizer normalization normalized normalizing",
    "vectorize vectorizer vectorization vectorized vectorizing",
    "lower lowest lowering slower slowest slowing",
    "the tokenizer tokenizes the tokens and the normalizer normalizes them",
    "tokenization and normalization and vectorization are pipeline stages",
    "고양이가 고양이를 고양이에게 고양이는 고양이와 고양이도 고양이만",
    "강아지가 강아지를 강아지에게 강아지는 강아지와 강아지도",
    "학생이 학생을 학생에게 학생은 학생과 학생도",
    "고양이가 학생을 보았고 강아지가 고양이를 보았다",
    "학생이 고양이에게 밥을 주었고 강아지에게 물을 주었다",
    "고양이는 자고 강아지는 뛰고 학생은 공부한다",
]

WORD_LEVEL_PROBLEM = ["고양이가", "고양이를", "고양이에게", "고양이는"]


if __name__ == "__main__":
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    counts = word_counts(CORPUS)
    print(f"training corpus: {sum(counts.values())} tokens, {len(counts)} word types\n")

    # 60, not "as many as possible" - see the over-merging section below.
    tokenizer = BPETokenizer(CORPUS, num_merges=60)
    print(f"learned {len(tokenizer.merges)} merges, {len(tokenizer)} symbols")
    print("  first 12 merges:")
    for i, (a, b) in enumerate(tokenizer.merges[:12], 1):
        print(f"    {i:>3}. {a!r} + {b!r} -> {a + b!r}")

    print("\nTHE KOREAN PAYOFF - promised since Day 1:")
    for word in WORD_LEVEL_PROBLEM:
        print(f"  {word} -> {tokenizer.encode_word(word)}")
    shared = tokenizer.shares_subword("고양이가", "고양이를")
    print(f"  고양이가 and 고양이를 now share: {sorted(shared)}")
    print("  at word level they shared nothing at all - four vocabulary entries,")
    print("  four separate counts, four separate IDF weights, no shared evidence")

    print("\nthe same thing for English suffixes:")
    for word in ("tokenize", "tokenizer", "tokenization", "tokens"):
        print(f"  {word:<14} -> {tokenizer.encode_word(word)}")

    print("\nthere is no out-of-vocabulary:")
    for word in ("tokenizers", "고양이처럼", "quixotic", "rhinoceros"):
        pieces = tokenizer.encode_word(word)
        known = sum(1 for p in pieces if p in tokenizer.vocabulary)
        print(f"  {word:<14} -> {pieces}")
        print(f"  {'':<14}    {known}/{len(pieces)} pieces already in the vocabulary")
    print("  every string encodes, falling back to single characters. Day 10's <unk>")
    print("  was a patch over a hole that this representation does not have.")

    print("\nmore merges means longer pieces and fewer of them:")
    words = sorted(counts)
    print(f"  {'merges':>8}{'symbols':>10}{'pieces/word':>14}")
    for num_merges in (0, 10, 30, 60, 120, 240):
        trial = BPETokenizer(CORPUS, num_merges=num_merges)
        print(f"  {len(trial.merges):>8}{len(trial):>10}{trial.pieces_per_word(words):>14.2f}")
    print("  the curve flattens - training stops once no pair repeats, so asking for")
    print("  240 merges does not get 240 of them")

    print("\nover-merging destroys the whole point:")
    largest = BPETokenizer(CORPUS, num_merges=400)
    for label, model in ((f"{len(tokenizer.merges)} merges", tokenizer),
                         (f"{len(largest.merges)} merges", largest)):
        pieces = model.encode_word("고양이가")
        shared = sorted(model.shares_subword("고양이가", "고양이를"))
        print(f"  at {label:<11} 고양이가 -> {pieces}")
        print(f"  {'':<15} shared with 고양이를: {shared if shared else 'nothing'}")
    print("  merge until nothing repeats and every word is one symbol again - which is")
    print("  word-level tokenization, the thing BPE was adopted to escape. The merge")
    print("  count is the hyperparameter that matters, and more is not better.")

    print("\nwhat BPE does not know:")
    for word in ("tokenization", "학생에게"):
        print(f"  {word:<14} -> {tokenizer.encode_word(word)}")
    print("  the pieces are frequent substrings, not morphemes. Where a boundary")
    print("  lands on a real suffix that is frequency agreeing with grammar,")
    print("  not the algorithm knowing any.")
