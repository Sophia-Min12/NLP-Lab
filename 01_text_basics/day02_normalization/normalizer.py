"""Day 2 — Text normalization and English stopwords.

Day 1 turned text into tokens. But ``"The"``, ``"the"`` and ``"the."`` are
three different strings, so they become three different vocabulary entries —
inflating every count, vector and index built on top of them. Normalization
collapses differences that do not matter *before* counting starts, and the
whole difficulty is deciding which differences do not matter.

Unicode normalization carries the most weight for the Korean half of this lab:
the syllable ``한`` typed on macOS is often stored decomposed (NFD, three jamo
code points) while the same syllable from a web form arrives composed (NFC,
one code point). They render identically and compare unequal.
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


# reused from day01
def tokenize_words(text: str) -> list[str]:
    """Word-level regex tokenization: words group, punctuation separates."""
    _require_str(text)
    text = text.replace("’", "'")  # curly apostrophe -> ASCII
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Put text into a canonical Unicode form, composed (``NFC``) by default.

    Without this, two visually identical Korean strings can be unequal:

    >>> composed = "한"
    >>> decomposed = unicodedata.normalize("NFD", composed)
    >>> len(composed), len(decomposed)
    (1, 3)
    >>> composed == decomposed
    False
    >>> normalize_unicode(composed) == normalize_unicode(decomposed)
    True
    """
    _require_str(text)
    if form not in {"NFC", "NFD", "NFKC", "NFKD"}:
        raise ValueError(f"unknown normalization form: {form!r}")
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """Collapse every run of whitespace to a single space and trim the ends.

    >>> normalize_whitespace("  hello   world  ")
    'hello world'
    """
    _require_str(text)
    return " ".join(text.split())


def to_lowercase(text: str) -> str:
    """Case-fold the text.

    ``str.casefold()`` rather than ``str.lower()``: it also folds cases
    ``lower()`` misses, such as German ``ß`` -> ``ss``. Korean has no case,
    so Hangul passes through untouched.

    >>> to_lowercase("STRASSE" ) == to_lowercase("Straße")
    True
    """
    _require_str(text)
    return text.casefold()


def strip_accents(text: str) -> str:
    """Drop diacritics: ``café`` -> ``cafe``, ``naïve`` -> ``naive``.

    Decomposes to NFD, removes the combining marks (Unicode category ``Mn``),
    then recomposes. Hangul jamo decompose to category ``Lo``, not ``Mn``, so
    Korean survives this intact — which is exactly why the recompose step is
    not optional.

    >>> strip_accents("café 한국어")
    'cafe 한국어'
    """
    _require_str(text)
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", without_marks)


#: High-frequency English function words. Hand-written rather than imported so
#: the list stays inspectable — every entry here is a deliberate choice, and
#: ``not`` in particular is a trap documented in the README.
STOPWORDS = frozenset(
    """
    a an the
    and but or nor so yet
    if then else than because while
    as at by for from in into of off on onto out over to under up with
    about after again against before below between down during further once
    here there when where why how
    all any both each few more most other some such only own same too very
    is am are was were be been being
    do does did doing done
    have has had having
    can could will would shall should may might must
    i me my myself we us our ours ourselves
    you your yours yourself yourselves
    he him his she her hers it its they them their theirs
    this that these those what which who whom
    no not nor none
    s t d ll m o re ve y
    """.split()
)


def remove_stopwords(tokens: list[str], stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """Drop function words. Comparison is case-insensitive.

    Pass a narrowed set when meaning depends on a listed word — see the
    negation example in the README:

    >>> remove_stopwords(["this", "is", "not", "good"], STOPWORDS - {"not"})
    ['not', 'good']
    """
    if isinstance(tokens, str):
        raise TypeError("expected a list of tokens, got str")
    return [t for t in tokens if t.casefold() not in stopwords]


def strip_punctuation(tokens: list[str]) -> list[str]:
    """Drop tokens with no alphanumeric character in them.

    Applied to tokens, not to raw text, so that ``don't`` survives while the
    standalone ``,`` does not.

    >>> strip_punctuation(["don't", ",", "stop", "!"])
    ["don't", 'stop']
    """
    if isinstance(tokens, str):
        raise TypeError("expected a list of tokens, got str")
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


def normalize(
    text: str,
    *,
    lowercase: bool = True,
    accents: bool = False,
    drop_punctuation: bool = True,
    drop_stopwords: bool = True,
    stopwords: frozenset[str] = STOPWORDS,
) -> list[str]:
    """Run the full pipeline and return normalized tokens.

    Order is deliberate: Unicode form first (so later comparisons are valid),
    then whitespace, then case, then tokenization, then the token-level
    filters. Every step is a keyword flag because none of them is universally
    correct.

    >>> normalize("The cats aren't in THE box.")
    ['cats', "aren't", 'box']
    """
    _require_str(text)
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    if lowercase:
        text = to_lowercase(text)
    if accents:
        text = strip_accents(text)
    tokens = tokenize_words(text)
    if drop_punctuation:
        tokens = strip_punctuation(tokens)
    if drop_stopwords:
        tokens = remove_stopwords(tokens, stopwords)
    return tokens


SAMPLE = (
    "The Cats   aren't  in THE box. "
    "A café in Seoul serves naïve tourists. "
    "이 문장은 정규화 이후에도 조사가 붙어 있다. "
    "Don't remove 'not' if you care about meaning!"
)


if __name__ == "__main__":
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    print("SAMPLE:", SAMPLE, "\n")

    raw = tokenize_words(SAMPLE)
    stages = [
        ("raw tokens (day 1)", raw),
        ("+ lowercase", tokenize_words(to_lowercase(SAMPLE))),
        ("+ no punctuation", normalize(SAMPLE, drop_stopwords=False)),
        ("+ no stopwords", normalize(SAMPLE)),
    ]
    for name, tokens in stages:
        print(f"{name:<20}: {len(tokens):>3} tokens, {len(set(tokens)):>3} unique")

    print("\nfinal tokens:", normalize(SAMPLE))

    print("\naccent folding (token counts unchanged, surface forms not):")
    for before, after in zip(normalize(SAMPLE), normalize(SAMPLE, accents=True)):
        if before != after:
            print(f"  {before} -> {after}")

    print("\nwhy 'not' is a trap:")
    sentence = "this movie is not good"
    print("  default    :", normalize(sentence))
    print("  keeping not:", normalize(sentence, stopwords=STOPWORDS - {"not"}))

    print("\nunicode forms that look identical:")
    composed, decomposed = "한국", unicodedata.normalize("NFD", "한국")
    print(f"  composed={len(composed)} chars, decomposed={len(decomposed)} chars")
    print("  equal as-is        :", composed == decomposed)
    print("  equal after NFC    :", normalize_unicode(composed) == normalize_unicode(decomposed))
