"""Day 1 — Rule-based tokenization in pure Python.

Tokenization is step zero of every NLP/LLM pipeline: before a model can
count, vectorize, or embed anything, text must be split into tokens.
Sample and test data deliberately mix English and Korean, since ``\\w`` in
Python's ``re`` is Unicode-aware.
"""

from __future__ import annotations

import re


def _require_str(text) -> None:
    """Shared type guard so all three tokenizers enforce the same contract."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")


def tokenize_whitespace(text: str) -> list[str]:
    """Split on any run of whitespace (spaces, tabs, newlines).

    >>> tokenize_whitespace("Hello   world")
    ['Hello', 'world']
    """
    _require_str(text)
    return text.split()


def tokenize_words(text: str) -> list[str]:
    """Word-level regex tokenization: words group, punctuation separates.

    ``\\w`` is Unicode-aware, so Korean Hangul syllables count as word
    characters; English contractions stay single tokens (both straight ``'``
    and the curly ``’`` apostrophe phones insert are handled).

    >>> tokenize_words("Don't stop! 자연어 처리는 재미있다.")
    ["Don't", 'stop', '!', '자연어', '처리는', '재미있다', '.']
    """
    _require_str(text)
    text = text.replace("’", "'")  # curly apostrophe -> ASCII
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences after ``.``, ``!`` or ``?``.

    A text with no terminator comes back as one sentence.

    >>> split_sentences("One. Two! Three?")
    ['One.', 'Two!', 'Three?']
    """
    _require_str(text)
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in pieces if p.strip()]


SAMPLE = (
    "Natural language processing starts with tokenization. "
    "Don't underestimate this step! "
    "자연어 처리는 토큰화에서 시작된다. "
    "이 단계를 절대 과소평가하지 마라! "
    "GPT-style models use subword tokenizers like BPE. "
    "우리는 Day 12에서 BPE를 직접 구현할 것이다."
)


if __name__ == "__main__":
    print("SAMPLE:", SAMPLE, "\n")
    for name, fn in (("whitespace", tokenize_whitespace), ("regex words", tokenize_words)):
        tokens = fn(SAMPLE)
        print(f"{name:<12}: {len(tokens):>3} tokens, {len(set(tokens)):>3} unique")
    print("\nsentences:")
    for i, sent in enumerate(split_sentences(SAMPLE), 1):
        print(f"  {i}. {sent}")
