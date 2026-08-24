"""Day 1 — Rule-based tokenization in pure Python.

Day 1 — 순수 파이썬 규칙 기반 토큰화.
Tokenization is step zero of every NLP/LLM pipeline: before a model can
count, vectorize, or embed anything, text must be split into tokens.
(토큰화는 모든 NLP/LLM 파이프라인의 0단계 — 세고, 벡터화하고, 임베딩하기
전에 텍스트를 토큰으로 쪼개야 한다.)
"""

from __future__ import annotations

import re


def _require_str(text) -> None:
    """Shared type guard so all three tokenizers enforce the same contract.

    세 함수가 동일한 타입 계약을 지키게 하는 공용 가드.
    """
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")


def tokenize_whitespace(text: str) -> list[str]:
    """Split on any run of whitespace (spaces, tabs, newlines).

    공백(스페이스·탭·줄바꿈) 연속을 하나의 구분자로 보고 자른다.

    >>> tokenize_whitespace("Hello   world")
    ['Hello', 'world']
    """
    _require_str(text)
    return text.split()


def tokenize_words(text: str) -> list[str]:
    """Word-level regex tokenization: words group, punctuation separates.

    단어 수준 정규식 토큰화 — 단어는 묶고, 문장부호는 독립 토큰으로 분리한다.
    ``\\w`` is Unicode-aware, so Korean 한글 syllables count as word
    characters; English contractions stay single tokens (both straight ``'``
    and the curly ``’`` apostrophe phones insert are handled).
    (한글 음절도 단어 문자로 취급되고, 영어 축약형은 곧은/둥근 아포스트로피
    모두 한 토큰으로 유지된다.)

    >>> tokenize_words("Don't stop! 자연어 처리는 재미있다.")
    ["Don't", 'stop', '!', '자연어', '처리는', '재미있다', '.']
    """
    _require_str(text)
    text = text.replace("’", "'")  # curly apostrophe -> ASCII
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences after ``.``, ``!`` or ``?``.

    마침표·느낌표·물음표 뒤에서 문장을 나눈다. 종결 부호가 없으면
    전체를 한 문장으로 돌려준다.

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
