"""Day 1 tests — tokenization edge cases, EN/KR mixed text, type contract.

Day 1 검증 — 토큰화 엣지케이스, 한영 혼합 텍스트, 타입 계약.
Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

from tokenizer import split_sentences, tokenize_whitespace, tokenize_words


class TestTokenizeWhitespace(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(tokenize_whitespace(""), [])

    def test_whitespace_only(self):
        self.assertEqual(tokenize_whitespace("   \n\t "), [])

    def test_internal_spaces_collapse(self):
        self.assertEqual(tokenize_whitespace("a  b"), ["a", "b"])

    def test_korean_eojeol_split(self):
        self.assertEqual(tokenize_whitespace("안녕하세요 세계"), ["안녕하세요", "세계"])

    def test_type_error_on_none(self):
        with self.assertRaises(TypeError):
            tokenize_whitespace(None)


class TestTokenizeWords(unittest.TestCase):
    def test_punctuation_separated(self):
        self.assertEqual(tokenize_words("Hello, world!"), ["Hello", ",", "world", "!"])

    def test_straight_contraction_preserved(self):
        self.assertEqual(tokenize_words("don't"), ["don't"])

    def test_curly_apostrophe_regression(self):
        # Phones and Word insert U+2019 — it must behave like the ASCII one.
        self.assertEqual(tokenize_words("don’t"), ["don't"])

    def test_mixed_english_korean(self):
        self.assertEqual(
            tokenize_words("Don't stop! 자연어 처리는 재미있다."),
            ["Don't", "stop", "!", "자연어", "처리는", "재미있다", "."],
        )

    def test_numbers_kept_as_tokens(self):
        self.assertEqual(tokenize_words("GPT-4 is here"), ["GPT", "-", "4", "is", "here"])

    def test_empty_string(self):
        self.assertEqual(tokenize_words(""), [])

    def test_type_error_on_float(self):
        with self.assertRaises(TypeError):
            tokenize_words(3.14)


class TestSplitSentences(unittest.TestCase):
    def test_three_english_sentences(self):
        result = split_sentences("One. Two! Three?")
        self.assertEqual(result, ["One.", "Two!", "Three?"])
        for sent, mark in zip(result, ".!?"):
            self.assertTrue(sent.endswith(mark))

    def test_no_terminator_returns_whole_text(self):
        self.assertEqual(split_sentences("no punctuation here"), ["no punctuation here"])

    def test_korean_sentences(self):
        self.assertEqual(
            split_sentences("토큰화는 기본이다. 오늘부터 시작한다."),
            ["토큰화는 기본이다.", "오늘부터 시작한다."],
        )

    def test_empty_string(self):
        self.assertEqual(split_sentences(""), [])

    def test_type_error_on_bytes(self):
        with self.assertRaises(TypeError):
            split_sentences(b"bytes")


if __name__ == "__main__":
    unittest.main()
