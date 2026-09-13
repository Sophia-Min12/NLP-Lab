"""Day 2 tests — normalization steps, the stopword list, and the full pipeline.

Korean strings below are functional test data (Unicode forms and the limits of
whitespace-level tokenization), not documentation. Runnable via `pytest` (repo
root) or `python -m unittest` (this folder).
"""

import unicodedata
import unittest

from normalizer import (
    STOPWORDS,
    normalize,
    normalize_unicode,
    normalize_whitespace,
    remove_stopwords,
    strip_accents,
    strip_punctuation,
    to_lowercase,
    tokenize_words,
)


class TestNormalizeUnicode(unittest.TestCase):
    def test_decomposed_korean_is_not_equal_before_normalizing(self):
        composed = "한국어"
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertNotEqual(composed, decomposed)
        self.assertEqual(len(composed), 3)
        # 한 and 국 decompose to 3 jamo each, 어 to 2 — it has no final
        # consonant. Character counts over Hangul are meaningless until the
        # form is normalized.
        self.assertEqual(len(decomposed), 8)

    def test_both_forms_converge_under_nfc(self):
        composed = "한국어"
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertEqual(normalize_unicode(composed), normalize_unicode(decomposed))

    def test_nfd_form_requested_explicitly(self):
        self.assertEqual(len(normalize_unicode("한", form="NFD")), 3)

    def test_ascii_unchanged(self):
        self.assertEqual(normalize_unicode("plain ascii"), "plain ascii")

    def test_empty_string(self):
        self.assertEqual(normalize_unicode(""), "")

    def test_unknown_form_rejected(self):
        with self.assertRaises(ValueError):
            normalize_unicode("한", form="NFZ")

    def test_type_error_on_none(self):
        with self.assertRaises(TypeError):
            normalize_unicode(None)


class TestNormalizeWhitespace(unittest.TestCase):
    def test_tabs_and_newlines_collapse(self):
        self.assertEqual(normalize_whitespace("  hello \t world \n"), "hello world")

    def test_leading_and_trailing_trimmed(self):
        self.assertEqual(normalize_whitespace("   padded   "), "padded")

    def test_whitespace_only_becomes_empty(self):
        self.assertEqual(normalize_whitespace(" \t\n "), "")

    def test_korean_spacing_preserved_between_eojeol(self):
        self.assertEqual(normalize_whitespace("한국어   처리"), "한국어 처리")

    def test_type_error_on_list(self):
        with self.assertRaises(TypeError):
            normalize_whitespace(["a"])


class TestToLowercase(unittest.TestCase):
    def test_basic_lowering(self):
        self.assertEqual(to_lowercase("THE Cats"), "the cats")

    def test_casefold_beats_lower_on_eszett(self):
        # str.lower() would leave "straße" and never match "STRASSE".
        self.assertEqual(to_lowercase("Straße"), to_lowercase("STRASSE"))

    def test_korean_has_no_case(self):
        self.assertEqual(to_lowercase("한국어"), "한국어")

    def test_type_error_on_int(self):
        with self.assertRaises(TypeError):
            to_lowercase(42)


class TestStripAccents(unittest.TestCase):
    def test_latin_diacritics_removed(self):
        self.assertEqual(strip_accents("café naïve résumé"), "cafe naive resume")

    def test_korean_survives_decompose_recompose(self):
        # Hangul jamo are category Lo, not Mn, so they must come back intact.
        text = "한국어 처리"
        self.assertEqual(strip_accents(text), text)
        self.assertEqual(len(strip_accents(text)), len(text))

    def test_ascii_unchanged(self):
        self.assertEqual(strip_accents("plain"), "plain")

    def test_type_error_on_none(self):
        with self.assertRaises(TypeError):
            strip_accents(None)


class TestRemoveStopwords(unittest.TestCase):
    def test_common_function_words_dropped(self):
        tokens = ["the", "cat", "is", "on", "the", "mat"]
        self.assertEqual(remove_stopwords(tokens), ["cat", "mat"])

    def test_comparison_is_case_insensitive(self):
        self.assertEqual(remove_stopwords(["The", "THE", "Cat"]), ["Cat"])

    def test_custom_set_can_keep_negation(self):
        tokens = ["this", "is", "not", "good"]
        self.assertEqual(remove_stopwords(tokens, STOPWORDS - {"not"}), ["not", "good"])

    def test_empty_list(self):
        self.assertEqual(remove_stopwords([]), [])

    def test_korean_tokens_untouched_by_english_list(self):
        tokens = ["이", "문장은", "그대로"]
        self.assertEqual(remove_stopwords(tokens), tokens)

    def test_type_error_on_str_input(self):
        with self.assertRaises(TypeError):
            remove_stopwords("the cat")


class TestStripPunctuation(unittest.TestCase):
    def test_standalone_marks_dropped(self):
        self.assertEqual(strip_punctuation(["hello", ",", "world", "!"]), ["hello", "world"])

    def test_contraction_survives(self):
        self.assertEqual(strip_punctuation(["don't", "."]), ["don't"])

    def test_numbers_kept(self):
        self.assertEqual(strip_punctuation(["GPT", "-", "4"]), ["GPT", "4"])

    def test_empty_list(self):
        self.assertEqual(strip_punctuation([]), [])

    def test_type_error_on_str_input(self):
        with self.assertRaises(TypeError):
            strip_punctuation("hello, world")


class TestNormalizePipeline(unittest.TestCase):
    def test_default_pipeline(self):
        self.assertEqual(normalize("The cats aren't in THE box."), ["cats", "aren't", "box"])

    def test_case_variants_collapse_to_one_token(self):
        tokens = normalize("Cat CAT cat", drop_stopwords=False)
        self.assertEqual(tokens, ["cat", "cat", "cat"])
        self.assertEqual(len(set(tokens)), 1)

    def test_normalization_shrinks_the_vocabulary(self):
        text = "The Cat sat. the cat SAT!"
        self.assertLess(len(set(normalize(text))), len(set(tokenize_words(text))))

    def test_all_stopwords_normalizes_to_nothing(self):
        # The honest failure mode: this sentence carries meaning, its
        # normalized form carries none.
        self.assertEqual(normalize("To be or not to be"), [])

    def test_flags_can_be_turned_off(self):
        self.assertEqual(
            normalize("The cat!", lowercase=False, drop_punctuation=False, drop_stopwords=False),
            ["The", "cat", "!"],
        )

    def test_accents_off_by_default(self):
        self.assertEqual(normalize("Café"), ["café"])
        self.assertEqual(normalize("Café", accents=True), ["cafe"])

    def test_decomposed_and_composed_korean_produce_identical_tokens(self):
        text = "정규화는 유니코드 형태를 통일한다"
        self.assertEqual(normalize(text), normalize(unicodedata.normalize("NFD", text)))

    def test_korean_particles_still_attached(self):
        # Day 1's open problem is not solved by normalization: 처리 + 는 stays
        # one token, so 처리는 and 처리를 remain separate vocabulary entries.
        tokens = normalize("자연어 처리는 어렵고 자연어 처리를 배운다")
        self.assertIn("처리는", tokens)
        self.assertIn("처리를", tokens)
        self.assertNotIn("처리", tokens)

    def test_empty_string(self):
        self.assertEqual(normalize(""), [])

    def test_type_error_on_bytes(self):
        with self.assertRaises(TypeError):
            normalize(b"bytes")


class TestStopwordList(unittest.TestCase):
    def test_list_is_immutable(self):
        self.assertIsInstance(STOPWORDS, frozenset)

    def test_entries_are_lowercase(self):
        for word in STOPWORDS:
            self.assertEqual(word, word.casefold())

    def test_contains_expected_function_words(self):
        for word in ("the", "is", "of", "and", "not"):
            self.assertIn(word, STOPWORDS)

    def test_does_not_contain_content_words(self):
        for word in ("cat", "tokenization", "seoul"):
            self.assertNotIn(word, STOPWORDS)


if __name__ == "__main__":
    unittest.main()
