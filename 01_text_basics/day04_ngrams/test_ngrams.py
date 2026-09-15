"""Day 4 tests — window extraction, padding, sparsity, and character n-grams.

Korean strings below are functional test data (character n-grams recovering
stems that eojeol-level windows miss), not documentation. Runnable via
`pytest` (repo root) or `python -m unittest` (this folder).
"""

import unicodedata
import unittest

from ngrams import (
    BOS,
    EOS,
    SAMPLE,
    bigrams,
    char_ngrams,
    ngram_frequencies,
    ngrams,
    normalize,
    pad_sequence,
    sparsity,
    top_ngrams,
    trigrams,
    word_frequencies,
)


class TestNgrams(unittest.TestCase):
    def test_bigrams_of_three_tokens(self):
        self.assertEqual(ngrams(["a", "b", "c"], 2), [("a", "b"), ("b", "c")])

    def test_unigrams_are_one_tuples(self):
        self.assertEqual(ngrams(["a", "b"], 1), [("a",), ("b",)])

    def test_count_is_len_minus_n_plus_one(self):
        tokens = list("abcdefgh")
        for n in range(1, 6):
            self.assertEqual(len(ngrams(tokens, n)), len(tokens) - n + 1)

    def test_window_longer_than_text_yields_nothing(self):
        self.assertEqual(ngrams(["a", "b"], 5), [])

    def test_window_exactly_as_long_as_text(self):
        self.assertEqual(ngrams(["a", "b"], 2), [("a", "b")])

    def test_empty_tokens(self):
        self.assertEqual(ngrams([], 2), [])

    def test_result_items_are_hashable_tuples(self):
        # They have to be usable as dictionary keys.
        gram = ngrams(["a", "b"], 2)[0]
        self.assertIsInstance(gram, tuple)
        self.assertEqual({gram: 1}[("a", "b")], 1)

    def test_input_list_not_mutated(self):
        tokens = ["a", "b", "c"]
        ngrams(tokens, 2)
        self.assertEqual(tokens, ["a", "b", "c"])

    def test_n_zero_rejected(self):
        with self.assertRaises(ValueError):
            ngrams(["a"], 0)

    def test_negative_n_rejected(self):
        with self.assertRaises(ValueError):
            ngrams(["a"], -2)

    def test_non_integer_n_rejected(self):
        with self.assertRaises(TypeError):
            ngrams(["a"], 2.0)

    def test_bool_n_rejected(self):
        # True is an int in Python and would silently mean n=1.
        with self.assertRaises(TypeError):
            ngrams(["a"], True)

    def test_type_error_on_str_input(self):
        with self.assertRaises(TypeError):
            ngrams("abc", 2)


class TestBigramsAndTrigrams(unittest.TestCase):
    def test_bigrams(self):
        self.assertEqual(bigrams(["the", "dog", "barked"]), [("the", "dog"), ("dog", "barked")])

    def test_trigrams(self):
        self.assertEqual(trigrams(["a", "b", "c", "d"]), [("a", "b", "c"), ("b", "c", "d")])

    def test_they_agree_with_ngrams(self):
        tokens = normalize(SAMPLE)
        self.assertEqual(bigrams(tokens), ngrams(tokens, 2))
        self.assertEqual(trigrams(tokens), ngrams(tokens, 3))


class TestWordOrderIsPreserved(unittest.TestCase):
    """The whole justification for n-grams over a bag of words."""

    def setUp(self):
        self.forward = normalize("the dog bit the man")
        self.reversed_roles = normalize("the man bit the dog")

    def test_unigram_counts_cannot_tell_them_apart(self):
        self.assertEqual(
            word_frequencies(self.forward),
            word_frequencies(self.reversed_roles),
        )

    def test_bigram_counts_can(self):
        self.assertNotEqual(
            ngram_frequencies(self.forward, 2),
            ngram_frequencies(self.reversed_roles, 2),
        )

    def test_the_distinguishing_bigram(self):
        self.assertIn(("dog", "bit"), ngram_frequencies(self.forward, 2))
        self.assertNotIn(("dog", "bit"), ngram_frequencies(self.reversed_roles, 2))


class TestPadSequence(unittest.TestCase):
    def test_bigram_padding_adds_one_marker_each_side(self):
        self.assertEqual(pad_sequence(["dogs", "bark"], 2), [BOS, "dogs", "bark", EOS])

    def test_trigram_padding_adds_two_start_markers(self):
        self.assertEqual(pad_sequence(["dogs", "bark"], 3), [BOS, BOS, "dogs", "bark", EOS])

    def test_unigram_padding_adds_only_the_end_marker(self):
        self.assertEqual(pad_sequence(["dogs"], 1), ["dogs", EOS])

    def test_length_grows_by_n(self):
        for n in range(1, 5):
            self.assertEqual(len(pad_sequence(["a", "b"], n)), 2 + n)

    def test_custom_markers(self):
        self.assertEqual(pad_sequence(["x"], 2, bos="[", eos="]"), ["[", "x", "]"])

    def test_empty_sequence_still_padded(self):
        self.assertEqual(pad_sequence([], 2), [BOS, EOS])

    def test_input_list_not_mutated(self):
        tokens = ["a"]
        pad_sequence(tokens, 3)
        self.assertEqual(tokens, ["a"])

    def test_padding_makes_first_word_a_target(self):
        # Unpadded, "dogs" is never the second element of a bigram, so a
        # model has no way to ask what starts a sentence.
        unpadded = ngrams(["dogs", "bark"], 2)
        padded = ngrams(pad_sequence(["dogs", "bark"], 2), 2)
        self.assertNotIn((BOS, "dogs"), unpadded)
        self.assertIn((BOS, "dogs"), padded)
        self.assertIn(("bark", EOS), padded)

    def test_type_error_on_str_input(self):
        with self.assertRaises(TypeError):
            pad_sequence("ab", 2)


class TestNgramFrequencies(unittest.TestCase):
    def test_counts_repeated_windows(self):
        self.assertEqual(
            ngram_frequencies(["a", "b", "a", "b"], 2),
            {("a", "b"): 2, ("b", "a"): 1},
        )

    def test_total_equals_number_of_windows(self):
        tokens = normalize(SAMPLE)
        freqs = ngram_frequencies(tokens, 3)
        self.assertEqual(sum(freqs.values()), len(ngrams(tokens, 3)))

    def test_padding_changes_the_totals(self):
        tokens = ["a", "b"]
        self.assertLess(
            sum(ngram_frequencies(tokens, 2).values()),
            sum(ngram_frequencies(tokens, 2, pad=True).values()),
        )

    def test_padded_table_contains_boundary_grams(self):
        freqs = ngram_frequencies(["a", "b"], 2, pad=True)
        self.assertIn((BOS, "a"), freqs)
        self.assertIn(("b", EOS), freqs)

    def test_empty_tokens(self):
        self.assertEqual(ngram_frequencies([], 2), {})


class TestTopNgrams(unittest.TestCase):
    def test_most_frequent_first(self):
        self.assertEqual(top_ngrams({("b", "c"): 1, ("a", "b"): 3}, 1), [(("a", "b"), 3)])

    def test_ties_break_on_the_ngram_itself(self):
        freqs = {("b", "b"): 2, ("a", "a"): 2}
        self.assertEqual([gram for gram, _ in top_ngrams(freqs, 2)], [("a", "a"), ("b", "b")])

    def test_k_larger_than_table(self):
        self.assertEqual(len(top_ngrams({("a", "b"): 1}, 99)), 1)

    def test_k_zero(self):
        self.assertEqual(top_ngrams({("a", "b"): 1}, 0), [])

    def test_negative_k_rejected(self):
        with self.assertRaises(ValueError):
            top_ngrams({("a", "b"): 1}, -1)


class TestCharNgrams(unittest.TestCase):
    def test_character_bigrams(self):
        self.assertEqual(char_ngrams("cat", 2), ["ca", "at"])

    def test_spaces_are_kept_by_default(self):
        self.assertEqual(char_ngrams("a b", 2), ["a ", " b"])

    def test_spaces_can_be_stripped(self):
        self.assertEqual(char_ngrams("a b", 2, strip_spaces=True), ["ab"])

    def test_window_longer_than_text(self):
        self.assertEqual(char_ngrams("ab", 5), [])

    def test_empty_text(self):
        self.assertEqual(char_ngrams("", 3), [])

    def test_korean_stem_recovered_across_particles(self):
        # The payoff: three distinct eojeol share one character trigram.
        text = "고양이가 고양이를 고양이에게"
        self.assertEqual(len(set(normalize(text))), 3)
        counts = word_frequencies(char_ngrams(text, 3, strip_spaces=True))
        self.assertEqual(counts["고양이"], 3)

    def test_decomposed_hangul_is_normalized_first(self):
        # Over NFD text the windows would slice syllables into jamo.
        composed = "한국어"
        decomposed = unicodedata.normalize("NFD", composed)
        self.assertEqual(char_ngrams(composed, 2), char_ngrams(decomposed, 2))

    def test_type_error_on_token_list(self):
        with self.assertRaises(TypeError):
            char_ngrams(["a", "b"], 2)

    def test_n_zero_rejected(self):
        with self.assertRaises(ValueError):
            char_ngrams("abc", 0)


class TestSparsity(unittest.TestCase):
    def test_half_seen_once(self):
        self.assertEqual(sparsity({("a", "b"): 1, ("b", "c"): 3}), 0.5)

    def test_all_seen_once(self):
        self.assertEqual(sparsity({("a",): 1, ("b",): 1}), 1.0)

    def test_none_seen_once(self):
        self.assertEqual(sparsity({("a",): 4}), 0.0)

    def test_empty_table_is_zero_not_a_division_error(self):
        self.assertEqual(sparsity({}), 0.0)

    def test_sparsity_climbs_with_n(self):
        # The case for smoothing (Day 10): longer windows repeat less, so
        # more of the table rests on a single observation.
        tokens = normalize(SAMPLE)
        measured = [sparsity(ngram_frequencies(tokens, n)) for n in (1, 2, 3, 4)]
        self.assertEqual(measured, sorted(measured))
        self.assertGreater(measured[-1], measured[0])

    def test_distinct_ngrams_grow_while_the_corpus_does_not(self):
        tokens = normalize(SAMPLE)
        self.assertGreater(
            len(ngram_frequencies(tokens, 3)),
            len(ngram_frequencies(tokens, 1)),
        )


if __name__ == "__main__":
    unittest.main()
