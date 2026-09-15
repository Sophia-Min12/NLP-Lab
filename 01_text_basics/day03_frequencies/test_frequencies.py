"""Day 3 tests — counting, ranking, and the Zipf measurements.

Korean strings below are functional test data (counting is script-agnostic,
but eojeol-level counting is not), not documentation. Runnable via `pytest`
(repo root) or `python -m unittest` (this folder).
"""

import unittest

from frequencies import (
    SAMPLE,
    coverage,
    hapax_legomena,
    normalize,
    rank_words,
    top_n,
    type_token_ratio,
    word_frequencies,
    zipf_exponent,
    zipf_table,
)


class TestWordFrequencies(unittest.TestCase):
    def test_counts_repeated_tokens(self):
        self.assertEqual(word_frequencies(["the", "cat", "the"]), {"the": 2, "cat": 1})

    def test_insertion_order_is_first_appearance(self):
        freqs = word_frequencies(["b", "a", "b", "c"])
        self.assertEqual(list(freqs), ["b", "a", "c"])

    def test_empty_token_list(self):
        self.assertEqual(word_frequencies([]), {})

    def test_total_count_equals_corpus_size(self):
        tokens = normalize(SAMPLE)
        self.assertEqual(sum(word_frequencies(tokens).values()), len(tokens))

    def test_korean_tokens_counted(self):
        freqs = word_frequencies(["고양이가", "개가", "고양이가"])
        self.assertEqual(freqs["고양이가"], 2)

    def test_type_error_on_str_input(self):
        # "hello" would otherwise be counted character by character.
        with self.assertRaises(TypeError):
            word_frequencies("hello")


class TestRankWords(unittest.TestCase):
    def test_sorted_by_count_descending(self):
        self.assertEqual(rank_words({"a": 1, "b": 5}), [("b", 5), ("a", 1)])

    def test_ties_break_alphabetically(self):
        self.assertEqual(rank_words({"b": 1, "a": 1, "c": 3}), [("c", 3), ("a", 1), ("b", 1)])

    def test_tie_order_is_independent_of_insertion_order(self):
        self.assertEqual(rank_words({"z": 2, "y": 2}), rank_words({"y": 2, "z": 2}))

    def test_empty_table(self):
        self.assertEqual(rank_words({}), [])


class TestTopN(unittest.TestCase):
    def test_returns_n_most_frequent(self):
        self.assertEqual(top_n({"a": 5, "b": 2, "c": 9}, 2), [("c", 9), ("a", 5)])

    def test_n_larger_than_vocabulary(self):
        self.assertEqual(len(top_n({"a": 1}, 10)), 1)

    def test_n_zero(self):
        self.assertEqual(top_n({"a": 1}, 0), [])

    def test_negative_n_rejected(self):
        with self.assertRaises(ValueError):
            top_n({"a": 1}, -1)


class TestHapaxLegomena(unittest.TestCase):
    def test_only_single_occurrence_words(self):
        self.assertEqual(hapax_legomena({"the": 9, "zebra": 1, "aardvark": 1}), ["aardvark", "zebra"])

    def test_none_when_everything_repeats(self):
        self.assertEqual(hapax_legomena({"a": 2, "b": 3}), [])

    def test_empty_table(self):
        self.assertEqual(hapax_legomena({}), [])

    def test_hapax_are_a_large_share_of_the_vocabulary(self):
        # The long flat tail: even in this tiny sample most types occur once.
        freqs = word_frequencies(normalize(SAMPLE))
        self.assertGreater(len(hapax_legomena(freqs)) / len(freqs), 0.4)


class TestTypeTokenRatio(unittest.TestCase):
    def test_basic_ratio(self):
        self.assertEqual(type_token_ratio(["a", "b", "a", "c"]), 0.75)

    def test_all_distinct_is_one(self):
        self.assertEqual(type_token_ratio(["a", "b"]), 1.0)

    def test_empty_is_zero_not_an_error(self):
        self.assertEqual(type_token_ratio([]), 0.0)

    def test_ratio_falls_as_text_repeats(self):
        short = ["a", "b"]
        longer = ["a", "b", "a", "b", "a", "b"]
        self.assertLess(type_token_ratio(longer), type_token_ratio(short))

    def test_type_error_on_str_input(self):
        with self.assertRaises(TypeError):
            type_token_ratio("abc")


class TestCoverage(unittest.TestCase):
    def test_single_type_share(self):
        self.assertAlmostEqual(coverage({"the": 7, "cat": 2, "sat": 1}, 1), 0.7)

    def test_full_vocabulary_covers_everything(self):
        freqs = {"a": 3, "b": 1}
        self.assertAlmostEqual(coverage(freqs, len(freqs)), 1.0)

    def test_zero_types_cover_nothing(self):
        self.assertEqual(coverage({"a": 3}, 0), 0.0)

    def test_empty_table_is_zero_not_a_division_error(self):
        self.assertEqual(coverage({}, 5), 0.0)

    def test_coverage_is_monotonic(self):
        freqs = word_frequencies(normalize(SAMPLE))
        shares = [coverage(freqs, k) for k in range(0, 10)]
        self.assertEqual(shares, sorted(shares))

    def test_negative_k_rejected(self):
        with self.assertRaises(ValueError):
            coverage({"a": 1}, -1)


class TestZipfTable(unittest.TestCase):
    def test_rank_one_prediction_matches_exactly(self):
        rows = zipf_table({"a": 10, "b": 6, "c": 2}, 3)
        rank, word, observed, predicted = rows[0]
        self.assertEqual((rank, word, observed), (1, "a", 10))
        self.assertAlmostEqual(predicted, 10.0)

    def test_prediction_is_top_count_over_rank(self):
        rows = zipf_table({"a": 10, "b": 6, "c": 2}, 3)
        self.assertAlmostEqual(rows[1][3], 5.0)
        self.assertAlmostEqual(rows[2][3], 10 / 3)

    def test_ranks_are_consecutive_from_one(self):
        rows = zipf_table(word_frequencies(normalize(SAMPLE)), 5)
        self.assertEqual([row[0] for row in rows], [1, 2, 3, 4, 5])

    def test_empty_table(self):
        self.assertEqual(zipf_table({}, 5), [])


class TestZipfExponent(unittest.TestCase):
    def test_perfect_zipf_gives_exponent_one(self):
        # counts exactly C / rank
        freqs = {chr(ord("a") + i): 1200 // (i + 1) for i in range(12)}
        self.assertAlmostEqual(zipf_exponent(freqs), 1.0, places=2)

    def test_flat_distribution_gives_exponent_near_zero(self):
        freqs = {"a": 5, "b": 5, "c": 5, "d": 5}
        self.assertAlmostEqual(zipf_exponent(freqs), 0.0, places=6)

    def test_steeper_falloff_gives_larger_exponent(self):
        gentle = {chr(ord("a") + i): 1200 // (i + 1) for i in range(12)}
        steep = {chr(ord("a") + i): 1200 // (i + 1) ** 2 for i in range(12)}
        self.assertGreater(zipf_exponent(steep), zipf_exponent(gentle))

    def test_single_word_cannot_be_fitted(self):
        with self.assertRaises(ValueError):
            zipf_exponent({"a": 1})

    def test_empty_table_cannot_be_fitted(self):
        with self.assertRaises(ValueError):
            zipf_exponent({})

    def test_hapax_tail_drags_the_full_range_fit_down(self):
        head = {f"w{i}": max(1, 10000 // (i + 1)) for i in range(200)}
        with_tail = {**head, **{f"t{j}": 1 for j in range(3000)}}

        self.assertAlmostEqual(zipf_exponent(head), 1.0, places=2)
        # The flat log(1) = 0 region flattens the slope...
        self.assertLess(zipf_exponent(with_tail), zipf_exponent(head) - 0.05)
        # ...and capping the fit at the head recovers it.
        self.assertAlmostEqual(zipf_exponent(with_tail, max_rank=200), zipf_exponent(head))

    def test_max_rank_larger_than_vocabulary_is_a_full_fit(self):
        freqs = {"a": 9, "b": 4, "c": 1}
        self.assertEqual(zipf_exponent(freqs, max_rank=999), zipf_exponent(freqs))

    def test_max_rank_below_two_rejected(self):
        with self.assertRaises(ValueError):
            zipf_exponent({"a": 9, "b": 4}, max_rank=1)


class TestNormalizeReusedFromDay02(unittest.TestCase):
    def test_case_and_punctuation_folded(self):
        self.assertEqual(normalize("The cat, THE Cat!"), ["the", "cat", "the", "cat"])

    def test_stopwords_deliberately_kept(self):
        # Day 3 needs the head of the curve, so "the" must survive.
        self.assertIn("the", normalize("the cat"))

    def test_type_error_on_bytes(self):
        with self.assertRaises(TypeError):
            normalize(b"bytes")


class TestZipfOnTheSample(unittest.TestCase):
    """The law should be visible even in the small bundled sample."""

    def setUp(self):
        self.tokens = normalize(SAMPLE)
        self.freqs = word_frequencies(self.tokens)

    def test_the_is_the_most_frequent_english_word(self):
        self.assertEqual(top_n(self.freqs, 1)[0][0], "the")

    def test_top_ten_types_cover_a_large_share_of_tokens(self):
        self.assertGreater(coverage(self.freqs, 10), 0.3)

    def test_vocabulary_is_smaller_than_the_corpus(self):
        self.assertLess(len(self.freqs), len(self.tokens))

    def test_korean_particles_split_the_counts(self):
        # 고양이가 and 고양이를 are the same noun with different particles,
        # so eojeol counting scatters what should be one entry. Unsolved
        # until BPE on Day 12.
        self.assertIn("고양이가", self.freqs)
        self.assertIn("고양이를", self.freqs)
        self.assertNotIn("고양이", self.freqs)


if __name__ == "__main__":
    unittest.main()
