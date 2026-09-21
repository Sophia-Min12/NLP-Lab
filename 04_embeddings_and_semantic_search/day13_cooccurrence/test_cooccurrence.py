"""Day 13 tests — windows, PMI arithmetic, and the demotion of function words.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from cooccurrence import (
    build_corpus,
    build_vocabulary,
    cooccurrence_matrix,
    normalize,
    pmi_matrix,
    sparsity,
    top_associations,
)


def _built(window=2, min_count=2):
    corpus = build_corpus()
    vocabulary = build_vocabulary(corpus, min_count=min_count)
    counts = cooccurrence_matrix(corpus, vocabulary, window=window)
    return corpus, vocabulary, counts, pmi_matrix(counts)


class TestBuildVocabulary(unittest.TestCase):
    def test_min_count_filters(self):
        self.assertEqual(build_vocabulary([["b", "a"], ["a"]], min_count=2), {"a": 0})

    def test_alphabetical_and_contiguous(self):
        vocabulary = build_vocabulary([["c", "a", "b"]])
        self.assertEqual(vocabulary, {"a": 0, "b": 1, "c": 2})

    def test_min_count_below_one_rejected(self):
        with self.assertRaises(ValueError):
            build_vocabulary([["a"]], min_count=0)

    def test_rejects_a_str_document(self):
        with self.assertRaises(TypeError):
            build_vocabulary(["a b"])


class TestCooccurrenceMatrix(unittest.TestCase):
    def test_adjacent_pair(self):
        matrix = cooccurrence_matrix([["a", "b"]], {"a": 0, "b": 1}, window=1)
        np.testing.assert_array_equal(matrix, [[0.0, 1.0], [1.0, 0.0]])

    def test_symmetric_by_construction(self):
        _, _, counts, _ = _built()
        np.testing.assert_allclose(counts, counts.T)

    def test_diagonal_is_zero(self):
        # A word is not its own context.
        _, _, counts, _ = _built()
        np.testing.assert_array_equal(np.diag(counts), np.zeros(counts.shape[0]))

    def test_window_does_not_cross_sentences(self):
        vocabulary = {"a": 0, "b": 1}
        matrix = cooccurrence_matrix([["a"], ["b"]], vocabulary, window=5)
        self.assertEqual(matrix[0, 1], 0.0)

    def test_wider_window_never_counts_less(self):
        _, vocabulary, narrow, _ = _built(window=1)
        corpus = build_corpus()
        wide = cooccurrence_matrix(corpus, vocabulary, window=5)
        self.assertTrue(np.all(wide >= narrow))

    def test_out_of_vocabulary_tokens_are_skipped(self):
        matrix = cooccurrence_matrix([["a", "zzz", "b"]], {"a": 0, "b": 1}, window=1)
        # "zzz" occupies a position but contributes no counts.
        self.assertEqual(matrix[0, 1], 0.0)

    def test_shape_matches_the_vocabulary(self):
        _, vocabulary, counts, _ = _built()
        self.assertEqual(counts.shape, (len(vocabulary), len(vocabulary)))

    def test_window_below_one_rejected(self):
        with self.assertRaises(ValueError):
            cooccurrence_matrix([["a"]], {"a": 0}, window=0)

    def test_rejects_a_str_sentence(self):
        with self.assertRaises(TypeError):
            cooccurrence_matrix(["a b"], {"a": 0})


class TestPmiMatrix(unittest.TestCase):
    def test_worked_example(self):
        counts = np.array([[0.0, 2.0], [2.0, 0.0]])
        # joint = 0.5 each off-diagonal; marginals 0.5; ratio 2 -> log 2
        np.testing.assert_allclose(pmi_matrix(counts), [[0.0, np.log(2)], [np.log(2), 0.0]])

    def test_ppmi_is_non_negative(self):
        _, _, _, ppmi = _built()
        self.assertGreaterEqual(ppmi.min(), 0.0)

    def test_signed_pmi_can_be_negative(self):
        _, _, counts, _ = _built()
        signed = pmi_matrix(counts, positive=False)
        self.assertLess(signed.min(), 0.0)

    def test_ppmi_is_the_clipped_signed_version(self):
        _, _, counts, ppmi = _built()
        signed = pmi_matrix(counts, positive=False)
        np.testing.assert_allclose(ppmi, np.maximum(signed, 0.0))

    def test_stays_finite_where_pairs_never_cooccur(self):
        _, _, counts, _ = _built()
        for positive in (True, False):
            matrix = pmi_matrix(counts, positive=positive)
            self.assertTrue(np.all(np.isfinite(matrix)))

    def test_symmetry_is_preserved(self):
        _, _, _, ppmi = _built()
        np.testing.assert_allclose(ppmi, ppmi.T)

    def test_clipping_increases_sparsity(self):
        _, _, counts, ppmi = _built()
        self.assertGreaterEqual(sparsity(ppmi), sparsity(counts))

    def test_all_zero_matrix(self):
        np.testing.assert_array_equal(pmi_matrix(np.zeros((2, 2))), np.zeros((2, 2)))

    def test_non_square_rejected(self):
        with self.assertRaises(ValueError):
            pmi_matrix(np.zeros((2, 3)))

    def test_negative_epsilon_rejected(self):
        with self.assertRaises(ValueError):
            pmi_matrix(np.zeros((2, 2)), epsilon=-1.0)


class TestFunctionWordsDemoteThemselves(unittest.TestCase):
    """Day 2's stopword list and Day 6's IDF, falling out of counting."""

    def setUp(self):
        _, self.vocabulary, self.counts, self.ppmi = _built()

    def test_the_has_the_largest_raw_row(self):
        totals = self.counts.sum(axis=1)
        self.assertEqual(int(np.argmax(totals)), self.vocabulary["the"])

    def test_the_has_the_most_partners(self):
        partners = np.count_nonzero(self.counts, axis=1)
        self.assertEqual(int(np.argmax(partners)), self.vocabulary["the"])

    def test_but_its_strongest_association_is_weak(self):
        strength = self.ppmi.max(axis=1)
        the = strength[self.vocabulary["the"]]
        for word in ("cat", "king", "fish", "ate"):
            self.assertLess(the, strength[self.vocabulary[word]], word)

    def test_the_does_not_appear_in_ppmi_top_associations(self):
        for word in ("cat", "king", "ate"):
            top = [t for t, _ in top_associations(self.ppmi, self.vocabulary, word, 4)]
            self.assertNotIn("the", top, word)

    def test_the_does_dominate_raw_count_associations(self):
        for word in ("cat", "king", "ate"):
            top = [t for t, _ in top_associations(self.counts, self.vocabulary, word, 1)]
            self.assertEqual(top, ["the"], word)


class TestDistributionalSimilarity(unittest.TestCase):
    """Words that never co-occur can still share contexts — the Day 7 gap."""

    def setUp(self):
        _, self.vocabulary, self.counts, self.ppmi = _built()

    def test_cat_and_dog_never_cooccur(self):
        self.assertEqual(self.counts[self.vocabulary["cat"], self.vocabulary["dog"]], 0.0)

    def test_yet_they_share_many_contexts(self):
        cat = self.ppmi[self.vocabulary["cat"]] > 0
        dog = self.ppmi[self.vocabulary["dog"]] > 0
        self.assertGreater(int(np.count_nonzero(cat & dog)), 4)

    def test_same_category_shares_more_than_across_categories(self):
        def shared(a, b):
            return int(np.count_nonzero(
                (self.ppmi[self.vocabulary[a]] > 0) & (self.ppmi[self.vocabulary[b]] > 0)
            ))

        self.assertGreater(shared("cat", "dog"), shared("cat", "king"))
        self.assertGreater(shared("king", "queen"), shared("king", "cat"))


class TestTopAssociations(unittest.TestCase):
    def setUp(self):
        _, self.vocabulary, self.counts, self.ppmi = _built()

    def test_returns_k_items(self):
        self.assertEqual(len(top_associations(self.ppmi, self.vocabulary, "cat", 3)), 3)

    def test_excludes_the_word_itself(self):
        picked = [t for t, _ in top_associations(self.ppmi, self.vocabulary, "cat", 10)]
        self.assertNotIn("cat", picked)

    def test_sorted_descending(self):
        values = [v for _, v in top_associations(self.ppmi, self.vocabulary, "king", 5)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_unknown_word_rejected(self):
        with self.assertRaises(KeyError):
            top_associations(self.ppmi, self.vocabulary, "zzz")

    def test_negative_k_rejected(self):
        with self.assertRaises(ValueError):
            top_associations(self.ppmi, self.vocabulary, "cat", -1)


class TestSparsity(unittest.TestCase):
    def test_counts_zeros(self):
        self.assertEqual(sparsity(np.array([[0.0, 1.0], [0.0, 0.0]])), 0.75)

    def test_dense(self):
        self.assertEqual(sparsity(np.ones((2, 2))), 0.0)

    def test_empty(self):
        self.assertEqual(sparsity(np.zeros((0, 0))), 0.0)


if __name__ == "__main__":
    unittest.main()
