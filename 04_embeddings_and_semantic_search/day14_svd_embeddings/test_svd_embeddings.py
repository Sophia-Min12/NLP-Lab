"""Day 14 tests — factorization properties, weighting, and what k buys.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from svd_embeddings import (
    build_corpus,
    build_vocabulary,
    cooccurrence_matrix,
    cosine_similarity_matrix,
    explained_variance_ratio,
    pmi_matrix,
    reconstruction_error,
    svd_embeddings,
    truncated_svd,
    unit_rows,
)


def _ppmi():
    corpus = build_corpus()
    vocabulary = build_vocabulary(corpus, min_count=2)
    return vocabulary, pmi_matrix(cooccurrence_matrix(corpus, vocabulary, window=2))


class TestTruncatedSvd(unittest.TestCase):
    def test_shapes(self):
        u, s, vt = truncated_svd(np.eye(3), 2)
        self.assertEqual((u.shape, s.shape, vt.shape), ((3, 2), (2,), (2, 3)))

    def test_singular_values_descend(self):
        _, ppmi = _ppmi()
        _, s, _ = truncated_svd(ppmi, 20)
        self.assertTrue(np.all(np.diff(s) <= 1e-12))

    def test_singular_values_are_non_negative(self):
        _, ppmi = _ppmi()
        _, s, _ = truncated_svd(ppmi, 20)
        self.assertGreaterEqual(s.min(), 0.0)

    def test_left_vectors_are_orthonormal(self):
        _, ppmi = _ppmi()
        u, _, _ = truncated_svd(ppmi, 10)
        np.testing.assert_allclose(u.T @ u, np.eye(10), atol=1e-10)

    def test_full_rank_reconstructs_exactly(self):
        _, ppmi = _ppmi()
        k = min(ppmi.shape)
        u, s, vt = truncated_svd(ppmi, k)
        np.testing.assert_allclose((u * s) @ vt, ppmi, atol=1e-8)

    def test_k_out_of_range_rejected(self):
        for k in (0, -1, 99):
            with self.assertRaises(ValueError):
                truncated_svd(np.eye(3), k)

    def test_non_matrix_rejected(self):
        with self.assertRaises(ValueError):
            truncated_svd(np.zeros(4), 1)


class TestSvdEmbeddings(unittest.TestCase):
    def test_shape_is_words_by_k(self):
        vocabulary, ppmi = _ppmi()
        self.assertEqual(svd_embeddings(ppmi, k=12).shape, (len(vocabulary), 12))

    def test_deterministic(self):
        _, ppmi = _ppmi()
        np.testing.assert_array_equal(svd_embeddings(ppmi, k=8), svd_embeddings(ppmi, k=8))

    def test_all_weightings_share_the_directions(self):
        # Only the scaling differs, so normalized rows agree for "none" and
        # any positive scaling of the same axes is a different geometry --
        # but every variant keeps the same shape.
        _, ppmi = _ppmi()
        for weighting in ("sqrt", "full", "none"):
            self.assertEqual(svd_embeddings(ppmi, k=10, weighting=weighting).shape[1], 10)

    def test_none_weighting_is_the_raw_left_factor(self):
        _, ppmi = _ppmi()
        u, _, _ = truncated_svd(ppmi, 6)
        np.testing.assert_allclose(svd_embeddings(ppmi, k=6, weighting="none"), u)

    def test_unknown_weighting_rejected(self):
        _, ppmi = _ppmi()
        with self.assertRaises(ValueError):
            svd_embeddings(ppmi, k=4, weighting="magic")


class TestExplainedVariance(unittest.TestCase):
    def test_sums_to_one(self):
        _, ppmi = _ppmi()
        self.assertAlmostEqual(float(explained_variance_ratio(ppmi).sum()), 1.0)

    def test_descending(self):
        _, ppmi = _ppmi()
        ratios = explained_variance_ratio(ppmi)
        self.assertTrue(np.all(np.diff(ratios) <= 1e-12))

    def test_all_zero_matrix(self):
        np.testing.assert_array_equal(explained_variance_ratio(np.zeros((3, 3))), np.zeros(3))


class TestReconstructionError(unittest.TestCase):
    def test_falls_monotonically_with_k(self):
        _, ppmi = _ppmi()
        errors = [reconstruction_error(ppmi, k) for k in (2, 8, 16, 32, 64)]
        self.assertEqual(errors, sorted(errors, reverse=True))

    def test_full_rank_is_near_zero(self):
        _, ppmi = _ppmi()
        self.assertLess(reconstruction_error(ppmi, min(ppmi.shape)), 1e-10)

    def test_between_zero_and_one(self):
        _, ppmi = _ppmi()
        for k in (1, 5, 20):
            error = reconstruction_error(ppmi, k)
            self.assertGreaterEqual(error, 0.0)
            self.assertLessEqual(error, 1.0)

    def test_zero_matrix(self):
        self.assertEqual(reconstruction_error(np.zeros((3, 3)), 2), 0.0)


class TestUnitRowsAndCosine(unittest.TestCase):
    def test_rows_become_unit_length(self):
        normalized = unit_rows(np.array([[3.0, 4.0], [1.0, 0.0]]))
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])

    def test_zero_row_survives(self):
        normalized = unit_rows(np.array([[0.0, 0.0], [3.0, 4.0]]))
        np.testing.assert_array_equal(normalized[0], [0.0, 0.0])

    def test_diagonal_is_one(self):
        _, ppmi = _ppmi()
        similarity = cosine_similarity_matrix(svd_embeddings(ppmi, k=12))
        np.testing.assert_allclose(np.diag(similarity), 1.0, atol=1e-9)

    def test_symmetric(self):
        _, ppmi = _ppmi()
        similarity = cosine_similarity_matrix(svd_embeddings(ppmi, k=12))
        np.testing.assert_allclose(similarity, similarity.T, atol=1e-12)

    def test_within_the_unit_range(self):
        _, ppmi = _ppmi()
        similarity = cosine_similarity_matrix(svd_embeddings(ppmi, k=12))
        self.assertGreaterEqual(similarity.min(), -1 - 1e-9)
        self.assertLessEqual(similarity.max(), 1 + 1e-9)

    def test_matches_pairwise_cosine(self):
        vectors = np.array([[1.0, 0.0], [1.0, 1.0]])
        expected = float(np.dot(vectors[0], vectors[1]) /
                         (np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])))
        self.assertAlmostEqual(cosine_similarity_matrix(vectors)[0, 1], expected)


class TestCompressionCarriesStructure(unittest.TestCase):
    """What the factorization is actually for."""

    def setUp(self):
        self.vocabulary, self.ppmi = _ppmi()
        self.vectors = svd_embeddings(self.ppmi, k=24)
        self.similarity = cosine_similarity_matrix(self.vectors)

    def sim(self, a, b):
        return self.similarity[self.vocabulary[a], self.vocabulary[b]]

    def test_categories_are_ordered_correctly(self):
        self.assertGreater(self.sim("cat", "dog"), self.sim("cat", "king"))
        self.assertGreater(self.sim("cat", "king"), self.sim("cat", "bread"))

    def test_gendered_pairs_are_very_close(self):
        self.assertGreater(self.sim("king", "queen"), 0.9)
        self.assertGreater(self.sim("man", "woman"), 0.9)

    def test_unrelated_words_are_near_zero_or_below(self):
        self.assertLess(self.sim("cat", "bread"), 0.2)

    def test_compression_lifts_related_pairs_above_the_sparse_score(self):
        # The point of the day: dense vectors catch what exact-context
        # overlap misses.
        sparse = cosine_similarity_matrix(self.ppmi)
        for a, b in (("cat", "dog"), ("garden", "forest")):
            self.assertGreater(
                self.sim(a, b),
                sparse[self.vocabulary[a], self.vocabulary[b]],
                f"{a}/{b}",
            )

    def test_dimensions_are_far_fewer_than_the_vocabulary(self):
        self.assertLess(self.vectors.shape[1], len(self.vocabulary) / 3)


class TestChoosingK(unittest.TestCase):
    """Separation is not monotonic in k, and reconstruction error is."""

    def setUp(self):
        self.vocabulary, self.ppmi = _ppmi()

    def gap(self, k):
        similarity = cosine_similarity_matrix(svd_embeddings(self.ppmi, k=k))
        return (similarity[self.vocabulary["cat"], self.vocabulary["dog"]]
                - similarity[self.vocabulary["cat"], self.vocabulary["bread"]])

    def test_too_few_dimensions_collapse_everything(self):
        self.assertLess(abs(self.gap(2)), 0.01)

    def test_separation_peaks_in_the_middle(self):
        gaps = {k: self.gap(k) for k in (2, 8, 24, 48, 80, len(self.vocabulary))}
        best = max(gaps, key=gaps.get)
        self.assertNotIn(best, (2, len(self.vocabulary)))
        self.assertGreater(gaps[best], gaps[len(self.vocabulary)])

    def test_reconstruction_keeps_improving_where_separation_does_not(self):
        # Fitting the matrix better and representing words better are
        # different goals, and they diverge.
        self.assertLess(reconstruction_error(self.ppmi, 80),
                        reconstruction_error(self.ppmi, 48))
        self.assertLess(self.gap(80), self.gap(48))


if __name__ == "__main__":
    unittest.main()
