"""Day 16 tests — normalization, fusion, and whether the hybrid earns its keep.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from hybrid import (
    DOCUMENTS,
    EVALUATION,
    EXACT_QUERIES,
    GAP_QUERIES,
    HybridSearcher,
    build_corpus,
    min_max,
    recall_at_k,
    reciprocal_rank_fusion,
)

_CORPUS = build_corpus()


def _searcher():
    return HybridSearcher(DOCUMENTS, embedding_corpus=_CORPUS)


class TestMinMax(unittest.TestCase):
    def test_rescales_to_unit_range(self):
        np.testing.assert_allclose(min_max(np.array([1.0, 3.0, 2.0])), [0.0, 1.0, 0.5])

    def test_all_equal_becomes_zeros(self):
        # A signal that cannot distinguish anything must contribute nothing.
        np.testing.assert_array_equal(min_max(np.array([2.0, 2.0])), [0.0, 0.0])

    def test_preserves_order(self):
        source = np.array([5.0, -1.0, 3.0])
        self.assertEqual(list(np.argsort(min_max(source))), list(np.argsort(source)))

    def test_handles_negatives(self):
        np.testing.assert_allclose(min_max(np.array([-2.0, 0.0])), [0.0, 1.0])

    def test_empty(self):
        self.assertEqual(min_max(np.array([])).size, 0)


class TestReciprocalRankFusion(unittest.TestCase):
    def test_symmetric_rankings_tie(self):
        scores = reciprocal_rank_fusion([[1, 2], [2, 1]], k=1.0)
        self.assertAlmostEqual(scores[1], scores[2])

    def test_agreed_top_wins(self):
        scores = reciprocal_rank_fusion([[1, 2], [1, 2]])
        self.assertGreater(scores[1], scores[2])

    def test_missing_from_one_ranking_still_scores(self):
        scores = reciprocal_rank_fusion([[1], [2]])
        self.assertIn(2, scores)

    def test_larger_k_flattens_the_difference(self):
        sharp = reciprocal_rank_fusion([[1, 2]], k=1.0)
        flat = reciprocal_rank_fusion([[1, 2]], k=1000.0)
        self.assertGreater(sharp[1] - sharp[2], flat[1] - flat[2])

    def test_non_positive_k_rejected(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([[1]], k=0.0)


class TestHybridSearcher(unittest.TestCase):
    def setUp(self):
        self.searcher = _searcher()

    def test_length(self):
        self.assertEqual(len(self.searcher), len(DOCUMENTS))

    def test_document_vectors_are_unit_length_or_zero(self):
        for matrix in (self.searcher.lexical_vectors, self.searcher.semantic_vectors):
            norms = np.linalg.norm(matrix, axis=1)
            for norm in norms:
                self.assertTrue(abs(norm - 1.0) < 1e-9 or norm == 0.0)

    def test_exact_match_ranks_first_under_lexical(self):
        self.assertEqual(self.searcher.search("windowsill", alpha=1.0, k=1)[0][0], 0)

    def test_unknown_query_returns_something_without_crashing(self):
        results = self.searcher.search("zzzqqq", alpha=0.5, k=3)
        self.assertEqual(len(results), 3)

    def test_empty_query(self):
        self.assertEqual(len(self.searcher.search("", alpha=0.5, k=3)), 3)

    def test_k_truncates(self):
        self.assertEqual(len(self.searcher.search("cat", k=2)), 2)

    def test_k_zero(self):
        self.assertEqual(self.searcher.search("cat", k=0), [])

    def test_alpha_out_of_range_rejected(self):
        for alpha in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                self.searcher.search("cat", alpha=alpha)

    def test_unknown_fusion_rejected(self):
        with self.assertRaises(ValueError):
            self.searcher.search("cat", fusion="magic")

    def test_rejects_a_token_list_as_query(self):
        with self.assertRaises(TypeError):
            self.searcher.search(["cat"])

    def test_rejects_a_bare_str_collection(self):
        with self.assertRaises(TypeError):
            HybridSearcher("one document")

    def test_empty_collection_rejected(self):
        with self.assertRaises(ValueError):
            HybridSearcher([])

    def test_deterministic(self):
        self.assertEqual(self.searcher.search("queen"), _searcher().search("queen"))


class TestTheTwoSignals(unittest.TestCase):
    def setUp(self):
        self.searcher = _searcher()

    def test_lexical_is_zero_when_no_word_coincides(self):
        # The Day 7 wall, still there.
        self.assertEqual(self.searcher.explain("queen", 6)["lexical"], 0.0)

    def test_semantic_crosses_that_gap(self):
        self.assertGreater(self.searcher.explain("queen", 6)["semantic"], 0.2)

    def test_the_signals_occupy_different_ranges(self):
        lexical = self.searcher.lexical_scores("queen")
        semantic = self.searcher.semantic_scores("queen")
        self.assertNotAlmostEqual(lexical.max(), semantic.max(), places=2)

    def test_explain_reports_both(self):
        report = self.searcher.explain("cat", 0)
        self.assertEqual(sorted(report), ["lexical", "semantic"])

    def test_explain_rejects_a_bad_document_id(self):
        with self.assertRaises(IndexError):
            self.searcher.explain("cat", 999)


class TestHybridBeatsBothExtremes(unittest.TestCase):
    """The claim the day exists to support."""

    def setUp(self):
        self.searcher = _searcher()

    def test_lexical_alone_fails_the_gap_queries(self):
        self.assertLess(recall_at_k(self.searcher, GAP_QUERIES, alpha=1.0, k=3), 0.5)

    def test_lexical_alone_is_perfect_on_exact_queries(self):
        self.assertEqual(recall_at_k(self.searcher, EXACT_QUERIES, alpha=1.0, k=3), 1.0)

    def test_semantic_alone_handles_the_gap_queries(self):
        self.assertEqual(recall_at_k(self.searcher, GAP_QUERIES, alpha=0.0, k=3), 1.0)

    def test_semantic_alone_slips_on_exact_queries(self):
        self.assertLess(recall_at_k(self.searcher, EXACT_QUERIES, alpha=0.0, k=3), 1.0)

    def test_a_blend_beats_both_extremes_overall(self):
        blended = recall_at_k(self.searcher, EVALUATION, alpha=0.5, k=3)
        lexical = recall_at_k(self.searcher, EVALUATION, alpha=1.0, k=3)
        semantic = recall_at_k(self.searcher, EVALUATION, alpha=0.0, k=3)
        self.assertGreater(blended, lexical)
        self.assertGreater(blended, semantic)

    def test_the_result_is_not_knife_edge(self):
        # If only one alpha worked, the result would be a tuning artefact.
        for alpha in (0.2, 0.4, 0.5, 0.6, 0.8):
            self.assertEqual(recall_at_k(self.searcher, EVALUATION, alpha=alpha, k=3), 1.0)

    def test_empty_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            recall_at_k(self.searcher, [], alpha=0.5)


class TestPretrainingIsNecessary(unittest.TestCase):
    """Twelve short documents cannot support their own word vectors."""

    def test_documents_alone_lose_rare_query_words(self):
        naive = HybridSearcher(DOCUMENTS)
        self.assertNotIn("queen", naive.embedding_vocabulary)

    def test_and_therefore_retrieve_worse(self):
        naive = HybridSearcher(DOCUMENTS)
        pretrained = _searcher()
        self.assertGreater(
            recall_at_k(pretrained, EVALUATION, alpha=0.5, k=3),
            recall_at_k(naive, EVALUATION, alpha=0.5, k=3),
        )

    def test_pretrained_vocabulary_covers_the_queries(self):
        searcher = _searcher()
        for query, _ in EVALUATION:
            if query in {"windowsill", "cart", "stick", "throne", "oven", "salty"}:
                continue  # lexical-only words, not required to have vectors
            self.assertIn(query, searcher.embedding_vocabulary, query)


class TestFusionStrategies(unittest.TestCase):
    def setUp(self):
        self.searcher = _searcher()

    def test_rrf_runs_and_returns_a_ranking(self):
        results = self.searcher.search("queen", fusion="rrf", k=3)
        self.assertEqual(len(results), 3)

    def test_rrf_ignores_alpha(self):
        self.assertEqual(
            self.searcher.search("queen", alpha=1.0, fusion="rrf"),
            self.searcher.search("queen", alpha=0.0, fusion="rrf"),
        )

    def test_linear_beats_rrf_here(self):
        # Discarding score magnitudes costs real information when one
        # signal returns mostly zeros.
        linear = recall_at_k(self.searcher, EVALUATION, alpha=0.5, k=3)
        rrf_hits = sum(
            bool({i for i, _ in self.searcher.search(q, fusion="rrf", k=3)} & relevant)
            for q, relevant in EVALUATION
        )
        self.assertGreater(linear, rrf_hits / len(EVALUATION))


if __name__ == "__main__":
    unittest.main()
