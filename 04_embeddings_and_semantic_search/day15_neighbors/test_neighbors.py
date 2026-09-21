"""Day 15 tests — neighbour search, analogy arithmetic, and the baselines.

Runnable via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import unittest

import numpy as np

from neighbors import (
    ANALOGY_CASES,
    WordVectors,
    evaluate_analogies,
    load_vectors,
)


def _toy():
    vocabulary = {"man": 0, "woman": 1, "king": 2, "queen": 3, "rock": 4}
    vectors = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, -1.0],
    ])
    return WordVectors(vocabulary, vectors)


class TestWordVectors(unittest.TestCase):
    def setUp(self):
        self.vectors = _toy()

    def test_length_and_membership(self):
        self.assertEqual(len(self.vectors), 5)
        self.assertIn("king", self.vectors)
        self.assertNotIn("zebra", self.vectors)

    def test_rows_are_unit_length(self):
        np.testing.assert_allclose(np.linalg.norm(self.vectors.vectors, axis=1), 1.0)

    def test_terms_follow_the_index_order(self):
        for term, index in self.vectors.vocabulary.items():
            self.assertEqual(self.vectors.terms[index], term)

    def test_self_similarity_is_one(self):
        self.assertAlmostEqual(self.vectors.similarity("king", "king"), 1.0)

    def test_similarity_is_symmetric(self):
        self.assertAlmostEqual(
            self.vectors.similarity("king", "queen"),
            self.vectors.similarity("queen", "king"),
        )

    def test_orthogonal_words_score_zero(self):
        self.assertAlmostEqual(self.vectors.similarity("man", "woman"), 0.0)

    def test_unknown_word_rejected(self):
        with self.assertRaises(KeyError):
            self.vectors.vector("zebra")

    def test_mismatched_sizes_rejected(self):
        with self.assertRaises(ValueError):
            WordVectors({"a": 0, "b": 1}, np.array([[1.0, 0.0]]))

    def test_non_matrix_rejected(self):
        with self.assertRaises(ValueError):
            WordVectors({"a": 0}, np.array([1.0, 0.0]))


class TestMostSimilar(unittest.TestCase):
    def setUp(self):
        self.vectors = _toy()

    def test_excludes_the_query_word(self):
        self.assertNotIn("king", [t for t, _ in self.vectors.most_similar("king", 4)])

    def test_returns_k_items(self):
        self.assertEqual(len(self.vectors.most_similar("king", 3)), 3)

    def test_k_zero(self):
        self.assertEqual(self.vectors.most_similar("king", 0), [])

    def test_negative_k_rejected(self):
        with self.assertRaises(ValueError):
            self.vectors.most_similar("king", -1)

    def test_scores_descend(self):
        scores = [s for _, s in self.vectors.most_similar("king", 4)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_opposite_direction_ranks_last(self):
        ranked = [t for t, _ in self.vectors.most_similar("king", 4)]
        self.assertEqual(ranked[-1], "rock")


class TestAnalogy(unittest.TestCase):
    def setUp(self):
        self.vectors = _toy()

    def test_the_toy_analogy_resolves(self):
        self.assertEqual(self.vectors.analogy("man", "king", "woman", k=1)[0][0], "queen")

    def test_inputs_are_excluded_by_default(self):
        picked = [t for t, _ in self.vectors.analogy("man", "king", "woman", k=4)]
        for word in ("man", "king", "woman"):
            self.assertNotIn(word, picked)

    def test_inputs_reappear_when_not_excluded(self):
        picked = [t for t, _ in self.vectors.analogy("man", "king", "woman", k=4,
                                                     exclude_inputs=False)]
        self.assertTrue({"man", "king", "woman"} & set(picked))

    def test_unknown_word_rejected(self):
        with self.assertRaises(KeyError):
            self.vectors.analogy("man", "king", "zebra")

    def test_negative_k_rejected(self):
        with self.assertRaises(ValueError):
            self.vectors.analogy("man", "king", "woman", k=-1)


class TestNearestToInput(unittest.TestCase):
    def setUp(self):
        self.vectors = _toy()

    def test_returns_a_word_outside_the_inputs(self):
        answer = self.vectors.nearest_to_input("man", "king", "woman", "c")
        self.assertNotIn(answer, {"man", "king", "woman"})

    def test_each_source_is_selectable(self):
        for which in ("a", "b", "c"):
            self.assertTrue(self.vectors.nearest_to_input("man", "king", "woman", which))

    def test_bad_source_rejected(self):
        with self.assertRaises(ValueError):
            self.vectors.nearest_to_input("man", "king", "woman", "d")


class TestOnTheRealVectors(unittest.TestCase):
    def setUp(self):
        self.vectors = load_vectors(k=24)

    def test_pipeline_produces_the_expected_shape(self):
        self.assertEqual(self.vectors.vectors.shape[1], 24)
        self.assertGreater(len(self.vectors), 50)

    def test_neighbours_are_sensible(self):
        self.assertIn("dog", [t for t, _ in self.vectors.most_similar("cat", 5)])
        self.assertIn("forest", [t for t, _ in self.vectors.most_similar("garden", 5)])

    def test_every_analogy_case_resolves_correctly(self):
        for a, b, c, expected in ANALOGY_CASES:
            answer = self.vectors.analogy(a, b, c, k=1)
            self.assertEqual(answer[0][0], expected, f"{a}:{b}::{c}")

    def test_deterministic(self):
        again = load_vectors(k=24)
        np.testing.assert_allclose(self.vectors.vectors, again.vectors)


class TestTheAnalogyResultIsNotWhatItLooksLike(unittest.TestCase):
    """Three separate reasons to discount the 100% score."""

    def setUp(self):
        self.vectors = load_vectors(k=24)

    def test_king_and_queen_are_nearly_the_same_vector(self):
        # So "queen" is what you get by asking for anything near "king".
        self.assertGreater(self.vectors.similarity("king", "queen"), 0.99)

    def test_without_exclusion_the_answer_is_an_input_word(self):
        # The convention that makes the task look solvable is doing the work.
        for a, b, c, _ in ANALOGY_CASES[:3]:
            top = self.vectors.analogy(a, b, c, k=1, exclude_inputs=False)[0][0]
            self.assertIn(top, {a, b, c}, f"{a}:{b}::{c}")

    def test_a_do_nothing_baseline_reproduces_every_answer(self):
        # If the arithmetic never beats "nearest neighbour of an input",
        # it has not been shown to contribute anything.
        scores = evaluate_analogies(self.vectors, ANALOGY_CASES)
        self.assertEqual(scores["arithmetic"], 1.0)
        self.assertEqual(scores["agrees_with_a_baseline"], 1.0)

    def test_the_baselines_alone_are_already_strong(self):
        scores = evaluate_analogies(self.vectors, ANALOGY_CASES)
        self.assertGreater(max(scores["nearest_b"], scores["nearest_c"]), 0.5)


class TestEvaluateAnalogies(unittest.TestCase):
    def test_reports_all_three_methods(self):
        scores = evaluate_analogies(load_vectors(k=24), ANALOGY_CASES)
        for key in ("arithmetic", "nearest_b", "nearest_c", "agrees_with_a_baseline"):
            self.assertIn(key, scores)
            self.assertGreaterEqual(scores[key], 0.0)
            self.assertLessEqual(scores[key], 1.0)

    def test_case_count_is_reported(self):
        scores = evaluate_analogies(load_vectors(k=24), ANALOGY_CASES)
        self.assertEqual(int(scores["cases"]), len(ANALOGY_CASES))

    def test_cases_with_unknown_words_are_skipped(self):
        vectors = load_vectors(k=24)
        padded = ANALOGY_CASES + [("zebra", "quagga", "okapi", "gnu")]
        scores = evaluate_analogies(vectors, padded)
        self.assertEqual(int(scores["cases"]), len(ANALOGY_CASES))

    def test_empty_case_list_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_analogies(load_vectors(k=24), [])

    def test_all_unknown_cases_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_analogies(load_vectors(k=24), [("zebra", "quagga", "okapi", "gnu")])


if __name__ == "__main__":
    unittest.main()
