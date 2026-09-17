"""Day 6 tests — TF schemes, IDF schemes, and the weighting they produce.

Korean strings below are functional test data, not documentation. Runnable
via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

from tfidf import (
    DOCUMENTS,
    IDF_SCHEMES,
    TF_SCHEMES,
    bag_of_words,
    build_vocabulary,
    document_frequencies,
    inverse_document_frequency,
    normalize,
    term_frequency,
    tfidf_matrix,
    tfidf_vector,
    top_terms,
)


class TestTermFrequency(unittest.TestCase):
    def test_raw_returns_the_counts(self):
        self.assertEqual(term_frequency([2, 0, 1], "raw"), [2.0, 0.0, 1.0])

    def test_relative_divides_by_document_length(self):
        self.assertEqual(term_frequency([3, 1], "relative"), [0.75, 0.25])

    def test_relative_sums_to_one(self):
        self.assertAlmostEqual(sum(term_frequency([5, 3, 2], "relative")), 1.0)

    def test_boolean_is_presence_only(self):
        self.assertEqual(term_frequency([4, 0, 9], "boolean"), [1.0, 0.0, 1.0])

    def test_log_leaves_zero_at_zero(self):
        self.assertEqual(term_frequency([0], "log"), [0.0])

    def test_log_of_one_is_one(self):
        self.assertEqual(term_frequency([1], "log"), [1.0])

    def test_log_compresses_repeated_terms(self):
        # Diminishing returns: the 10th occurrence adds less than the 2nd.
        # The comparison is between marginal gains, not totals.
        one, two, nine, ten = term_frequency([1, 2, 9, 10], "log")
        self.assertGreater(two - one, ten - nine)

    def test_raw_counts_have_no_diminishing_returns(self):
        one, two, nine, ten = term_frequency([1, 2, 9, 10], "raw")
        self.assertAlmostEqual(two - one, ten - nine)

    def test_every_scheme_maps_zero_to_zero(self):
        for scheme in TF_SCHEMES:
            self.assertEqual(term_frequency([0, 5], scheme)[0], 0.0, scheme)

    def test_empty_vector(self):
        for scheme in TF_SCHEMES:
            self.assertEqual(term_frequency([], scheme), [], scheme)

    def test_all_zero_counts_do_not_divide_by_zero(self):
        self.assertEqual(term_frequency([0, 0], "relative"), [0.0, 0.0])

    def test_unknown_scheme_rejected(self):
        with self.assertRaises(ValueError):
            term_frequency([1], "tf-idf-ish")


class TestInverseDocumentFrequency(unittest.TestCase):
    def setUp(self):
        self.documents = [normalize(text) for text in DOCUMENTS]
        self.vocabulary = build_vocabulary(self.documents)

    def test_one_weight_per_vocabulary_term(self):
        for scheme in ("smooth",):
            weights = inverse_document_frequency(self.documents, self.vocabulary, scheme)
            self.assertEqual(len(weights), len(self.vocabulary))

    def test_weights_follow_vocabulary_order(self):
        vocabulary = {"common": 0, "rare": 1}
        documents = [["common", "rare"], ["common"], ["common"]]
        weights = inverse_document_frequency(documents, vocabulary, "classic")
        self.assertAlmostEqual(weights[0], 0.0)
        self.assertAlmostEqual(weights[1], math.log(3))

    def test_rarer_terms_never_weigh_less(self):
        # The defining property: idf is non-increasing in document frequency.
        dfs = document_frequencies(self.documents)
        for scheme in IDF_SCHEMES:
            if scheme == "probabilistic":
                continue  # undefined at df == N, covered separately
            weights = inverse_document_frequency(self.documents, self.vocabulary, scheme)
            pairs = sorted((dfs[t], weights[i]) for t, i in self.vocabulary.items())
            for (df_a, w_a), (df_b, w_b) in zip(pairs, pairs[1:]):
                if df_a < df_b:
                    self.assertGreaterEqual(w_a, w_b, f"{scheme}: df {df_a} vs {df_b}")

    def test_classic_erases_a_term_in_every_document(self):
        documents = [["x", "a"], ["x", "b"]]
        vocabulary = build_vocabulary(documents)
        weights = inverse_document_frequency(documents, vocabulary, "classic")
        self.assertEqual(weights[vocabulary["x"]], 0.0)

    def test_smooth_keeps_a_ubiquitous_term_positive(self):
        documents = [["x", "a"], ["x", "b"]]
        vocabulary = build_vocabulary(documents)
        weights = inverse_document_frequency(documents, vocabulary, "smooth")
        self.assertGreater(weights[vocabulary["x"]], 0.0)

    def test_smooth_survives_an_unseen_term(self):
        # An open vocabulary is the normal case for a query (Day 7).
        weights = inverse_document_frequency([["a"]], {"a": 0, "unseen": 1}, "smooth")
        self.assertGreater(weights[1], weights[0])

    def test_classic_rejects_an_unseen_term(self):
        with self.assertRaises(ValueError):
            inverse_document_frequency([["a"]], {"a": 0, "unseen": 1}, "classic")

    def test_probabilistic_goes_negative_past_half(self):
        documents = [["x"], ["x"], ["x"], ["y"]]
        vocabulary = build_vocabulary(documents)
        weights = inverse_document_frequency(documents, vocabulary, "probabilistic")
        self.assertLess(weights[vocabulary["x"]], 0.0)
        self.assertGreater(weights[vocabulary["y"]], 0.0)

    def test_probabilistic_rejects_a_ubiquitous_term(self):
        documents = [["x"], ["x"]]
        with self.assertRaises(ValueError):
            inverse_document_frequency(documents, build_vocabulary(documents), "probabilistic")

    def test_unknown_scheme_rejected(self):
        with self.assertRaises(ValueError):
            inverse_document_frequency([["a"]], {"a": 0}, "inverse-ish")

    def test_rejects_a_str_document(self):
        with self.assertRaises(TypeError):
            inverse_document_frequency(["a document"], {"a": 0})


class TestTfidfVector(unittest.TestCase):
    def test_multiplies_tf_by_idf(self):
        self.assertEqual(tfidf_vector(["a", "a"], {"a": 0, "b": 1}, [2.0, 5.0]), [2.0, 0.0])

    def test_absent_term_stays_zero_however_large_its_idf(self):
        vector = tfidf_vector(["a"], {"a": 0, "b": 1}, [1.0, 99.0])
        self.assertEqual(vector[1], 0.0)

    def test_zero_idf_erases_a_present_term(self):
        vector = tfidf_vector(["a", "b"], {"a": 0, "b": 1}, [0.0, 1.0])
        self.assertEqual(vector[0], 0.0)
        self.assertGreater(vector[1], 0.0)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            tfidf_vector(["a"], {"a": 0, "b": 1}, [1.0])

    def test_empty_document_gives_all_zeros(self):
        self.assertEqual(tfidf_vector([], {"a": 0}, [3.0]), [0.0])


class TestTfidfMatrix(unittest.TestCase):
    def setUp(self):
        self.documents = [normalize(text) for text in DOCUMENTS]
        self.vocabulary = build_vocabulary(self.documents)

    def test_shape_and_idf_length(self):
        matrix, idf = tfidf_matrix(self.documents, self.vocabulary)
        self.assertEqual(len(matrix), len(self.documents))
        self.assertEqual(len(idf), len(self.vocabulary))
        for row in matrix:
            self.assertEqual(len(row), len(self.vocabulary))

    def test_returned_idf_reproduces_the_rows(self):
        # Day 7 weights a query with these same numbers, so they have to be
        # the ones that produced the matrix.
        matrix, idf = tfidf_matrix(self.documents, self.vocabulary)
        rebuilt = tfidf_vector(self.documents[0], self.vocabulary, idf)
        self.assertEqual(rebuilt, matrix[0])

    def test_all_weights_non_negative_under_defaults(self):
        matrix, _ = tfidf_matrix(self.documents, self.vocabulary)
        for row in matrix:
            for weight in row:
                self.assertGreaterEqual(weight, 0.0)

    def test_empty_collection(self):
        matrix, idf = tfidf_matrix([], {})
        self.assertEqual(matrix, [])
        self.assertEqual(idf, [])


class TestTopTerms(unittest.TestCase):
    def test_heaviest_first(self):
        self.assertEqual(top_terms([0.5, 0.9], {"a": 0, "b": 1}, 1), [("b", 0.9)])

    def test_ties_break_alphabetically(self):
        picked = top_terms([1.0, 1.0], {"b": 1, "a": 0}, 2)
        self.assertEqual([term for term, _ in picked], ["a", "b"])

    def test_k_larger_than_vocabulary(self):
        self.assertEqual(len(top_terms([1.0], {"a": 0}, 9)), 1)

    def test_k_zero(self):
        self.assertEqual(top_terms([1.0], {"a": 0}, 0), [])

    def test_negative_k_rejected(self):
        with self.assertRaises(ValueError):
            top_terms([1.0], {"a": 0}, -1)


class TestWeightingChangesWhatIsLoudest(unittest.TestCase):
    """The point of the day, measured on the shared collection."""

    def setUp(self):
        self.documents = [normalize(text) for text in DOCUMENTS]
        self.vocabulary = build_vocabulary(self.documents)

    def test_raw_counts_make_the_as_loud_as_the_topic(self):
        counts = bag_of_words(self.documents[3], self.vocabulary)
        self.assertEqual(counts[self.vocabulary["the"]], counts[self.vocabulary["soup"]])

    def test_tfidf_puts_the_topic_above_the_function_word(self):
        matrix, _ = tfidf_matrix(self.documents, self.vocabulary)
        row = matrix[3]
        self.assertGreater(row[self.vocabulary["soup"]], row[self.vocabulary["the"]])

    def test_classic_idf_removes_the_from_the_top_terms(self):
        # Smooth idf cannot reach zero, so on a collection this small "the"
        # stays competitive; classic suppresses it properly.
        smooth, _ = tfidf_matrix(self.documents, self.vocabulary, idf_scheme="smooth")
        classic, _ = tfidf_matrix(self.documents, self.vocabulary, idf_scheme="classic")
        self.assertIn("the", [t for t, _ in top_terms(smooth[4], self.vocabulary, 3)])
        self.assertNotIn("the", [t for t, _ in top_terms(classic[4], self.vocabulary, 3)])

    def test_smooth_idf_range_is_narrow_on_a_tiny_collection(self):
        _, idf = tfidf_matrix(self.documents, self.vocabulary)
        self.assertLess(max(idf) / min(idf), 2.0)

    def test_korean_documents_get_korean_top_terms(self):
        matrix, _ = tfidf_matrix(self.documents, self.vocabulary)
        picked = [term for term, _ in top_terms(matrix[7], self.vocabulary, 3)]
        for term in picked:
            self.assertTrue(any("가" <= ch <= "힣" for ch in term), term)


if __name__ == "__main__":
    unittest.main()
