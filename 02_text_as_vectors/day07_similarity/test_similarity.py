"""Day 7 tests — vector algebra, the cosine properties, and ranking.

Korean strings below are functional test data (particle mismatch defeating
exact-term retrieval), not documentation. Runnable via `pytest` (repo root)
or `python -m unittest` (this folder).
"""

import unittest

from similarity import (
    DOCUMENTS,
    bag_of_words,
    build_vocabulary,
    cosine_similarity,
    dot,
    euclidean_distance,
    inverse_document_frequency,
    norm,
    normalize,
    rank_documents,
    tfidf_vector,
)


def _collection():
    documents = [normalize(text) for text in DOCUMENTS]
    vocabulary = build_vocabulary(documents)
    idf = inverse_document_frequency(documents, vocabulary)
    matrix = [tfidf_vector(d, vocabulary, idf) for d in documents]
    return documents, vocabulary, idf, matrix


class TestDot(unittest.TestCase):
    def test_sum_of_products(self):
        self.assertEqual(dot([1.0, 2.0], [3.0, 4.0]), 11.0)

    def test_orthogonal_vectors_give_zero(self):
        self.assertEqual(dot([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_only_shared_terms_contribute(self):
        # The fact Day 8's inverted index is built on.
        self.assertEqual(dot([5.0, 0.0], [0.0, 9.0]), 0.0)

    def test_empty_vectors(self):
        self.assertEqual(dot([], []), 0.0)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            dot([1.0], [1.0, 2.0])


class TestNorm(unittest.TestCase):
    def test_three_four_five(self):
        self.assertEqual(norm([3.0, 4.0]), 5.0)

    def test_zero_vector(self):
        self.assertEqual(norm([0.0, 0.0]), 0.0)

    def test_empty_vector(self):
        self.assertEqual(norm([]), 0.0)

    def test_scales_linearly(self):
        self.assertAlmostEqual(norm([6.0, 8.0]), 2 * norm([3.0, 4.0]))


class TestCosineSimilarity(unittest.TestCase):
    def test_identical_direction_is_one(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_is_zero(self):
        self.assertEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_invariant_under_scaling(self):
        # The property that makes it the right choice for documents.
        self.assertAlmostEqual(cosine_similarity([1.0, 1.0], [2.0, 2.0]), 1.0)
        self.assertAlmostEqual(
            cosine_similarity([1.0, 2.0], [3.0, 4.0]),
            cosine_similarity([10.0, 20.0], [3.0, 4.0]),
        )

    def test_symmetric(self):
        a, b = [1.0, 2.0, 3.0], [4.0, 0.0, 1.0]
        self.assertAlmostEqual(cosine_similarity(a, b), cosine_similarity(b, a))

    def test_zero_vector_is_zero_not_a_division_error(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_both_zero_vectors(self):
        self.assertEqual(cosine_similarity([0.0], [0.0]), 0.0)

    def test_stays_within_the_unit_range_on_real_data(self):
        # Mathematically in [0, 1] for non-negative vectors; floating point
        # makes that only approximate, so the check carries a tolerance.
        _, _, _, matrix = _collection()
        for a in matrix:
            for b in matrix:
                score = cosine_similarity(a, b)
                self.assertGreaterEqual(score, -1e-12)
                self.assertLessEqual(score, 1 + 1e-12)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


class TestEuclideanContrast(unittest.TestCase):
    """Why the day does not use distance."""

    def test_distance_basics(self):
        self.assertEqual(euclidean_distance([0.0, 0.0], [3.0, 4.0]), 5.0)

    def test_distance_of_a_vector_from_itself(self):
        self.assertEqual(euclidean_distance([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_distance_is_not_scale_invariant(self):
        # Same topic, one document twenty times longer: cosine says
        # identical, distance says far apart.
        short = [1.0, 1.0]
        long = [20.0, 20.0]
        self.assertAlmostEqual(cosine_similarity(short, long), 1.0)
        self.assertGreater(euclidean_distance(short, long), 20.0)

    def test_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            euclidean_distance([1.0], [1.0, 2.0])


class TestRankDocuments(unittest.TestCase):
    def setUp(self):
        self.documents, self.vocabulary, self.idf, self.matrix = _collection()

    def rank(self, text, **kwargs):
        return rank_documents(normalize(text), self.matrix, self.vocabulary, self.idf, **kwargs)

    def test_best_match_first(self):
        self.assertEqual(self.rank("tokenization tokens")[0][0], 0)

    def test_scores_are_descending(self):
        scores = [score for _, score in self.rank("search query documents")]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_every_document_is_scored(self):
        self.assertEqual(len(self.rank("soup")), len(self.matrix))

    def test_k_truncates(self):
        self.assertEqual(len(self.rank("soup", k=3)), 3)

    def test_k_zero(self):
        self.assertEqual(self.rank("soup", k=0), [])

    def test_negative_k_rejected(self):
        with self.assertRaises(ValueError):
            self.rank("soup", k=-1)

    def test_topic_queries_find_their_documents(self):
        for query, expected in (
            ("tokenization tokens", 0),
            ("soup salt", 3),
            ("search query documents", 2),
            ("rain wind coast", 5),
        ):
            self.assertEqual(self.rank(query)[0][0], expected, query)

    def test_unrelated_documents_score_exactly_zero(self):
        scores = dict(self.rank("soup salt"))
        self.assertGreater(scores[3], 0.0)
        self.assertEqual(scores[0], 0.0)

    def test_out_of_vocabulary_query_scores_nothing_anywhere(self):
        ranked = self.rank("quantum helicopter")
        self.assertTrue(all(score == 0.0 for _, score in ranked))

    def test_ties_break_on_document_id(self):
        # All-zero scores must still come back in a stable order.
        ranked = self.rank("quantum helicopter")
        self.assertEqual([doc_id for doc_id, _ in ranked], list(range(len(self.matrix))))

    def test_repeated_calls_agree(self):
        self.assertEqual(self.rank("search query"), self.rank("search query"))

    def test_collection_idf_separates_a_real_match_from_a_stopword_match(self):
        # Query "the soup": doc 3 is about soup, doc 4 merely contains "the".
        # Collection IDF (df: the=5, soup=1) should pull them apart; flat
        # weights treat the two query terms as equally informative.
        #
        # A query whose terms all share one df would show nothing here --
        # equal weights leave the query's *direction* unchanged, and cosine
        # ignores its scale.
        flat = [1.0] * len(self.vocabulary)
        weighted = dict(self.rank("the soup"))
        unweighted = dict(
            rank_documents(normalize("the soup"), self.matrix, self.vocabulary, flat)
        )
        self.assertGreater(
            weighted[3] / weighted[4],
            unweighted[3] / unweighted[4],
        )

    def test_idf_weighting_changes_the_scores(self):
        flat = [1.0] * len(self.vocabulary)
        self.assertNotEqual(
            self.rank("the soup"),
            rank_documents(normalize("the soup"), self.matrix, self.vocabulary, flat),
        )

    def test_rejects_a_str_query(self):
        with self.assertRaises(TypeError):
            rank_documents("soup", self.matrix, self.vocabulary, self.idf)


class TestTheLexicalCeiling(unittest.TestCase):
    """What this representation fundamentally cannot do."""

    def setUp(self):
        self.documents, self.vocabulary, self.idf, self.matrix = _collection()

    def test_a_synonym_scores_nothing(self):
        # "word segmentation" means what "tokenization" means. Cosine over
        # term counts has no way to know that; Level 4 embeddings do.
        ranked = rank_documents(
            normalize("word segmentation"), self.matrix, self.vocabulary, self.idf, k=1
        )
        self.assertEqual(ranked[0][1], 0.0)

    def test_the_literal_term_scores_well(self):
        ranked = rank_documents(
            normalize("tokenization"), self.matrix, self.vocabulary, self.idf, k=1
        )
        self.assertEqual(ranked[0][0], 0)
        self.assertGreater(ranked[0][1], 0.5)


class TestKoreanParticleMismatch(unittest.TestCase):
    """Retrieval inherits the tokenization problem, now as missed matches."""

    def setUp(self):
        self.documents, self.vocabulary, self.idf, self.matrix = _collection()

    def test_query_term_misses_its_inflected_form(self):
        # Document 7 contains 문서의 and 문서를 but not the bare 문서, so a
        # query for 문서 matches nothing in it.
        self.assertIn("문서의", self.vocabulary)
        self.assertIn("문서를", self.vocabulary)
        self.assertNotIn("문서", self.vocabulary)

    def test_only_the_uninflected_half_of_the_query_lands(self):
        both = rank_documents(normalize("검색 문서"), self.matrix, self.vocabulary, self.idf, k=1)
        only = rank_documents(normalize("검색"), self.matrix, self.vocabulary, self.idf, k=1)
        self.assertEqual(both[0][0], 7)
        # Adding 문서 to the query does not improve the match, because the
        # term contributes nothing the document can answer.
        self.assertLessEqual(both[0][1], only[0][1])


if __name__ == "__main__":
    unittest.main()
