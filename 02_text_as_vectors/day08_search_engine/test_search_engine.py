"""Day 8 tests — the index, the candidate guarantee, and the engine.

Korean strings below are functional test data, not documentation. Runnable
via `pytest` (repo root) or `python -m unittest` (this folder).

The test that matters most is TestIndexMatchesFullScan: the index is only
allowed to change *which* documents are visited, never the answer.
"""

import unittest

from search_engine import (
    DOCUMENTS,
    SearchEngine,
    boolean_and,
    build_index,
    build_vocabulary,
    candidates,
    cosine_similarity,
    inverse_document_frequency,
    normalize,
    tfidf_vector,
)


class TestBuildIndex(unittest.TestCase):
    def test_term_to_documents_with_counts(self):
        self.assertEqual(build_index([["a", "b", "a"], ["b"]]), {"a": {0: 2}, "b": {0: 1, 1: 1}})

    def test_every_term_appears(self):
        documents = [normalize(text) for text in DOCUMENTS]
        index = build_index(documents)
        for document in documents:
            for term in document:
                self.assertIn(term, index)

    def test_postings_agree_with_the_documents(self):
        documents = [normalize(text) for text in DOCUMENTS]
        index = build_index(documents)
        for term, postings in index.items():
            for document_id, count in postings.items():
                self.assertEqual(documents[document_id].count(term), count)

    def test_posting_list_length_is_document_frequency(self):
        documents = [normalize(text) for text in DOCUMENTS]
        index = build_index(documents)
        expected = sum(1 for d in documents if "the" in d)
        self.assertEqual(len(index["the"]), expected)

    def test_stores_less_than_a_dense_matrix(self):
        documents = [normalize(text) for text in DOCUMENTS]
        index = build_index(documents)
        vocabulary = build_vocabulary(documents)
        stored = sum(len(postings) for postings in index.values())
        self.assertLess(stored, len(documents) * len(vocabulary))

    def test_empty_collection(self):
        self.assertEqual(build_index([]), {})

    def test_empty_documents_contribute_nothing(self):
        self.assertEqual(build_index([[], []]), {})

    def test_rejects_a_str_document(self):
        with self.assertRaises(TypeError):
            build_index(["the cat"])


class TestCandidates(unittest.TestCase):
    def test_union_over_query_terms(self):
        index = {"a": {0: 1}, "b": {1: 1}}
        self.assertEqual(candidates(["a", "b"], index), {0, 1})

    def test_unknown_terms_contribute_nothing(self):
        self.assertEqual(candidates(["a", "z"], {"a": {0: 1}, "b": {1: 1}}), {0})

    def test_empty_query(self):
        self.assertEqual(candidates([], {"a": {0: 1}}), set())

    def test_query_with_no_known_term(self):
        self.assertEqual(candidates(["z"], {"a": {0: 1}}), set())

    def test_rejects_a_str_query(self):
        with self.assertRaises(TypeError):
            candidates("ab", {})


class TestCandidateGuarantee(unittest.TestCase):
    """Skipping non-candidates must cost nothing, or the index is a bug."""

    def setUp(self):
        self.documents = [normalize(text) for text in DOCUMENTS]
        self.vocabulary = build_vocabulary(self.documents)
        self.idf = inverse_document_frequency(self.documents, self.vocabulary)
        self.vectors = [tfidf_vector(d, self.vocabulary, self.idf) for d in self.documents]
        self.index = build_index(self.documents)

    def test_every_skipped_document_really_scores_zero(self):
        for query_text in ("tokenization tokens", "soup salt", "the", "검색 문서", "rain"):
            tokens = normalize(query_text)
            query_vector = tfidf_vector(tokens, self.vocabulary, self.idf)
            kept = candidates(tokens, self.index)
            for document_id, vector in enumerate(self.vectors):
                if document_id not in kept:
                    self.assertEqual(
                        cosine_similarity(query_vector, vector), 0.0,
                        f"{query_text!r} skipped doc {document_id} but it scores above zero",
                    )


class TestBooleanAnd(unittest.TestCase):
    def test_intersection_of_postings(self):
        index = {"a": {0: 1, 1: 1}, "b": {1: 1}}
        self.assertEqual(boolean_and(["a", "b"], index), {1})

    def test_one_missing_term_empties_the_result(self):
        index = {"a": {0: 1, 1: 1}, "b": {1: 1}}
        self.assertEqual(boolean_and(["a", "z"], index), set())

    def test_empty_query_matches_nothing(self):
        # Never a surprise, unlike "an empty AND matches everything".
        self.assertEqual(boolean_and([], {"a": {0: 1}}), set())

    def test_single_term_is_just_its_postings(self):
        self.assertEqual(boolean_and(["a"], {"a": {0: 1, 2: 1}}), {0, 2})

    def test_stricter_than_candidates(self):
        documents = [normalize(text) for text in DOCUMENTS]
        index = build_index(documents)
        query = normalize("soup tokenization")
        self.assertTrue(boolean_and(query, index).issubset(candidates(query, index)))
        self.assertEqual(boolean_and(query, index), set())
        self.assertNotEqual(candidates(query, index), set())

    def test_rejects_a_str_query(self):
        with self.assertRaises(TypeError):
            boolean_and("ab", {})


class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine(DOCUMENTS)

    def test_length_is_the_collection_size(self):
        self.assertEqual(len(self.engine), len(DOCUMENTS))

    def test_topic_queries_find_their_documents(self):
        for query, expected in (
            ("tokenization tokens", 0),
            ("soup salt", 3),
            ("search query documents", 2),
            ("rain wind coast", 5),
        ):
            self.assertEqual(self.engine.search(query)[0][0], expected, query)

    def test_results_are_sorted_by_score(self):
        scores = [score for _, score in self.engine.search("tokenization tokens")]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_zero_scoring_documents_are_not_returned(self):
        for _, score in self.engine.search("soup"):
            self.assertGreater(score, 0.0)

    def test_no_match_returns_an_empty_list(self):
        self.assertEqual(self.engine.search("quantum helicopter"), [])

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.engine.search(""), [])

    def test_k_truncates(self):
        self.assertLessEqual(len(self.engine.search("the", k=2)), 2)

    def test_k_zero(self):
        self.assertEqual(self.engine.search("soup", k=0), [])

    def test_negative_k_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.search("soup", k=-1)

    def test_repeated_searches_agree(self):
        self.assertEqual(self.engine.search("the"), self.engine.search("the"))

    def test_case_and_punctuation_do_not_matter(self):
        self.assertEqual(self.engine.search("Soup, SALT!"), self.engine.search("soup salt"))

    def test_rejects_a_token_list_as_query(self):
        with self.assertRaises(TypeError):
            self.engine.search(["soup"])

    def test_rejects_a_bare_str_collection(self):
        with self.assertRaises(TypeError):
            SearchEngine("one document")

    def test_min_df_is_passed_through(self):
        narrow = SearchEngine(DOCUMENTS, min_df=2)
        self.assertLess(len(narrow.vocabulary), len(self.engine.vocabulary))


class TestScoredDocumentCount(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine(DOCUMENTS)

    def test_rare_term_scores_one_document(self):
        self.assertEqual(self.engine.scored_document_count("tokenization"), 1)

    def test_common_term_scores_more(self):
        self.assertGreater(
            self.engine.scored_document_count("the"),
            self.engine.scored_document_count("tokenization"),
        )

    def test_unknown_query_scores_nothing(self):
        self.assertEqual(self.engine.scored_document_count("quantum helicopter"), 0)

    def test_never_exceeds_the_collection(self):
        for query in ("the", "soup", "tokenization tokens text"):
            self.assertLessEqual(self.engine.scored_document_count(query), len(self.engine))


class TestExplain(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine(DOCUMENTS)

    def test_lists_the_contributing_terms(self):
        terms = [term for term, _ in self.engine.explain("search query documents", 2)]
        self.assertEqual(sorted(terms), ["documents", "query", "search"])

    def test_sorted_by_contribution(self):
        contributions = [c for _, c in self.engine.explain("search query documents", 2)]
        self.assertEqual(contributions, sorted(contributions, reverse=True))

    def test_contributions_sum_to_the_score(self):
        query = "search query documents"
        total = sum(c for _, c in self.engine.explain(query, 2))
        score = dict(self.engine.search(query))[2]
        self.assertAlmostEqual(total, score)

    def test_non_matching_document_explains_nothing(self):
        self.assertEqual(self.engine.explain("soup", 0), [])

    def test_unknown_query_explains_nothing(self):
        self.assertEqual(self.engine.explain("quantum", 0), [])

    def test_out_of_range_document_rejected(self):
        with self.assertRaises(IndexError):
            self.engine.explain("soup", 99)


class TestIndexMatchesFullScan(unittest.TestCase):
    """The index may change what is visited, never what is returned."""

    def setUp(self):
        self.engine = SearchEngine(DOCUMENTS)

    def full_scan(self, query_text):
        """Day 7's approach: score every document, keep the non-zero ones."""
        tokens = normalize(query_text)
        query_vector = tfidf_vector(tokens, self.engine.vocabulary, self.engine.idf)
        scored = [
            (document_id, cosine_similarity(query_vector, vector))
            for document_id, vector in enumerate(self.engine.vectors)
        ]
        hits = [(document_id, score) for document_id, score in scored if score > 0]
        hits.sort(key=lambda pair: (-pair[1], pair[0]))
        return hits

    def test_identical_results_for_every_query(self):
        for query_text in (
            "tokenization tokens",
            "soup salt",
            "search query documents",
            "the",
            "rain wind coast",
            "검색 문서",
            "quantum helicopter",
            "",
        ):
            self.assertEqual(
                self.engine.search(query_text, k=len(self.engine)),
                self.full_scan(query_text),
                query_text,
            )


class TestKoreanRetrieval(unittest.TestCase):
    def setUp(self):
        self.engine = SearchEngine(DOCUMENTS)

    def test_korean_query_finds_the_korean_document(self):
        self.assertEqual(self.engine.search("검색 문서")[0][0], 7)

    def test_inflected_form_is_not_matched_by_its_stem(self):
        # 문서 contributes nothing: the document only writes 문서의/문서를.
        contributions = dict(self.engine.explain("검색 문서", 7))
        self.assertIn("검색", contributions)
        self.assertNotIn("문서", contributions)

    def test_searching_the_exact_surface_form_works(self):
        self.assertEqual(self.engine.search("문서를")[0][0], 7)


if __name__ == "__main__":
    unittest.main()
