"""Day 5 tests — vocabulary contracts, vectorization, sparsity, OOV.

Korean strings below are functional test data (particle-driven vocabulary
growth), not documentation. Runnable via `pytest` (repo root) or
`python -m unittest` (this folder).
"""

import unittest

from bag_of_words import (
    DOCUMENTS,
    bag_of_words,
    build_vocabulary,
    document_frequencies,
    document_term_matrix,
    matrix_sparsity,
    normalize,
    oov_rate,
    to_sparse,
)


class TestDocumentFrequencies(unittest.TestCase):
    def test_counts_documents_not_occurrences(self):
        self.assertEqual(document_frequencies([["a", "a"], ["a", "b"]]), {"a": 2, "b": 1})

    def test_term_in_every_document(self):
        freqs = document_frequencies([["x"], ["x"], ["x"]])
        self.assertEqual(freqs["x"], 3)

    def test_empty_collection(self):
        self.assertEqual(document_frequencies([]), {})

    def test_empty_documents_contribute_nothing(self):
        self.assertEqual(document_frequencies([[], []]), {})

    def test_never_exceeds_collection_size(self):
        documents = [normalize(text) for text in DOCUMENTS]
        for term, df in document_frequencies(documents).items():
            self.assertLessEqual(df, len(documents), term)

    def test_rejects_a_str_document(self):
        # A raw string would be counted character by character.
        with self.assertRaises(TypeError):
            document_frequencies(["the cat"])

    def test_rejects_a_bare_str_collection(self):
        with self.assertRaises(TypeError):
            document_frequencies("the cat")


class TestBuildVocabulary(unittest.TestCase):
    def test_terms_numbered_alphabetically(self):
        self.assertEqual(build_vocabulary([["b", "a"], ["c"]]), {"a": 0, "b": 1, "c": 2})

    def test_indices_are_contiguous_from_zero(self):
        vocabulary = build_vocabulary([normalize(text) for text in DOCUMENTS])
        self.assertEqual(sorted(vocabulary.values()), list(range(len(vocabulary))))

    def test_independent_of_document_order(self):
        # The index assignment is a contract shared by every vector, so it
        # must not depend on which document arrived first.
        documents = [normalize(text) for text in DOCUMENTS]
        self.assertEqual(build_vocabulary(documents), build_vocabulary(documents[::-1]))

    def test_min_df_drops_single_document_terms(self):
        self.assertEqual(build_vocabulary([["a", "b"], ["b", "c"]], min_df=2), {"b": 0})

    def test_max_df_drops_ubiquitous_terms(self):
        # "b" is in both documents, so max_df=0.5 excludes it.
        self.assertEqual(build_vocabulary([["a", "b"], ["b", "c"]], max_df=0.5), {"a": 0, "c": 1})

    def test_max_df_one_keeps_everything(self):
        documents = [normalize(text) for text in DOCUMENTS]
        self.assertEqual(build_vocabulary(documents, max_df=1.0), build_vocabulary(documents))

    def test_filters_only_shrink_the_vocabulary(self):
        documents = [normalize(text) for text in DOCUMENTS]
        full = build_vocabulary(documents)
        for narrowed in (
            build_vocabulary(documents, min_df=2),
            build_vocabulary(documents, max_df=0.5),
        ):
            self.assertLessEqual(len(narrowed), len(full))
            self.assertTrue(set(narrowed).issubset(set(full)))

    def test_reindexed_after_filtering(self):
        # Filtering must renumber, not leave gaps where terms were removed.
        vocabulary = build_vocabulary([["a", "b"], ["b", "c"]], min_df=2)
        self.assertEqual(sorted(vocabulary.values()), list(range(len(vocabulary))))

    def test_empty_collection_gives_empty_vocabulary(self):
        self.assertEqual(build_vocabulary([]), {})

    def test_min_df_below_one_rejected(self):
        with self.assertRaises(ValueError):
            build_vocabulary([["a"]], min_df=0)

    def test_max_df_out_of_range_rejected(self):
        for bad in (0, 1.5, -0.2):
            with self.assertRaises(ValueError):
                build_vocabulary([["a"]], max_df=bad)


class TestBagOfWords(unittest.TestCase):
    def test_counts_by_vocabulary_position(self):
        self.assertEqual(bag_of_words(["a", "a", "c"], {"a": 0, "b": 1, "c": 2}), [2, 0, 1])

    def test_length_always_matches_the_vocabulary(self):
        vocabulary = {"a": 0, "b": 1, "c": 2}
        for tokens in ([], ["a"], ["z", "z"], ["a", "b", "c"]):
            self.assertEqual(len(bag_of_words(tokens, vocabulary)), len(vocabulary))

    def test_unknown_terms_dropped_silently(self):
        self.assertEqual(bag_of_words(["z"], {"a": 0}), [0])

    def test_empty_vocabulary_gives_empty_vector(self):
        self.assertEqual(bag_of_words(["a"], {}), [])

    def test_total_equals_in_vocabulary_token_count(self):
        vocabulary = {"a": 0, "b": 1}
        tokens = ["a", "a", "b", "z"]
        self.assertEqual(sum(bag_of_words(tokens, vocabulary)), 3)

    def test_rejects_a_str_document(self):
        with self.assertRaises(TypeError):
            bag_of_words("abc", {"a": 0})


class TestDocumentTermMatrix(unittest.TestCase):
    def test_shape_is_documents_by_terms(self):
        documents = [normalize(text) for text in DOCUMENTS]
        vocabulary = build_vocabulary(documents)
        matrix = document_term_matrix(documents, vocabulary)
        self.assertEqual(len(matrix), len(documents))
        for row in matrix:
            self.assertEqual(len(row), len(vocabulary))

    def test_rows_match_bag_of_words(self):
        documents = [["a"], ["b", "b"]]
        vocabulary = {"a": 0, "b": 1}
        self.assertEqual(document_term_matrix(documents, vocabulary), [[1, 0], [0, 2]])

    def test_column_totals_equal_corpus_counts(self):
        documents = [normalize(text) for text in DOCUMENTS]
        vocabulary = build_vocabulary(documents)
        matrix = document_term_matrix(documents, vocabulary)
        column = vocabulary["the"]
        expected = sum(document.count("the") for document in documents)
        self.assertEqual(sum(row[column] for row in matrix), expected)

    def test_empty_collection(self):
        self.assertEqual(document_term_matrix([], {"a": 0}), [])


class TestWordOrderIsLost(unittest.TestCase):
    """The known cost of this representation, carried over from Day 4."""

    def test_reordered_sentences_share_a_vector(self):
        pair = [normalize("the dog bit the man"), normalize("the man bit the dog")]
        vocabulary = build_vocabulary(pair)
        rows = document_term_matrix(pair, vocabulary)
        self.assertEqual(rows[0], rows[1])

    def test_ngram_terms_recover_the_difference(self):
        # The functions never assume a term is one word, so bigram strings
        # work as vocabulary entries unchanged.
        def bigram_terms(text):
            words = normalize(text)
            return [f"{a}_{b}" for a, b in zip(words, words[1:])]

        pair = [bigram_terms("the dog bit the man"), bigram_terms("the man bit the dog")]
        vocabulary = build_vocabulary(pair)
        rows = document_term_matrix(pair, vocabulary)
        self.assertNotEqual(rows[0], rows[1])


class TestOovRate(unittest.TestCase):
    def test_half_the_tokens_missing(self):
        self.assertEqual(oov_rate(["a", "z"], {"a": 0}), 0.5)

    def test_nothing_missing(self):
        self.assertEqual(oov_rate(["a"], {"a": 0}), 0.0)

    def test_everything_missing(self):
        self.assertEqual(oov_rate(["z"], {"a": 0}), 1.0)

    def test_counted_over_running_tokens_not_types(self):
        # One missing type occurring three times costs three tokens.
        self.assertAlmostEqual(oov_rate(["a", "z", "z", "z"], {"a": 0}), 0.75)

    def test_empty_document_is_zero_not_a_division_error(self):
        self.assertEqual(oov_rate([], {"a": 0}), 0.0)

    def test_rejects_a_str_document(self):
        with self.assertRaises(TypeError):
            oov_rate("abc", {"a": 0})


class TestMatrixSparsity(unittest.TestCase):
    def test_three_of_four_cells_zero(self):
        self.assertEqual(matrix_sparsity([[1, 0], [0, 0]]), 0.75)

    def test_dense_matrix(self):
        self.assertEqual(matrix_sparsity([[1, 2], [3, 4]]), 0.0)

    def test_all_zero_matrix(self):
        self.assertEqual(matrix_sparsity([[0, 0]]), 1.0)

    def test_empty_matrix_is_zero_not_a_division_error(self):
        self.assertEqual(matrix_sparsity([]), 0.0)

    def test_real_collection_is_mostly_zeros(self):
        documents = [normalize(text) for text in DOCUMENTS]
        matrix = document_term_matrix(documents, build_vocabulary(documents))
        self.assertGreater(matrix_sparsity(matrix), 0.8)


class TestToSparse(unittest.TestCase):
    def test_keeps_only_non_zero_entries(self):
        self.assertEqual(to_sparse([2, 0, 1]), {0: 2, 2: 1})

    def test_all_zero_vector(self):
        self.assertEqual(to_sparse([0, 0]), {})

    def test_empty_vector(self):
        self.assertEqual(to_sparse([]), {})

    def test_round_trips_back_to_dense(self):
        dense = [0, 3, 0, 1]
        sparse = to_sparse(dense)
        rebuilt = [sparse.get(i, 0) for i in range(len(dense))]
        self.assertEqual(rebuilt, dense)


class TestKoreanVocabularyGrowth(unittest.TestCase):
    """Particles inflate the vocabulary and starve every document filter."""

    def test_one_noun_occupies_three_dimensions(self):
        documents = [normalize("고양이가 잔다"), normalize("고양이를 본다"), normalize("고양이에게 준다")]
        vocabulary = build_vocabulary(documents)
        for form in ("고양이가", "고양이를", "고양이에게"):
            self.assertIn(form, vocabulary)
        self.assertNotIn("고양이", vocabulary)

    def test_min_df_erases_the_korean_documents(self):
        # No Korean term survives min_df=2, because each occurrence of a
        # stem carries a different particle. Dropping single-document terms
        # therefore discards the Korean half of the collection entirely.
        documents = [normalize(text) for text in DOCUMENTS]
        narrow = build_vocabulary(documents, min_df=2)
        self.assertEqual(oov_rate(documents[6], narrow), 1.0)
        self.assertEqual(oov_rate(documents[7], narrow), 1.0)
        self.assertLess(oov_rate(documents[0], narrow), 1.0)


if __name__ == "__main__":
    unittest.main()
