"""Day 17 tests — analyzers, the bilingual result, and the CLI.

Korean strings below are functional test data: they are the problem the
whole curriculum has been deferring. Runnable via `pytest` (repo root) or
`python -m unittest` (this folder).
"""

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from capstone import (
    BPE_CORPUS,
    DOCUMENTS,
    END,
    EVALUATION,
    Analyzer,
    Engine,
    build_analyzers,
    build_parser,
    char_ngrams,
    encode_word,
    main,
    normalize,
    recall_at_k,
    train_bpe,
)

_ANALYZERS = build_analyzers()
_ENGINES = {name: Engine(DOCUMENTS, analyzer) for name, analyzer in _ANALYZERS.items()}


class TestAnalyzer(unittest.TestCase):
    def test_word_analyzer(self):
        self.assertEqual(Analyzer("word").terms("The cat sat"), ["the", "cat", "sat"])

    def test_char_analyzer(self):
        self.assertEqual(Analyzer("char", n=2).terms("cat"), ["ca", "at"])

    def test_char_analyzer_ignores_spaces(self):
        self.assertIn("ab", Analyzer("char", n=2).terms("a b"))

    def test_subword_analyzer_needs_a_corpus(self):
        with self.assertRaises(ValueError):
            Analyzer("subword")

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            Analyzer("magic")

    def test_rejects_a_token_list(self):
        with self.assertRaises(TypeError):
            Analyzer("word").terms(["cat"])

    def test_every_analyzer_returns_terms_for_both_scripts(self):
        for name, analyzer in _ANALYZERS.items():
            for text in ("the cat sat", "고양이가 잔다"):
                self.assertTrue(analyzer.terms(text), f"{name} on {text!r}")


class TestBpe(unittest.TestCase):
    def test_boundary_marker_present_or_absent(self):
        with_boundary = train_bpe(BPE_CORPUS, 200, boundary=True)
        without = train_bpe(BPE_CORPUS, 200, boundary=False)
        self.assertTrue(any(END in pair for pair in with_boundary))
        self.assertFalse(any(END in pair for pair in without))

    def test_pieces_rejoin_to_the_word(self):
        merges = train_bpe(BPE_CORPUS, 200, boundary=False)
        for word in ("문서의", "고양이가", "tokenization"):
            self.assertEqual("".join(encode_word(word, merges, boundary=False)), word)

    def test_deterministic(self):
        self.assertEqual(train_bpe(BPE_CORPUS, 100), train_bpe(BPE_CORPUS, 100))

    def test_negative_merges_rejected(self):
        with self.assertRaises(ValueError):
            train_bpe(BPE_CORPUS, -1)

    def test_zero_merges_gives_characters(self):
        self.assertEqual(encode_word("cat", [], boundary=False), ["c", "a", "t"])


class TestTheBoundaryMarkerHurtsRetrieval(unittest.TestCase):
    """Day 12's deliberate design decision, wrong for this task."""

    def setUp(self):
        self.with_boundary = _ANALYZERS["subword(+</w>)"]
        self.without = _ANALYZERS["subword"]

    def test_without_the_marker_a_bare_stem_is_one_piece(self):
        self.assertEqual(self.without.terms("문서"), ["문서"])

    def test_and_an_inflected_form_starts_with_that_same_piece(self):
        self.assertEqual(self.without.terms("문서의")[0], "문서")

    def test_so_query_and_document_share_the_stem(self):
        shared = set(self.without.terms("문서")) & set(self.without.terms("문서의"))
        self.assertIn("문서", shared)

    def test_with_the_marker_the_standalone_word_segments_differently(self):
        # A bare query word carries </w>; the same stem as a prefix does
        # not, so the two are segmented under different pair statistics.
        standalone = self.with_boundary.terms("문서")
        inflected = self.with_boundary.terms("문서의")
        self.assertNotEqual(standalone[0], inflected[0])

    def test_and_the_stem_is_no_longer_shared(self):
        shared = set(self.with_boundary.terms("문서")) & set(self.with_boundary.terms("문서의"))
        self.assertNotIn("문서", shared)


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.engine = _ENGINES["subword"]

    def test_length(self):
        self.assertEqual(len(self.engine), len(DOCUMENTS))

    def test_vocabulary_indices_are_contiguous(self):
        self.assertEqual(
            sorted(self.engine.vocabulary.values()),
            list(range(len(self.engine.vocabulary))),
        )

    def test_search_returns_only_positive_scores(self):
        for _, score in self.engine.search("검색", k=5):
            self.assertGreater(score, 0.0)

    def test_a_nonsense_query_still_matches_under_subwords(self):
        # Day 12's no-OOV guarantee, with its bill attached: "zzzqqq"
        # decomposes to single characters, some of which occur in the
        # English documents, so it never cleanly misses. See the class
        # below.
        self.assertNotEqual(self.engine.search("zzzqqq"), [])

    def test_empty_query(self):
        self.assertEqual(self.engine.search(""), [])

    def test_k_zero(self):
        self.assertEqual(self.engine.search("검색", k=0), [])

    def test_negative_k_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.search("검색", k=-1)

    def test_shared_terms_reports_the_overlap(self):
        self.assertIn("문서", self.engine.shared_terms("문서", 0))

    def test_shared_terms_rejects_a_bad_id(self):
        with self.assertRaises(IndexError):
            self.engine.shared_terms("문서", 99)

    def test_rejects_a_bare_str_collection(self):
        with self.assertRaises(TypeError):
            Engine("one document", Analyzer("word"))

    def test_empty_collection_rejected(self):
        with self.assertRaises(ValueError):
            Engine([], Analyzer("word"))

    def test_deterministic(self):
        again = Engine(DOCUMENTS, _ANALYZERS["subword"])
        self.assertEqual(self.engine.search("검색"), again.search("검색"))


class TestTheBilingualResult(unittest.TestCase):
    """The question the whole curriculum deferred, answered."""

    def korean(self):
        return [(q, r) for q, r in EVALUATION if not q.isascii()]

    def english(self):
        return [(q, r) for q, r in EVALUATION if q.isascii()]

    def test_word_level_handles_english_and_fails_korean(self):
        engine = _ENGINES["word"]
        self.assertEqual(recall_at_k(engine, self.english()), 1.0)
        self.assertLess(recall_at_k(engine, self.korean()), 0.25)

    def test_subwords_close_the_gap(self):
        engine = _ENGINES["subword"]
        self.assertEqual(recall_at_k(engine, self.korean()), 1.0)
        self.assertEqual(recall_at_k(engine, self.english()), 1.0)

    def test_character_ngrams_also_close_it(self):
        self.assertEqual(recall_at_k(_ENGINES["char2"], self.korean()), 1.0)

    def test_boundary_marker_costs_recall(self):
        self.assertLess(
            recall_at_k(_ENGINES["subword(+</w>)"], EVALUATION),
            recall_at_k(_ENGINES["subword"], EVALUATION),
        )

    def test_english_never_suffers_from_the_change(self):
        # Whatever helps Korean must not cost English.
        for name in ("word", "subword", "char2"):
            self.assertEqual(recall_at_k(_ENGINES[name], self.english()), 1.0, name)

    def test_char_ngrams_cost_vocabulary_size(self):
        # They work, but at 2x the terms of subwords.
        self.assertGreater(
            len(_ENGINES["char2"].vocabulary),
            2 * len(_ENGINES["subword"].vocabulary) // 3,
        )

    def test_empty_evaluation_rejected(self):
        with self.assertRaises(ValueError):
            recall_at_k(_ENGINES["word"], [])


class TestNoOutOfVocabularyHasACost(unittest.TestCase):
    """Day 12 sold this as a pure win. It is a trade."""

    def test_word_level_can_say_not_found(self):
        self.assertEqual(_ENGINES["word"].search("zzzqqq"), [])

    def test_subwords_cannot(self):
        self.assertNotEqual(_ENGINES["subword"].search("zzzqqq"), [])

    def test_the_match_is_on_a_single_character(self):
        shared = _ENGINES["subword"].shared_terms("zzzqqq", 4)
        self.assertTrue(shared)
        self.assertTrue(all(len(term) == 1 for term in shared))

    def test_char_ngrams_are_less_affected_here(self):
        # Bigrams of "zzzqqq" happen not to occur in the collection, but
        # that is luck rather than a guarantee: a shorter n would match.
        self.assertEqual(_ENGINES["char2"].search("zzzqqq"), [])

    def test_the_spurious_score_is_small(self):
        # Low, but not zero - so a threshold is needed where word-level
        # retrieval needed none.
        results = _ENGINES["subword"].search("zzzqqq")
        self.assertLess(results[0][1], 0.2)


class TestCli(unittest.TestCase):
    def run_cli(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = main(argv)
        return code, buffer.getvalue()

    def test_search_finds_a_korean_stem(self):
        code, output = self.run_cli(["search", "문서"])
        self.assertEqual(code, 0)
        self.assertIn("검색 엔진은", output)

    def test_search_explain_shows_the_matched_terms(self):
        _, output = self.run_cli(["search", "문서", "--explain"])
        self.assertIn("matched on", output)

    def test_search_reports_a_miss_without_crashing(self):
        # The word analyzer is the one that can actually miss.
        code, output = self.run_cli(["search", "zzzqqq", "-a", "word"])
        self.assertEqual(code, 0)
        self.assertIn("no match", output)

    def test_search_rejects_an_unknown_analyzer(self):
        code, output = self.run_cli(["search", "문서", "-a", "magic"])
        self.assertEqual(code, 2)
        self.assertIn("unknown analyzer", output)

    def test_compare_lists_every_analyzer(self):
        _, output = self.run_cli(["compare", "문서"])
        for name in build_analyzers():
            self.assertIn(name, output)

    def test_tokenize_shows_every_analyzer(self):
        _, output = self.run_cli(["tokenize", "고양이가 잔다"])
        self.assertIn("고양이", output)

    def test_evaluate_runs(self):
        code, output = self.run_cli(["evaluate"])
        self.assertEqual(code, 0)
        self.assertIn("recall@3", output)

    def test_parser_requires_a_command(self):
        # argparse writes its usage message to stderr on the way out.
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args([])


class TestReusedPrimitives(unittest.TestCase):
    """Spot checks that the copied-forward helpers still behave."""

    def test_normalize(self):
        self.assertEqual(normalize("The Cat, THE cat!"), ["the", "cat", "the", "cat"])

    def test_char_ngrams_window_longer_than_text(self):
        self.assertEqual(char_ngrams("ab", 5), [])

    def test_char_ngrams_reject_bad_n(self):
        with self.assertRaises(ValueError):
            char_ngrams("abc", 0)


if __name__ == "__main__":
    unittest.main()
