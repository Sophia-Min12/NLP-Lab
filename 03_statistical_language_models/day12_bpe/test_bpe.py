"""Day 12 tests — merge mechanics, determinism, and the properties BPE buys.

Korean strings below are functional test data — they are the problem this
day exists to solve. Runnable via `pytest` (repo root) or
`python -m unittest` (this folder).
"""

import unittest

from bpe import (
    CORPUS,
    END,
    BPETokenizer,
    encode,
    encode_word,
    initial_splits,
    merge_symbols,
    normalize,
    pair_counts,
    train_bpe,
    word_counts,
)


class TestWordCounts(unittest.TestCase):
    def test_counts_across_documents(self):
        self.assertEqual(word_counts(["a cat", "a dog"]), {"a": 2, "cat": 1, "dog": 1})

    def test_normalization_is_applied(self):
        self.assertEqual(word_counts(["The THE the!"]), {"the": 3})

    def test_empty_corpus(self):
        self.assertEqual(word_counts([]), {})

    def test_rejects_a_bare_str(self):
        with self.assertRaises(TypeError):
            word_counts("a cat")


class TestInitialSplits(unittest.TestCase):
    def test_characters_plus_end_marker(self):
        self.assertEqual(initial_splits({"at": 2}), {("a", "t", END): 2})

    def test_weights_are_carried_through(self):
        self.assertEqual(initial_splits({"a": 7})[("a", END)], 7)

    def test_korean_splits_into_syllables(self):
        # NFC syllables, not jamo - Day 2's normalization is why.
        self.assertEqual(initial_splits({"고양이": 1}), {("고", "양", "이", END): 1})


class TestPairCounts(unittest.TestCase):
    def test_adjacent_pairs_weighted(self):
        self.assertEqual(pair_counts({("a", "t", END): 2}), {("a", "t"): 2, ("t", END): 2})

    def test_pairs_accumulate_across_words(self):
        counts = pair_counts({("a", "t", END): 2, ("c", "a", "t", END): 3})
        self.assertEqual(counts[("a", "t")], 5)

    def test_single_symbol_word_has_no_pairs(self):
        self.assertEqual(pair_counts({("a",): 1}), {})

    def test_empty_splits(self):
        self.assertEqual(pair_counts({}), {})


class TestMergeSymbols(unittest.TestCase):
    def test_joins_the_pair(self):
        self.assertEqual(merge_symbols(("a", "t", END), ("a", "t")), ("at", END))

    def test_merges_every_occurrence(self):
        self.assertEqual(merge_symbols(("a", "b", "a", "b"), ("a", "b")), ("ab", "ab"))

    def test_overlaps_are_taken_left_to_right(self):
        # "aaa" with (a, a) must be ("aa", "a"), never ("a", "aa").
        self.assertEqual(merge_symbols(("a", "a", "a"), ("a", "a")), ("aa", "a"))

    def test_absent_pair_changes_nothing(self):
        symbols = ("a", "t", END)
        self.assertEqual(merge_symbols(symbols, ("x", "y")), symbols)

    def test_empty_symbols(self):
        self.assertEqual(merge_symbols((), ("a", "b")), ())

    def test_bad_pair_rejected(self):
        with self.assertRaises(ValueError):
            merge_symbols(("a",), ("a", "b", "c"))


class TestTrainBpe(unittest.TestCase):
    def test_deterministic(self):
        first, _ = train_bpe(CORPUS, num_merges=30)
        second, _ = train_bpe(CORPUS, num_merges=30)
        self.assertEqual(first, second)

    def test_merges_the_most_frequent_pair_first(self):
        merges, _ = train_bpe(["low low lower"], num_merges=1)
        self.assertEqual(merges[0], ("l", "o"))

    def test_zero_merges_gives_the_character_vocabulary(self):
        merges, vocabulary = train_bpe(CORPUS, num_merges=0)
        self.assertEqual(merges, [])
        self.assertTrue(all(len(s) == 1 or s == END for s in vocabulary))

    def test_stops_early_when_no_pair_repeats(self):
        # Asking for far more merges than exist yields the same tokenizer.
        many, _ = train_bpe(CORPUS, num_merges=400)
        more, _ = train_bpe(CORPUS, num_merges=4000)
        self.assertEqual(many, more)
        self.assertLess(len(many), 400)

    def test_vocabulary_grows_by_one_per_merge(self):
        _, base = train_bpe(CORPUS, num_merges=0)
        merges, grown = train_bpe(CORPUS, num_merges=20)
        self.assertEqual(len(grown), len(base) + len(merges))

    def test_every_merged_symbol_is_in_the_vocabulary(self):
        merges, vocabulary = train_bpe(CORPUS, num_merges=40)
        for a, b in merges:
            self.assertIn(a + b, vocabulary)

    def test_negative_merges_rejected(self):
        with self.assertRaises(ValueError):
            train_bpe(CORPUS, num_merges=-1)


class TestEncodeWord(unittest.TestCase):
    def setUp(self):
        self.merges, _ = train_bpe(CORPUS, num_merges=60)

    def test_encoding_is_deterministic(self):
        self.assertEqual(encode_word("고양이가", self.merges), encode_word("고양이가", self.merges))

    def test_pieces_rejoin_to_the_original_word(self):
        for word in ("tokenizer", "고양이에게", "rhinoceros", "학생을"):
            joined = "".join(encode_word(word, self.merges)).replace(END, "")
            self.assertEqual(joined, word)

    def test_every_word_ends_with_the_end_marker(self):
        for word in ("token", "고양이", "zzz"):
            self.assertTrue("".join(encode_word(word, self.merges)).endswith(END))

    def test_no_merges_gives_characters(self):
        self.assertEqual(encode_word("at", []), ["a", "t", END])

    def test_empty_word_is_just_the_marker(self):
        self.assertEqual(encode_word("", self.merges), [END])

    def test_rejects_a_token_list(self):
        with self.assertRaises(TypeError):
            encode_word(["a"], self.merges)

    def test_encode_runs_over_a_whole_string(self):
        pieces = encode("the tokenizer", self.merges)
        self.assertEqual("".join(pieces).replace(END, " ").strip(), "the tokenizer")


class TestNoOutOfVocabulary(unittest.TestCase):
    """The property that retires Day 10's <unk>."""

    def setUp(self):
        self.tokenizer = BPETokenizer(CORPUS, num_merges=60)

    def test_an_entirely_unseen_word_still_encodes(self):
        for word in ("quixotic", "rhinoceros", "zzzzz", "고양이처럼"):
            pieces = self.tokenizer.encode_word(word)
            self.assertTrue(pieces)
            self.assertEqual("".join(pieces).replace(END, ""), word)

    def test_unseen_words_reuse_known_pieces(self):
        # "tokenizers" was never in the corpus; every piece of it was.
        pieces = self.tokenizer.encode_word("tokenizers")
        for piece in pieces:
            self.assertIn(piece, self.tokenizer.vocabulary)

    def test_worst_case_is_single_characters(self):
        pieces = self.tokenizer.encode_word("xqz")
        self.assertEqual(pieces, ["x", "q", "z", END])


class TestKoreanPayoff(unittest.TestCase):
    """The fix promised since Day 1, stated as assertions."""

    def setUp(self):
        self.tokenizer = BPETokenizer(CORPUS, num_merges=60)

    def test_the_stem_is_a_learned_symbol(self):
        self.assertIn("고양이", self.tokenizer.vocabulary)

    def test_every_inflected_form_starts_with_the_stem(self):
        for word in ("고양이가", "고양이를", "고양이에게", "고양이는"):
            self.assertEqual(self.tokenizer.encode_word(word)[0], "고양이")

    def test_different_particles_share_the_stem(self):
        self.assertEqual(self.tokenizer.shares_subword("고양이가", "고양이를"), {"고양이"})

    def test_word_level_tokenization_shares_nothing(self):
        # The state of affairs from Day 1 to Day 11, for contrast.
        self.assertEqual(set(normalize("고양이가")) & set(normalize("고양이를")), set())

    def test_other_stems_are_learned_too(self):
        for stem in ("강아지", "학생"):
            self.assertIn(stem, self.tokenizer.vocabulary)

    def test_english_suffixes_split_as_well(self):
        self.assertEqual(self.tokenizer.encode_word("tokenize"), ["token", "ize" + END])


class TestOverMerging(unittest.TestCase):
    """More merges is not better, and the failure is total."""

    def test_too_many_merges_rebuilds_whole_words(self):
        largest = BPETokenizer(CORPUS, num_merges=400)
        self.assertEqual(largest.encode_word("고양이가"), ["고양이가" + END])

    def test_and_therefore_shares_nothing_again(self):
        largest = BPETokenizer(CORPUS, num_merges=400)
        self.assertEqual(largest.shares_subword("고양이가", "고양이를"), set())

    def test_the_tuned_tokenizer_still_shares(self):
        tuned = BPETokenizer(CORPUS, num_merges=60)
        self.assertTrue(tuned.shares_subword("고양이가", "고양이를"))

    def test_pieces_per_word_falls_toward_one(self):
        words = sorted(word_counts(CORPUS))
        measured = [
            BPETokenizer(CORPUS, num_merges=m).pieces_per_word(words)
            for m in (0, 10, 30, 60)
        ]
        self.assertEqual(measured, sorted(measured, reverse=True))
        self.assertGreater(measured[0], 5.0)
        self.assertLess(measured[-1], 3.0)

    def test_pieces_per_word_of_an_empty_list(self):
        self.assertEqual(BPETokenizer(CORPUS, num_merges=10).pieces_per_word([]), 0.0)


if __name__ == "__main__":
    unittest.main()
