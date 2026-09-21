"""Day 10 tests — counting, the distribution property, zeros, and smoothing cost.

Korean strings below are functional test data, not documentation. Runnable
via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

from ngram_lm import (
    BOS,
    EOS,
    UNK,
    NgramModel,
    build_corpus,
    normalize,
    pad_sequence,
    replace_rare,
    split_corpus,
)


def _trained(n=2, alpha=1.0):
    train, test = split_corpus(build_corpus(), held_out=20)
    train = replace_rare(train, min_count=2)
    model = NgramModel(n, alpha=alpha)
    model.fit(train)
    return model, train, test


class TestReplaceRare(unittest.TestCase):
    def test_folds_below_the_threshold(self):
        self.assertEqual(replace_rare([["a", "a", "b"]], min_count=2), [["a", "a", "<unk>"]])

    def test_counts_across_sentences_not_within(self):
        self.assertEqual(replace_rare([["a"], ["a"]], min_count=2), [["a"], ["a"]])

    def test_min_count_one_changes_nothing(self):
        sentences = [["a", "b"], ["c"]]
        self.assertEqual(replace_rare(sentences, min_count=1), sentences)

    def test_empty_corpus(self):
        self.assertEqual(replace_rare([], min_count=2), [])

    def test_rejects_a_str_sentence(self):
        with self.assertRaises(TypeError):
            replace_rare(["a b"], min_count=2)


class TestSplitCorpus(unittest.TestCase):
    def test_sizes(self):
        train, test = split_corpus([[str(i)] for i in range(10)], held_out=2)
        self.assertEqual((len(train), len(test)), (8, 2))

    def test_partition_is_complete_and_disjoint(self):
        corpus = [[str(i)] for i in range(50)]
        train, test = split_corpus(corpus, held_out=10)
        self.assertEqual(len(train) + len(test), len(corpus))
        for sentence in test:
            self.assertNotIn(sentence, train)

    def test_deterministic(self):
        corpus = build_corpus()
        self.assertEqual(split_corpus(corpus, 20), split_corpus(corpus, 20))

    def test_held_out_drawn_from_across_the_corpus(self):
        # A tail slice would hand the model an unseen template block.
        corpus = [[str(i)] for i in range(100)]
        _, test = split_corpus(corpus, held_out=10)
        indices = [int(s[0]) for s in test]
        self.assertLess(min(indices), 20)
        self.assertGreater(max(indices), 80)

    def test_zero_held_out(self):
        train, test = split_corpus([["a"], ["b"]], held_out=0)
        self.assertEqual(len(train), 2)
        self.assertEqual(test, [])

    def test_held_out_too_large_rejected(self):
        with self.assertRaises(ValueError):
            split_corpus([["a"], ["b"]], held_out=2)

    def test_negative_held_out_rejected(self):
        with self.assertRaises(ValueError):
            split_corpus([["a"], ["b"]], held_out=-1)


class TestBuildCorpus(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(build_corpus(), build_corpus())

    def test_large_enough_to_model(self):
        corpus = build_corpus()
        self.assertGreater(sum(len(s) for s in corpus), 1000)

    def test_sentences_are_token_lists(self):
        for sentence in build_corpus():
            self.assertIsInstance(sentence, list)
            self.assertTrue(all(isinstance(t, str) for t in sentence))

    def test_contains_both_scripts(self):
        flat = [t for s in build_corpus() for t in s]
        self.assertIn("king", flat)
        self.assertTrue(any("가" <= c <= "힣" for t in flat for c in t))


class TestFit(unittest.TestCase):
    def test_counts_padded_ngrams(self):
        model = NgramModel(2)
        model.fit([["a", "b"]])
        self.assertEqual(model.ngram_counts[(BOS, "a")], 1)
        self.assertEqual(model.ngram_counts[("b", EOS)], 1)

    def test_eos_is_in_the_vocabulary(self):
        model = NgramModel(2)
        model.fit([["a"]])
        self.assertIn(EOS, model.vocabulary)

    def test_bos_is_not_in_the_vocabulary(self):
        # It is conditioned on, never generated.
        model = NgramModel(2)
        model.fit([["a"]])
        self.assertNotIn(BOS, model.vocabulary)

    def test_sentences_are_padded_separately(self):
        # Otherwise the end of one sentence would predict the start of the next.
        model = NgramModel(2)
        model.fit([["a"], ["b"]])
        self.assertNotIn(("a", "b"), model.ngram_counts)
        self.assertEqual(model.ngram_counts[(BOS, "a")], 1)
        self.assertEqual(model.ngram_counts[(BOS, "b")], 1)

    def test_not_fitted_before_fit(self):
        self.assertFalse(NgramModel(2).fitted)

    def test_probability_before_fitting_raises(self):
        with self.assertRaises(RuntimeError):
            NgramModel(2).probability("a")

    def test_bad_n_rejected(self):
        with self.assertRaises(ValueError):
            NgramModel(0)
        with self.assertRaises(TypeError):
            NgramModel(True)

    def test_negative_alpha_rejected(self):
        with self.assertRaises(ValueError):
            NgramModel(2, alpha=-0.5)

    def test_rejects_a_str_sentence(self):
        with self.assertRaises(TypeError):
            NgramModel(2).fit(["a b"])


class TestProbability(unittest.TestCase):
    def test_worked_example(self):
        model = NgramModel(2)
        model.fit([["the", "cat", "sat"], ["the", "cat", "ran"]])
        # count(the cat)=2, count(the)=2, |V|={the,cat,sat,ran,</s>}=5
        self.assertAlmostEqual(model.probability("cat", ("the",)), 3 / 7)

    def test_continuations_form_a_distribution(self):
        model, _, _ = _trained()
        for context in (("the",), ("king",), ("ate",)):
            total = sum(model.continuations(context).values())
            self.assertAlmostEqual(total, 1.0, places=9)

    def test_unigram_model_ignores_context(self):
        model = NgramModel(1)
        model.fit([["a", "b"]])
        self.assertEqual(model.probability("a"), model.probability("a", ("anything",)))

    def test_context_is_truncated_to_the_model_order(self):
        model, _, _ = _trained(n=2)
        short = model.probability("king", ("the",))
        long = model.probability("king", ("walked", "to", "the"))
        self.assertEqual(short, long)

    def test_mle_gives_zero_to_unseen_continuations(self):
        model = NgramModel(2, alpha=0.0)
        model.fit([["a", "b"]])
        self.assertEqual(model.probability("zzz", ("a",)), 0.0)

    def test_smoothing_gives_unseen_continuations_positive_mass(self):
        model = NgramModel(2, alpha=1.0)
        model.fit([["a", "b"]])
        self.assertGreater(model.probability("b", ("b",)), 0.0)

    def test_unseen_context_returns_uniform_under_smoothing(self):
        model = NgramModel(2, alpha=1.0)
        model.fit([["a", "b"]])
        probabilities = {t: model.probability(t, ("zzz",)) for t in model.vocabulary}
        self.assertAlmostEqual(min(probabilities.values()), max(probabilities.values()))


class TestLogProbability(unittest.TestCase):
    def test_negative_and_finite_under_smoothing(self):
        model, _, test = _trained()
        for sentence in test[:5]:
            value = model.log_probability(sentence)
            self.assertTrue(math.isfinite(value))
            self.assertLess(value, 0.0)

    def test_mle_returns_negative_infinity_on_an_unseen_bigram(self):
        model = NgramModel(2, alpha=0.0)
        model.fit([["a", "b"]])
        self.assertEqual(model.log_probability(["b", "a"]), float("-inf"))

    def test_higher_order_models_hit_more_zeros(self):
        # Day 4's sparsity curve, arriving as impossibility.
        _, train, test = _trained()
        zeros = []
        for n in (2, 3, 4, 5):
            model = NgramModel(n, alpha=0.0)
            model.fit(train)
            zeros.append(sum(1 for s in test if model.log_probability(s) == float("-inf")))
        self.assertEqual(zeros, sorted(zeros))
        self.assertGreater(zeros[-1], zeros[0])

    def test_the_two_causes_of_a_zero_are_different(self):
        # A low-order model only fails on an unseen *word*; a high-order one
        # also fails on a known word in an unseen *sequence*.
        model = NgramModel(2, alpha=0.0)
        model.fit([["the", "cat", "sat"], ["the", "dog", "ran"]])
        self.assertEqual(model.log_probability(["the", "zebra"]), float("-inf"))

        trigram = NgramModel(3, alpha=0.0)
        trigram.fit([["the", "cat", "sat"], ["the", "dog", "ran"]])
        # Every word is known, but "cat ran" was never seen.
        self.assertEqual(trigram.log_probability(["the", "cat", "ran"]), float("-inf"))

    def test_smoothing_removes_every_zero(self):
        _, train, test = _trained()
        for n in (2, 3, 4, 5):
            model = NgramModel(n, alpha=1.0)
            model.fit(train)
            for sentence in test:
                self.assertTrue(math.isfinite(model.log_probability(sentence)), n)

    def test_rejects_a_str_sentence(self):
        model, _, _ = _trained()
        with self.assertRaises(TypeError):
            model.log_probability("the king")


class TestSmoothingCost(unittest.TestCase):
    def test_mass_splits_into_observed_and_unseen(self):
        model, _, _ = _trained()
        observed, unseen = model.smoothing_cost(("the",))
        self.assertAlmostEqual(observed + unseen, 1.0)

    def test_add_one_moves_a_lot_of_mass(self):
        # A context seen 34 times with one continuation still loses most of
        # its mass under add-one.
        model, _, _ = _trained(alpha=1.0)
        observed, unseen = model.smoothing_cost(("ate",))
        self.assertGreater(unseen, 0.4)

    def test_smaller_alpha_keeps_more_mass_on_the_data(self):
        masses = []
        for alpha in (1.0, 0.1, 0.01):
            model, _, _ = _trained(alpha=alpha)
            observed, _ = model.smoothing_cost(("ate",))
            masses.append(observed)
        self.assertEqual(masses, sorted(masses))

    def test_mle_keeps_all_of_it(self):
        model, _, _ = _trained(alpha=0.0)
        observed, unseen = model.smoothing_cost(("the",))
        self.assertAlmostEqual(observed, 1.0)
        self.assertAlmostEqual(unseen, 0.0)

    def test_contexts_with_many_continuations_lose_less(self):
        model, _, _ = _trained()
        wide, _ = model.smoothing_cost(("the",))
        narrow, _ = model.smoothing_cost(("ate",))
        self.assertGreater(wide, narrow)


class TestUnknownWords(unittest.TestCase):
    def test_an_unseen_word_folded_to_unk_is_scoreable(self):
        model, _, _ = _trained()
        sentence = normalize("the wizard ate the bread")
        folded = [t if t in model.vocabulary else UNK for t in sentence]
        self.assertIn(UNK, folded)
        self.assertTrue(math.isfinite(model.log_probability(folded)))

    def test_unk_is_in_the_vocabulary_after_folding(self):
        model, _, _ = _trained()
        self.assertIn(UNK, model.vocabulary)


class TestPadSequence(unittest.TestCase):
    def test_bigram_padding(self):
        self.assertEqual(pad_sequence(["a"], 2), [BOS, "a", EOS])

    def test_trigram_padding(self):
        self.assertEqual(pad_sequence(["a"], 3), [BOS, BOS, "a", EOS])

    def test_rejects_a_str(self):
        with self.assertRaises(TypeError):
            pad_sequence("a", 2)


if __name__ == "__main__":
    unittest.main()
