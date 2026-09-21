"""Day 11 tests — sampling knobs, reproducibility, and perplexity.

Korean strings below are functional test data, not documentation. Runnable
via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

from generation import (
    EOS,
    NgramModel,
    apply_temperature,
    build_corpus,
    generate,
    perplexity,
    replace_rare,
    split_corpus,
    top_k_filter,
)


def _trained(n=2, alpha=0.1):
    train, test = split_corpus(build_corpus(), held_out=20)
    train = replace_rare(train, min_count=2)
    model = NgramModel(n, alpha=alpha)
    model.fit(train)
    return model, train, test


class TestApplyTemperature(unittest.TestCase):
    def test_result_is_a_distribution(self):
        out = apply_temperature({"a": 0.6, "b": 0.4}, 0.5)
        self.assertAlmostEqual(sum(out.values()), 1.0)

    def test_temperature_one_is_a_no_op(self):
        source = {"a": 0.6, "b": 0.4}
        for token, p in apply_temperature(source, 1.0).items():
            self.assertAlmostEqual(p, source[token])

    def test_low_temperature_sharpens(self):
        source = {"a": 0.6, "b": 0.4}
        self.assertGreater(apply_temperature(source, 0.3)["a"], source["a"])

    def test_high_temperature_flattens(self):
        source = {"a": 0.6, "b": 0.4}
        self.assertLess(apply_temperature(source, 5.0)["a"], source["a"])

    def test_ordering_is_preserved(self):
        source = {"a": 0.5, "b": 0.3, "c": 0.2}
        for temperature in (0.2, 1.0, 4.0):
            out = apply_temperature(source, temperature)
            self.assertEqual(
                sorted(out, key=lambda t: -out[t]),
                sorted(source, key=lambda t: -source[t]),
            )

    def test_zero_temperature_rejected(self):
        with self.assertRaises(ValueError):
            apply_temperature({"a": 1.0}, 0)

    def test_negative_temperature_rejected(self):
        with self.assertRaises(ValueError):
            apply_temperature({"a": 1.0}, -1)


class TestTopKFilter(unittest.TestCase):
    def test_keeps_k_and_renormalizes(self):
        out = top_k_filter({"a": 0.5, "b": 0.3, "c": 0.2}, 2)
        self.assertEqual(sorted(out), ["a", "b"])
        self.assertAlmostEqual(sum(out.values()), 1.0)

    def test_k_one_is_the_argmax(self):
        self.assertEqual(top_k_filter({"a": 0.5, "b": 0.3}, 1), {"a": 1.0})

    def test_k_larger_than_the_distribution(self):
        source = {"a": 0.5, "b": 0.5}
        self.assertEqual(sorted(top_k_filter(source, 99)), ["a", "b"])

    def test_ties_break_deterministically(self):
        self.assertEqual(sorted(top_k_filter({"b": 0.5, "a": 0.5}, 1)), ["a"])

    def test_k_below_one_rejected(self):
        with self.assertRaises(ValueError):
            top_k_filter({"a": 1.0}, 0)


class TestGenerate(unittest.TestCase):
    def test_same_seed_gives_same_output(self):
        model, _, _ = _trained()
        self.assertEqual(generate(model, seed=42), generate(model, seed=42))

    def test_different_seeds_usually_differ(self):
        model, _, _ = _trained()
        outputs = {tuple(generate(model, max_tokens=12, seed=s)) for s in range(8)}
        self.assertGreater(len(outputs), 1)

    def test_respects_max_tokens(self):
        model, _, _ = _trained()
        self.assertLessEqual(len(generate(model, max_tokens=5, seed=1)), 5)

    def test_max_tokens_zero(self):
        model, _, _ = _trained()
        self.assertEqual(generate(model, max_tokens=0, seed=1), [])

    def test_end_marker_is_never_emitted(self):
        model, _, _ = _trained()
        for seed in range(6):
            self.assertNotIn(EOS, generate(model, max_tokens=20, seed=seed))

    def test_only_vocabulary_tokens_appear(self):
        model, _, _ = _trained()
        for token in generate(model, max_tokens=20, seed=2):
            self.assertIn(token, model.vocabulary)

    def test_greedy_needs_no_seed_and_is_stable(self):
        model, _, _ = _trained()
        self.assertEqual(generate(model, greedy=True), generate(model, greedy=True))

    def test_greedy_on_a_single_path_corpus(self):
        model = NgramModel(2, alpha=0.0)
        model.fit([["the", "cat", "sat"]])
        self.assertEqual(generate(model, greedy=True), ["the", "cat", "sat"])

    def test_top_k_one_matches_greedy(self):
        model, _, _ = _trained()
        self.assertEqual(
            generate(model, max_tokens=10, seed=1, top_k=1),
            generate(model, max_tokens=10, greedy=True),
        )

    def test_unfitted_model_rejected(self):
        with self.assertRaises(RuntimeError):
            generate(NgramModel(2))

    def test_negative_max_tokens_rejected(self):
        model, _, _ = _trained()
        with self.assertRaises(ValueError):
            generate(model, max_tokens=-1)


class TestSmoothingShapesGeneration(unittest.TestCase):
    """High-order models only look fluent when they are memorizing."""

    def test_unsmoothed_high_order_reproduces_training_data(self):
        model, train, _ = _trained(n=5, alpha=0.0)
        produced = generate(model, max_tokens=14, seed=7)
        self.assertIn(produced, train)

    def test_smoothed_high_order_wanders_off(self):
        model, train, _ = _trained(n=5, alpha=0.1)
        produced = generate(model, max_tokens=14, seed=7)
        self.assertNotIn(produced, train)

    def test_unsmoothed_output_only_uses_seen_transitions(self):
        model, train, _ = _trained(n=2, alpha=0.0)
        produced = generate(model, max_tokens=14, seed=3)
        for a, b in zip(produced, produced[1:]):
            self.assertIn((a, b), model.ngram_counts)


class TestPerplexity(unittest.TestCase):
    def test_worked_example(self):
        # Unigram, alpha=1, corpus ["a","b"]: V={a,b,</s>}, each token
        # gets (1+1)/(2+3) = 0.4 except </s> at (1+1)/(2+3); 3 predicted
        # tokens, so PP = (0.4^-3)^(1/3) = 2.5... check the real number.
        model = NgramModel(1, alpha=1.0)
        model.fit([["a", "b"]])
        self.assertAlmostEqual(perplexity(model, [["a", "b"]]), 3.0)

    def test_positive_and_finite_under_smoothing(self):
        model, _, test = _trained()
        value = perplexity(model, test)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 1.0)

    def test_infinite_when_the_model_calls_text_impossible(self):
        model = NgramModel(2, alpha=0.0)
        model.fit([["a", "b"]])
        self.assertEqual(perplexity(model, [["b", "a"]]), float("inf"))

    def test_a_confident_model_is_less_perplexed(self):
        # One sentence repeated: the model should barely be surprised.
        model = NgramModel(2, alpha=0.01)
        model.fit([["a", "b", "c"]] * 20)
        self.assertLess(perplexity(model, [["a", "b", "c"]]), 2.0)

    def test_counts_the_end_marker_and_not_the_padding(self):
        # Two sentences of 3 tokens => 8 predicted tokens, not 6 or 10.
        model = NgramModel(2, alpha=1.0)
        sentences = [["a", "b", "c"], ["a", "b", "c"]]
        model.fit(sentences)
        total_log = sum(model.log_probability(s) for s in sentences)
        expected = math.exp(-total_log / 8)
        self.assertAlmostEqual(perplexity(model, sentences), expected)

    def test_held_out_perplexity_exceeds_training_perplexity(self):
        for n in (2, 3, 4, 5):
            model, train, test = _trained(n=n)
            self.assertGreater(perplexity(model, test), perplexity(model, train), n)

    def test_the_generalization_gap_widens_with_order(self):
        # The memorization signal: the ratio climbs monotonically.
        ratios = []
        for n in (1, 2, 3, 4, 5):
            model, train, test = _trained(n=n)
            ratios.append(perplexity(model, test) / perplexity(model, train))
        self.assertEqual(ratios, sorted(ratios))
        self.assertGreater(ratios[-1], 1.5)

    def test_held_out_perplexity_is_not_monotonic_in_n(self):
        # It falls, then rises: more context is not always better.
        values = []
        for n in (1, 2, 3, 4, 5):
            model, _, test = _trained(n=n)
            values.append(perplexity(model, test))
        self.assertNotEqual(values, sorted(values))
        self.assertEqual(min(values), values[2])  # n=3 is the best

    def test_too_much_smoothing_hurts(self):
        heavy, _, test = _trained(alpha=1.0)
        light, _, _ = _trained(alpha=0.01)
        self.assertGreater(perplexity(heavy, test), perplexity(light, test))

    def test_empty_collection_rejected(self):
        model, _, _ = _trained()
        with self.assertRaises(ValueError):
            perplexity(model, [])

    def test_rejects_a_str(self):
        model, _, _ = _trained()
        with self.assertRaises(TypeError):
            perplexity(model, "a sentence")

    def test_unfitted_model_rejected(self):
        with self.assertRaises(RuntimeError):
            perplexity(NgramModel(2), [["a"]])


class TestPerplexityIsNotQuality(unittest.TestCase):
    """The metric's blind spot, stated as a test."""

    def test_a_memorizing_model_scores_well_on_what_it_memorized(self):
        # Perfect training perplexity, and it has learned nothing.
        model = NgramModel(5, alpha=0.0)
        sentences = [["the", "cat", "sat", "on", "the", "mat"]]
        model.fit(sentences)
        self.assertLess(perplexity(model, sentences), 1.5)
        # ...and is infinitely perplexed by one word changed.
        self.assertEqual(
            perplexity(model, [["the", "dog", "sat", "on", "the", "mat"]]),
            float("inf"),
        )


if __name__ == "__main__":
    unittest.main()
