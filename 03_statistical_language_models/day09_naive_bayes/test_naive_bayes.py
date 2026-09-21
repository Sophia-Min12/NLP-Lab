"""Day 9 tests — counting, log-space scoring, smoothing, and the failure mode.

Korean strings below are functional test data, not documentation. Runnable
via `pytest` (repo root) or `python -m unittest` (this folder).
"""

import math
import unittest

from naive_bayes import (
    TEST_DATA,
    TRAINING_DATA,
    NaiveBayesClassifier,
    accuracy,
    confusion_matrix,
    normalize,
    with_bigrams,
)


def _trained(alpha=1.0, features=lambda d: d):
    documents = [features(normalize(text)) for text, _ in TRAINING_DATA]
    labels = [label for _, label in TRAINING_DATA]
    classifier = NaiveBayesClassifier(alpha=alpha)
    classifier.fit(documents, labels)
    return classifier


class TestFit(unittest.TestCase):
    def test_classes_are_sorted(self):
        classifier = NaiveBayesClassifier()
        classifier.fit([["a"], ["b"], ["c"]], ["z", "a", "m"])
        self.assertEqual(classifier.classes, ["a", "m", "z"])

    def test_priors_reflect_class_balance(self):
        classifier = NaiveBayesClassifier()
        classifier.fit([["a"], ["a"], ["b"]], ["x", "x", "y"])
        self.assertAlmostEqual(math.exp(classifier.log_prior["x"]), 2 / 3)
        self.assertAlmostEqual(math.exp(classifier.log_prior["y"]), 1 / 3)

    def test_priors_are_a_distribution(self):
        classifier = _trained()
        total = sum(math.exp(lp) for lp in classifier.log_prior.values())
        self.assertAlmostEqual(total, 1.0)

    def test_vocabulary_is_the_union_of_documents(self):
        classifier = NaiveBayesClassifier()
        classifier.fit([["a", "b"], ["b", "c"]], ["x", "y"])
        self.assertEqual(classifier.vocabulary, {"a", "b", "c"})

    def test_not_fitted_before_fit(self):
        self.assertFalse(NaiveBayesClassifier().fitted)

    def test_predicting_before_fitting_raises(self):
        with self.assertRaises(RuntimeError):
            NaiveBayesClassifier().predict(["a"])

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            NaiveBayesClassifier().fit([["a"], ["b"]], ["x"])

    def test_empty_collection_rejected(self):
        with self.assertRaises(ValueError):
            NaiveBayesClassifier().fit([], [])

    def test_str_document_rejected(self):
        with self.assertRaises(TypeError):
            NaiveBayesClassifier().fit(["a document"], ["x"])

    def test_refitting_replaces_the_old_model(self):
        classifier = NaiveBayesClassifier()
        classifier.fit([["a"]], ["x"])
        classifier.fit([["b"]], ["y"])
        self.assertEqual(classifier.classes, ["y"])
        self.assertEqual(classifier.vocabulary, {"b"})


class TestAlpha(unittest.TestCase):
    def test_zero_alpha_rejected(self):
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(alpha=0)

    def test_negative_alpha_rejected(self):
        with self.assertRaises(ValueError):
            NaiveBayesClassifier(alpha=-1)

    def test_smoothing_keeps_unseen_class_term_combinations_finite(self):
        # "great" never occurs with "neg"; without smoothing that term
        # would send the class to negative infinity.
        classifier = NaiveBayesClassifier()
        classifier.fit([["great"], ["awful"]], ["pos", "neg"])
        scores = classifier.predict_log_scores(["great"])
        for label, score in scores.items():
            self.assertTrue(math.isfinite(score), label)

    def test_larger_alpha_flattens_the_decision(self):
        # More smoothing pulls the classes together.
        sharp = _trained(alpha=0.01).predict_proba(normalize("excellent superb"))
        flat = _trained(alpha=50.0).predict_proba(normalize("excellent superb"))
        self.assertGreater(max(sharp.values()), max(flat.values()))


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.classifier = _trained()

    def test_scores_are_negative_log_values(self):
        for score in self.classifier.predict_log_scores(normalize("excellent film")).values():
            self.assertLess(score, 0.0)

    def test_probabilities_sum_to_one(self):
        probabilities = self.classifier.predict_proba(normalize("a brilliant film"))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_probabilities_are_in_range(self):
        for p in self.classifier.predict_proba(normalize("a dreadful film")).values():
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_predict_agrees_with_the_highest_probability(self):
        tokens = normalize("a brilliant and moving film")
        best = max(self.classifier.predict_proba(tokens).items(), key=lambda kv: kv[1])[0]
        self.assertEqual(self.classifier.predict(tokens), best)

    def test_unknown_terms_are_skipped_not_scored(self):
        # An unseen term carries identical evidence for every class, so it
        # must not change the decision.
        plain = self.classifier.predict_log_scores(normalize("excellent"))
        padded = self.classifier.predict_log_scores(normalize("excellent zzzqqq"))
        self.assertEqual(plain, padded)

    def test_empty_document_falls_back_to_the_prior(self):
        scores = self.classifier.predict_log_scores([])
        self.assertEqual(scores, self.classifier.log_prior)

    def test_all_unknown_document_falls_back_to_the_prior(self):
        scores = self.classifier.predict_log_scores(["zzzqqq", "wwwxxx"])
        self.assertEqual(scores, self.classifier.log_prior)

    def test_long_document_does_not_underflow(self):
        # 500 terms would drive a plain product to exactly 0.0.
        tokens = normalize("excellent superb brilliant ") * 500
        scores = self.classifier.predict_log_scores(tokens)
        self.assertTrue(all(math.isfinite(s) for s in scores.values()))
        probabilities = self.classifier.predict_proba(tokens)
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_rejects_a_str_document(self):
        with self.assertRaises(TypeError):
            self.classifier.predict("excellent")


class TestClassification(unittest.TestCase):
    def setUp(self):
        self.classifier = _trained()

    def test_clear_sentiment_is_classified_correctly(self):
        self.assertEqual(self.classifier.predict(normalize("excellent superb brilliant")), "positive")
        self.assertEqual(self.classifier.predict(normalize("awful dreadful terrible")), "negative")

    def test_korean_sentiment(self):
        self.assertEqual(self.classifier.predict(normalize("훌륭하고 아름다운")), "positive")
        self.assertEqual(self.classifier.predict(normalize("지루하고 형편없는")), "negative")

    def test_beats_chance_on_held_out_data(self):
        documents = [normalize(text) for text, _ in TEST_DATA]
        labels = [label for _, label in TEST_DATA]
        predictions = [self.classifier.predict(d) for d in documents]
        self.assertGreater(accuracy(predictions, labels), 0.5)


class TestTopFeatures(unittest.TestCase):
    def setUp(self):
        self.classifier = _trained()

    def test_returns_k_terms(self):
        self.assertEqual(len(self.classifier.top_features("positive", 3)), 3)

    def test_sorted_by_log_ratio(self):
        ratios = [r for _, r in self.classifier.top_features("negative", 5)]
        self.assertEqual(ratios, sorted(ratios, reverse=True))

    def test_sentiment_words_surface(self):
        positive = [t for t, _ in self.classifier.top_features("positive", 8)]
        negative = [t for t, _ in self.classifier.top_features("negative", 8)]
        self.assertTrue({"brilliant", "beautiful"} & set(positive))
        self.assertTrue({"awful", "dreadful"} & set(negative))

    def test_unknown_class_rejected(self):
        with self.assertRaises(KeyError):
            self.classifier.top_features("neutral")

    def test_negative_k_rejected(self):
        with self.assertRaises(ValueError):
            self.classifier.top_features("positive", -1)

    def test_unfitted_rejected(self):
        with self.assertRaises(RuntimeError):
            NaiveBayesClassifier().top_features("x")


class TestMetrics(unittest.TestCase):
    def test_accuracy(self):
        self.assertEqual(accuracy(["a", "b"], ["a", "a"]), 0.5)

    def test_perfect_and_zero_accuracy(self):
        self.assertEqual(accuracy(["a"], ["a"]), 1.0)
        self.assertEqual(accuracy(["a"], ["b"]), 0.0)

    def test_empty_accuracy_is_zero_not_a_division_error(self):
        self.assertEqual(accuracy([], []), 0.0)

    def test_accuracy_length_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            accuracy(["a"], ["a", "b"])

    def test_confusion_matrix_counts(self):
        matrix = confusion_matrix(["a", "b"], ["a", "a"], ["a", "b"])
        self.assertEqual(matrix, {"a": {"a": 1, "b": 1}, "b": {"a": 0, "b": 0}})

    def test_confusion_diagonal_sums_to_correct_predictions(self):
        predicted = ["a", "b", "a"]
        actual = ["a", "b", "b"]
        matrix = confusion_matrix(predicted, actual, ["a", "b"])
        diagonal = sum(matrix[c][c] for c in ("a", "b"))
        self.assertEqual(diagonal, 2)

    def test_confusion_total_equals_sample_count(self):
        matrix = confusion_matrix(["a", "b", "a"], ["a", "a", "b"], ["a", "b"])
        total = sum(v for row in matrix.values() for v in row.values())
        self.assertEqual(total, 3)


class TestNegationDefeatsUnigrams(unittest.TestCase):
    """The independence assumption, failing in the way it always fails."""

    def setUp(self):
        self.unigram = _trained()
        self.bigram = _trained(features=with_bigrams)

    def test_not_is_uninformative_on_its_own(self):
        # Balanced across classes by construction, so it carries no signal.
        self.assertEqual(
            self.unigram._counts["positive"].get("not"),
            self.unigram._counts["negative"].get("not"),
        )

    def test_good_leans_positive_on_its_own(self):
        self.assertGreater(
            self.unigram._counts["positive"].get("good", 0),
            self.unigram._counts["negative"].get("good", 0),
        )

    def test_unigrams_get_the_negation_wrong(self):
        self.assertEqual(self.unigram.predict(normalize("good")), "positive")
        self.assertEqual(self.unigram.predict(normalize("not good")), "positive")

    def test_bigram_features_fix_it(self):
        self.assertEqual(self.bigram.predict(with_bigrams(normalize("good"))), "positive")
        self.assertEqual(self.bigram.predict(with_bigrams(normalize("not good"))), "negative")

    def test_with_bigrams_keeps_the_unigrams(self):
        self.assertEqual(with_bigrams(["not", "good"]), ["not", "good", "not_good"])

    def test_with_bigrams_on_a_single_token(self):
        self.assertEqual(with_bigrams(["alone"]), ["alone"])

    def test_with_bigrams_rejects_a_str(self):
        with self.assertRaises(TypeError):
            with_bigrams("not good")


if __name__ == "__main__":
    unittest.main()
