"""Day 9 — Naive Bayes text classifier.

Level 2 ranked documents against a query. This day assigns them a *label*,
using the first genuinely probabilistic model in the curriculum.

Bayes' rule says ``P(class | document) ∝ P(class) · P(document | class)``.
The second factor is hopeless as written — no collection contains enough
examples of a whole document to estimate it — so Naive Bayes assumes each
term is independent of the others given the class, turning it into a
product over terms. That assumption is **false**: words in real sentences
are heavily dependent. The model works anyway, because ranking classes
correctly is a much weaker requirement than estimating probabilities
correctly.

Two implementation facts carry most of the weight:

- **Everything happens in log space.** Multiplying a hundred probabilities
  underflows float64 to exactly 0.0, and 0.0 for every class means the
  argmax is whichever class happened to be checked first. Sums of logs do
  not underflow.
- **Unseen terms need smoothing.** One term never seen with a class sends
  that class's probability to zero and no amount of other evidence can
  recover it. Add-alpha smoothing is the fix, and Day 10 examines what it
  costs.
"""

from __future__ import annotations

import math
import re
import sys
import unicodedata


# reused from day01
def _require_str(text) -> None:
    """Shared type guard so every function enforces the same contract."""
    if not isinstance(text, str):
        raise TypeError(f"expected str, got {type(text).__name__}")


# reused from day03
def _require_tokens(tokens) -> None:
    """Reject a bare string where a token list is expected."""
    if isinstance(tokens, str):
        raise TypeError("expected a list of tokens, got str")


# reused from day01
def tokenize_words(text: str) -> list[str]:
    """Word-level regex tokenization: words group, punctuation separates."""
    _require_str(text)
    text = text.replace("’", "'")  # curly apostrophe -> ASCII
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)


# reused from day03 onward
def normalize(text: str) -> list[str]:
    """NFC, casefold, tokenize, drop standalone punctuation."""
    _require_str(text)
    text = unicodedata.normalize("NFC", text)
    tokens = tokenize_words(text.casefold())
    return [t for t in tokens if any(ch.isalnum() for ch in t)]


# reused from day04
def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    """Every window of n adjacent tokens, as tuples."""
    _require_tokens(tokens)
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    tokens = list(tokens)
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def with_bigrams(tokens: list[str]) -> list[str]:
    """Unigrams plus joined bigrams, as a single feature list.

    Day 4's repair for the bag-of-words order problem, in the form a
    classifier can use: ``not_good`` becomes its own feature, so the model
    can learn that it means something different from ``not`` and ``good``
    apart.

    >>> with_bigrams(["not", "good"])
    ['not', 'good', 'not_good']
    """
    _require_tokens(tokens)
    return list(tokens) + [f"{a}_{b}" for a, b in ngrams(tokens, 2)]


class NaiveBayesClassifier:
    """Multinomial Naive Bayes over token counts.

    ``alpha`` is the smoothing constant added to every term count. It must
    be positive: at ``alpha=0`` a single unseen term drives a class's log
    probability to negative infinity, which is exactly the failure the
    smoothing exists to prevent.

    >>> clf = NaiveBayesClassifier()
    >>> clf.fit([["good", "great"], ["bad", "awful"]], ["pos", "neg"])
    >>> clf.predict(["good"])
    'pos'
    >>> clf.predict(["awful"])
    'neg'
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.alpha = alpha
        self.classes: list[str] = []
        self.log_prior: dict[str, float] = {}
        self.vocabulary: set[str] = set()
        self._counts: dict[str, dict[str, int]] = {}
        self._totals: dict[str, int] = {}

    @property
    def fitted(self) -> bool:
        return bool(self.classes)

    def fit(self, documents: list[list[str]], labels: list[str]) -> None:
        """Estimate class priors and per-class term counts.

        Both come straight from counting — there is no iterative training
        step. That is why Naive Bayes remains a sensible baseline: it costs
        one pass over the data.
        """
        if isinstance(documents, str):
            raise TypeError("expected a list of documents, got str")
        if len(documents) != len(labels):
            raise ValueError(
                f"{len(documents)} documents but {len(labels)} labels; they must match"
            )
        if not documents:
            raise ValueError("cannot fit on an empty collection")

        self.classes = sorted(set(labels))
        self.vocabulary = set()
        self._counts = {label: {} for label in self.classes}
        self._totals = {label: 0 for label in self.classes}
        document_counts = {label: 0 for label in self.classes}

        for document, label in zip(documents, labels):
            if isinstance(document, str):
                raise TypeError("documents must be token lists, not str")
            document_counts[label] += 1
            counts = self._counts[label]
            for term in document:
                counts[term] = counts.get(term, 0) + 1
                self._totals[label] += 1
                self.vocabulary.add(term)

        total = len(documents)
        self.log_prior = {
            label: math.log(document_counts[label] / total) for label in self.classes
        }

    def _log_likelihood(self, term: str, label: str) -> float:
        """log P(term | class), add-alpha smoothed over the vocabulary."""
        numerator = self._counts[label].get(term, 0) + self.alpha
        denominator = self._totals[label] + self.alpha * len(self.vocabulary)
        return math.log(numerator / denominator)

    def predict_log_scores(self, tokens: list[str]) -> dict[str, float]:
        """Unnormalized log P(class | document), one entry per class.

        Terms outside the training vocabulary are **skipped**, not smoothed.
        A term never seen in training carries the same evidence for every
        class, so including it would add an identical constant to each score
        and change nothing except the arithmetic.
        """
        _require_tokens(tokens)
        if not self.fitted:
            raise RuntimeError("classifier is not fitted; call fit() first")

        scores = {}
        for label in self.classes:
            score = self.log_prior[label]
            for term in tokens:
                if term in self.vocabulary:
                    score += self._log_likelihood(term, label)
            scores[label] = score
        return scores

    def predict_proba(self, tokens: list[str]) -> dict[str, float]:
        """Normalized class probabilities, summing to 1.

        Exponentiating is done after subtracting the maximum log score —
        the standard trick that keeps ``exp`` away from underflow while
        leaving the ratios untouched.
        """
        scores = self.predict_log_scores(tokens)
        highest = max(scores.values())
        weights = {label: math.exp(score - highest) for label, score in scores.items()}
        total = sum(weights.values())
        return {label: weight / total for label, weight in weights.items()}

    def predict(self, tokens: list[str]) -> str:
        """The highest-scoring class. Ties break alphabetically."""
        scores = self.predict_log_scores(tokens)
        return min(scores, key=lambda label: (-scores[label], label))

    def top_features(self, label: str, k: int = 5) -> list[tuple[str, float]]:
        """Terms that most favour ``label`` over the best alternative.

        The score is a log ratio, so it rewards terms that are frequent in
        this class *and* rare in the others — a term common everywhere
        scores near zero. Bare per-class frequency would just surface
        stopwords, the Day 3 problem again.
        """
        if not self.fitted:
            raise RuntimeError("classifier is not fitted; call fit() first")
        if label not in self.classes:
            raise KeyError(f"unknown class {label!r}; known: {self.classes}")
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        others = [c for c in self.classes if c != label]
        ratios = []
        for term in self.vocabulary:
            mine = self._log_likelihood(term, label)
            best_other = max((self._log_likelihood(term, c) for c in others), default=0.0)
            ratios.append((term, mine - best_other))
        return sorted(ratios, key=lambda pair: (-pair[1], pair[0]))[:k]


def accuracy(predicted: list[str], actual: list[str]) -> float:
    """Share of predictions that are correct; 0.0 on empty input.

    >>> accuracy(["a", "b"], ["a", "a"])
    0.5
    """
    if len(predicted) != len(actual):
        raise ValueError(f"{len(predicted)} predictions but {len(actual)} labels")
    if not predicted:
        return 0.0
    return sum(1 for p, a in zip(predicted, actual) if p == a) / len(predicted)


def confusion_matrix(
    predicted: list[str],
    actual: list[str],
    classes: list[str],
) -> dict[str, dict[str, int]]:
    """``matrix[actual][predicted]`` counts.

    Accuracy alone hides which way the errors go, and with unbalanced
    classes it can look respectable while the model never predicts the
    minority class at all.

    >>> confusion_matrix(["a", "b"], ["a", "a"], ["a", "b"])
    {'a': {'a': 1, 'b': 1}, 'b': {'a': 0, 'b': 0}}
    """
    if len(predicted) != len(actual):
        raise ValueError(f"{len(predicted)} predictions but {len(actual)} labels")
    matrix = {row: {column: 0 for column in classes} for row in classes}
    for p, a in zip(predicted, actual):
        matrix[a][p] += 1
    return matrix


#: Short bilingual sentiment reviews. Deliberately small and deliberately
#: including negations, which is where the independence assumption shows.
TRAINING_DATA = [
    ("The movie was excellent and the acting was superb", "positive"),
    ("A wonderful film with a brilliant script", "positive"),
    ("I loved every minute of this delightful story", "positive"),
    ("Great performances and a beautiful ending", "positive"),
    ("A good film, good acting, a good script and a good ending", "positive"),
    ("The best film I have seen this year and really good fun", "positive"),
    ("Charming, funny and clever, not boring", "positive"),
    ("The story was not predictable, not dull and not tedious", "positive"),
    ("이 영화는 정말 훌륭하고 재미있었다", "positive"),
    ("배우들의 연기가 뛰어나고 아름다운 영화였다", "positive"),
    ("The movie was terrible and the acting was awful", "negative"),
    ("A dreadful film with a boring script", "negative"),
    ("I hated every minute of this tedious story", "negative"),
    ("Weak performances and a disappointing ending", "negative"),
    ("The film was not good and not clever", "negative"),
    ("The worst film I have seen this year", "negative"),
    ("Dreary, humourless and utterly predictable", "negative"),
    ("It was not good, just not good at all", "negative"),
    ("이 영화는 정말 지루하고 실망스러웠다", "negative"),
    ("배우들의 연기가 어색하고 형편없는 영화였다", "negative"),
]

TEST_DATA = [
    ("A brilliant and moving film", "positive"),
    ("The script was dreadful and boring", "negative"),
    ("Excellent acting and a clever story", "positive"),
    ("A dull, tedious and weak piece of work", "negative"),
    ("이 영화는 훌륭하고 아름다웠다", "positive"),
    ("이 영화는 지루하고 형편없었다", "negative"),
]


if __name__ == "__main__":
    # Windows consoles default to cp949/cp1252 and choke on accents and Hangul.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):  # pragma: no cover - platform dependent
        pass

    train_documents = [normalize(text) for text, _ in TRAINING_DATA]
    train_labels = [label for _, label in TRAINING_DATA]
    test_documents = [normalize(text) for text, _ in TEST_DATA]
    test_labels = [label for _, label in TEST_DATA]

    classifier = NaiveBayesClassifier()
    classifier.fit(train_documents, train_labels)

    print(f"trained on {len(train_documents)} reviews, "
          f"{len(classifier.vocabulary)} distinct terms, "
          f"classes {classifier.classes}\n")

    predictions = [classifier.predict(document) for document in test_documents]
    print(f"held-out accuracy: {accuracy(predictions, test_labels):.0%}")
    for (text, actual), predicted in zip(TEST_DATA, predictions):
        mark = "ok " if predicted == actual else "MISS"
        probability = classifier.predict_proba(normalize(text))[predicted]
        print(f"  {mark} {predicted:<8} p={probability:.2f}  {text}")

    print("\nconfusion matrix (rows = actual, columns = predicted):")
    matrix = confusion_matrix(predictions, test_labels, classifier.classes)
    print(f"  {'':<10}" + "".join(f"{c:>10}" for c in classifier.classes))
    for actual in classifier.classes:
        print(f"  {actual:<10}" + "".join(f"{matrix[actual][p]:>10}" for p in classifier.classes))

    print("\nmost discriminative terms:")
    for label in classifier.classes:
        picked = ", ".join(term for term, _ in classifier.top_features(label, 5))
        print(f"  {label:<9}: {picked}")

    print("\nwhy log space - multiply 0.001 by itself until float64 gives up:")
    product, steps = 1.0, 0
    while product > 0:
        product *= 0.001
        steps += 1
    print(f"  after {steps} terms the product is exactly 0.0")
    print("  every class would then score 0.0, and the argmax is whichever was checked first")
    print(f"  the same {steps} terms summed in log space -> {steps * math.log(0.001):.1f}  (no trouble)")

    bigram_classifier = NaiveBayesClassifier()
    bigram_classifier.fit([with_bigrams(d) for d in train_documents], train_labels)

    positive_not = classifier._counts["positive"].get("not", 0)
    negative_not = classifier._counts["negative"].get("not", 0)
    positive_good = classifier._counts["positive"].get("good", 0)
    negative_good = classifier._counts["negative"].get("good", 0)

    print("\nthe independence assumption meets negation:")
    print(f"  'not'  appears {positive_not}x positive / {negative_not}x negative "
          "-> on its own it says nothing")
    print(f"  'good' appears {positive_good}x positive / {negative_good}x negative "
          "-> on its own it leans positive")
    print(f"  {'':<12}{'unigram':>12}{'bigram':>12}")
    for text in ("good", "not good"):
        tokens = normalize(text)
        print(f"  {text!r:<12}{classifier.predict(tokens):>12}"
              f"{bigram_classifier.predict(with_bigrams(tokens)):>12}")
    print("  unigrams cannot represent the flip - 'not' and 'good' are independent")
    print("  given the class, so their order is invisible. 'not_good' is a feature that is not.")
