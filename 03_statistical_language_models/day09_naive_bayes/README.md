# Day 9 · Naive Bayes Text Classifier

> Level 2 ranked documents against a query. This day assigns them a **label**, with the first genuinely probabilistic model in the curriculum.

Bayes' rule gives `P(class | document) ∝ P(class) · P(document | class)`. The second factor is hopeless as written — no collection holds enough examples of a whole document to estimate it — so Naive Bayes assumes each term is **independent of the others given the class**, turning it into a product over terms.

That assumption is flatly false. Words in real sentences depend on each other heavily. The model works anyway, because choosing the highest-scoring class is a far weaker requirement than estimating the probabilities correctly.

## The functions

| Piece | What it does |
|---|---|
| `NaiveBayesClassifier(alpha)` | multinomial NB with add-alpha smoothing |
| `.fit(documents, labels)` | count priors and per-class term frequencies |
| `.predict(tokens)` | the highest-scoring class |
| `.predict_proba(tokens)` | normalized class probabilities |
| `.top_features(label, k)` | the terms that most favour a class |
| `accuracy` / `confusion_matrix` | evaluation |
| `with_bigrams(tokens)` | unigrams + joined bigrams as features |

## Everything happens in log space

Not as a style preference — as a correctness requirement. The demo measures the failure instead of asserting it:

```
multiply 0.001 by itself until float64 gives up:
  after 108 terms the product is exactly 0.0
  the same 108 terms summed in log space -> -746.0  (no trouble)
```

Once the product underflows, **every class scores 0.0**, and the argmax silently becomes whichever class happened to be checked first. A 108-term document is a short news article. Sums of logs never underflow, and a test pushes a 1500-token document through to confirm the scores stay finite.

`predict_proba` then exponentiates only after subtracting the maximum log score — the standard trick that keeps `exp` in range while leaving the ratios untouched.

## Smoothing, and what gets skipped

`alpha` must be **positive**. At `alpha=0`, one term never seen with a class drives that class to probability zero, and no amount of other evidence recovers it.

Terms outside the training vocabulary are **skipped rather than smoothed**. An unseen term carries identical evidence for every class, so including it would add the same constant to each score and change nothing but the arithmetic. A test asserts that appending gibberish to a document leaves the scores bit-identical.

## Where it breaks: negation

This is the day's real content, and the training data was built deliberately so the failure is visible rather than accidental.

```
'not'  appears 4x positive / 4x negative -> on its own it says nothing
'good' appears 5x positive / 3x negative -> on its own it leans positive

                 unigram      bigram
'good'          positive    positive
'not good'      positive    negative
```

The unigram model gets `not good` **wrong**. It has to: `not` and `good` are independent given the class, so their *order* is invisible to the model. Seeing both is the same as seeing each separately, and `good` is the stronger cue.

`with_bigrams` makes `not_good` a feature in its own right, and the flip is captured. This is Day 4's n-gram repair applied to classification — the same fix, three levels on.

> The balance in `not` is engineered. With `not` leaning negative the unigram model would get `not good` right *for the wrong reason* — reading `not` as a negative word rather than as a negation. Constructing the data so the cue is genuinely uninformative is what makes the demonstration mean anything.

## The held-out miss is not noise

Accuracy is **83%**, five of six, and the one failure is worth more than the five successes:

```
MISS positive p=0.62  A dull, tedious and weak piece of work
```

That review is negative. It gets called positive because the training set contains *"The story was not predictable, not dull and not tedious"* — a **positive** review. Under unigram features that example teaches the model that `dull` and `tedious` occur in positive reviews, since the `not` in front of them is a separate, uninformative feature.

So the negation problem is not confined to the toy `not good` case. It quietly corrupts the feature weights of every negated word in the training set, and the damage shows up on unrelated examples later. Adding bigram features addresses the cause rather than the symptom.

## `top_features` uses a ratio, not a frequency

The score is `log P(term | this class) − log P(term | best other class)`. A term common in *every* class scores near zero and does not surface. Ranking by bare per-class frequency would just return `the`, `and`, `a` — Day 3's problem, arriving again.

On 20 training reviews the output is still a little noisy (`at`, `all` sneak in from *"not ... at all"*), which is what a 20-document training set looks like honestly reported.

## Korean

Classification works — both Korean test reviews are labelled correctly — but on the thinnest possible evidence. The training set has four Korean reviews, and because particles attach to stems (Day 5), `훌륭하고` and `훌륭한` would be entirely separate features. The model is matching whole eojeol it happened to see before, not learning Korean sentiment. **Day 12's BPE** is what would let these forms share evidence.

## Run it

```bash
# demo (accuracy, confusion matrix, top features, underflow, negation)
python naive_bayes.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 03_statistical_language_models/day09_naive_bayes

# tests — from inside this folder
python -m unittest
python test_naive_bayes.py

# the docstring examples are runnable too
python -m doctest naive_bayes.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Naive Bayes estimates `P(term | class)` and throws word order away. **Day 10** estimates `P(term | previous terms)` and keeps it, which is a language model rather than a classifier — and it runs straight into the zero-probability problem that add-alpha smoothing papers over here. **Day 11** then uses that model to generate text and to measure how surprised it is by text it has not seen.
