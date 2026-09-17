# Day 6 · TF-IDF from Scratch

> Day 5's raw counts make `the` the loudest number in every document, which is exactly backwards: a term appearing everywhere cannot tell any two documents apart. TF-IDF is the standard repair, and it is two ideas multiplied together.

**TF** asks *how much is this document about the term*. **IDF** asks *how much does knowing the term narrow the collection down*. A term in every document narrows nothing, so its IDF is near zero, so the product collapses and the term drops out on its own.

## The functions

| Function | What it does |
|---|---|
| `term_frequency(counts, scheme)` | rescale one document's counts |
| `inverse_document_frequency(docs, vocab, scheme)` | one weight per term |
| `tfidf_vector(tokens, vocab, idf, scheme)` | one document, weighted |
| `tfidf_matrix(docs, vocab, ...)` | the collection — returns `(matrix, idf)` |
| `top_terms(vector, vocab, k)` | the heaviest terms in a vector |

## Four TF schemes

| Scheme | Formula | Says |
|---|---|---|
| `raw` | `count` | a 1000-word document outranks a 50-word one on everything |
| `relative` | `count / length` | **default** — removes the length bias |
| `log` | `1 + log(count)`, 0 at 0 | the 10th occurrence matters less than the 2nd |
| `boolean` | `1` or `0` | repetition says nothing (short titles) |

The `log` scheme's claim is about *marginal* gain, and the test says so precisely: `tf(2) − tf(1) > tf(10) − tf(9)`. Under `raw` those two increments are equal, which is the whole difference.

The choice is not cosmetic. Run the demo and the heaviest document changes with the scheme — document 4 under `raw`, document 0 under `relative` — because `raw` is measuring length and `relative` is measuring concentration.

## Three IDF schemes

| Scheme | Formula | Behaviour at the extremes |
|---|---|---|
| `smooth` | `log((1+N)/(1+df)) + 1` | **default**; unseen terms are safe, ubiquitous terms stay positive |
| `classic` | `log(N/df)` | textbook; df = N gives exactly **0**, df = 0 divides by zero |
| `probabilistic` | `log((N−df)/df)` | goes **negative** past half the collection |

The offsets in `smooth` pretend one extra document contains every term. That does two jobs: an unseen term cannot divide by zero — which matters because a *query* routinely contains terms the collection lacks (Day 7) — and the trailing `+ 1` keeps every weight strictly positive.

`classic` and `probabilistic` raise `ValueError` where they are undefined rather than returning a silently wrong number. Being unusable on an open vocabulary is a real property of those schemes, not a bug to paper over.

## Does TF-IDF retire the stopword list? Not at N=8

This is the claim everyone repeats, and on this collection it is **only half true** — worth reporting rather than smoothing over.

```
doc 4  smooth : ['the', 'add', 'butter']      ← 'the' is still the top term
       classic: ['add', 'butter', 'fry']
```

Two reasons, both structural:

1. **`smooth` cannot reach zero.** The `+ 1` floor guarantees a positive weight, so a ubiquitous term is *demoted* but never *erased*. `classic` can hit exactly 0 and does.
2. **Eight documents is not enough range.** The smooth IDF here spans only **1.41 to 2.50 — a ratio of 1.8×**. A term appearing 3 times in a short document easily wins on TF alone against that. With thousands of documents the spread widens by an order of magnitude and the effect people describe actually appears.

Both are pinned by tests. The honest statement: *TF-IDF demotes function words in proportion to how ubiquitous they are, and on a large collection that is enough to replace a stopword list — but "large" is doing real work in that sentence.*

## Why `tfidf_matrix` hands back the IDF

Because the weights are fitted on **this** collection and a query has to be weighted with the **same** ones to be comparable. Recomputing IDF from a one-document query would make every term equally rare and destroy the entire signal. Day 7 depends on this, and a test checks that the returned weights reproduce the matrix rows exactly.

This is the same discipline as Day 5's vocabulary: quantities derived from the collection are part of the shared contract, not per-document details.

## Why it works at all

Because of Day 3. If word frequencies were uniform, "rare" would carry no information and IDF would be a constant. They are violently non-uniform — that is Zipf's law — so document frequency is a genuinely strong signal about which terms are worth attending to. TF-IDF is Zipf's law turned into a weighting.

## Korean

Weighting is script-agnostic and works correctly here: the Korean documents get Korean top terms, pinned by a test. But it inherits Day 5's problem unchanged. Because `고양이가` and `고양이를` are separate terms, each carries its own document frequency, so the stem's evidence is **split across several IDF weights** instead of concentrating into one. Every particle-bearing form looks rarer than the stem actually is, which inflates its IDF — rare-looking for the wrong reason. Subword units (**Day 12**) are still the fix.

## Run it

```bash
# demo (idf extremes, raw vs weighted, scheme comparison, the N=8 caveat)
python tfidf.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_text_as_vectors/day06_tfidf

# tests — from inside this folder
python -m unittest
python test_tfidf.py

# the docstring examples are runnable too
python -m doctest tfidf.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Documents are now weighted vectors, and the obvious question is how to compare two of them. **Day 7** answers it with the angle between them rather than the distance, which finally makes document length irrelevant. **Day 8** keeps the same arithmetic but stops visiting documents that cannot possibly score above zero.
