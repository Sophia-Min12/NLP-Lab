# Day 7 · Cosine Similarity and Document Ranking

> Days 5 and 6 turned documents into weighted vectors. Comparing them is now a geometry question — and the obvious answer is the wrong one.

## Why not distance

Euclidean distance asks *how far apart* two vectors are. A long document about soup and a short document about soup then look unrelated, because the long one simply has bigger numbers in every coordinate:

```
short vector      : [1.0, 1.0]
long vector       : [20.0, 20.0]     ← same words, 20× the length
euclidean distance: 26.870           ← looks unrelated
cosine similarity : 1.000            ← identical direction
```

Cosine asks about the **angle**, which is a question about proportions: what share of this document's weight sits on each term. Doubling a document's length leaves its direction untouched. That single property is why cosine is the default for text, and a test pins both halves of the contrast.

## The functions

| Function | What it does |
|---|---|
| `dot(a, b)` | sum of element-wise products |
| `norm(v)` | Euclidean length |
| `cosine_similarity(a, b)` | angle-based similarity, `0` … `1` |
| `euclidean_distance(a, b)` | kept for contrast, not used for ranking |
| `rank_documents(query, matrix, vocab, idf, k)` | score everything, sort, truncate |

## Three edge cases that are not edge cases

**A zero vector has no direction**, so `cosine_similarity` returns `0.0` rather than dividing by zero. This is not exotic — it is precisely what an all-out-of-vocabulary query looks like, and a search box gets those constantly.

**Vectors of different lengths cannot be compared at all.** They belong to different vocabularies, which means coordinate 7 means different things in each. `_require_same_length` raises rather than silently truncating with `zip`.

**Ties must break deterministically.** A query matching nothing scores every document 0.0, and those still have to come back in a stable order. `rank_documents` sorts on `(-score, document_id)`; a test checks that a nonsense query returns documents in id order rather than whatever the sort happened to do.

## Floating point: cosine is only approximately in [0, 1]

Parallel vectors come back as `0.9999999999999998`, and the division can occasionally land a hair *above* 1.0. Nothing here clamps the value — the formula stays a formula — so the test asserting the range carries a `1e-12` tolerance, and callers should compare with a tolerance rather than `== 1.0`. Worth knowing before it surfaces as a mysteriously failing assertion.

## The query must be weighted with the collection's IDF

`rank_documents` takes `idf` as an argument rather than computing it, because the weights have to be the ones that produced the matrix (Day 6). Fitting fresh IDF on a one-document query would make every term equally rare and erase the signal entirely.

Demonstrating this needs a carefully chosen query, and getting it wrong is instructive. A query like `search query documents` shows **nothing** — all three terms have df=1 here, so they receive equal weights, and equal weights leave the query's *direction* unchanged. Cosine ignores the query vector's scale, so the ranking is bit-for-bit identical.

The query that does show it is `the soup` — `the` has df=5, `soup` has df=1:

| | doc 3 (about soup) ÷ doc 4 (merely contains "the") |
|---|---|
| collection IDF | **3.57×** |
| flat weights | 2.38× |

IDF is what keeps a stopword in the query from dragging unrelated documents up the ranking. The test asserts that ratio relationship rather than mere inequality.

## The lexical ceiling

This is the honest limit of everything in Level 2:

```
'tokenization'      -> best 0.568 (doc 0)
'word segmentation' -> best 0.000
```

Those two queries mean the same thing. The second scores **exactly zero** against every document, because cosine over term counts can only reward terms that literally coincide. No amount of reweighting fixes it — the information simply is not in the representation.

That gap is the reason Level 4 exists. Embeddings place `tokenization` and `segmentation` near each other in a space where similarity is not spelling, and **Day 16's hybrid scorer** combines the two signals rather than choosing between them — lexical matching is precise when the words line up, and useless when they do not.

## Korean: half the query lands

The particle problem stops being abstract here and becomes a missed search result. Document 7 contains `문서의` and `문서를`, but not the bare `문서`:

```
query '검색 문서' -> doc 7, score 0.354
```

`검색` matches exactly. `문서` matches **nothing**, because the document only ever uses inflected forms. The query is half-wasted, and a test pins that adding `문서` to the query does not improve the match at all.

This is Day 5's vocabulary explosion and Day 6's split IDF, arriving finally as the symptom a user would actually notice: *searching for a word that is plainly in the document returns a weaker match than it should.* **Day 12's BPE** is the fix; **Day 8** builds the index that makes the search fast regardless.

## Run it

```bash
# demo (cosine vs distance, four ranked queries, the synonym failure)
python similarity.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_text_as_vectors/day07_similarity

# tests — from inside this folder
python -m unittest
python test_similarity.py

# the docstring examples are runnable too
python -m doctest similarity.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Ranking now works, but `rank_documents` scores **every** document in the collection — and the demo output shows most of those scores are exactly `0.000`. Since only terms present in *both* vectors contribute anything to a dot product, visiting a document that shares no term with the query is pure waste. **Day 8** builds the index that skips them, closing Level 2 with a working search engine.
