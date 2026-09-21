# Day 16 · Capstone I — Hybrid Ranking Scorer

> Two representations now exist, and each fails exactly where the other works.

**Lexical** (Days 5–8) is precise. TF-IDF over an inverted index knows an exact match when it sees one, and can say which term earned the score. It returns **exactly zero** when the words do not coincide, however close the meaning.

**Semantic** (Days 13–15) crosses that gap — `cat` and `dog` never co-occur and come out neighbours — but it is vague, and Day 15 showed how easily it can look better than it is.

Combining them is not a compromise. Lexical precision where the words line up, semantic recall where they do not: that is the hybrid retrieval RAG systems are built on.

## The functions

| Piece | What it does |
|---|---|
| `HybridSearcher(texts, embedding_corpus, ...)` | both signals over one collection |
| `.lexical_scores(query)` / `.semantic_scores(query)` | the raw signals |
| `.search(query, alpha, k, fusion)` | `alpha=1` lexical, `0` semantic |
| `.explain(query, doc_id)` | both raw signals for one document |
| `min_max(scores)` | rescale to `[0, 1]` |
| `reciprocal_rank_fusion(rankings)` | combine orderings instead of scores |
| `recall_at_k(searcher, cases, alpha, k)` | evaluation |

## The hard part is that the scores are not comparable

```
'queen'  lexical  +0.000 .. +0.417
         semantic +0.115 .. +0.689
'dog'    lexical  +0.000 .. +0.339
         semantic -0.005 .. +0.821
```

Both are cosines, both notionally in `[-1, 1]`, and they occupy completely different ranges on any given query. A raw weighted sum would silently let whichever range is wider dominate, and `alpha` would not mean what it says.

`min_max` rescales each signal per query before weighting. One detail that is a decision rather than a default: **an all-equal input maps to zeros, not ones.** If a signal cannot distinguish any document, it should contribute nothing — mapping it to 1.0 everywhere would add a constant that survives the weighting and shifts the blend for no reason.

## The pretraining corpus is not a convenience

My first version trained the word vectors on the twelve documents being searched, and **the semantic half was dead** — every score exactly 0.000. The cause: `queen` occurs once across twelve short documents, so `min_count=2` dropped it from the embedding vocabulary, and the query embedded to a zero vector.

That is not a bug to work around, it is the actual constraint. A document collection is almost never large enough to learn distributional vectors from. Real systems **pretrain vectors on a large corpus** and apply them to whatever they are asked to search, and `HybridSearcher` now takes `embedding_corpus` separately to make that explicit.

```
vectors from the 12 documents only:  alpha=0.5 -> 67%,  alpha=0.0 -> 42%
vectors pretrained on the corpus:    alpha=0.5 -> 100%
```

A test pins that `queen` is missing from the naive searcher's embedding vocabulary, so the failure cannot quietly return.

## The evaluation, and how it was nearly rigged

The first evaluation set was six queries whose target document does **not** contain the query word — exactly the gap lexical search cannot cross. Pure semantic scored 100% on it and pure lexical 33%.

That measures nothing. A test set made only of cases the semantic half exists to solve will show the semantic half solving them; the answer was never in doubt. So the set now has two halves — `GAP_QUERIES` and `EXACT_QUERIES`, where the query word *is* in the target:

```
 alpha     gap   exact   overall
   1.0     33%    100%       67%  <- pure lexical
   0.8    100%    100%      100%
   0.6    100%    100%      100%
   0.5    100%    100%      100%
   0.4    100%    100%      100%
   0.2    100%    100%      100%
   0.0    100%     83%       92%  <- pure semantic
```

Now the shape is real. **Each extreme fails on the half the other handles, and every blend beats both.** Lexical is perfect on exact matches and near chance on the gap; semantic crosses the gap and slips on exact matches, because a vague signal will happily prefer a loosely related document when an exact one exists.

The result also **holds across the whole range 0.2–0.8**, which matters: if only one value of `alpha` worked, that would be a tuning artefact rather than a finding. A test asserts the whole range.

### The caveat, stated plainly

12 queries over 12 documents, where recall@3 by chance alone is **25%**. The direction is right and matches what larger studies find; the numbers themselves are worth very little. This is a demonstration that the mechanism works, not a measurement of how well.

## Reciprocal rank fusion loses to the linear blend

```
linear (alpha=0.5)  100%
rrf                  67%
```

RRF throws the scores away and combines only the orderings, so nothing needs to be commensurable — which is why it is a common production default. Here it does **worse**, and the reason is instructive: lexical scores are zero for most documents on most queries, so the lexical *ordering* below the first hit or two is arbitrary. RRF treats that noise as a real ranking and gives it equal weight. The linear blend, having kept the magnitudes, can tell that those documents scored nothing.

Keeping score magnitudes costs a normalization step and buys the ability to know when a signal has nothing to say.

## Document vectors are IDF-weighted

`_embed` averages the word vectors of a document's tokens, weighted by IDF. A plain mean would let `the` dominate every document vector for the reason Day 6 gave — it appears everywhere, so it pulls every document toward the same direction. Weighting by IDF lets the rare contentful words steer: the same fix as TF-IDF, applied to a mean instead of a coordinate.

## What this closes

Day 7 ended with `word segmentation` scoring 0.000 against a document about tokenization, and said the gap was what Level 4 existed to fix. It is fixed here, in the sense that matters: the system now retrieves documents whose words do not appear in the query, without giving up exact matching when the words do appear.

`explain()` still reports both raw signals, so a result can always be traced to the half that produced it.

## Run it

```bash
# demo (scale mismatch, alpha sweep, pretraining, fusion strategies)
python hybrid.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 04_embeddings_and_semantic_search/day16_hybrid_scorer

# tests — from inside this folder
python -m unittest
python test_hybrid.py

# the docstring examples are runnable too
python -m doctest hybrid.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

**Day 17** puts this behind a command-line interface, runs it bilingually, and writes the honest account of what these seventeen days built — including the parts that do not work and the measurements that should not be believed.
