# Day 8 · Inverted Index Mini Search Engine

> Day 7 ranked documents correctly and wastefully. It scored **every** document against every query, and its own output showed most of those scores were exactly `0.000`. This day stops computing them.

## The observation the whole day rests on

A dot product only adds terms present in **both** vectors. So a document sharing no term with the query contributes nothing — and could have been skipped without ever being looked at.

The **inverted index** makes skipping possible by turning the matrix around. Day 5 stored *document → terms*, with a cell for every term each document does **not** contain. This stores *term → documents*, which is the question a query actually asks:

```python
build_index([["a", "b", "a"], ["b"]])
# {'a': {0: 2}, 'b': {0: 1, 1: 1}}
```

## The functions

| Function | What it does |
|---|---|
| `build_index(docs)` | `term → {document_id: count}` |
| `candidates(query, index)` | documents holding **any** query term (OR) |
| `boolean_and(query, index)` | documents holding **every** query term (AND) |
| `SearchEngine(texts)` | Days 5–7 assembled behind one `search` call |

## The saving

```
'tokenization'       scores 1/8 -> skips 88%
'soup'               scores 1/8 -> skips 88%
'the'                scores 5/8 -> skips 38%
'quantum helicopter' scores 0/8 -> skips 100%
```

The pattern is the point: the saving is **largest for exactly the queries people actually type**. Rare, informative terms — the ones with high IDF — have short postings lists, so they name few candidates. Common terms like `the` save least, and they are also the terms carrying the least information. Zipf's law (Day 3) is what makes an inverted index worth building.

## Correctness is not optional here

A faster search that quietly ranks differently is not a faster search. `TestIndexMatchesFullScan` runs Day 7's brute-force scan alongside the indexed engine over eight queries — English, Korean, common-term, no-match, empty — and requires byte-identical results.

A second test states the underlying guarantee directly: for every query, **every document the index skipped is verified to score exactly 0.0**. That is what makes the skip free rather than approximate.

## Design decisions

**Indexing happens once, in `__init__`.** The vocabulary, IDF weights, document vectors and index are all fitted on the collection and reused for every query. That split — expensive work once, cheap work per query — is what makes something a search engine rather than a loop.

**Zero-scoring documents are not returned.** Day 7's `rank_documents` returns all 8 documents including six at `0.000`; `search` returns only real hits, and `[]` when there are none. A search box showing a page of zero-score results is showing noise.

**An empty `boolean_and` query matches nothing.** Set intersection over no sets is arguably "everything", but nobody typing an empty query wants the entire collection. The never-surprising reading wins.

**`explain()` exists.** Retrieval that cannot say *why* is hard to debug and harder to trust:

```
why did doc 2 match 'search query documents'?
  documents    0.306
  query        0.153
  search       0.153
```

`documents` contributes double because the document uses it twice. A test checks that the contributions sum to exactly the reported score — an explanation that doesn't add up is worse than none.

## Boolean AND, for contrast

The strictest classical model, and useful for seeing what ranking buys:

```
'soup salt'         -> [3]
'soup tokenization' -> no document has every term
```

It is **unranked** — it cannot say which of two matches is better — and it goes empty the moment one term is missing. Ranked retrieval degrades gracefully where boolean retrieval falls off a cliff.

## Korean: the index is fine, the units are not

Retrieval works, and searching the exact surface form works perfectly:

```python
engine.search("문서를")   # -> doc 7
```

But `explain("검색 문서", 7)` lists **only** `검색`. The term `문서` contributes nothing, because the document writes `문서의` and `문서를` and never the bare stem. The index faithfully stores what the tokenizer gave it; the tokenizer gave it the wrong units.

Nothing in Level 2 can fix this — not weighting, not geometry, not the index. It is a tokenization problem, and it waits for **Day 12's BPE**.

## Run it

```bash
# demo (the index, ranked queries, work avoided, explain, boolean AND)
python search_engine.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_text_as_vectors/day08_search_engine

# tests — from inside this folder
python -m unittest
python test_search_engine.py

# the docstring examples are runnable too
python -m doctest search_engine.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Level 2 closed

Text is now a search engine. Five days of primitives compose into one object:

```python
engine = SearchEngine(DOCUMENTS)
engine.search("tokenization tokens")   # [(0, 0.588), (1, 0.175)]
```

Day 1 split it, Day 2 normalized it, Day 3 counted it, Day 4 windowed it, Day 5 vectorized it, Day 6 weighted it, Day 7 compared it, Day 8 indexed it.

**What it still cannot do** is the honest summary, and it is one sentence: *it can only match words that literally coincide.* `word segmentation` scores zero against a document about tokenization (Day 7), and `문서` scores zero against a document full of `문서를`. Both failures are the same failure — the representation knows spelling, not meaning.

**Level 3** goes after the probabilities behind the counts, and **Level 4** finally attacks the lexical gap with embeddings. **Day 16** then combines the two: this engine's exact-match precision, plus semantic recall, which is the hybrid scorer RAG-Lab starts from.
