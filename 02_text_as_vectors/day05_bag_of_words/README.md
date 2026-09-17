# Day 5 · Bag-of-Words and Vocabulary Building

> Level 1 ended with text as countable units. Turning those counts into **vectors** is what lets a machine compare two documents at all — once a document is a list of numbers, similarity becomes geometry (Day 7) and retrieval becomes arithmetic (Day 8).

## The functions

| Function | What it does |
|---|---|
| `document_frequencies(docs)` | in how many documents does each term appear? |
| `build_vocabulary(docs, min_df, max_df)` | term → column index, after filtering |
| `bag_of_words(tokens, vocab)` | one document → one count vector |
| `document_term_matrix(docs, vocab)` | the whole collection as rows |
| `oov_rate(tokens, vocab)` | what share of a document can't be represented? |
| `matrix_sparsity(matrix)` | what share of the matrix is zero? |
| `to_sparse(vector)` | drop the zeros, keep `{column: count}` |

## The vocabulary is a contract

Building the vocabulary is a **separate step** from vectorizing, and that separation is the main idea of the day. Two documents are comparable only if position 7 means the same term in both. So the index assignment has to be fixed once, shared by every vector in the collection, and independent of which document happened to arrive first.

`build_vocabulary` sorts terms alphabetically before numbering them. Any deterministic order would do — alphabetical is just the one that is obviously reproducible. A test builds the vocabulary from the collection and from the collection reversed and requires the two to be identical.

Filtering has to **renumber**, not leave holes where terms were dropped, or the vectors would carry dead columns. Also pinned by a test.

## `min_df` and `max_df`

- `min_df` is an absolute **document count**. `min_df=2` drops terms appearing in only one document — typos, hapax, one-off proper nouns.
- `max_df` is a **proportion** of the collection. `max_df=0.5` drops terms appearing in more than half of the documents.

The asymmetry is deliberate: it is how these two get used in practice. "Appears at least twice" is a statement about absolute evidence; "appears nearly everywhere" is a statement about relative ubiquity.

What `max_df` removes is worth staring at. On this collection it deletes exactly one term — `the` — which is to say **`max_df` derives a stopword list from the corpus instead of importing a hand-written one**. Day 2's list was a guess about English in general; this is a measurement of the documents actually in hand. Day 6 takes the same idea further and makes the down-weighting continuous rather than a yes/no cut.

## Out-of-vocabulary is a measurement, not an accident

`bag_of_words` drops terms outside the vocabulary silently — that is simply what a fixed vocabulary means. The honest response is not to pretend it does not happen but to measure it, which is what `oov_rate` is for. It counts **running tokens, not types**: dropping a rare word once costs less than dropping a common word ten times.

## Two things this representation destroys

**Word order.** `"the dog bit the man"` and `"the man bit the dog"` produce the identical vector — the failure Day 4 already diagnosed. The demo shows it directly.

The repair costs nothing structurally, because none of these functions assumes a term is a single word. Feed them n-gram terms and the same code separates the two sentences:

```python
def bigram_terms(text):
    words = normalize(text)
    return [f"{a}_{b}" for a, b in zip(words, words[1:])]
```

A test pins both halves: identical vectors from word terms, different vectors from bigram terms.

**Space.** The matrix here is 8 × 71 and about **86% zeros**; real collections run well past 99%. Every one of those zeros gets stored, and on Day 7 multiplied. `to_sparse` shows the alternative — space proportional to what the document actually uses — and that idea, taken to the collection level, *is* the inverted index of Day 8.

## Korean: the vocabulary explodes, and `min_df` erases it

This is the sharpest measurement in the day. `고양이가`, `고양이를`, `고양이에게` are one noun with three particles, so they occupy **three separate dimensions** and the stem `고양이` occupies none.

The consequence for filtering is severe. Under `min_df=2`:

```
doc 0 (English): 82% of tokens dropped
doc 3 (English): 69% of tokens dropped
doc 6 (Korean) : 100% of tokens dropped
```

**No Korean term survives at all.** A term needs to appear in two documents to pass, and a Korean stem essentially never appears twice in the same surface form, because each occurrence carries whatever particle its grammatical role demands. A filter that reads as sensible housekeeping in English silently deletes the Korean half of the collection. A test pins exactly this.

Two honest notes about that 100%: this collection has only two Korean documents on different subjects, so a larger Korean corpus would fare better — and the high English rates show `min_df=2` is aggressive on any tiny collection. But the *gap* between 82% and 100% is the real finding, and it does not close by adding documents of this kind. The fix is subword units — **Day 12's BPE**.

## Run it

```bash
# demo (sparsity, a document as a vector, filters, OOV rates)
python bag_of_words.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 02_text_as_vectors/day05_bag_of_words

# tests — from inside this folder
python -m unittest
python test_bag_of_words.py

# the docstring examples are runnable too
python -m doctest bag_of_words.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Raw counts make `the` the loudest signal in every document, which is useless for telling documents apart — exactly the Zipf head measured on Day 3. **Day 6** fixes it by weighting each count by how rare the term is across the collection, **Day 7** compares the resulting vectors by angle rather than magnitude, and **Day 8** stops materializing the zeros. The `DOCUMENTS` collection defined here is carried through all three.
