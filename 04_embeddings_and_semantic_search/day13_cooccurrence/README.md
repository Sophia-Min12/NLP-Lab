# Day 13 · Co-occurrence Matrix and PPMI

> Day 7 found the wall: `word segmentation` scored **exactly 0.000** against a document about tokenization. No reweighting closes that gap, because the strings genuinely never coincide. The information is not in the representation.

The distributional hypothesis says where to look instead: **a word is characterised by the company it keeps.** Two words are similar if they appear in similar contexts — even when they never appear together. So stop counting how often a word occurs and start counting **what occurs near it**.

Day 5 built a document × word matrix. This builds a **word × word** one.

## The functions

| Function | What it does |
|---|---|
| `cooccurrence_matrix(sentences, vocab, window)` | how often each pair appears within `window` |
| `pmi_matrix(counts, positive=True)` | PMI, clipped at zero (PPMI) |
| `top_associations(matrix, vocab, word, k)` | the strongest neighbours of a word |
| `sparsity(matrix)` | share of exact zeros |

NumPy arrives here, as Level 4 allows. Everything below is still a count and a logarithm. **CI installs it from this day on**; Levels 1–3 remain standard-library only and still run without it.

## A note that matters more from here on

`build_corpus()` is synthetic. Its distributional structure is **built in, not discovered** — animals share contexts with animals and people with people because the templates say so.

What follows therefore demonstrates that these methods *find structure that exists*. It is not evidence that comparable structure would emerge from a comparable amount of real text. It would not: real corpora for this work are measured in millions of tokens, and this one is about 1,500. Day 15 is where that distinction becomes sharp enough to change a conclusion, and it is stated again there.

## Why raw counts fail, and why PMI is not just IDF again

Raw co-occurrence counts are dominated by whatever is common:

```
cat     the(31), through(12), near(6), saw(6)
king    the(28), ate(4), saw(4), to(4)
ate     the(70), bread(10), soup(10), rice(8)
```

`the` wins every row, which tells us nothing. Day 6 met this and answered with IDF — down-weight by how many documents a term appears in. **PMI asks a sharper question**: does this pair co-occur *more than chance would predict*?

```
PMI(x, y) = log( P(x, y) / (P(x) · P(y)) )
```

The denominator is what independence predicts. A word that sits next to everything has exactly the co-occurrences independence already accounts for, so its PMI collapses toward zero — not because it was down-weighted, but because **there was never any surprise in it**.

```
cat     slept(2.77), on(2.08), near(1.38), ran(1.38)
king    ruled(2.46), saw(1.36), to(1.36), ate(0.99)
ate     soup(2.03), rice(1.95), fish(1.89), bread(1.07)
```

`the` disappears from every list, and a test pins that it dominates the raw-count lists and appears in none of the PPMI ones.

## `the` demotes itself, measured properly

My first attempt at this measurement was wrong, and the wrong version is instructive. I compared the *mean PPMI of the `the` row* against the matrix mean and found it **higher** — apparently no demotion at all. The mean over a full row is diluted by zeros, and `the` has fewer zeros than anything else precisely because it co-occurs with everything.

Averaging only over the words it actually co-occurs with gives the real picture:

```
word     raw total  partners  max PPMI  mean PPMI*
the           1129        59      1.04        0.47
ate            140        15      2.03        1.08
cat             75        10      2.77        1.42
king            51         8      2.46        1.16
fish            36         8      3.50        1.71
```

A clean inversion. `the` leads on raw count by **15×** and comes **last** on association strength. It has the most partners and the weakest tie to any of them — which is what being a function word *is*. `fish` is rare and occurs in few contexts, so it scores highest.

Day 2's hand-written stopword list and Day 6's IDF both fall out of counting, with nothing told to the algorithm.

## Why *positive* PMI

Negative PMI claims two words co-occur *less* than chance. In any realistic corpus that estimate is dominated by sampling noise: most pairs simply never co-occur, and "never" is not evidence of repulsion — it is usually evidence of a small corpus. Clipping also keeps the matrix non-negative and very sparse (**94% zeros** here), which is what Day 14's factorization wants.

Pairs that never co-occur would give `log 0`. They are set to zero *before* clipping, so `positive=False` stays finite too; a test checks every entry is finite under both settings.

## The window is the model

```
window=1: cat ~ slept, near, ran, wandered
window=2: cat ~ slept, on, near, ran
window=5: cat ~ windowsill, slept, warm, on
```

Narrow windows capture what can grammatically sit beside a word; wide ones blur toward topic. Neither is more correct — they answer different questions, and the answer propagates directly into what Day 14's embeddings consider "similar".

Counting stays **inside a sentence**, for the same reason Day 10 padded sentences separately: a window running off the end would invent adjacency that never occurred. A test checks that two one-word sentences produce no co-occurrence at any window size.

## The Day 7 gap, finally addressable

```
cat and dog co-occur 0 times
  cat    vs dog   : 7 shared context words
  king   vs queen : 8 shared context words
  cat    vs king  : 3 shared context words
```

`cat` and `dog` **never** appear together. Under every representation so far — bag of words, TF-IDF, cosine, the inverted index — their similarity was necessarily zero. Here they share seven context words, and same-category pairs share more than cross-category ones. A test asserts that ordering.

That overlap is the signal. It is still 100 dimensions of mostly zeros, though, and comparing sparse rows directly is fragile: two words can be similar and share no *exact* context. **Day 14** compresses these rows into short dense vectors where that fragility goes away.

## Korean

The Korean sentences are too few to survive `min_count=2` in any useful number, so they contribute almost nothing to this matrix. That is the honest state: distributional methods need many occurrences of each unit, and eojeol-level Korean splits every stem across particles, so it needs *more* data than English rather than less. Day 12's BPE is what would make the units shareable; combining the two is beyond what this corpus can demonstrate, and Day 17 says so plainly.

## Run it

```bash
# demo (raw vs PPMI, the demotion table, window effects, shared contexts)
python cooccurrence.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 04_embeddings_and_semantic_search/day13_cooccurrence

# tests — from inside this folder
python -m unittest
python test_cooccurrence.py

# the docstring examples are runnable too
python -m doctest cooccurrence.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

A PPMI row *is* a word vector — 100 dimensions, 94% of them zero. **Day 14** factorizes the matrix with SVD to get a short dense vector per word, which is both more robust and small enough to compare quickly. **Day 15** then asks what those vectors actually know.
