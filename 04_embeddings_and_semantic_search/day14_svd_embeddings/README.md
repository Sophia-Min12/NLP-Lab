# Day 14 · SVD Word Embeddings

> Day 13 produced a PPMI row per word: 100 numbers, 94% of them zero. That is already a word vector, and it has two problems.

**It is fragile.** Two words can be similar and share no *exact* context. If one occurs beside `walked` and the other beside `strolled`, the sparse rows overlap nowhere and similarity reads as zero — the Day 7 failure, one level down.

**It is large.** One dimension per vocabulary word means 50,000 mostly-zero dimensions per word on a real corpus.

The singular value decomposition fixes both at once. It factorizes the matrix into orthogonal directions ordered by how much variance each explains; keeping the first `k` gives a short dense vector per word. Words that occurred in *related* contexts end up close even with no exact context shared, because the retained directions capture the pattern rather than the individual cells.

This is latent semantic analysis, and it is the direct ancestor of word2vec and GloVe — both later shown to be implicitly factorizing a matrix much like this one.

## The functions

| Function | What it does |
|---|---|
| `truncated_svd(matrix, k)` | `(U_k, S_k, Vt_k)` |
| `svd_embeddings(matrix, k, weighting)` | one dense vector per word |
| `explained_variance_ratio(matrix)` | what share each direction carries |
| `reconstruction_error(matrix, k)` | relative Frobenius error at rank `k` |
| `unit_rows(vectors)` / `cosine_similarity_matrix(vectors)` | all pairs at once |

Day 7 compared two vectors at a time in pure Python. Normalizing rows and multiplying by the transpose does every pair in one call — the same arithmetic, rearranged, and it is what makes Day 15's neighbour search a single matrix multiply.

**Singular vector signs are arbitrary.** `(-u, -v)` describes the same direction as `(u, v)`, so never compare raw coordinates across runs or libraries. Compare distances and angles, which are sign-invariant.

## The defect this day exposed in the corpus

The first version of this day reported `cat`/`dog` similarity of **exactly 1.000** — and so did the *sparse* PPMI space, which meant SVD was contributing nothing at all.

The cause was not the factorization. `build_corpus()` applied every template to every word of a category, which made those words perfectly interchangeable: **34 pairs of word types had bit-identical co-occurrence rows**. Every similarity was 1.0 because the vectors were literally equal, and Day 15's analogies would have been vacuous.

The corpus now uses rotated template slices and gives each subject, place and food one context of its own. Identical pairs drop from 34 to 3, and the survivors are honest — `ran`/`wandered` really are interchangeable here.

Worth stating plainly: **a synthetic corpus can be too clean to demonstrate anything.** The defect was invisible in Days 10–13, which only needed the corpus to be a plausible token stream, and it only surfaced when something depended on words being *different*.

## What compression buys

```
pair                sparse PPMI   SVD k=24
cat / dog                 0.497      0.681
king / queen              0.991      0.999
garden / forest           0.556      0.670
```

Every related pair scores **higher** after compression. That is the fragility argument made concrete: sparse rows only agree where contexts coincide exactly, while the dense space rewards contexts that are merely *related*. A test requires this lift for `cat`/`dog` and `garden`/`forest`.

The resulting space is ordered the way it should be:

```
cat vs dog     +0.681
king vs queen  +0.999
cat vs king    +0.398
bread vs soup  +0.536
cat vs bread   -0.042
```

Same category > adjacent category > unrelated, with a test on the ordering.

## Choosing k, honestly

```
   k   cat/dog   cat/bread      gap   recon err
   2    +1.000      +1.000   +0.000       0.917
   4    +0.930      +0.990   -0.060       0.864
   8    +0.981      +0.577   +0.404       0.768
  16    +0.590      +0.254   +0.336       0.649
  24    +0.681      -0.042   +0.723       0.544
  32    +0.737      -0.044   +0.782       0.442
  48    +0.813      -0.105   +0.918       0.271
  64    +0.475      -0.010   +0.486       0.156
  80    +0.254      -0.011   +0.265       0.060
 100    +0.254      -0.009   +0.263       0.000
```

Separation **peaks at k=48 and collapses after**. At k=2 every word shares one direction and everything is similar to everything; at k=4 the gap is actually *negative*, meaning `cat` is closer to `bread` than to `dog`. Past the peak the noise directions return and drown the structure in per-word detail.

The last column is the important one. **Reconstruction error falls the whole way** — the rank-100 approximation is exact — while the separation it is supposed to serve rises and then collapses. *Fitting the matrix better and representing the words better are different goals, and here they diverge.* A test pins that divergence directly.

I set the demo default to **k=24** rather than the peak at 48, because 48 is nearly half the vocabulary and tuning a hyperparameter against the one pair you intend to quote is how benchmark numbers get made. k=24 gives a clean ordering without being chosen to flatter it.

## The weighting

```
sqrt   cat/dog +0.681   cat/the +0.497
full   cat/dog +0.682   cat/the +0.546
none   cat/dog +0.684   cat/the +0.445
```

`sqrt` (`U · √S`) is the usual choice: it splits the scaling symmetrically between the word and context sides. `full` (`U · S`) lets the leading direction dominate, which on a PPMI matrix largely encodes overall word frequency — note `cat/the` is highest there, which is exactly the frequency artefact. `none` uses `U` alone and counts the noisiest directions as heavily as the strongest.

On this corpus the differences are small. On a large one they are not.

## How many dimensions actually matter

```
  k   this one   cumulative  reconstruction
  1      9.7%         9.7%           0.950
  2      6.2%        16.0%           0.917
  4      4.6%        25.4%           0.864
  8      2.9%        41.0%           0.768
 16      1.7%        57.9%           0.649
 32      1.1%        80.5%           0.442
```

No single direction dominates — the leading one carries under 10%. On a real corpus the first few directions carry far more, because there is genuine large-scale structure to find. Here the variance is spread thin, which is what a small, largely uniform corpus looks like from the inside.

## Run it

```bash
# demo (variance spectrum, compression gains, weighting, choosing k)
python svd_embeddings.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 04_embeddings_and_semantic_search/day14_svd_embeddings

# tests — from inside this folder
python -m unittest
python test_svd_embeddings.py

# the docstring examples are runnable too
python -m doctest svd_embeddings.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Every word is now a short dense vector, and the obvious question is what those vectors know. **Day 15** asks it directly — nearest neighbours, and the famous analogy arithmetic — and is where the synthetic corpus caveat finally changes a conclusion rather than merely qualifying one.
