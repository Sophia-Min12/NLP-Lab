# Day 3 · Word Frequencies and Zipf's Law

> Counting is the last step before text becomes numbers. A frequency table *is* the unnormalized bag-of-words vector of Day 5, and the document counts TF-IDF needs on Day 6. It also exposes the one statistical fact that shapes every design decision downstream.

## Zipf's law

Rank the words of any corpus by frequency. The count of the word at rank *r* falls off roughly as **C / r** — the 2nd word appears about half as often as the 1st, the 10th about a tenth as often. Plotted on log-log axes it is close to a straight line with slope −1.

This is not a curiosity. It is why vocabularies are huge but useless at the tail, why inverted indexes compress so well, why unsmoothed language models assign zero probability to so much text, and why subword tokenizers exist at all.

## The functions

| Function | What it answers |
|---|---|
| `word_frequencies(tokens)` | how often does each word occur? |
| `rank_words(freqs)` | what is the frequency order? |
| `top_n(freqs, n)` | what are the *n* most common words? |
| `hapax_legomena(freqs)` | which words occur exactly once? |
| `type_token_ratio(tokens)` | how varied is the vocabulary? |
| `coverage(freqs, k)` | what share of the corpus do the top *k* types hold? |
| `zipf_table(freqs, n)` | observed counts vs the `C / rank` prediction |
| `zipf_exponent(freqs)` | least-squares slope on log-log axes |

`word_frequencies` is a plain loop over a dictionary rather than `collections.Counter`. The counting dict is worth seeing once; `Counter` is the same thing and is what production code should use.

## Types and tokens

Two words that get used interchangeably in ordinary speech and must not be here:

- **tokens** — running words. `"the cat the"` is 3 tokens.
- **types** — distinct words. `"the cat the"` is 2 types.

Vocabulary size is a count of types. Corpus size is a count of tokens. Zipf's law is a statement about how the two relate.

## Why stopwords are *not* removed here

Day 2 could remove `the`, `of`, `and`. Day 3 deliberately does not, because **the stopword list is simply the head of the Zipf curve** — that is the whole justification for having one. Removing them before counting would hide the very shape this day is about. Run the demo and the top ten is almost entirely function words.

## Ties must break deterministically

`rank_words` sorts by `(-count, word)`. Without that second key, words with equal counts would come out in dictionary insertion order, and every rank-dependent number — the Zipf table, the fitted exponent, `top_n` — would shift between runs on the same input. A test pins that `{"z": 2, "y": 2}` and `{"y": 2, "z": 2}` rank identically.

## Fitting the exponent without NumPy

`zipf_exponent` takes logs of rank and count, then computes the ordinary least-squares slope by hand:

```
s = -Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²      where x = log(rank), y = log(count)
```

Two things bias this fit, and it is worth being exact about which one dominates.

**The hapax tail flattens the slope.** Most *types* sit in the flat tail where the count is 1 and `log(count)` is 0. Averaged over thousands of such ranks, the line is dragged toward horizontal. This is measurable: a synthetic perfect-Zipf head fits at **1.003** on its own, drops to **0.91** once 3000 hapax are appended, and returns to **1.003** when `max_rank` caps the fit at the head. A test pins exactly that.

**Corpus size dominates anyway.** The textbook exponent of ~1.0 is *asymptotic* — it needs large corpora. This README is around a thousand tokens and fits near **0.7** whichever rank window you choose; trimming ranks moves it by a few hundredths, not by 0.3. The bundled 46-token `SAMPLE` fits near **0.61**.

So: the *shape* is unmistakable at this scale — look at how closely the `C / r` column tracks the observed counts in the top ranks — but the *exponent* is not, and quoting 1.0 off a thousand-token text would be making the number up.

> The demo counting its own README is deliberate self-reference, not a shortcut — it keeps the folder self-contained with no downloads, and it is genuine text. The numbers move when this file is edited; re-run rather than trusting a number quoted here.

## What the demo shows

```bash
python frequencies.py
```

Three things worth reading off the output:

1. **Coverage climbs brutally fast.** A handful of types account for a large share of all tokens, and a few dozen more add almost nothing. This is why an index with a stopword list is so much smaller.
2. **Roughly half the vocabulary is hapax legomena** — words occurring exactly once. The tail is enormous and nearly empty of information, which is precisely the data-sparsity problem Day 10 must smooth away.
3. **The Zipf prediction tracks reality well in the middle ranks** and drifts at both ends. Real corpora bend away from the pure `C / r` line at the very top and in the deep tail; the law is a strong approximation, not an identity.

## Korean: counts scatter across particles

`고양이가` and `고양이를` are the same noun carrying different particles, so eojeol-level counting files them as two unrelated types. The noun's true frequency is split across every particle it appears with, which inflates the vocabulary and deflates every individual count. A test pins that `고양이` never appears as a type on its own.

This is the same limitation Days 1 and 2 hit, stated in the vocabulary of counts: **Zipf's law holds for Korean, but the units it holds over are not the units you want.** The practical fix is subword tokenization on **Day 12 (BPE)**.

## Run it

```bash
# demo (Zipf table, fitted exponent, coverage curve)
python frequencies.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_text_basics/day03_frequencies

# tests — from inside this folder
python -m unittest
python test_frequencies.py

# the docstring examples are runnable too
python -m doctest frequencies.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

The frequency table is the raw material of everything in Level 2: **Day 4** counts n-grams instead of single words, **Day 5** turns the table into a vector, and **Day 6** reweights it by how rare each word is across documents. That reweighting is TF-IDF, and the reason it works is the shape measured today — frequent words are frequent everywhere, so frequency alone cannot tell documents apart.
