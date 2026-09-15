# Day 4 · N-gram Extraction

> Day 3 counted words in isolation, and a bag of words cannot tell **"the dog bit the man"** from **"the man bit the dog"** — identical counts, opposite meaning. An n-gram is the cheapest possible repair: a window of *n* adjacent tokens, slid one position at a time, so a little local word order survives into the counts.

## The functions

| Function | What it does |
|---|---|
| `ngrams(tokens, n)` | every window of *n* adjacent tokens, as tuples |
| `bigrams(tokens)` / `trigrams(tokens)` | the two cases worth naming |
| `pad_sequence(tokens, n)` | wrap in `n-1` × `<s>` and one `</s>` |
| `ngram_frequencies(tokens, n, pad=False)` | count them |
| `top_ngrams(freqs, k)` | the *k* most common |
| `char_ngrams(text, n, strip_spaces=False)` | windows over *characters*, not words |
| `sparsity(freqs)` | share of n-grams seen exactly once |

## Three decisions worth defending

**N-grams are tuples, not joined strings.** An n-gram must be hashable to serve as a dictionary key, and `" ".join(...)` would make the bigram `("new", "york")` indistinguishable from a single token `"new york"`. Tuples keep the boundary.

**`n` rejects `bool`.** `True` is an `int` in Python, so `ngrams(tokens, True)` would quietly compute unigrams. The guard is `isinstance(n, bool) or not isinstance(n, int)`, and a test pins it.

**A sequence of *L* tokens yields *L − n + 1* n-grams**, and an empty list when the window is longer than the text. Returning nothing is correct, not an error — short documents genuinely have no 5-grams.

## Padding: why `<s>` and `</s>` exist

Without padding, the first word of a sentence is never the *target* of an n-gram — it only ever appears as context. A model built from those counts cannot answer "what tends to start a sentence?", nor "what tends to end one?".

```python
ngrams(["dogs", "bark"], 2)                     # [('dogs', 'bark')]
ngrams(pad_sequence(["dogs", "bark"], 2), 2)    # [('<s>', 'dogs'), ('dogs', 'bark'), ('bark', '</s>')]
```

The count is `n-1` start markers, not one, because that is exactly how much left context an n-gram model conditions on. A trigram model needs two words of history before the first real word, so `pad_sequence(tokens, 3)` supplies `['<s>', '<s>', ...]`. Day 10 depends on this.

## The sparsity explosion

Raising *n* by one multiplies the number of *possible* sequences by the vocabulary size, while the corpus stays exactly as big. Run the demo and watch the last column — the corpus is this README, roughly a thousand tokens:

```
   n   distinct   seen once
   1     ~424        63%
   2     ~866        92%
   3     ~946        98%
   4     ~963        99%
   5     ~966       100%
```

The counts are approximate because editing this file changes them; the percentages barely move, and they are the point. By n=5 *every* distinct window occurs exactly once. A maximum-likelihood model estimates each of those probabilities from a **single observation**, and assigns probability **zero** to every window it happens not to have seen — including perfectly ordinary English. That is not a fixable bug in the counting; it is a property of language plus finite data, and it is precisely what **Laplace smoothing on Day 10** exists to patch. The tests pin that sparsity rises monotonically with *n*.

This is Day 3's Zipf tail again, one level up: n-grams are far more Zipfian than words, because combinations are rarer than their parts.

## The most common bigrams are function-word pairs

`the dog`, `the man`, `and the`, `at the`. Frequency alone surfaces *grammar*, not *content* — the same finding as Day 3's top-ten, and the same reason raw counts are a weak ranking signal.

> The demo shows this on the bundled `SAMPLE` rather than on the README it otherwise counts, and the reason is worth noting: a technical document's top bigrams are its own jargon (`n grams`, `python m`), which is a fact about *this file*, not about language. Claims about ordinary prose should be demonstrated on ordinary prose. Finding bigrams that are genuinely informative means asking whether two words co-occur **more than chance would predict**, which is pointwise mutual information on **Day 13**.

## Korean: character n-grams are the real answer

This is where the day pays off for the bilingual half of the lab. `고양이가`, `고양이를` and `고양이에게` are one noun with three particles. Word-level counting files them as three unrelated types, so the noun's frequency is split three ways and no window ever links them:

```python
normalize("고양이가 고양이를 고양이에게")
# ['고양이가', '고양이를', '고양이에게']   ← 3 distinct types, nothing shared

word_frequencies(char_ngrams("고양이가 고양이를 고양이에게", 3, strip_spaces=True))["고양이"]
# 3                                      ← the stem, found without any morphology
```

Character n-grams need no word boundaries at all, which is why they are the standard fallback for Korean, Chinese and Japanese retrieval. They are crude — they also produce meaningless windows spanning word edges — but they are language-agnostic and they work.

`char_ngrams` NFC-normalizes first. Over decomposed Hangul (Day 2) the windows would slice syllables into jamo and the trigram `고양이` would never appear at all. A test pins that composed and decomposed input give identical output.

## Run it

```bash
# demo (bag-of-words failure, sparsity table, padding, Korean char n-grams)
python ngrams.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_text_basics/day04_ngrams

# tests — from inside this folder
python -m unittest
python test_ngrams.py

# the docstring examples are runnable too
python -m doctest ngrams.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

This closes Level 1: raw text is now a countable, order-aware collection of units. **Level 2** turns those counts into geometry — `ngram_frequencies` is already the vector Day 5 builds, just without the shared vocabulary that lets two documents be compared. **Day 10** reads the same tables as conditional probabilities, and **Day 12**'s BPE runs the character-level version of this over a corpus to *learn* its units instead of fixing them in advance.
