# Day 12 · Byte-Pair Encoding Tokenizer

> Every unresolved problem in this curriculum traces back to one decision made on Day 1: that the unit of text is **the word**.

Day 5 watched the vocabulary explode. Day 6 watched a stem's evidence split across its inflected forms. Day 7 watched a query miss a document that plainly contained the word. Day 10 had no way at all to score a word it had never seen. Every one of those READMEs ended by pointing here.

**Byte-Pair Encoding** stops treating words as atoms. Start from characters, repeatedly merge the most frequent adjacent pair, and the units are *learned from the corpus* instead of fixed in advance. Frequent words become single symbols; rare words decompose into pieces that are themselves frequent.

## The algorithm

| Function | What it does |
|---|---|
| `word_counts(corpus)` | word types with frequencies |
| `initial_splits(counts)` | each word → characters + `</w>` |
| `pair_counts(splits)` | adjacent pairs, weighted by word frequency |
| `merge_symbols(symbols, pair)` | replace a pair with the joined symbol |
| `train_bpe(corpus, num_merges)` | the loop: count, merge the best, repeat |
| `encode_word(word, merges)` | replay merges in learned order |
| `BPETokenizer(corpus, num_merges)` | the trained thing |

Three details that matter:

**`</w>` marks word ends.** It stops merges running across word boundaries, and keeps a word-final piece distinct from the same letters mid-word — `er</w>` in *faster* is a suffix, `er` in *ergonomic* is not.

**Merging scans left to right.** `aaa` with pair `(a,a)` becomes `aa a`, never `a aa`. Training and encoding call the *same* `merge_symbols`, so they cannot drift apart.

**Ties break on the pair itself.** Without it, two training runs on the same corpus could produce different tokenizers.

Training also stops early once no pair occurs more than once — merging a one-off pair creates a symbol that will never be reused. On this corpus that caps training at **79 merges**, so asking for 400 gets 79.

## The payoff

```
고양이가   -> ['고양이', '가</w>']
고양이를   -> ['고양이', '를</w>']
고양이에게 -> ['고양이', '에게</w>']
고양이는   -> ['고양이', '는</w>']

고양이가 and 고양이를 now share: ['고양이']
```

At word level those four shared **nothing** — four vocabulary entries, four separate counts, four separate IDF weights, no shared evidence between them. Merge #7 and #8 built `고` + `양` → `고양` → `고양이`, and the stem became a thing the rest of the pipeline can count.

The same happens for English suffixes (`tokenize` → `['token', 'ize</w>']`), and the algorithm did not need to be told that Korean has particles or that English has suffixes. It only counted.

## There is no out-of-vocabulary

```
tokenizers  -> ['token', 'ize', 'r', 's</w>']      4/4 pieces known
고양이처럼    -> ['고양이', '처', '럼', '</w>']         2/4 pieces known
quixotic    -> ['q','u','i','x','o','t','i','c','</w>']
```

Every string encodes, falling back to single characters in the worst case. Day 10's `<unk>` was a patch over a hole this representation does not have. A test checks the round trip: pieces rejoin to the original word exactly, for Korean and English alike.

## The hyperparameter that matters — and the failure I hit

My first run of this day used 120 merges and **the payoff did not happen**:

```
at 60 merges   고양이가 -> ['고양이', '가</w>']
               shared with 고양이를: ['고양이']
at 79 merges   고양이가 -> ['고양이가</w>']
               shared with 고양이를: nothing
```

Merge until nothing repeats and every word becomes a single symbol again — which is word-level tokenization, *the exact thing BPE was adopted to escape*. The stem `고양이` is still in the vocabulary; it is simply never used, because a longer symbol always matches first.

So the merge count is not a knob to max out. Too few and everything is characters; too many and everything is words; the useful behaviour lives in between:

```
  merges   symbols   pieces/word
       0        51          6.88
      10        61          5.34
      30        81          3.71
      60       111          2.32
      79       130          1.95
```

Real tokenizers choose this by vocabulary budget — GPT-2 uses 50,257 symbols — and the budget is doing exactly this trade-off at scale. Both ends of the failure are pinned by tests, so the tuned value cannot silently drift into the degenerate one.

## What BPE does not know

```
tokenization -> ['tokenization</w>']
학생에게      -> ['학생', '에게</w>']
```

`학생/에게` happens to be the right morphological split. `tokenization` staying whole is not wrong either — it is frequent in this corpus, so it earned its own symbol. But neither outcome reflects any grammatical knowledge. **BPE is a compression algorithm applied to tokenization.** Where its boundaries land on real morphemes, that is frequency agreeing with linguistics, not the algorithm understanding any.

It also inherits whatever the corpus contains. Train it on this corpus and `고양이` is a unit; train it on English-only text and Korean would shatter into characters. A tokenizer is a model fitted to data, with all that implies.

## What this fixes retroactively

| Day | The problem | With BPE |
|---|---|---|
| 5 | four vocabulary entries per Korean stem | one shared `고양이` piece |
| 6 | IDF split across inflected forms | evidence concentrates on the stem |
| 7 | query `문서` misses `문서를` | both contain the `문서` piece |
| 8 | index stores unrelated surface forms | postings share a stem symbol |
| 10 | unseen word is unscoreable | no OOV exists |

Level 2's engine is not rebuilt here; **Day 16** is where subword units get combined with the semantic signal into one ranking.

## Run it

```bash
# demo (merges learned, the Korean payoff, no-OOV, over-merging)
python bpe.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 03_statistical_language_models/day12_bpe

# tests — from inside this folder
python -m unittest
python test_bpe.py

# the docstring examples are runnable too
python -m doctest bpe.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Level 3 closed

Three days of probability and one of representation. Naive Bayes classified by counting terms per class; the n-gram model scored sentences by counting terms per context; generation sampled from those counts and perplexity measured how badly they fit unseen text. Every one of them was limited by the units it counted, and Day 12 finally changed the units.

**Level 4** attacks the last gap — the lexical ceiling of Day 7, where `word segmentation` scored exactly zero against a document about tokenization. Counting cannot close it, because the words genuinely never coincide. **Day 13** starts by counting what words appear *near* each other instead.
