# Day 10 · N-gram Language Model with Laplace Smoothing

> Day 9 estimated `P(term | class)` and threw word order away. A **language model** estimates `P(term | the terms before it)` and keeps it. Same counting, one change of conditioning variable — and the result can score how likely a sentence is, which is what spelling correction, speech recognition, translation and every autoregressive generator are built on.

## The corpus

Levels 1–2 counted real text. This day needs *enough* text to estimate conditional probabilities, and Level 3 is standard-library only — there is nothing here to download with. So `build_corpus()` generates ~1,400 tokens from templates.

That has a consequence worth stating before any number below is read: **the distributional structure is built in, not discovered.** Animals share contexts with animals, people with people, and the possessive tracks the person. This is useful for showing what these methods do when the structure exists, and it is *not* evidence they would find it in real text. Days 13–15 depend on this corpus and the point matters more there; it is flagged again wherever it does.

The corpus also carries a deliberate **hapax tail** — eight words appearing exactly once. Without it every token would be frequent, `<unk>` would have nothing to train on, and Day 3's finding that real vocabularies are mostly rare words would be quietly contradicted by the data.

## The functions

| Piece | What it does |
|---|---|
| `NgramModel(n, alpha)` | counts padded n-grams; `alpha=0` is MLE, `1` is Laplace |
| `.probability(token, context)` | `P(token \| context)`, context truncated to order |
| `.log_probability(sentence)` | total log probability, or `-inf` |
| `.continuations(context)` | the full distribution over the vocabulary |
| `.smoothing_cost(context)` | mass split between observed and unseen |
| `replace_rare(sentences, min_count)` | fold rare words into `<unk>` |
| `build_corpus()` / `split_corpus(...)` | the data and a deterministic held-out split |

## Two decisions in `fit`

**Sentences are padded separately.** Otherwise the end of one sentence would predict the start of the next, and `<s>` would stop meaning "a sentence begins here". A test checks that two one-word sentences produce two `(<s>, word)` counts and no bigram spanning them.

**`</s>` is in the vocabulary; `<s>` is not.** Ending is an event the model must be able to predict — without it, probabilities would not sum to 1 over the possible continuations and nothing would ever stop generating (Day 11). `<s>` is only ever conditioned on, never generated.

## The zero, measured

Maximum likelihood assigns probability 0 to any continuation it never saw, and one zero makes an entire sentence impossible. How often that happens depends entirely on the model's order:

```
 n               MLE     Laplace
 2            1/20        0/20
 3            4/20        0/20
 4            9/20        0/20
 5           15/20        0/20
```

The two ends of that table fail for **different reasons**, and the distinction is the real content:

- At **n=2** the single failure is an out-of-vocabulary *word* — one held-out sentence uses a word the training split never contained.
- From **n=3 up**, the extra failures are known words in sequences never seen. This is Day 4's sparsity curve arriving as impossibility instead of as a percentage.

A test pins both causes separately, using a two-sentence corpus where `the zebra` fails on the word and `the cat ran` fails on the sequence while every word is familiar.

That bigrams and trigrams survive at all is an artefact of the templated corpus, which generalises far better than real text would. On real data the zeros start much earlier.

## What Laplace costs

Smoothing removes every zero — the right-hand column above is all zeros. The question nobody asks often enough is what it charges for that:

```
context         seen  count   observed   unseen
the               27    374     84.2%    15.8%
king               5     15     17.1%    82.9%
ate                1     32     24.6%    75.4%
wandered           1     15     13.7%    86.3%
```

Read the `ate` row carefully. That context was observed **32 times** and always with the same continuation. Add-one still hands **75.4% of its probability mass** to continuations that were never seen — three times what it leaves on the evidence it actually has. The model is now mostly describing things that did not happen.

The reason is structural: add-one adds `alpha × |V|` to every denominator, so the damage scales with vocabulary size and hits *narrow, well-attested* contexts hardest. Exactly the contexts a language model should be most confident about.

```
alpha=1.0    observed  24.6%   unseen  75.4%
alpha=0.1    observed  76.1%   unseen  23.9%
alpha=0.01   observed  96.9%   unseen   3.1%
```

Add-k with a small k is the usual retreat, and it is a retreat rather than a solution — it just tunes how much is stolen. The real answers redistribute mass according to how often *unseen things happen*, estimated from the hapax rate: **Good-Turing**, and **Kneser-Ney**, which additionally asks how many distinct contexts a word appears in rather than how often. Neither is implemented here; knowing that add-one is a placeholder is the point.

## `<unk>`

`replace_rare(sentences, min_count=2)` folds the corpus's own rare words into `<unk>` before training. The symbol then has a probability estimated from real data, and any unseen word at test time can be mapped to it and scored:

```
the wizard ate the bread
-> the <unk> ate the bread
log P = -19.10   (finite, so it is scoreable)
```

This is the Day 3 hapax tail put to work rather than discarded — and it only works because the corpus has one.

## Korean

The Korean sentences are present for continuity and are too few to model. With three sentences per subject, the model memorizes them; it learns nothing about Korean. Worse, `고양이가` and `고양이를` would be unrelated tokens, so even with more data an eojeol-level language model would need many times the data an English one does to reach the same coverage. That multiplier is the argument for subword units, and **Day 12** finally builds them.

## Run it

```bash
# demo (distribution, zeros by n, smoothing cost, <unk>)
python ngram_lm.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 03_statistical_language_models/day10_ngram_lm

# tests — from inside this folder
python -m unittest
python test_ngram_lm.py

# the docstring examples are runnable too
python -m doctest ngram_lm.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

The model is now a genuine distribution over continuations, which means two things become possible. **Day 11** samples from it to generate text, and measures **perplexity** — how surprised the model is by text it has not seen — which finally gives a number for whether smoothing choices like the one above are any good.
