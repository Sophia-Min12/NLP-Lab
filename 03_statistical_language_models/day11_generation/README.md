# Day 11 · Text Generation and Perplexity

> Day 10 built a distribution over continuations. Two things follow, and they are the two halves of how language models are used and judged: **sampling from it**, and **measuring how surprised it is**.

Every autoregressive text generator runs the loop in `generate()` — sample a token, append it to the context, repeat. The difference between an n-gram model and a modern one is entirely in how `P(next | context)` is estimated, not in the loop.

## The functions

| Piece | What it does |
|---|---|
| `apply_temperature(dist, t)` | sharpen (`t<1`) or flatten (`t>1`), then renormalize |
| `top_k_filter(dist, k)` | keep the `k` likeliest, renormalize |
| `sample_next(model, context, rng, ...)` | draw one token |
| `generate(model, max_tokens, seed, ...)` | sample until `</s>` or the limit |
| `perplexity(model, sentences)` | `exp(-1/N · Σ log P)` |

Sampling takes an explicit `random.Random`, so a seed makes output exactly reproducible. That is what lets a test assert on generated text, and what lets a README quote a specific sentence.

## Two counting details in `perplexity`

`N` counts **predicted** tokens: each `</s>` is included (the model really does predict it) and the `<s>` padding is excluded (it never does). Getting this wrong shifts every number silently, and it is the usual reason two implementations of "the same" metric disagree. A test pins that two 3-token sentences give `N = 8`.

An unsmoothed model on unseen text returns `inf`. That is the correct answer, not an error — a model that assigns a sentence probability zero is infinitely perplexed by it.

## What generation actually shows

```
n=1: forest ate the
n=2: the bread fresh bought bread at caught salty
n=3: the boy saw bought on great ate near and in behind bought hot village
n=5: the boy saw bought on great ate near and in behind bought hot village
```

`n=1` has no context at all. `n=2` is locally fluent and globally aimless — each adjacent pair is plausible, the sentence is not. And then `n=5` is **no better than `n=3`**, which is not what the usual story predicts.

The cause is smoothing, not the order:

```
alpha=0.1   the boy saw bought on great ate near and in behind bought hot village
            -> not in the training data
alpha=0.0   the boy saw the dog near the river
            -> verbatim training sentence
```

A smoothed high-order model spreads mass over continuations its context never had. One step off a seen path and it falls into near-uniform noise — and with a 5-token context, nearly every path is off. Unsmoothed, it can only walk transitions it actually saw, so it is fluent **because it is reciting**.

So the two ways to make a high-order n-gram model produce good text are: memorize, or add noise. Neither is generalization. Tests pin both behaviours.

## Decoding knobs

**Temperature** raises each probability to `1/t`. Below 1 the peaks grow; above 1 everything moves toward uniform. It preserves the *ordering* of tokens — a test checks that — so it changes how much the model hedges, never what it prefers.

**Greedy** takes the likeliest token every time. It needs no seed, and it stops early:

```
the forest
```

The likeliest continuation of the likeliest first word turns out to be `</s>`. Locally optimal, globally short — which is exactly why real systems use beam search rather than greedy decoding.

**Top-k** refuses the smoothed tail. Day 10 measured one context giving 75.4% of its mass to continuations never observed; sampling from the full distribution keeps drawing from that. `k=1` is provably identical to greedy, and a test asserts it.

## Perplexity, and the gap it exposes

```
 n       train    held-out    ratio
 1        35.7        37.8     1.06
 2         4.7         5.2     1.10
 3         4.0         5.1     1.25
 4         4.4         6.3     1.46
 5         4.9         9.4     1.91
```

Held-out perplexity **falls then rises**, bottoming out at n=3. More context is not always better: past some order the model has too little data per context to estimate anything, and smoothing does the rest.

The right-hand column is the important one. The ratio climbs monotonically and never turns around — the model fits data it has seen better than data it has not, by a widening margin. That is **memorization, measured**, and it is the entire reason for holding data out. Both facts are pinned by tests.

Smoothing shows the same shape:

```
alpha=1.0    13.1
alpha=0.1     5.2
alpha=0.01    3.9
alpha=0.0     inf   <- one unseen word makes it infinite
```

Add-one is over three times as perplexed as add-0.01 — Day 10's "what Laplace costs" table, now as a single number. And `alpha=0` is infinite, from **one** out-of-vocabulary word in twenty sentences.

## Perplexity is not quality

The metric's blind spot deserves stating plainly, and there is a test for it:

```python
model = NgramModel(5, alpha=0.0)
model.fit([["the", "cat", "sat", "on", "the", "mat"]])

perplexity(model, [["the", "cat", "sat", "on", "the", "mat"]])   # < 1.5  — excellent
perplexity(model, [["the", "dog", "sat", "on", "the", "mat"]])   # inf    — one word changed
```

A model that has learned exactly nothing scores near-perfectly on the text it memorized and collapses on a single substitution. Perplexity measures **fit to a distribution**, not usefulness, fluency, or truth. It is comparable only between models scored on the *same held-out text with the same vocabulary* — which is also why perplexity numbers quoted across papers are usually not comparable at all.

## Korean

Korean tokens appear in generated output (`지나갔다` turns up at T=1.0 and T=2.0) purely as vocabulary items. With three Korean sentences the model has no Korean structure to learn, and since every particle-bearing form is a separate token, an eojeol-level model would need far more data than an English one to reach the same coverage. **Day 12** is where that finally gets addressed.

## Run it

```bash
# demo (generation by order, smoothing's effect, decoding knobs, perplexity)
python generation.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 03_statistical_language_models/day11_generation

# tests — from inside this folder
python -m unittest
python test_generation.py

# the docstring examples are runnable too
python -m doctest generation.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Every failure in Levels 1–3 has traced back to the same place: the units. Korean stems fragment across particles, unseen words are unscoreable, and the vocabulary grows without bound. **Day 12** stops treating words as atoms and *learns* subword units from the corpus — which is what GPT-style tokenizers do, and the fix promised since Day 1.
