# Day 15 · Nearest Neighbours and Word Analogies

> Day 14 turned every word into a short dense vector. This day asks what those vectors know — and finds that the famous answer is worth much less than it looks.

## The functions

| Piece | What it does |
|---|---|
| `WordVectors(vocab, vectors)` | unit-normalized rows, so similarity is a dot product |
| `.most_similar(word, k)` | the `k` nearest words |
| `.analogy(a, b, c, k)` | rank words near `b − a + c` |
| `.nearest_to_input(a, b, c, which)` | **baseline**: nearest neighbour of one input |
| `evaluate_analogies(vectors, cases)` | arithmetic vs the baselines |
| `load_vectors(k, window, min_count)` | the whole Level 4 pipeline in one call |

## Nearest neighbours

```
cat      slept(0.76), river(0.72), dog(0.68), day(0.66)
king     queen(1.00), man(0.92), girl(0.90), boy(0.90)
garden   forest(0.67), bright(0.63), of(0.62), village(0.58)
the      through(0.82), near(0.81), saw(0.81), ran(0.79)
```

This mostly works. `cat` finds `dog`, `garden` finds `forest`, `king` finds `queen` — and `cat`/`dog` never co-occur even once, so this is the Day 7 lexical gap genuinely crossed for the first time in the curriculum.

It is also visibly noisy. `cat` is closer to `slept` and `river` than to `dog`, because those words appear in its signature sentences and the corpus is small. `the` has no meaningful neighbours at all, just the words it happens to sit beside. This is what distributional vectors look like with ~1,500 tokens.

## The famous result

```
ok   man : king :: woman : ?  ->  queen(0.99), girl(0.89), boy(0.89)
ok   king : queen :: man : ?  ->  woman(0.99), girl(0.94), boy(0.93)
ok   woman : queen :: man : ?  ->  king(0.99), girl(0.90), boy(0.89)
```

**100% on all six cases.** A test asserts every one of them.

And it means almost nothing. Here are three independent reasons, each measured rather than asserted.

### 1. Excluding the input words is doing the work

Standard practice is to drop `a`, `b` and `c` from the results. Without it:

```
man : king :: woman : ?  ->  king, queen, man
king : queen :: man : ?  ->  man, woman, girl
woman : queen :: man : ?  ->  queen, king, girl
```

The top answer is **always an input word**. `b − a + c` lands nearest whichever vector contributed most to it, which is exactly what you would expect from adding and subtracting three unit vectors. The exclusion convention is not a neutral formatting choice — it is what makes the task look solvable at all. A test pins that the un-excluded answer is always an input.

### 2. `king` and `queen` are nearly the same vector

```
similarity(king, queen) = 0.999
```

So `queen` is what you get by asking for anything near `king`. The pair does not test the arithmetic; it tests whether the vectors know that kings and queens appear in similar sentences, which is a far weaker claim.

### 3. A baseline that does no arithmetic gets the same answers

This is the one that settles it. Answer each analogy by returning the nearest neighbour of an *input word*, ignoring the arithmetic entirely:

```
analogy                     arithmetic    nn(b)    nn(c)     want
man : king :: woman : ?          queen    queen      boy    queen
king : queen :: man : ?          woman     girl    woman    woman
woman : queen :: man : ?          king     king     girl     king
his : her :: king : ?            queen    heavy    queen    queen
boy : girl :: king : ?           queen      man    queen    queen
king : queen :: boy : ?           girl      man     girl     girl

over 6 cases:
  arithmetic                100%
  nearest neighbour of b     33%
  nearest neighbour of c     67%
  arithmetic agreed with a baseline in 100% of cases
```

**In every single case the arithmetic returned an answer that one of the do-nothing baselines also returned.** It never produced an answer that required the vector arithmetic. A 100% score, and zero demonstrated contribution from the thing being evaluated.

This is not a novel observation — it is essentially the critique Linzen and others made of analogy benchmarks, which is why the field stopped treating analogy accuracy as evidence of much. Reproducing it on six cases and 100 words costs nothing and is worth more than the headline.

### And the corpus was built to contain the answer

```
'the king ruled the kingdom from his throne'
'the queen ruled the kingdom from her throne'
```

`his`/`her` is the *only* systematic difference between those two contexts. The gender relation the analogy recovers was put there on purpose, by me, in Day 10's `build_corpus()`. The arithmetic recovering it demonstrates that the arithmetic works when the structure exists. It says nothing whatsoever about whether such structure emerges from real language at this scale.

## So what is actually established

Worth separating, because it is easy to throw out too much:

- **Distributional similarity works.** `cat` and `dog` never co-occur and come out neighbours. That is real, and it is the Day 7 gap crossed.
- **Compression helps.** Day 14 measured related pairs scoring higher after SVD than before.
- **The analogy result is not evidence of anything** on this corpus, by three independent measurements.
- **Nothing here generalizes to real text** without far more data. ~1,500 tokens is three orders of magnitude short.

A method that is right for the wrong reason is still worth having — but only if you know which reason you are relying on.

## Korean

Absent from this day. The Korean sentences do not survive `min_count=2` in useful numbers, so there are no Korean vectors to find neighbours for. Combining Day 12's BPE pieces with distributional vectors is the obvious next step and is beyond what this corpus supports; **Day 17** states that as an open item rather than pretending otherwise.

## Run it

```bash
# demo (neighbours, analogies, the exclusion effect, the baselines)
python neighbors.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 04_embeddings_and_semantic_search/day15_neighbors

# tests — from inside this folder
python -m unittest
python test_neighbors.py

# the docstring examples are runnable too
python -m doctest neighbors.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Two representations now exist side by side, and each fails where the other works. Level 2's TF-IDF engine is precise when the words line up and scores **exactly zero** when they do not. These embeddings cross that gap but are vague, noisy, and easily fooled. **Day 16** combines them into one ranking instead of choosing — which is the hybrid scorer RAG-Lab starts from.
