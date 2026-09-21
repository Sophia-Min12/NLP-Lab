# Day 17 · Capstone II — Bilingual Search, a CLI, and the Honest Account

> Seventeen days of primitives, assembled into one command-line tool — and used to settle the question the whole curriculum kept deferring: **can any of this find a Korean document by its stem?**

Yes. By two routes. And the route everything pointed at needs an amendment.

## The CLI

```bash
python capstone.py evaluate                    # recall@3 for every analyzer
python capstone.py compare 문서                 # one query through all four
python capstone.py search 고양이 --explain       # search, with matched terms
python capstone.py tokenize "고양이가 잔다"       # how each analyzer splits text
```

The only moving part is the **analyzer** — the thing that turns text into index terms. Everything downstream (Day 5's vocabulary contract, Day 6's IDF, Day 7's cosine) is indifferent to whether a term is a word, a subword or a character bigram. Swapping the units is a one-line change, which is the payoff for having kept them separate all along.

## The result

```
recall@3 over 12 queries (9 Korean, 3 English)

  analyzer          overall   Korean  English   terms
  word                 33%      11%     100%      61
  subword(+</w>)       92%      89%     100%      90
  subword             100%     100%     100%      85
  char2               100%     100%     100%     199
```

**Word-level retrieval gets 100% on English and 11% on Korean.** That single row is what Days 1–11 kept running into, finally measured on the same task with the same code. Every Korean query here is a bare stem, and every Korean document contains that stem only with a particle attached, so exact-term matching has nothing to match.

Subwords close it completely. Character bigrams also close it, at **2.3× the vocabulary** — they are the crude, reliable option, exactly as Day 4 described.

## The amendment: `</w>` is wrong for retrieval

Day 12 introduced the end-of-word marker deliberately and defended it: it stops merges crossing word boundaries and keeps a word-final piece distinct from the same letters mid-word. All true. For retrieval it is a **bug**.

```
query 문서   with </w>  ->  ['문', '서</w>']
doc   문서의  with </w>  ->  ['문서', '의</w>']        shared: nothing

query 문서   without    ->  ['문서']
doc   문서의  without    ->  ['문서', '의']            shared: 문서
```

A bare query word is a *complete word*, so its final piece carries `</w>`. The same stem inside a longer word is a *prefix*, and carries none. The two are therefore segmented under different pair statistics and need not agree — and here they do not.

The visible cost:

```
compare 문서
  subword(+</w>)   0.191  [3] 학생이 학교에서 선생에게 질문을 했고...   via 문
  subword          0.448  [0] 검색 엔진은 질의와 문서의 유사도를...    via 문서
```

With the marker, the query matches the **wrong document**, on the single character `문`. Without it, the right document, on the stem. Five tests pin both halves.

This is the capstone's real lesson, and it is not about Korean: **a design decision that is correct for one task can be exactly wrong for the next one, and nothing warns you.** Day 12 had good reasons and the reasons still hold — for language modelling, where word boundaries matter. Retrieval wants stem matching and would rather not know where words end.

## And "no out-of-vocabulary" has a bill

Day 12 sold this as a pure win. It is a trade:

```python
Engine(DOCUMENTS, Analyzer("word")).search("zzzqqq")      # []
Engine(DOCUMENTS, Analyzer("subword", ...)).search("zzzqqq")
# [(4, 0.123), (7, 0.074)]   matched on: 'q'
```

Because every string encodes, **nothing ever cleanly misses**. `zzzqqq` decomposes to single characters, `q` occurs in "query", and the engine returns a document. Word-level retrieval can say *not found*; subword retrieval cannot, and needs a score threshold where word-level needed none.

The scores are low (0.12), so a threshold works. But the guarantee that Day 12 presented as free turns out to cost the ability to answer "no".

---

# The honest account

## What works, and is real

- **The pipeline composes.** Seventeen days of primitives, each self-contained, assemble into a working bilingual search CLI where swapping the tokenizer is one line. That is the strongest claim here and it is fully supported.
- **Subword units solve the Korean problem.** 11% → 100% on Korean queries, with English unchanged at 100%. The mechanism promised on Day 1 delivers on Day 17, with a caveat nobody mentioned along the way.
- **Distributional similarity crosses the lexical gap.** `cat` and `dog` never co-occur and come out neighbours (Day 13–15). `queen` retrieves a document that only says `king` (Day 16).
- **Hybrid beats either half.** Lexical 67%, semantic 92%, any blend 100% (Day 16), and across a wide `alpha` range rather than one tuned value.

## What works but should not be believed

- **Day 15's analogies score 100% and demonstrate nothing.** A baseline that does no arithmetic at all reproduced *every* answer; `king` and `queen` have similarity 0.999; and without the convention of excluding input words, the top answer is always an input. Three independent measurements, all pointing the same way.
- **Day 16's evaluation is 12 queries over 12 documents**, where recall@3 by chance is 25%. The direction matches what larger studies find. The numbers are worth very little.
- **Day 17's evaluation is 12 queries over 8 documents.** Same caveat, doubled.
- **The embedding corpus is synthetic and its structure was built in by me.** Animals share contexts with animals because the templates say so. The `his`/`her` distinction the analogies recover was placed there deliberately. Day 14 caught the extreme version of this: the first corpus made same-category words *bit-identical*, so every similarity was exactly 1.000 and SVD contributed nothing.

## What does not work

- **Korean embeddings.** The Korean sentences never survive `min_count=2` in useful numbers, so Days 13–15 are English-only in practice. The obvious fix — run the distributional pipeline over BPE pieces rather than words — is the natural next step and is not done here.
- **Real-corpus scale.** Everything in Levels 3–4 runs on ~1,500 tokens. Distributional semantics needs millions. Nothing measured in Level 4 should be assumed to transfer.
- **Korean beyond retrieval.** Subwords fixed *matching*. They were not carried back into the language model (Day 10), the classifier (Day 9), or the embeddings (Day 13). Each of those still has the eojeol problem.

## Mistakes worth keeping

Five things went wrong that were more informative than the successes, each preserved in the day that hit it:

1. **Day 12** — 120 merges rebuilt every word as a single symbol, so the stem was shared with nothing. Over-merging turns BPE back into word-level tokenization, the exact thing it was adopted to escape.
2. **Day 13** — measuring `the`'s demotion as a row mean showed it *higher* than average. A full-row mean is diluted by zeros, and `the` has the fewest zeros of any word.
3. **Day 14** — the synthetic corpus was too clean to demonstrate anything, and the defect was invisible until something depended on words being different.
4. **Day 16** — training word vectors on the twelve searched documents killed the semantic half entirely; and the first evaluation set contained only the cases the semantic half exists to solve.
5. **Day 17** — the end-of-word marker, correct on Day 12, silently broke stem matching.

Every one was found by a measurement disagreeing with an expectation. None would have been caught by the tests alone, because the tests were written to confirm the intended behaviour.

## What I would do next

- Run Days 13–15 over BPE pieces rather than words, and check whether Korean stems get usable vectors.
- Replace add-one smoothing (Day 10) with Kneser-Ney and re-measure perplexity.
- Find a real bilingual corpus — the synthetic one has earned its retirement.
- Carry the subword units back into Days 9–11.

---

## Run it

```bash
# the CLI
python capstone.py evaluate
python capstone.py compare 문서
python capstone.py search 고양이 --explain
python capstone.py tokenize "고양이가 잔다"
python capstone.py --help

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 04_embeddings_and_semantic_search/day17_capstone

# tests — from inside this folder
python -m unittest
python test_capstone.py

# the docstring examples are runnable too
python -m doctest capstone.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

The curriculum ends where **RAG-Lab** begins: a working hybrid retriever, honestly measured, with its limits written down. The next system puts a generator behind it — and everything above about what retrieval can and cannot find becomes the thing that decides what the generator has to work with.
