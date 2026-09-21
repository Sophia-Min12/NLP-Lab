# NLP-Lab

![tests](https://github.com/Sophia-Min12/NLP-Lab/actions/workflows/tests.yml/badge.svg)

**One day, one concept, one commit — classical NLP from first principles, written in code, not words.**

> Tokenization, vectors, language models, embeddings — the exact primitives every LLM and RAG system is built on. This lab studies them one small unit per day, ending where RAG-Lab will begin: a working hybrid semantic search engine.

**Environment**: Python 3.10+ · standard library only through Level 3, NumPy from Level 4 · `pytest` as the test runner.

---

## ✅ Finished — start here

All 17 days are complete. The end of the road is a bilingual search CLI built entirely from the primitives:

```bash
cd 04_embeddings_and_semantic_search/day17_capstone
python capstone.py evaluate          # recall@3 for four tokenization strategies
python capstone.py compare 문서       # one query through all of them
python capstone.py search 고양이 -e   # search, showing which terms matched
```

```
  analyzer          overall   Korean  English   terms
  word                 33%      11%     100%      61
  subword             100%     100%     100%      85
  char2               100%     100%     100%     199
```

**Word-level retrieval gets 100% on English and 11% on Korean.** That gap is the thread running through the whole curriculum, and closing it is what Day 12's subword tokenizer is for.

Two documents are worth reading before the code:

- **[Day 17 — the honest account](04_embeddings_and_semantic_search/day17_capstone/README.md)** — what works, what works but should not be believed, what does not work, and the five mistakes that taught more than the successes.
- **[Day 12 — Byte-Pair Encoding](03_statistical_language_models/day12_bpe/README.md)** — the fix every earlier day points at.

---

## 🗺️ Curriculum Roadmap

**Level 0 · Repo Setup**
- [x] **Day 0** — Repo scaffold, CI, and curriculum roadmap

**Level 1 · Text Basics**
- [x] **Day 1** — Rule-based tokenization
- [x] **Day 2** — Text normalization and English stopwords
- [x] **Day 3** — Word frequencies and Zipf's law
- [x] **Day 4** — N-gram extraction

**Level 2 · Text as Vectors**
- [x] **Day 5** — Bag-of-words and vocabulary building
- [x] **Day 6** — TF-IDF from scratch
- [x] **Day 7** — Cosine similarity and document ranking
- [x] **Day 8** — Inverted index mini search engine

**Level 3 · Statistical Language Models**
- [x] **Day 9** — Naive Bayes text classifier
- [x] **Day 10** — N-gram language model with Laplace smoothing
- [x] **Day 11** — Text generation and perplexity
- [x] **Day 12** — Byte-Pair Encoding tokenizer (the algorithm family GPT tokenizers use)

**Level 4 · Embeddings and Semantic Search**
- [x] **Day 13** — Co-occurrence matrix and PPMI
- [x] **Day 14** — SVD word embeddings
- [x] **Day 15** — Nearest neighbors and word analogies
- [x] **Day 16** — Capstone I: hybrid ranking scorer
- [x] **Day 17** — Capstone II: bilingual demo, CLI, and honest writeup

---

## 📐 Conventions

- Every day folder `NN_topic/dayNN_name/` is **self-contained**: small helpers from earlier days are copied forward with a `# reused from dayNN` comment, so no test ever needs a cross-folder import.
- Every code day ships with a runnable test; `pytest` from the repo root runs everything (this is what CI runs).
- Test data deliberately includes mixed English–Korean text: handling both scripts correctly is part of the curriculum, and the Korean-specific limits of each technique are documented honestly as they appear.

## 🔗 Sibling labs

- [ArgMin-Lab](https://github.com/Sophia-Min12/ArgMin-Lab) — search & optimization: document ranking here is just *scoring + argmax* there.

## License

[MIT](LICENSE) © Sophia Min
