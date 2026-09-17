# NLP-Lab

![tests](https://github.com/Sophia-Min12/NLP-Lab/actions/workflows/tests.yml/badge.svg)

**One day, one concept, one commit — classical NLP from first principles, written in code, not words.**

> Tokenization, vectors, language models, embeddings — the exact primitives every LLM and RAG system is built on. This lab studies them one small unit per day, ending where RAG-Lab will begin: a working hybrid semantic search engine.

**Environment**: Python 3.10+ · standard library only through Level 3, NumPy from Level 4 · `pytest` as the test runner.

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
- [ ] **Day 8** — Inverted index mini search engine

**Level 3 · Statistical Language Models**
- [ ] **Day 9** — Naive Bayes text classifier
- [ ] **Day 10** — N-gram language model with Laplace smoothing
- [ ] **Day 11** — Text generation and perplexity
- [ ] **Day 12** — Byte-Pair Encoding tokenizer (the algorithm family GPT tokenizers use)

**Level 4 · Embeddings and Semantic Search**
- [ ] **Day 13** — Co-occurrence matrix and PPMI
- [ ] **Day 14** — SVD word embeddings
- [ ] **Day 15** — Nearest neighbors and word analogies
- [ ] **Day 16** — Capstone I: hybrid ranking scorer
- [ ] **Day 17** — Capstone II: bilingual demo, CLI, and honest writeup

---

## 📐 Conventions

- Every day folder `NN_topic/dayNN_name/` is **self-contained**: small helpers from earlier days are copied forward with a `# reused from dayNN` comment, so no test ever needs a cross-folder import.
- Every code day ships with a runnable test; `pytest` from the repo root runs everything (this is what CI runs).
- Test data deliberately includes mixed English–Korean text: handling both scripts correctly is part of the curriculum, and the Korean-specific limits of each technique are documented honestly as they appear.

## 🔗 Sibling labs

- [ArgMin-Lab](https://github.com/Sophia-Min12/ArgMin-Lab) — search & optimization: document ranking here is just *scoring + argmax* there.

## License

[MIT](LICENSE) © Sophia Min
