# NLP-Lab

![tests](https://github.com/Sophia-Min12/NLP-Lab/actions/workflows/tests.yml/badge.svg)

**One day, one concept, one commit — classical NLP from first principles, written in code, not words.**
**하루에 하나의 개념, 하나의 커밋 — 고전 NLP를 밑바닥부터, 말이 아닌 코드로.**

> Tokenization, vectors, language models, embeddings — the exact primitives every LLM and RAG system is built on. This lab studies them one small unit per day, ending where [RAG-Lab] will begin: a working hybrid semantic search engine.
> <sub>토큰화, 벡터, 언어 모델, 임베딩 — 모든 LLM과 RAG 시스템의 기본 재료들. 이 랩은 매일 작은 단위 하나씩 공부하며, 마지막에는 하이브리드 시맨틱 검색 엔진으로 [RAG-Lab]의 출발점에 도착한다.</sub>

**Environment · 환경**: Python 3.10+ · standard library only through Level 3, NumPy from Level 4 · `pytest` as the test runner.
<sub>레벨 3까지는 표준 라이브러리만, 레벨 4부터 NumPy 사용 · 테스트는 `pytest`.</sub>

---

## 🗺️ Curriculum Roadmap · 커리큘럼 로드맵

**Level 0 · Repo Setup · 저장소 설정**
- [x] **Day 0** — Repo scaffold, CI, and bilingual roadmap <sub>· 저장소 스캐폴드, CI, 이중 언어 로드맵</sub>

**Level 1 · Text Basics · 텍스트 기초**
- [ ] **Day 1** — Rule-based tokenization <sub>· 규칙 기반 토큰화</sub>
- [ ] **Day 2** — Text normalization and English stopwords <sub>· 텍스트 정규화와 영어 불용어</sub>
- [ ] **Day 3** — Word frequencies and Zipf's law <sub>· 단어 빈도와 지프의 법칙</sub>
- [ ] **Day 4** — N-gram extraction <sub>· N-그램 추출</sub>

**Level 2 · Text as Vectors · 텍스트의 벡터 표현**
- [ ] **Day 5** — Bag-of-words and vocabulary building <sub>· 단어 가방 모델과 어휘 사전</sub>
- [ ] **Day 6** — TF-IDF from scratch <sub>· TF-IDF 직접 구현</sub>
- [ ] **Day 7** — Cosine similarity and document ranking <sub>· 코사인 유사도와 문서 랭킹</sub>
- [ ] **Day 8** — Inverted index mini search engine <sub>· 역색인 미니 검색 엔진</sub>

**Level 3 · Statistical Language Models · 통계적 언어 모델**
- [ ] **Day 9** — Naive Bayes text classifier <sub>· 나이브 베이즈 텍스트 분류기</sub>
- [ ] **Day 10** — N-gram language model with Laplace smoothing <sub>· 라플라스 스무딩 N-그램 언어 모델</sub>
- [ ] **Day 11** — Text generation and perplexity <sub>· 텍스트 생성과 퍼플렉서티</sub>
- [ ] **Day 12** — Byte-Pair Encoding tokenizer <sub>· BPE 토크나이저 (GPT 계열 토크나이저의 원리)</sub>

**Level 4 · Embeddings and Semantic Search · 임베딩과 시맨틱 검색**
- [ ] **Day 13** — Co-occurrence matrix and PPMI <sub>· 동시출현 행렬과 PPMI</sub>
- [ ] **Day 14** — SVD word embeddings <sub>· SVD 단어 임베딩</sub>
- [ ] **Day 15** — Nearest neighbors and word analogies <sub>· 최근접 이웃과 단어 유추</sub>
- [ ] **Day 16** — Capstone I: hybrid ranking scorer <sub>· 캡스톤 I: 하이브리드 랭킹 스코어러</sub>
- [ ] **Day 17** — Capstone II: bilingual demo, CLI, and honest writeup <sub>· 캡스톤 II: 이중 언어 데모, CLI, 결과 정리</sub>

---

## 📐 Conventions · 규칙

- Every day folder `NN_topic/dayNN_name/` is **self-contained**: small helpers from earlier days are copied forward with a `# reused from dayNN` comment, so no test ever needs a cross-folder import.
  <sub>모든 일차 폴더는 **자기 완결적** — 이전 날의 헬퍼는 `# reused from dayNN` 주석과 함께 복사해 온다. 테스트가 폴더 간 import에 의존하지 않는다.</sub>
- Every code day ships with a runnable test; `pytest` from the repo root runs everything (this is what CI runs).
  <sub>코드가 있는 날은 반드시 실행 가능한 테스트와 함께 — 저장소 루트에서 `pytest` 하나로 전부 실행 (CI와 동일).</sub>

## 🔗 Sibling labs · 자매 랩

- [ArgMin-Lab](https://github.com/Sophia-Min12/ArgMin-Lab) — search & optimization: document ranking here is just *scoring + argmax* there.

## License

[MIT](LICENSE) © Sophia Min (경01 Min)
