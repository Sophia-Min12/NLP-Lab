# Day 1 · Rule-based Tokenization · 규칙 기반 토큰화

> A **token** is the smallest unit a pipeline counts, vectorizes, or embeds. Tokenization is step zero of every NLP/LLM system — get it wrong and everything downstream inherits the mistake.
> <sub>**토큰**은 파이프라인이 세고, 벡터화하고, 임베딩하는 최소 단위다. 토큰화는 모든 NLP/LLM 시스템의 0단계 — 여기서 틀리면 이후 전부가 그 실수를 물려받는다.</sub>

## Three functions · 세 가지 함수

| Function | Example |
|---|---|
| `tokenize_whitespace(text)` — split on any whitespace run <sub>· 공백 연속 기준 분리</sub> | `"Hello   world"` → `['Hello', 'world']` |
| `tokenize_words(text)` — words group, punctuation separates <sub>· 단어는 묶고 문장부호는 분리</sub> | `"Hello, world!"` → `['Hello', ',', 'world', '!']` |
| `split_sentences(text)` — split after `.` `!` `?` <sub>· 종결 부호 뒤에서 문장 분리</sub> | `"One. Two!"` → `['One.', 'Two!']` |

## Why the details matter · 디테일이 중요한 이유

- **Punctuation**: `"world!"` as one token and `"world"` as another would double the vocabulary for the same word. <sub>(문장부호가 붙으면 같은 단어가 서로 다른 토큰이 되어 어휘가 불어난다)</sub>
- **Curly apostrophe `U+2019`**: phones and Word silently replace `don't` with `don’t`. We normalize it first, so both produce `["don't"]`. <sub>(휴대폰·워드가 몰래 바꾸는 둥근 아포스트로피를 먼저 정규화한다)</sub>
- **Korean note · 한국어 주의**: word-level tokens keep particles attached — `처리는` stays one token (`처리` + `는` is *not* separated). The consequence is documented on Day 2, and a partial fix arrives with **BPE on Day 12**. <sub>(단어 수준 토큰은 조사가 붙은 채로 남는다 — Day 2에서 그 결과를, Day 12의 BPE에서 부분적 해결을 다룬다)</sub>
- **Type contract**: all three functions raise `TypeError` on non-`str` input via one shared `_require_str()` helper. <sub>(세 함수 모두 동일한 타입 계약을 하나의 헬퍼로 강제한다)</sub>

## Run it · 실행

```bash
# demo (token/vocabulary comparison) · 데모
python tokenizer.py

# tests — from the repo root (what CI runs) · 저장소 루트에서
pytest
python -m unittest discover -s 01_text_basics/day01_tokenization

# tests — from inside this folder · 이 폴더 안에서
python -m unittest
python test_tokenizer.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.
> <sub>루트에서 옵션 없는 `python -m unittest discover`는 이 구조에서 테스트를 하나도 찾지 못한다 — 위 명령을 사용할 것.</sub>

## Where this leads · 다음 단계

Same paragraph, different tokenizer → different token count and vocabulary size (run the demo). That gap motivates **Day 2 normalization** and, eventually, the **subword tokenizers (BPE, Day 12)** that GPT-style models actually use.
<sub>같은 문단도 토크나이저에 따라 토큰 수·어휘 크기가 달라진다(데모 실행). 이 차이가 **Day 2 정규화**와 GPT 계열 모델이 실제로 쓰는 **서브워드 토크나이저(Day 12 BPE)**로 이어진다.</sub>
