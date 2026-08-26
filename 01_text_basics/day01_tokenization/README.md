# Day 1 · Rule-based Tokenization

> A **token** is the smallest unit a pipeline counts, vectorizes, or embeds. Tokenization is step zero of every NLP/LLM system — get it wrong and everything downstream inherits the mistake.

## Three functions

| Function | Example |
|---|---|
| `tokenize_whitespace(text)` — split on any whitespace run | `"Hello   world"` → `['Hello', 'world']` |
| `tokenize_words(text)` — words group, punctuation separates | `"Hello, world!"` → `['Hello', ',', 'world', '!']` |
| `split_sentences(text)` — split after `.` `!` `?` | `"One. Two!"` → `['One.', 'Two!']` |

## Why the details matter

- **Punctuation**: `"world!"` as one token and `"world"` as another would double the vocabulary for the same word.
- **Curly apostrophe `U+2019`**: phones and Word silently replace `don't` with `don’t`. We normalize it first, so both produce `["don't"]`.
- **Korean note**: word-level tokens keep particles attached — `처리는` ("processing" + topic particle) stays one token; `처리` + `는` is *not* separated. The consequence is documented on Day 2, and a partial fix arrives with **BPE on Day 12**. The sample and test data deliberately mix English and Korean because `\w` in Python's `re` is Unicode-aware.
- **Type contract**: all three functions raise `TypeError` on non-`str` input via one shared `_require_str()` helper.

## Run it

```bash
# demo (token/vocabulary comparison)
python tokenizer.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_text_basics/day01_tokenization

# tests — from inside this folder
python -m unittest
python test_tokenizer.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Same paragraph, different tokenizer → different token count and vocabulary size (run the demo). That gap motivates **Day 2 normalization** and, eventually, the **subword tokenizers (BPE, Day 12)** that GPT-style models actually use.
