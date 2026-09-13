# Day 2 · Text Normalization and English Stopwords

> Day 1 produced tokens. But `"The"`, `"the"` and `"the."` are three different strings, so they become three different vocabulary entries — and every count, vector and index built on top inherits the split. Normalization collapses differences that don't matter. The entire difficulty is deciding which differences don't matter.

## The pipeline

| Step | Function | Example |
|---|---|---|
| Unicode form | `normalize_unicode(text)` | decomposed `한` (3 code points) → composed `한` (1) |
| Whitespace | `normalize_whitespace(text)` | `"  a \t b \n"` → `"a b"` |
| Case | `to_lowercase(text)` | `"THE Cats"` → `"the cats"` |
| Accents *(opt-in)* | `strip_accents(text)` | `"café"` → `"cafe"` |
| Punctuation | `strip_punctuation(tokens)` | `["hi", ","]` → `["hi"]` |
| Stopwords | `remove_stopwords(tokens)` | `["the", "cat"]` → `["cat"]` |

`normalize(text)` runs all of it and returns tokens. Every step is a keyword flag, because none of them is universally correct:

```python
normalize("The cats aren't in THE box.")          # ['cats', "aren't", 'box']
normalize("Café", accents=True)                   # ['cafe']
normalize("The cat!", lowercase=False,
          drop_punctuation=False,
          drop_stopwords=False)                   # ['The', 'cat', '!']
```

## Why the order is what it is

Unicode form first — every later comparison assumes two equal-looking strings really are equal. Then whitespace, then case, then tokenization, then the token-level filters. Punctuation is stripped *after* tokenizing, not before, so `don't` survives while a standalone `,` does not.

## Three things worth knowing

**1. Unicode normalization is not optional for Korean.** The syllable `한` typed on macOS is often stored decomposed (NFD — three jamo code points); the same syllable from a web form arrives composed (NFC — one). They render identically and compare unequal, so a search index built without NFC silently misses half its matches. Note also that `len("한국어")` is 3 composed but 8 decomposed — `한` and `국` become 3 jamo each, `어` only 2, since it has no final consonant.

**2. `casefold()`, not `lower()`.** `"Straße".lower()` is `"straße"`, which never matches `"STRASSE"`. `casefold()` folds it to `"strasse"` and both sides meet. Hangul has no case and passes through untouched.

**3. `strip_accents` has to recompose.** It decomposes to NFD, drops combining marks (Unicode category `Mn`), then returns to NFC. Hangul jamo are category `Lo`, not `Mn` — so Korean survives the mark-dropping step, but only the recompose puts the syllables back together. A regression test pins this.

## The stopword trap

Stopword removal is lossy, and the loss is not always harmless:

```python
normalize("To be or not to be")     # []  ← the whole sentence
normalize("this movie is not good") # ['movie', 'good']  ← now it reads positive
```

`not` is in `STOPWORDS` because standard lists include it. When meaning depends on it, narrow the set instead of editing the constant:

```python
remove_stopwords(tokens, STOPWORDS - {"not"})
```

Rule of thumb: drop stopwords for **topic**-driven tasks (keywords, document similarity, TF-IDF on Day 6). Keep them for anything where syntax carries meaning — sentiment, negation, the language models of Level 3.

The list is hand-written rather than imported, so every entry stays inspectable and the repo keeps its standard-library-only promise.

## Korean: what this day does *not* fix

`처리는` ("processing" + topic particle) is still one token, and `처리를` is another. Normalization cannot separate them — it only collapses surface variants of the *same* string. Korean needs morphological analysis, and the practical fix in this curriculum is subword tokenization on **Day 12 (BPE)**. Meanwhile, the English stopword list correctly leaves Korean tokens alone; there is no Korean equivalent here, because Korean function words are attached to stems rather than standing alone.

## Run it

```bash
# demo (per-stage token/vocabulary counts, the 'not' trap, the NFC/NFD gap)
python normalizer.py

# tests — from the repo root (what CI runs)
pytest
python -m unittest discover -s 01_text_basics/day02_normalization

# tests — from inside this folder
python -m unittest
python test_normalizer.py

# the docstring examples are runnable too
python -m doctest normalizer.py
```

> ⚠ Bare `python -m unittest discover` from the repo **root** silently finds zero tests with this layout — use one of the commands above.

## Where this leads

Run the demo: the same paragraph goes from 34 tokens / 30 types to 19 / 19. That vocabulary is what **Day 3** counts to find Zipf's law, and what **Day 5** turns into a bag-of-words vector. Every decision made here — which case, which punctuation, which stopwords — silently shapes all of it.
