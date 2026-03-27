# BLEU

**Bilingual Evaluation Understudy** — an n-gram precision metric originally designed to evaluate machine translation.

---

## What it measures

BLEU counts how many n-grams (1-gram to 4-gram) in the **candidate** response appear in the **reference** answer, then computes a geometric mean across n-gram orders and applies a *brevity penalty* to punish responses that are shorter than the reference.

In plain English: **how many of the candidate's words/phrases also appear in the reference?**

---

## Use BLEU when…

- You have a **single authoritative reference** answer (e.g. a ground-truth translation or a templated expected output).
- Exact or near-exact phrasing is important (e.g. machine translation, slot-filling, constrained generation).
- You want a **fast, deterministic** metric with no API calls.
- You are doing **regression testing** — detecting whether a new model version produces notably different text.

---

## Do not use BLEU when…

- Multiple valid answers exist (e.g. open-ended QA, creative writing) — BLEU will penalise correct paraphrases.
- The task is **abstractive summarisation** where rewriting is expected and desirable.
- You care about **semantic correctness** rather than surface-form overlap.
- Your reference is very short (brevity penalty becomes extreme).
- You are evaluating **long-form generation** — sentence-level BLEU is unreliable on paragraphs.

---

## Interpreting scores

| Score | Typical interpretation |
|-------|----------------------|
| > 0.6 | High lexical overlap; likely very close to reference |
| 0.3 – 0.6 | Moderate overlap; paraphrased but covering key content |
| 0.1 – 0.3 | Low overlap; semantically related but differently worded |
| < 0.1 | Very different wording; could still be semantically correct |

!!! warning "BLEU is not a proxy for quality"
    A response can score 0 on BLEU and still be a perfectly correct answer — just worded differently from the reference.

---

## Failure modes

- **Synonym problem** — "automobile" vs "car" → 0 n-gram overlap despite identical meaning.
- **Short candidate penalty** — very concise correct answers get penalised by the brevity penalty.
- **Reference gaming** — a response that copies the reference verbatim scores 1.0 but may not be useful.
- **Single reference limitation** — score is unreliable when only one reference is available; multi-reference BLEU is more robust but not implemented in this framework.

---

## How this framework computes it

**Implementation:** `src/llm_eval/metrics/classical.py` → `BleuMetric`

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BleuMetric(BaseMetric):
    def compute(self, query, response, contexts=None, reference=None):
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        candidate_tokens = nltk.word_tokenize(response.lower())
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(reference_tokens, candidate_tokens,
                               smoothing_function=smoothing)
        return MetricResult(name="BLEU", score=score)
```

Key implementation details:

| Detail | Value |
|--------|-------|
| Library | `nltk.translate.bleu_score.sentence_bleu` |
| N-gram weights | Default: equal weights for 1–4 grams |
| Smoothing | `SmoothingFunction().method1` (avoids zero-count crashes for short inputs) |
| Tokenisation | `nltk.word_tokenize` with `.lower()` normalisation |
| Score range | 0–1 (natural BLEU range) |
| Reference field | `expected_answer` in the JSONL |

!!! note "Sentence-level vs corpus-level"
    This implementation computes **sentence-level BLEU** (one score per row). Corpus-level BLEU (aggregated over all examples) would be more statistically reliable but is not the default. The framework summarises by taking the mean across rows.

---

## Config key

```yaml
metrics:
  - bleu
```

---

## Example

```
Reference:  "The capital of France is Paris."
Candidate:  "Paris is the capital city of France."

Tokenised reference:  [the, capital, of, france, is, paris]
Tokenised candidate:  [paris, is, the, capital, city, of, france]

1-gram precision: 6/7 = 0.857
(some n-grams penalised by brevity differences)
Final BLEU ≈ 0.63
```
