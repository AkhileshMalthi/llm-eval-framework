# ROUGE-L

**Recall-Oriented Understudy for Gisting Evaluation** — a family of recall-focused metrics for summarisation evaluation. This framework implements the **ROUGE-L** variant (longest common subsequence).

---

## What it measures

ROUGE-L measures the **longest common subsequence (LCS)** between the candidate response and the reference.  
Unlike BLEU (which requires contiguous n-grams), LCS allows tokens to be non-contiguous — it captures **in-order word overlap** even when the wording is rearranged.

The framework returns the **F-measure** of ROUGE-L, which balances precision and recall:

```
F = (1 + β²) × Precision × Recall / (β² × Precision + Recall)
```

where `β = 1` (equal weight), with stemming enabled.

In plain English: **how much of the reference content is covered by the candidate, in the right order?**

---

## Use ROUGE-L when…

- You are evaluating **summarisation** — you want to know whether the key content of a reference summary is present.
- **Recall matters** more than precision (you care about coverage, not verbosity).
- The candidate may reorder phrases relative to the reference.
- You want a **fast, deterministic** metric with no API calls.
- You are running it alongside BLEU to get both precision (BLEU) and recall (ROUGE-L) perspectives.

---

## Do not use ROUGE-L when…

- You are doing **open-ended QA** where many valid phrasings exist — ROUGE-L still penalises legitimate paraphrases.
- You need to detect **semantic correctness** without surface overlap (use BERTScore instead).
- Your reference is a single sentence and the candidate is much longer — precision component can hide verbosity.
- You are evaluating **creative writing** or tasks with no definitive reference.

---

## Interpreting scores

| Score | Typical interpretation |
|-------|----------------------|
| > 0.5 | Strong recall of reference content; good coverage |
| 0.3 – 0.5 | Moderate recall; main points present but possibly incomplete |
| 0.1 – 0.3 | Weak recall; significant content missing or reworded |
| < 0.1 | Very little overlap with reference |

!!! tip "ROUGE-L vs BLEU together"
    BLEU emphasises **precision** (how much of what the model says is in the reference).  
    ROUGE-L emphasises **recall** (how much of the reference is in what the model says).  
    Use both for a fuller picture.

---

## Failure modes

- **Synonym blindness** — like BLEU, ROUGE-L treats synonyms as misses (mitigated partially by the stemming option).
- **Length insensitivity** — a very long verbose response can get a high recall without being concise.
- **Order sensitivity** — LCS requires tokens in the same order; ROUGE-2 (not implemented here) would catch more reorderings.
- **Single-reference limitation** — multi-reference ROUGE averages over all references for a fairer score.

---

## How this framework computes it

**Implementation:** `src/llm_eval/metrics/classical.py` → `RougeLMetric`

```python
from rouge_score import rouge_scorer

class RougeLMetric(BaseMetric):
    def compute(self, query, response, contexts=None, reference=None):
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, response)
        return MetricResult(name="ROUGE-L", score=scores['rougeL'].fmeasure)
```

Key implementation details:

| Detail | Value |
|--------|-------|
| Library | `rouge_score.rouge_scorer.RougeScorer` |
| Variant | `rougeL` (longest common subsequence) |
| Stemming | `use_stemmer=True` (Porter stemmer — reduces "running"/"runs" to same stem) |
| Score returned | F-measure (`scores['rougeL'].fmeasure`) |
| Score range | 0–1 |
| Reference field | `expected_answer` in the JSONL |

!!! note "`scorer.score(reference, response)` argument order"
    The `rouge_score` library uses `score(target, prediction)` order — so the reference is the first argument and the candidate is the second. This is already correctly handled in the implementation.

---

## Config key

```yaml
metrics:
  - rouge
```

---

## Example

```
Reference:  "Water boils at 100 degrees Celsius at sea level."
Candidate:  "At sea level, water reaches boiling point at 100°C."

LCS: "water", "at", "100", ... (order preserved)
ROUGE-L F ≈ 0.55

(Stemming helps "boiling"/"boils" partially match; "degrees Celsius"/"°C" will not match)
```
