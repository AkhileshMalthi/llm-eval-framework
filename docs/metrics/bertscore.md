# BERTScore

**BERTScore** measures semantic similarity between the candidate and reference by comparing their contextual embeddings rather than surface-form n-grams.

---

## What it measures

BERTScore generates token-level embeddings for both the candidate and the reference using a pre-trained transformer (DistilBERT in this framework), then computes **greedy token matching** based on cosine similarity.

It produces three sub-scores:

| Sub-score | Measures |
|-----------|---------|
| **Precision** | How much of the candidate is semantically covered by the reference |
| **Recall** | How much of the reference is semantically covered by the candidate |
| **F1** | Harmonic mean of P and R |

This framework returns the **F1** score.

In plain English: **how semantically similar are the two texts, even if they use different words?**

---

## Use BERTScore when…

- You need **semantic matching** and the wording may legitimately vary.
- You are evaluating tasks with **multiple valid phrasings** (paraphrase detection, abstractive summarisation).
- BLEU/ROUGE are too strict — you care about meaning, not exact words.
- You are comparing **model outputs to human references** for tasks like summarisation or dialogue.

---

## Do not use BERTScore when…

- You need a **fast, lightweight** metric — BERTScore requires loading a transformer model (slow on first call, though cached thereafter).
- You are in a **resource-constrained environment** (embedded, very small RAM).
- You need a metric that is **human-interpretable** step-by-step — the embedding math is opaque compared to BLEU/ROUGE.
- The task requires **factual accuracy** — BERTScore measures linguistic similarity, not factual correctness.

---

## Interpreting scores

BERTScore F1 is bounded [0, 1] but in practice scores tend to be high even for poor outputs (the embedding space is dense).

| Score | Typical interpretation |
|-------|----------------------|
| > 0.9 | Very high semantic similarity |
| 0.85 – 0.9 | Good semantic match; minor differences |
| 0.75 – 0.85 | Moderate similarity; related but divergent |
| < 0.75 | Low similarity; different content or topic |

!!! warning "Scores are not directly comparable across model versions"
    BERTScore values depend on which underlying model is used. The framework uses DistilBERT — if you change to RoBERTa or DeBERTa, the absolute values will shift.

---

## Failure modes

- **Embedding collapse** — if two semantically opposite sentences use similar vocabulary, embeddings may be close.
- **High floor effect** — unrelated sentences often still score 0.7+ because all English uses similar common words.
- **Reference required** — like BLEU/ROUGE, you still need a reference; it cannot assess quality without one.
- **Model dependency** — results depend on the embedding model; change the model, change the scores.

---

## How this framework computes it

**Implementation:** `src/llm_eval/metrics/semantic.py` → `BERTScoreMetric`

Key implementation details:

| Detail | Value |
|--------|-------|
| Library | `bert_score` |
| Model | DistilBERT (`distilbert-base-uncased`) |
| Score returned | F1 |
| Score range | 0–1 |
| Reference field | `expected_answer` in the JSONL |
| Caching | HuggingFace model cached after first run |

---

## Config key

```yaml
metrics:
  - bertscore
```
