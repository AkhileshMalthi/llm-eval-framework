# Metrics Overview — Cheat Sheet

Quick reference for all 7 built-in metrics.  
Click a metric name for the full detail page.

---

## Comparison table

| Metric | Needs reference? | Needs LLM? | Best for | Not good for |
|--------|:---:|:---:|----------|--------------|
| [BLEU](bleu.md) | ✅ | ❌ | Translation quality, exact phrasing | Open-ended generation, diverse valid answers |
| [ROUGE-L](rouge.md) | ✅ | ❌ | Summarisation, recall of key content | Creative tasks, tasks with many valid answers |
| [BERTScore](bertscore.md) | ✅ | ❌ | Semantic similarity with flexible wording | Speed-critical pipelines (requires DistilBERT) |
| [Faithfulness](rag.md#faithfulness) | ❌ | ✅ | Hallucination detection in RAG | Fact-checking without a retrieval context |
| [Context Relevancy](rag.md#context-relevancy) | ❌ | ✅ | Retrieval quality (is the context useful?) | Classical QA with a fixed answer key |
| [Answer Relevancy](rag.md#answer-relevancy) | ❌ | ✅ | Does the response actually answer the question? | Grammatical correctness, stylistic quality |
| [LLM Judge](llm-judge.md) | ❌ | ✅ | Holistic quality (coherence, relevance, safety) | High-throughput pipelines (slow + costs tokens) |

---

## Score interpretation guide

All scores are normalised to **[0, 1]**.

| Range | Interpretation |
|-------|---------------|
| 0.0 – 0.3 | Poor — significant quality issue |
| 0.3 – 0.6 | Fair — acceptable but room for improvement |
| 0.6 – 0.8 | Good — solid quality |
| 0.8 – 1.0 | Excellent — near-ideal output |

!!! warning "Absolute thresholds vary by task"
    A BLEU of 0.4 can be excellent for abstractive summarisation but poor for machine translation.  
    Always compare against a baseline or compare across model versions rather than treating any single number as absolute ground truth.

---

## Choosing a metric for your task

```
Do you have a reference (expected) answer?
    YES → Does exact wording matter?
              YES → Use BLEU (+ ROUGE for recall coverage)
              NO  → Use BERTScore (semantic similarity)
    NO  → Is this a RAG pipeline?
              YES → Use Faithfulness + Context Relevancy + Answer Relevancy
              NO  → Use LLM Judge (coherence / relevance / safety)
```

---

## When to use multiple metrics together

A single metric is almost never sufficient. Recommended combinations:

| Use case | Recommended metrics |
|----------|-------------------|
| RAG pipeline end-to-end | Faithfulness + Context Relevancy + Answer Relevancy + ROUGE-L |
| Summarisation quality | ROUGE-L + BERTScore + LLM Judge (coherence) |
| Translation | BLEU + BERTScore |
| Open-ended chat | LLM Judge (all dimensions) |
| Regression test (fast, no API) | BLEU + ROUGE-L |

---

## Config key reference

| YAML key | Metric |
|----------|--------|
| `bleu` | BLEU |
| `rouge` | ROUGE-L |
| `bertscore` | BERTScore |
| `faithfulness` | Faithfulness |
| `context_relevancy` | Context Relevancy |
| `answer_relevancy` | Answer Relevancy |
| `judge` | LLM-as-Judge (multi-dimensional) |
