# RAG Metrics

The framework includes three metrics designed specifically for **Retrieval-Augmented Generation (RAG)** pipelines. All three are **LLM-graded** — they call an external LLM (OpenAI or Groq) to evaluate quality.

---

## Overview

| Metric | Question it answers | Required fields |
|--------|--------------------|----|
| [Faithfulness](#faithfulness) | Is the response grounded in the retrieved contexts? | `response`, `retrieved_contexts` |
| [Context Relevancy](#context-relevancy) | Are the retrieved contexts actually useful for the query? | `query`, `retrieved_contexts` |
| [Answer Relevancy](#answer-relevancy) | Does the response actually address the query? | `query`, `response` |

Used together, these three metrics give a full picture of a RAG pipeline's quality.

---

## Faithfulness

### What it measures

Faithfulness checks whether every claim in the **response** is supported by the **retrieved contexts**. A response that invents facts not found in the contexts is "unfaithful" — this is the primary hallucination signal for RAG.

Score: **0 (not faithful) or 1 (faithful)** — binary, graded by the LLM.

### Use when…

- You want to detect **hallucinations** in a RAG pipeline.
- You have a retrieval step and want to know if the generator is staying within its retrieval window.
- You are debugging "why did the model make up a fact?"

### Do not use when…

- You have no `retrieved_contexts` — the metric returns 0.0 with reasoning "No contexts provided".
- You want a **continuous measure of quality** — the binary output is coarse; use Answer Relevancy for that.
- The task has no retrieval step (e.g. pure generation, summarisation of a known document).

### Interpreting scores

Since this is binary, interpretation is direct:

| Score | Meaning |
|-------|---------|
| 1.0 | Response is fully supported by contexts |
| 0.0 | Response contains unsupported claims |

!!! warning "LLM grading is probabilistic"
    The same response may receive different scores on different runs. Aggregate over many samples for reliable averages.

### Implementation

**File:** `src/llm_eval/metrics/rag.py` → `FaithfulnessMetric`

The LLM receives this prompt:

```
Verify if the response is supported by the context.
CONTEXT: {context_str}
RESPONSE: {response}
Output ONLY:
SCORE: [0 or 1]
REASON: [Short explanation]
```

The score is extracted via `extract_score_from_response()` with no normalisation (already 0–1).

### Config key

```yaml
metrics:
  - faithfulness
```

---

## Context Relevancy

### What it measures

Context Relevancy evaluates whether the **retrieved contexts** are actually useful for answering the **query**. A retriever that returns tangentially related documents scores low here.

Score: **0–10 (LLM-graded), normalised to 0–1** by dividing by 10.

### Use when…

- You want to measure **retrieval quality** independently of generation quality.
- You are debugging a RAG pipeline and want to know if the problem is in retrieval or generation.
- You are comparing different chunking strategies or embedding models.

### Do not use when…

- You have no `retrieved_contexts`.
- You want to measure whether the **response is good** (that's Answer Relevancy or Faithfulness).

### Interpreting scores

| Score | Meaning |
|-------|---------|
| > 0.7 | Contexts are highly relevant to the query |
| 0.4 – 0.7 | Contexts are partially relevant |
| < 0.4 | Retrieval is returning off-topic content |

### Implementation

**File:** `src/llm_eval/metrics/rag.py` → `ContextRelevancyMetric`

Prompt:

```
Query: {query}
Context: {context_str}
Rate how useful this context is to answer the query. Score 0-10. Output: SCORE: [val]
```

Score extracted via `extract_score_from_response(raw, max_score=10.0)` → divided by 10.

### Config key

```yaml
metrics:
  - context_relevancy
```

---

## Answer Relevancy

### What it measures

Answer Relevancy evaluates whether the **response** actually addresses the **query**. A response can be factually correct but still be off-topic or fail to answer the question asked.

Score: **0–10 (LLM-graded), normalised to 0–1**.

### Use when…

- You want to know if the model is actually answering the question or going off on a tangent.
- You are measuring **response quality** when no reference answer exists.
- You want a complement to Faithfulness (faithful but irrelevant responses are also bad).

### Do not use when…

- You care about **factual accuracy** vs the source context — use Faithfulness instead.
- You need a **reference-based** measure of correctness.

### Interpreting scores

| Score | Meaning |
|-------|---------|
| > 0.7 | Response clearly addresses the query |
| 0.4 – 0.7 | Partially addresses the query |
| < 0.4 | Response does not adequately answer the question |

### Implementation

**File:** `src/llm_eval/metrics/rag.py` → `AnswerRelevancyMetric`

Prompt:

```
Query: {query}
Response: {response}
Rate how well the response addresses the query. Score 0-10. Output: SCORE: [val]
```

Score extracted via `extract_score_from_response(raw, max_score=10.0)` → divided by 10.

### Config key

```yaml
metrics:
  - answer_relevancy
```

---

## Using all three together

The three RAG metrics diagnose different failure modes:

```
Low Faithfulness       → hallucination (generator ignoring context)
Low Context Relevancy  → retrieval failure (wrong documents retrieved)
Low Answer Relevancy   → query understanding failure (response doesn't address the question)
```

Recommended config for a complete RAG evaluation:

```yaml
metrics:
  - faithfulness
  - context_relevancy
  - answer_relevancy
  - rouge  # add a reference-based metric if you have expected answers
```
