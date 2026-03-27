# Data Model

This page describes the input and output schemas used throughout the framework.

---

## Input: JSONL dataset

The evaluator reads a **JSONL** (JSON Lines) file where every line is a valid JSON object.

### Fields

| Field | Type | Required | Used by |
|-------|------|----------|---------|
| `query` | `string` | **Yes** | all metrics |
| `response` | `string` | **Yes** | all metrics |
| `expected_answer` | `string` | For reference-based metrics | BLEU, ROUGE-L, BERTScore |
| `retrieved_contexts` | `list[string]` | For RAG metrics | Faithfulness, Context Relevancy, Answer Relevancy |

!!! note "Missing fields"
    If a metric requires a field that is absent, it raises a `ValueError` at compute time.  
    Classical metrics (`bleu`, `rouge`, `bertscore`) raise `ValueError("Reference text is required…")` when `expected_answer` is missing.

### Example row

```json
{
  "query": "What causes rain?",
  "response": "Rain is caused by water vapour condensing into droplets and falling from clouds.",
  "expected_answer": "Rain forms when water vapour in clouds condenses and becomes heavy enough to fall.",
  "retrieved_contexts": [
    "The water cycle describes how water evaporates, condenses, and falls as precipitation.",
    "Clouds form when moist air rises and cools, causing water vapour to condense around particles."
  ]
}
```

---

## How the evaluator loads data

`Evaluator.run(dataset_path)` calls `pandas.read_json(dataset_path, lines=True)`.  
The resulting DataFrame is iterated row-by-row, extracting the four fields above and passing them to each metric's `compute()` method.

The field mapping inside `_process_row`:

```python
query     = row.get("query", "")
response  = row.get("response", "")
reference = row.get("expected_answer")          # None if absent
contexts  = row.get("retrieved_contexts")       # None if absent
```

---

## Metric input interface

Every metric implements:

```python
def compute(
    self,
    query: str,
    response: str,
    contexts: Optional[list[str]] = None,
    reference: Optional[str] = None,
) -> MetricResult
```

This uniform interface means all metrics can be used polymorphically regardless of whether they need `contexts` or `reference`.

---

## Output: `MetricResult`

Each `compute()` call returns a `MetricResult` Pydantic model:

```python
class MetricResult(BaseModel):
    name: str                          # e.g. "BLEU", "ROUGE-L", "Faithfulness"
    score: float                       # always in [0, 1]
    reasoning: Optional[str] = None   # LLM-based metrics include a rationale string
```

### Score normalisation

All scores are normalised to **[0, 1]** regardless of the raw metric range:

| Metric | Raw range | Normalisation |
|--------|-----------|---------------|
| BLEU | 0–1 | identity |
| ROUGE-L | 0–1 (F-measure) | identity |
| BERTScore | 0–1 (F1) | identity |
| Faithfulness | 0–1 (LLM prompt) | identity |
| Context Relevancy | 0–10 (LLM prompt) | `÷ 10` |
| Answer Relevancy | 0–10 (LLM prompt) | `÷ 10` |
| LLM Judge | 1–5 per dimension | `sum(scores) / (5 × count)` |

---

## Output: results DataFrame

`Evaluator.run()` returns a `pandas.DataFrame` where:

- **rows** = dataset items (one per JSONL line)
- **columns** = metric names (e.g. `"BLEU"`, `"ROUGE-L"`, `"BERTScore"`)
- **values** = `float` scores in [0, 1]

Example (3 rows, 2 metrics):

```
      BLEU    ROUGE-L
0   0.4231     0.5812
1   0.0000     0.1234
2   0.9123     0.9876
```

---

## Output: saved artefacts

After `Evaluator.run()`, the CLI writes to `output_dir/`:

| File | Format | Contents |
|------|--------|---------|
| `results.json` | JSON array | Full DataFrame serialised with `orient="records"` |
| `results.md` | Markdown | Human-readable summary with mean/min/max per metric |
| `radar_chart.png` | PNG | Spider-web chart of mean scores across all metrics |
| `dist_<MetricName>.png` | PNG | Score distribution histogram per metric |

### `results.json` structure

```json
[
  {
    "BLEU": 0.4231,
    "ROUGE-L": 0.5812,
    "BERTScore": 0.7654
  },
  ...
]
```

### `results.md` structure

```markdown
# Evaluation Report: My Evaluation

## Summary Statistics

| Metric  | Mean   | Min    | Max    |
|---------|--------|--------|--------|
| BLEU    | 0.4231 | 0.0000 | 0.9123 |
| ROUGE-L | 0.5812 | 0.1234 | 0.9876 |
```
