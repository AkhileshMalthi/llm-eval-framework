# Quickstart

This page gets you from zero to a running evaluation in under five minutes.

---

## 1. Prepare your dataset

The framework reads a **JSONL** file where every line is a JSON object.

Required fields:

| Field | Type | Required by |
|-------|------|-------------|
| `query` | `string` | all metrics |
| `response` | `string` | all metrics |
| `expected_answer` | `string` | BLEU, ROUGE-L, BERTScore |
| `retrieved_contexts` | `list[string]` | RAG metrics, faithfulness |

Minimal example (`my_data.jsonl`):

```json
{"query": "What is the capital of France?", "response": "The capital of France is Paris.", "expected_answer": "Paris", "retrieved_contexts": ["France is a country in Western Europe. Its capital is Paris."]}
```

Sample benchmarks are included in `benchmarks/`:

- `rag_benchmark.jsonl` — 28 RAG evaluation samples
- `single_item_benchmark.jsonl` — single-item smoke test

---

## 2. Create a config file

```yaml title="config.yaml"
eval_name: "My First Evaluation"
dataset_path: "benchmarks/rag_benchmark.jsonl"
output_dir: "results"
max_workers: 4

metrics:
  - bleu
  - rouge
  - bertscore
  - faithfulness
  - context_relevancy
  - answer_relevancy
  - judge

llm_judge:
  provider: "groq"          # or "openai"
  model: "llama-3.3-70b-versatile"   # or "gpt-4o-mini"
```

!!! tip "Start small"
    Running just `bleu` and `rouge` requires no API key and is a good first step.

---

## 3. Run via the CLI

```bash
llm-eval run --config config.yaml
```

Override options without editing the YAML:

```bash
# Custom output directory
llm-eval run --config config.yaml --output-dir /tmp/eval-results

# Swap the dataset file
llm-eval run --config config.yaml --benchmark my_data.jsonl

# Use more parallel workers
llm-eval run --config config.yaml --max-workers 8
```

---

## 4. Understand the output

After a successful run, `results/` contains:

```
results/
├── results.json         ← full metric scores for every row (JSON)
├── results.md           ← Markdown summary with mean/min/max per metric
├── radar_chart.png      ← spider-web chart across all metrics
├── dist_BLEU.png        ← score distribution histogram for BLEU
├── dist_ROUGE-L.png
└── ...
```

The CLI also prints a summary table:

```
╭────────────┬────────┬────────┬────────╮
│ Stat       │  BLEU  │ROUGE-L │  ...   │
├────────────┼────────┼────────┼────────┤
│ mean       │ 0.4231 │ 0.5812 │  ...   │
│ min        │ 0.0000 │ 0.1234 │  ...   │
│ max        │ 0.9123 │ 0.9876 │  ...   │
╰────────────┴────────┴────────┴────────╯
```

---

## 5. Python API usage

You can also call the framework directly from Python:

```python
from pathlib import Path
from llm_eval import Evaluator
from llm_eval.metrics.classical import BleuMetric, RougeLMetric
from llm_eval.metrics.semantic import BERTScoreMetric
from llm_eval.reporting.markdown_gen import generate_markdown_report
from llm_eval.reporting.visualizer import generate_radar_chart

# Build metric list
metrics = [BleuMetric(), RougeLMetric(), BERTScoreMetric()]

# Run evaluation
evaluator = Evaluator(metrics=metrics, max_workers=4)
results_df = evaluator.run("benchmarks/rag_benchmark.jsonl")

# Generate artefacts
print(generate_markdown_report(results_df, "My Evaluation"))
generate_radar_chart(results_df, Path("radar_chart.png"))
```

---

## Next steps

- Learn what each metric actually measures → [Metrics Overview](../metrics/overview.md)
- Understand the full data schema → [Concepts → Data Model](../concepts/data-model.md)
- Add your own metric → [How-To → Add a Metric](../how-to/add-a-metric.md)
