# FAQ

Answers to common questions about the framework.

---

## General

### Why does `llm-eval --help` respond instantly but `llm-eval run` takes longer to start?

The CLI uses **lazy imports** — heavy ML libraries (`torch`, `transformers`, `bert_score`) are only imported when `run` actually executes. The `--help` flag never loads those libraries, so it is near-instantaneous.

### Can I use the framework without an API key?

Yes. The classical and semantic metrics (`bleu`, `rouge`, `bertscore`) require no API key. You only need an OpenAI or Groq key for LLM-based metrics (`faithfulness`, `context_relevancy`, `answer_relevancy`, `judge`).

### What Python version is required?

Python **3.13** or **3.14**. See the `pyproject.toml` for the exact constraint: `python = ">=3.13,<3.15"`.

---

## Metrics

### I forgot what BLEU / ROUGE measures — where is the quick reminder?

See the [Metrics Overview cheat sheet](metrics/overview.md), or jump directly to [BLEU](metrics/bleu.md) or [ROUGE-L](metrics/rouge.md).

### My BLEU scores are all very low. Is something wrong?

Not necessarily. BLEU is strict about surface-form overlap. Common reasons for low scores:

- The candidate uses synonyms or paraphrases not present in the reference.
- The reference is very short (brevity penalty kicks in).
- You are evaluating abstractive summarisation or open-ended generation — BLEU is not appropriate for those tasks.

Try [BERTScore](metrics/bertscore.md) if you care about semantic similarity rather than exact word overlap.

### Why do LLM-based metrics return different scores on repeated runs?

LLMs are non-deterministic by default. The framework does not set `temperature=0`, so scores can vary slightly between runs. For reliable aggregate statistics, run on a large enough dataset (at least 50–100 samples) and compare means rather than individual scores.

### Faithfulness returned 0.0 with "No contexts provided". Why?

The `retrieved_contexts` field was absent or empty in your JSONL dataset. Faithfulness requires context documents. Check your dataset has a `retrieved_contexts` field with a non-empty list.

### Can I customise the judge dimensions (coherence/relevance/safety)?

The `LLMJudgeConfig.dimensions` field is stored but the `MultiDimensionalJudge` prompt currently hardcodes three dimensions. To use different dimensions, subclass `MultiDimensionalJudge` and override the `compute()` method with a customised prompt. See [How-To → Add a Metric](how-to/add-a-metric.md).

---

## Dataset

### What format does the dataset need to be in?

**JSONL** (JSON Lines) — one JSON object per line. Each object should contain `query`, `response`, and optionally `expected_answer` and `retrieved_contexts`. See [Concepts → Data Model](concepts/data-model.md) for the full schema.

### Can I use CSV instead of JSONL?

Not directly — the `Evaluator` calls `pandas.read_json(path, lines=True)`. Convert your CSV to JSONL first:

```python
import pandas as pd
df = pd.read_csv("data.csv")
df.to_json("data.jsonl", orient="records", lines=True)
```

### How large can the dataset be?

The framework uses `ThreadPoolExecutor` with configurable `max_workers`. For LLM-based metrics, performance depends on API rate limits. For classical metrics, datasets of thousands of rows run in seconds.

---

## Configuration

### How do I change the LLM provider mid-evaluation?

Each metric is initialised once at startup. To switch providers, change `llm_judge.provider` in your YAML config and re-run.

### What happens if a metric key in config is not recognised?

Unrecognised keys are silently ignored by the registry lookup:

```python
return [registry[m] for m in config.metrics if m in registry]
```

No error is raised — the unrecognised metric is simply not run. Double-check spelling against the [valid metric keys table](reference/cli.md#valid-metric-keys).

---

## Output

### Where are the results saved?

By default in `results/` (or whatever `output_dir` is set to in your config). Contents: `results.json`, `results.md`, `radar_chart.png`, and one `dist_<Metric>.png` per metric.

### How do I compare two evaluation runs?

Load both `results.json` files into pandas DataFrames and compare the `.describe()` statistics:

```python
import pandas as pd

run1 = pd.read_json("results_v1/results.json")
run2 = pd.read_json("results_v2/results.json")

print("Run 1 means:")
print(run1.mean())
print("Run 2 means:")
print(run2.mean())
print("Difference:")
print(run2.mean() - run1.mean())
```
