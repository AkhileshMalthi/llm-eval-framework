# CLI Reference

The framework ships a **Typer**-based CLI installed as the `llm-eval` command.

---

## Entry point

```
llm-eval = "llm_eval.cli:app"
```

After installation (`pip install -e .` or `poetry install`), the `llm-eval` command is available on your `PATH`.

---

## Commands

### `llm-eval run`

Run an evaluation pipeline from a YAML config file.

```
llm-eval run [OPTIONS]
```

#### Options

| Option | Short | Type | Default | Description |
|--------|-------|------|---------|-------------|
| `--config` | `-c` | `Path` | *(required)* | Path to the YAML config file |
| `--output-dir` | `-o` | `Path` | *(from config)* | Directory to write output artefacts (overrides `output_dir` in config) |
| `--benchmark` | `-b` | `Path` | *(from config)* | JSONL benchmark file to evaluate (overrides `dataset_path` in config) |
| `--max-workers` | `-w` | `int` | *(from config)* | Number of parallel evaluation workers (overrides `max_workers` in config) |
| `--help` | | | | Show this message and exit |

#### Examples

```bash
# Basic run with a config file
llm-eval run --config examples/config.yaml

# Override output directory
llm-eval run --config examples/config.yaml --output-dir /tmp/results

# Use a different dataset file without editing the config
llm-eval run --config examples/config.yaml --benchmark my_data.jsonl

# Increase parallelism
llm-eval run --config examples/config.yaml --max-workers 8

# Use short flags
llm-eval run -c examples/config.yaml -o /tmp/results -b my_data.jsonl -w 8
```

---

## Config file reference

The `--config` option points to a YAML file with the following structure:

```yaml title="config.yaml"
# Required
eval_name: "My Evaluation"         # human-readable label
dataset_path: "data/eval.jsonl"    # path to JSONL benchmark file

# Optional
output_dir: "results"              # default: "results"
max_workers: 4                     # default: 4

# Metrics to run (choose from the keys below)
metrics:
  - bleu
  - rouge
  - bertscore
  - faithfulness
  - context_relevancy
  - answer_relevancy
  - judge

# Required only when using faithfulness, context_relevancy, answer_relevancy, or judge
llm_judge:
  provider: "groq"                         # "openai" or "groq"
  model: "llama-3.3-70b-versatile"         # model name for the provider
  dimensions: ["coherence", "relevance", "safety"]  # used by judge metric
```

### Valid metric keys

| Key | Metric | Requires reference | Requires LLM |
|-----|--------|:-:|:-:|
| `bleu` | BLEU | ✅ | ❌ |
| `rouge` | ROUGE-L | ✅ | ❌ |
| `bertscore` | BERTScore | ✅ | ❌ |
| `faithfulness` | Faithfulness | ❌ | ✅ |
| `context_relevancy` | Context Relevancy | ❌ | ✅ |
| `answer_relevancy` | Answer Relevancy | ❌ | ✅ |
| `judge` | LLM-as-Judge | ❌ | ✅ |

---

## Output artefacts

After a successful run, the output directory contains:

| File | Format | Description |
|------|--------|-------------|
| `results.json` | JSON array | Raw scores for every row, serialised with `orient="records"` |
| `results.md` | Markdown | Summary report with mean/min/max per metric |
| `radar_chart.png` | PNG | Spider-web chart of mean scores across all metrics |
| `dist_<MetricName>.png` | PNG | Score distribution histogram per metric |

---

## Exit codes

| Code | Meaning |
|------|---------|
| `0` | Success |
| `1` | Config file not found, or benchmark file not found |

---

## Environment variables

LLM-based metrics require API keys. The CLI automatically loads a `.env` file from the current directory via `python-dotenv`:

```dotenv
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...
```

These can also be set as shell environment variables before running `llm-eval run`.

---

## Implementation notes

The CLI is implemented with **lazy imports** — all heavy ML libraries (`torch`, `transformers`, `bert_score`) are only imported when `llm-eval run` is actually executed, not when `llm-eval --help` is called. This keeps the `--help` response near-instantaneous.
