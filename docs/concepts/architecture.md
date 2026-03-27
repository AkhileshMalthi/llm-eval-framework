# Architecture

This page explains how the framework is organised and how its main pieces interact.

---

## High-level flow

```
YAML config
    │
    ▼
EvalConfig (Pydantic)          ← validates and types the config
    │
    ▼
CLI / Python API
    │
    ├─ builds list of BaseMetric instances
    │
    ▼
Evaluator
    │  (ThreadPoolExecutor, configurable workers)
    │
    ├─ reads JSONL dataset row by row
    │
    ├─ for each row calls metric.compute(query, response, contexts, reference)
    │
    ▼
pandas DataFrame  (rows = dataset items, columns = metric scores)
    │
    ├─ generate_markdown_report()  → results.md
    ├─ generate_radar_chart()      → radar_chart.png
    └─ generate_score_histograms() → dist_<metric>.png
```

---

## Key components

### `EvalConfig` and `LLMJudgeConfig`

Defined in `src/llm_eval/config.py` using **Pydantic v2**.  
`EvalConfig` is the root model that holds:

- `eval_name` — human-readable label for the run
- `dataset_path` — path to the JSONL file
- `output_dir` — where artefacts are written (default: `results/`)
- `metrics` — list of metric keys (e.g. `["bleu", "rouge", "faithfulness"]`)
- `max_workers` — parallelism (default: `4`)
- `llm_judge` — optional `LLMJudgeConfig` for LLM-backed metrics

`LLMJudgeConfig` holds:

- `provider` — `"openai"` or `"groq"`
- `model` — model name (e.g. `"gpt-4o-mini"`, `"llama-3.3-70b-versatile"`)
- `dimensions` — judge dimensions (default: `["coherence", "relevance", "safety"]`)

---

### `BaseMetric` and the metric registry

All metrics live in `src/llm_eval/metrics/` and inherit from `BaseMetric`:

```python
class BaseMetric(ABC):
    @abstractmethod
    def compute(
        self,
        query: str,
        response: str,
        contexts: Optional[list[str]] = None,
        reference: Optional[str] = None,
    ) -> MetricResult: ...
```

`MetricResult` is a Pydantic model with three fields:

```python
class MetricResult(BaseModel):
    name: str
    score: float          # always normalised to [0, 1]
    reasoning: Optional[str] = None  # for LLM-based metrics
```

The CLI maps metric key strings to instances in `cli.get_metric_instances()`:

| Key | Class | Module |
|-----|-------|--------|
| `bleu` | `BleuMetric` | `metrics.classical` |
| `rouge` | `RougeLMetric` | `metrics.classical` |
| `bertscore` | `BERTScoreMetric` | `metrics.semantic` |
| `faithfulness` | `FaithfulnessMetric` | `metrics.rag` |
| `context_relevancy` | `ContextRelevancyMetric` | `metrics.rag` |
| `answer_relevancy` | `AnswerRelevancyMetric` | `metrics.rag` |
| `judge` | `MultiDimensionalJudge` | `metrics.judge` |

---

### `Evaluator`

Defined in `src/llm_eval/evaluator.py`.

- Accepts a list of `BaseMetric` instances and `max_workers`.
- `run(dataset_path)` loads the JSONL file into a DataFrame, then uses `concurrent.futures.ThreadPoolExecutor` to call `_process_row()` in parallel.
- Returns a `pd.DataFrame` with one column per metric.

---

### `JudgeClient`

Defined in `src/llm_eval/utils/llm_client.py`.

- Wraps the **OpenAI** and **Groq** Python SDKs with a single `ask_judge(prompt)` method.
- Retries automatically (exponential back-off, max 3 attempts) via `tenacity`.
- LLM-based metrics (`FaithfulnessMetric`, `ContextRelevancyMetric`, `AnswerRelevancyMetric`, `MultiDimensionalJudge`) all compose a `JudgeClient`.

---

### Reporting

Two modules in `src/llm_eval/reporting/`:

| Module | Functions | Output |
|--------|-----------|--------|
| `markdown_gen` | `generate_markdown_report(df, eval_name)` | Markdown string (written to `results.md`) |
| `visualizer` | `generate_radar_chart(df, path)`, `generate_score_histograms(df, dir)` | PNG files |

---

## Module map

```
src/
└── llm_eval/
    ├── __init__.py          # public exports: Evaluator, EvalConfig, …
    ├── cli.py               # Typer app + metric registry
    ├── config.py            # EvalConfig, LLMJudgeConfig
    ├── constants.py         # DEFAULT_LLM_PROVIDER = "openai"
    ├── evaluator.py         # Evaluator class
    ├── metrics/
    │   ├── base.py          # BaseMetric, MetricResult
    │   ├── classical.py     # BleuMetric, RougeLMetric
    │   ├── semantic.py      # BERTScoreMetric
    │   ├── rag.py           # FaithfulnessMetric, ContextRelevancyMetric, AnswerRelevancyMetric
    │   └── judge.py         # MultiDimensionalJudge
    ├── reporting/
    │   ├── markdown_gen.py  # Markdown report generator
    │   └── visualizer.py    # Radar charts & histograms
    └── utils/
        ├── llm_client.py    # JudgeClient (OpenAI / Groq)
        └── parsing.py       # extract_json_from_response, extract_score_from_response
```

---

## Design decisions

**Why JSONL for the dataset?**  
JSONL is streamable (no need to load the full file), works with `pandas.read_json(..., lines=True)`, and each row is self-describing.

**Why `ThreadPoolExecutor` and not `asyncio`?**  
The LLM SDKs used (openai, groq) have synchronous and async clients; the synchronous path is simpler and sufficient for evaluation workloads where the bottleneck is network I/O per sample, not Python concurrency.

**Why are all metric scores normalised to [0, 1]?**  
Uniform range makes radar charts meaningful and allows averaging across metric types.

**Why lazy imports in the CLI?**  
ML libraries (`torch`, `transformers`, `bert_score`) take several seconds to import. Lazy imports keep `llm-eval --help` instantaneous.
